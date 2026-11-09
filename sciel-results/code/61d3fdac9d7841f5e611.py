import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def _compute_coverage(residues, primes, M):
    """Compute coverage count for each integer 1..M. Returns array of length M where index i-1 is coverage of i."""
    cov = np.zeros(M, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        # x in [1, M] such that x % p == r
        # x = r + k*p, so x >= 1 means k >= ceil((1-r)/p)
        # x <= M means k <= floor((M-r)/p)
        # The first x >= 1 with x % p == r:
        if r == 0:
            first = p  # smallest positive multiple of p
        else:
            first = r
        if first > M:
            continue
        # All x = first, first+p, first+2p, ... <= M
        # In 0-indexed array (index = x-1): indices are first-1, first+p-1, ...
        indices = np.arange(first - 1, M, p)
        cov[indices] += 1
    return cov

def _find_first_gap_from_cov(cov, M):
    """Find smallest x in 1..M with cov[x-1] < 2. Returns x or M+1."""
    # Find first index where cov < 2
    below = np.where(cov < 2)[0]
    if len(below) == 0:
        return M + 1
    return below[0] + 1

def _score(residues, primes, M):
    cov = _compute_coverage(residues, primes, M)
    gap = _find_first_gap_from_cov(cov, M)
    return gap - 1

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Choose a working window size for search
    # We want to be able to evaluate quickly. M=50000 gives coverage array of 50000 ints.
    # Each _compute_coverage is O(M * sum(1/p)) which is manageable.
    M_search = 50000
    
    # Initialize with a good heuristic:
    # For small primes, set residues to cover small numbers densely.
    # Strategy: for prime p, set r_p = 0 if p <= 10, else random.
    # Actually, let's try: r_p = 0 for all p. This covers multiples of each p.
    # Then x=1 is covered by no prime (1%p != 0 for p>1), so that's bad.
    # Better: r_p = 1 for all p? Then x=1 is covered by all 168 primes. x=2 is covered by primes p where 2%p==1, i.e., p=... 2%p=1 only if p divides 1, impossible. So x=2 covered by 0 primes. Bad.
    # Let's just use random init with multiple seeds.
    
    best_residues = None
    best_score = -1
    
    for seed in [42, 123, 999, 2024, 7777]:
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        score = _score(residues, primes, M_search)
        if score > best_score:
            best_score = score
            best_residues = residues[:]
    
    current_residues = best_residues[:]
    current_score = best_score
    
    # Conflict-driven local repair
    stagnant_count = 0
    stagnation_threshold = 200
    no_improve_count = 0
    
    while time.time() < deadline:
        # Compute coverage
        cov = _compute_coverage(current_residues, primes, M_search)
        gap = _find_first_gap_from_cov(cov, M_search)
        current_score = gap - 1
        
        if gap > M_search:
            # We've covered the entire search window. Try to extend or just stop.
            # For now, try random perturbations to see if we can do better.
            idx = random.randint(0, n_primes - 1)
            old_r = current_residues[idx]
            new_r = random.randint(0, primes[idx] - 1)
            current_residues[idx] = new_r
            new_cov = _compute_coverage(current_residues, primes, M_search)
            new_gap = _find_first_gap_from_cov(new_cov, M_search)
            new_score = new_gap - 1
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues[:]
                stagnant_count = 0
                no_improve_count = 0
            else:
                current_residues[idx] = old_r
                stagnant_count += 1
                no_improve_count += 1
            if stagnant_count > stagnation_threshold:
                stagnant_count = 0
                current_residues = best_residues[:]
            continue
        
        # Find all positions with coverage < 2 in [1, gap]
        # Actually, gap is the first such position. But there might be others.
        # Let's find all gaps in [1, M_search]
        below = np.where(cov < 2)[0]
        if len(below) == 0:
            # No gaps, should have been caught above
            continue
        
        # Focus on the first gap
        gap_pos = gap  # 1-indexed
        gap_idx = gap_pos - 1  # 0-indexed
        
        # Identify primes that CAN cover gap_pos
        # These are primes p where current_residues[i] != gap_pos % p
        candidates = []
        for i, p in enumerate(primes):
            target_r = gap_pos % p
            if current_residues[i] != target_r:
                candidates.append((i, target_r))
        
        if not candidates:
            # All primes already cover gap_pos, but coverage < 2? Impossible with 168 primes.
            break
        
        # Try to fix by changing one residue
        random.shuffle(candidates)
        
        improved = False
        # Try a subset of candidates
        num_to_try = min(30, len(candidates))
        for idx, target_r in candidates[:num_to_try]:
            old_r = current_residues[idx]
            current_residues[idx] = target_r
            new_cov = _compute_coverage(current_residues, primes, M_search)
            new_gap = _find_first_gap_from_cov(new_cov, M_search)
            new_score = new_gap - 1
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
                    no_improve_count = 0
                break
            else:
                current_residues[idx] = old_r
        
        if not improved:
            # Try random perturbation
            idx = random.randint(0, n_primes - 1)
            old_r = current_residues[idx]
            new_r = random.randint(0, primes[idx] - 1)
            current_residues[idx] = new_r
            new_cov = _compute_coverage(current_residues, primes, M_search)
            new_gap = _find_first_gap_from_cov(new_cov, M_search)
            new_score = new_gap - 1
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
                    no_improve_count = 0
            else:
                if random.random() < 0.03:
                    current_score = new_score
                else:
                    current_residues[idx] = old_r
                stagnant_count += 1
                no_improve_count += 1
            
            if stagnant_count > stagnation_threshold:
                stagnant_count = 0
                current_residues = best_residues[:]
                current_score = best_score
    
    # Final evaluation with full 100000 cap
    final_cov = _compute_coverage(best_residues, primes, 100000)
    final_gap = _find_first_gap_from_cov(final_cov, 100000)
    # Score is min(final_gap - 1, 100000)
    
    return best_residues
