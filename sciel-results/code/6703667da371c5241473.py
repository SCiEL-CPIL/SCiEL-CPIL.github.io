"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _compute_counts(residues, primes, max_check=2000):
    """Compute coverage counts for x=1..max_check."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    return counts

def _find_L(counts):
    """Find largest L such that all x in 1..L have count >= 2."""
    below = np.where(counts[1:] < 2)[0]
    if len(below) == 0:
        return len(counts) - 1
    return below[0]

def _score_full(residues, primes, max_check=2000):
    """Full score computation."""
    counts = _compute_counts(residues, primes, max_check)
    return _find_L(counts)

def _update_counts(counts, old_r, new_r, p, max_check=2000):
    """Update counts when changing residue for prime p from old_r to new_r."""
    old_start = old_r if old_r > 0 else p
    if old_start <= max_check:
        counts[old_start::p] -= 1
    new_start = new_r if new_r > 0 else p
    if new_start <= max_check:
        counts[new_start::p] += 1

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    max_check = 2000
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # === PHASE 1: Coverage-driven greedy initialization ===
    # Goal: find residues such that positions 1..target are each covered >= 2 times
    # Process positions in order; for each under-covered position, adjust residues
    
    # Start with random initialization
    np.random.seed(42)
    residues = np.array([np.random.randint(0, p) for p in primes], dtype=np.int32)
    
    # Compute initial counts
    counts = _compute_counts(residues, primes, max_check)
    current_L = _find_L(counts)
    
    # Greedy: for each position from 1 to max_check, if count < 2, try to fix it
    # by changing the residue of a prime that could cover it
    for x in range(1, min(max_check + 1, 500)):
        if (time.time() - start_time) > deadline:
            break
        if counts[x] >= 2:
            continue
        
        # Find primes p where x mod p could be a residue
        # For each prime, check if changing its residue to x % p would help
        best_gain = 0
        best_idx = -1
        best_new_r = residues[0]
        
        for i, p in enumerate(primes):
            if p > max_check:
                break
            new_r = x % p
            if new_r == residues[i]:
                continue
            
            # Simulate: what's the net gain?
            # We need to check how many positions improve vs worsen
            # Quick estimate: count positions in 1..max_check affected
            old_start = residues[i] if residues[i] > 0 else p
            new_start = new_r if new_r > 0 else p
            
            # Count positions that go from <2 to >=2 (gain) and >=2 to <2 (loss)
            # This is expensive, so use a heuristic: just check if it helps x and nearby
            gain = 0
            for dx in range(-5, 6):
                pos = x + dx
                if 1 <= pos <= max_check:
                    # Would this position be covered by the new residue?
                    covered_new = (pos % p == new_r) or (new_r == 0 and pos % p == 0)
                    covered_old = (pos % p == residues[i]) or (residues[i] == 0 and pos % p == 0)
                    if covered_new and not covered_old:
                        if counts[pos] < 2:
                            gain += 1
                    elif not covered_new and covered_old:
                        if counts[pos] <= 2:
                            gain -= 2  # Penalty for losing coverage
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_new_r = new_r
        
        if best_idx >= 0 and best_gain > 0:
            _update_counts(counts, residues[best_idx], best_new_r, primes[best_idx], max_check)
            residues[best_idx] = best_new_r
            current_L = _find_L(counts)
    
    # === PHASE 2: Simulated annealing ===
    best_residues = residues.copy()
    best_counts = counts.copy()
    best_L = current_L
    
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling = 0.995
    max_iters = 50000
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if (time.time() - start_time) > deadline:
            break
        
        # Pick a random prime index
        i = rng.randint(n)
        p = primes[i]
        old_r = best_residues[i]
        
        # Generate new residue: either random or targeted at bottleneck
        if rng.random() < 0.3 and best_L < max_check:
            # Target the frontier
            target_pos = best_L + 1
            new_r = target_pos % p
        else:
            new_r = rng.randint(0, p)
        
        if new_r == old_r:
            continue
        
        # Apply change
        _update_counts(best_counts, old_r, new_r, p, max_check)
        new_L = _find_L(best_counts)
        
        # Acceptance criterion
        if new_L >= best_L or rng.random() < np.exp((new_L - best_L) / max(T, 1e-10)):
            best_residues[i] = new_r
            best_L = new_L
        else:
            # Revert
            _update_counts(best_counts, new_r, old_r, p, max_check)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # === PHASE 3: Final local refinement ===
    improved = True
    no_improve = 0
    max_no_improve = 20
    
    while improved and no_improve < max_no_improve and (time.time() - start_time) < deadline:
        improved = False
        no_improve += 1
        
        for i in range(n):
            if (time.time() - start_time) > deadline:
                break
            
            p = primes[i]
            current_r = best_residues[i]
            
            # Generate candidates focused on bottleneck
            candidates = set()
            if best_L < max_check:
                frontier = best_L + 1
                for delta in range(-3, 4):
                    pos = frontier + delta
                    if 1 <= pos <= max_check:
                        candidates.add(pos % p)
            
            # Add some random candidates
            for _ in range(3):
                candidates.add(rng.randint(0, p))
            
            best_candidate = current_r
            best_candidate_L = best_L
            
            for new_r in candidates:
                if new_r == current_r:
                    continue
                
                counts_copy = best_counts.copy()
                _update_counts(counts_copy, current_r, new_r, p, max_check)
                new_L = _find_L(counts_copy)
                
                if new_L > best_candidate_L:
                    best_candidate = new_r
                    best_candidate_L = new_L
            
            if best_candidate != current_r:
                _update_counts(best_counts, current_r, best_candidate, p, max_check)
                best_residues[i] = best_candidate
                best_L = best_candidate_L
                improved = True
                no_improve = 0
    
    # Extend check if we've reached 1000
    if best_L >= 1000:
        extended_max = 5000
        counts_ext = _compute_counts(best_residues, primes, extended_max)
        ext_L = _find_L(counts_ext)
        best_L = ext_L
    
    return list(best_residues)
# EVOLVE-BLOCK-END
