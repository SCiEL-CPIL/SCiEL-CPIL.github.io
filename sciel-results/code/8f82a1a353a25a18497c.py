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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Precompute x % p for x in 1..MAX_X, for each prime
    # We'll use a chunked approach to limit memory
    MAX_X = 5000
    # coverage[x] = number of primes p where x % p == r_p
    # Vectorized: for each prime p, the residues are 0, p, 2p, ... 
    # x satisfies x % p == r_p iff x is in {r_p, r_p+p, r_p+2p, ...}
    
    def compute_coverage(residues, max_x):
        """Return array cov[1..max_x] where cov[x] = count of primes covering x."""
        cov = np.zeros(max_x + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            # positions: r, r+p, r+2p, ... <= max_x
            # But we need x >= 1, so if r == 0, positions are p, 2p, ...
            # Actually x % p == r means x = r + kp for k >= 0, but x >= 1
            # If r == 0: x = p, 2p, 3p, ... (x=0 not in range)
            # If r > 0: x = r, r+p, r+2p, ...
            if r == 0:
                start = p
            else:
                start = r
            if start > max_x:
                continue
            # All positions from start to max_x step p
            positions = np.arange(start, max_x + 1, p)
            cov[positions] += 1
        return cov
    
    def find_first_gap(cov, max_x):
        """Find smallest x in 1..max_x where cov[x] < 2. Return x or max_x+1."""
        # cov[0] is unused
        bad = np.where(cov[1:max_x+1] < 2)[0]
        if len(bad) == 0:
            return max_x + 1
        return bad[0] + 1
    
    def score_with_cov(residues, max_x):
        cov = compute_coverage(residues, max_x)
        gap = find_first_gap(cov, max_x)
        return gap - 1
    
    # Initialize: try several random starts and pick best
    best_residues = None
    best_score = -1
    best_cov = None
    best_max_x = 2000
    
    for seed in [42, 123, 999, 2024, 7, 13, 99, 314]:
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        cov = compute_coverage(residues, 2000)
        gap = find_first_gap(cov, 2000)
        score = gap - 1
        if score > best_score:
            best_score = score
            best_residues = residues[:]
            best_cov = cov
            best_max_x = 2000
    
    current_residues = best_residues[:]
    current_score = best_score
    current_cov = best_cov
    
    # Local search: repair-based hill climbing
    stagnant_count = 0
    stagnation_threshold = 300
    iteration = 0
    max_iter = 50000
    
    while iteration < max_iter:
        iteration += 1
        if time.time() > deadline:
            break
        
        # Find first gap in current coverage
        gap = find_first_gap(current_cov, best_max_x)
        
        if gap > best_max_x:
            # We passed the current max. Try to extend.
            new_max = min(best_max_x * 2, 100000)
            new_cov = compute_coverage(current_residues, new_max)
            new_gap = find_first_gap(new_cov, new_max)
            new_score = new_gap - 1
            if new_score > current_score:
                current_score = new_score
                current_cov = new_cov
                best_max_x = new_max
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
            continue
        
        # gap <= best_max_x. We need to fix it.
        # The gap is at position `gap`. We need to increase coverage at `gap`.
        # Currently cov[gap] < 2. We need to change some r_p so that gap % p == r_p.
        # Candidates: primes p where current r_p != gap % p.
        
        gap_val = gap
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            target_r = gap_val % p
            if current_residues[i] != target_r:
                # Check if this prime currently covers gap (it doesn't, since r_p != gap%p)
                # Changing r_p to target_r will add 1 to coverage at gap.
                # But it may remove coverage at other positions.
                candidates.append((i, target_r))
        
        if not candidates:
            # All primes already cover gap, but cov[gap] < 2? Impossible with 168 primes.
            # Unless my logic is wrong. Break.
            break
        
        # Try candidates. Prefer primes with larger p (they affect fewer positions).
        # Sort by p descending so we try large primes first (less disruptive).
        candidates.sort(key=lambda x: -primes[x[0]])
        
        improved = False
        num_to_try = min(len(candidates), 15)
        
        for idx, target_r in candidates[:num_to_try]:
            old_r = current_residues[idx]
            current_residues[idx] = target_r
            
            # Update coverage efficiently: 
            # Old residue old_r covered positions {old_r, old_r+p, ...}
            # New residue target_r covers positions {target_r, target_r+p, ...}
            p = primes[idx]
            
            # Remove old coverage
            if old_r == 0:
                old_positions = np.arange(p, best_max_x + 1, p)
            else:
                old_positions = np.arange(old_r, best_max_x + 1, p)
            current_cov[old_positions] -= 1
            
            # Add new coverage
            if target_r == 0:
                new_positions = np.arange(p, best_max_x + 1, p)
            else:
                new_positions = np.arange(target_r, best_max_x + 1, p)
            current_cov[new_positions] += 1
            
            # Find new first gap
            new_gap = find_first_gap(current_cov, best_max_x)
            new_score = new_gap - 1
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
                break
            else:
                # Revert
                current_cov[old_positions] += 1
                current_cov[new_positions] -= 1
                current_residues[idx] = old_r
        
        if not improved:
            # No single repair worked. Try a random perturbation.
            idx = random.randint(0, n_primes - 1)
            old_r = current_residues[idx]
            new_r = random.randint(0, primes[idx] - 1)
            if new_r == old_r:
                continue
            current_residues[idx] = new_r
            p = primes[idx]
            
            if old_r == 0:
                old_positions = np.arange(p, best_max_x + 1, p)
            else:
                old_positions = np.arange(old_r, best_max_x + 1, p)
            current_cov[old_positions] -= 1
            
            if new_r == 0:
                new_positions = np.arange(p, best_max_x + 1, p)
            else:
                new_positions = np.arange(new_r, best_max_x + 1, p)
            current_cov[new_positions] += 1
            
            new_gap = find_first_gap(current_cov, best_max_x)
            new_score = new_gap - 1
            
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
            else:
                # Revert
                current_cov[old_positions] += 1
                current_cov[new_positions] -= 1
                current_residues[idx] = old_r
                stagnant_count += 1
            
            # If stagnant too long, restart from best
            if stagnant_count > stagnation_threshold:
                stagnant_count = 0
                current_residues = best_residues[:]
                current_score = best_score
                current_cov = compute_coverage(current_residues, best_max_x)
        
        # Occasionally try to extend max_x if we've been stable
        if stagnant_count == 0 and best_max_x < 100000 and iteration % 500 == 0:
            new_max = min(best_max_x * 2, 100000)
            new_cov = compute_coverage(current_residues, new_max)
            new_gap = find_first_gap(new_cov, new_max)
            new_score = new_gap - 1
            if new_score >= current_score:
                current_score = new_score
                current_cov = new_cov
                best_max_x = new_max
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
    
    # Final: verify best_residues with a larger check
    final_max = 100000
    final_cov = compute_coverage(best_residues, final_max)
    final_gap = find_first_gap(final_cov, final_max)
    final_score = final_gap - 1
    
    # If final score is better than what we tracked, great. Otherwise use best_residues.
    # best_residues should already be optimal for the range we searched.
    
    return best_residues
