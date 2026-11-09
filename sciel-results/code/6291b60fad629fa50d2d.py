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

def _compute_coverage(residues, primes, start=1, end=5000):
    """Compute for each integer in [start, end] how many primes cover it."""
    length = end - start + 1
    coverage = np.zeros(length, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        # Find all x in [start, end] such that x % p == r
        # x = r + k*p, so k*p in [start-r, end-r]
        # k in [ceil((start-r)/p), floor((end-r)/p)]
        k_min = (start - r + p - 1) // p if start > r else (start - r) // p
        if k_min < 0:
            k_min = 0
        k_max = (end - r) // p
        if k_max >= k_min:
            indices = (r + np.arange(k_min, k_max + 1) * p) - start
            valid = (indices >= 0) & (indices < length)
            coverage[indices[valid]] += 1
    return coverage

def _find_first_gap(residues, primes, max_check=5000):
    """Find the smallest x >= 1 such that coverage < 2."""
    coverage = _compute_coverage(residues, primes, 1, max_check)
    gaps = np.where(coverage < 2)[0]
    if len(gaps) == 0:
        return max_check + 1
    return gaps[0] + 1

def _score(residues, primes, limit=5000):
    """Returns the length L."""
    gap = _find_first_gap(residues, primes, limit)
    return gap - 1

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Better initialization: try to cover small numbers densely
    # Strategy: for small primes, choose residues that cover many small integers
    # For p=2: r=0 covers evens, r=1 covers odds
    # For p=3: r=0 covers 3,6,9..., r=1 covers 1,4,7..., r=2 covers 2,5,8...
    # We want double coverage, so let's try a greedy approach
    
    best_residues = None
    best_score = -1
    
    # Try multiple random initializations
    for seed in [42, 123, 999, 2024, 7, 11, 13, 17]:
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        score = _score(residues, primes, 5000)
        if score > best_score:
            best_score = score
            best_residues = residues[:]
    
    current_residues = best_residues[:]
    current_score = best_score
    
    # Local search with efficient evaluation
    stagnant_count = 0
    stagnation_threshold = 200
    
    while time.time() < deadline:
        # Find first gap
        gap = _find_first_gap(current_residues, primes, 5000)
        
        if gap > 5000:
            # We have at least 5000 coverage
            # Try random moves to see if we can extend further
            idx = random.randint(0, n_primes - 1)
            old_r = current_residues[idx]
            new_r = random.randint(0, primes[idx] - 1)
            current_residues[idx] = new_r
            new_score = _score(current_residues, primes, 5000)
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues[:]
                stagnant_count = 0
            else:
                current_residues[idx] = old_r
                stagnant_count += 1
            
            if stagnant_count > stagnation_threshold:
                stagnant_count = 0
                current_residues = best_residues[:]
                current_score = best_score
            continue
        
        # gap <= 5000, need to fix it
        # Find primes that could cover `gap`
        target_r = gap % primes  # vectorized: for each prime, what residue would cover gap
        
        # Candidates are indices where current_residues[i] != target_r[i]
        candidates = np.where(np.array(current_residues) != target_r)[0]
        
        if len(candidates) == 0:
            # All primes already cover gap, but coverage < 2? Shouldn't happen
            break
        
        # Try a few random candidates
        improved = False
        num_to_try = min(30, len(candidates))
        selected = np.random.choice(candidates, size=num_to_try, replace=False)
        
        for idx in selected:
            idx = int(idx)
            old_r = current_residues[idx]
            current_residues[idx] = int(target_r[idx])
            new_score = _score(current_residues, primes, 5000)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
                break
            else:
                current_residues[idx] = old_r
        
        if not improved:
            # Try random perturbation
            idx = random.randint(0, n_primes - 1)
            old_r = current_residues[idx]
            new_r = random.randint(0, primes[idx] - 1)
            current_residues[idx] = new_r
            new_score = _score(current_residues, primes, 5000)
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
            else:
                current_residues[idx] = old_r
                stagnant_count += 1
            
            if stagnant_count > stagnation_threshold:
                stagnant_count = 0
                current_residues = best_residues[:]
                current_score = best_score
    
    # Final evaluation with higher limit
    final_gap = _find_first_gap(best_residues, primes, 100000)
    
    return best_residues
