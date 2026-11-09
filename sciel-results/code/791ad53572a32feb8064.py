"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from math import isqrt

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return np.where(sieve)[0]

def _construct_with_threshold(primes, threshold, use_least_populated, random_seed=None):
    """Try a specific sieve threshold and residue strategy. Returns the resulting set size."""
    S_mask = np.ones(N + 1, dtype=bool)
    rng = np.random.default_rng(random_seed)
    
    # Sieve small primes up to threshold
    small_primes = primes[primes <= threshold]
    for p in small_primes:
        if use_least_populated:
            # Find least populated residue mod p in current S
            S_indices = np.where(S_mask)[0]
            residues = S_indices % p
            counts = np.bincount(residues, minlength=p)
            
            # Randomized tie-breaking
            min_count = np.min(counts)
            candidates = np.where(counts == min_count)[0]
            if len(candidates) > 1 and random_seed is not None:
                best_r = int(rng.choice(candidates))
            else:
                best_r = int(candidates[0])
                
            to_remove = S_indices[residues == best_r]
            S_mask[to_remove] = False
        else:
            # Remove residue 0
            multiples = np.arange(0, N + 1, p)
            S_mask[multiples] = False
    
    remaining_primes = primes[primes > threshold]
    
    changed = True
    while changed:
        S_indices = np.where(S_mask)[0]
        if len(S_indices) == 0:
            break
        
        candidates = remaining_primes[remaining_primes <= len(S_indices)]
        
        bad_p = None
        min_removal_count = float('inf')
        best_r = -1
        best_p_val = -1
        best_candidates = None
        
        for p in candidates:
            residues = S_indices % p
            # Optimization: Check if all residues are present efficiently
            # unique_res = np.unique(residues) # Expensive
            # if len(unique_res) == p:
            
            counts = np.bincount(residues, minlength=p)
            if np.min(counts) > 0:
                min_count = int(np.min(counts))
                if min_count < min_removal_count:
                    min_removal_count = min_count
                    # Find all residues with min count for tie-breaking
                    best_r_candidates = np.where(counts == min_count)[0]
                    best_p_val = p
                    bad_p = p
                    best_candidates = best_r_candidates
                    # Note: We don't break here because we want to find the global min removal count
                    # across all bad primes to choose the "best" prime to fix.
        
        if bad_p is not None:
            # Randomized tie-breaking for residue selection within the best prime
            if len(best_candidates) > 1 and random_seed is not None:
                best_r = int(rng.choice(best_candidates))
            else:
                best_r = int(best_candidates[0])
                
            to_remove_mask = (S_indices % best_p_val) == best_r
            values_to_remove = S_indices[to_remove_mask]
            S_mask[values_to_remove] = False
            changed = True
        else:
            changed = False
    
    return np.where(S_mask)[0].tolist()

def construct():
    start_time = time.time()
    deadline = 295.0  # Safety margin
    
    primes = _primes(N)
    
    # Try multiple thresholds and strategies
    best_result = None
    best_size = 0
    
    # First pass: Deterministic baseline
    thresholds = [50, 75, 100, 125, 150, 200]
    strategies = [False, True]  # False = residue 0, True = least populated
    
    for threshold in thresholds:
        for use_least in strategies:
            if time.time() - start_time > deadline:
                return best_result if best_result else list(range(N + 1))
            
            result = _construct_with_threshold(primes, threshold, use_least, random_seed=None)
            size = len(result)
            if size > best_size:
                best_size = size
                best_result = result
                
    # Second pass: Randomized exploration
    # Use the best threshold/strategy combination found, or explore a mix
    # We will run randomized trials on the most promising settings
    # Based on previous runs, "least populated" with higher thresholds often works well.
    
    # Define a set of (threshold, use_least) pairs to explore randomly
    exploration_configs = [
        (100, True), (150, True), (200, True), 
        (75, True), (125, True),
        (100, False), (150, False)
    ]
    
    # Shuffle configs to vary which ones get more trials
    rng_master = np.random.default_rng(42)
    
    # Estimate time per run to determine how many random trials we can afford
    # Rough estimate: 10-50ms per run depending on S size. 
    # Let's aim for ~50-100 random runs if fast, or fewer if slow.
    
    max_random_trials = 100
    trials_run = 0
    
    while trials_run < max_random_trials:
        if time.time() - start_time > deadline - 5.0: # Leave 5s margin
            break
            
        config = exploration_configs[trials_run % len(exploration_configs)]
        threshold, use_least = config
        
        # Unique seed for each trial
        seed = 1000 + trials_run
        
        result = _construct_with_threshold(primes, threshold, use_least, random_seed=seed)
        size = len(result)
        
        if size > best_size:
            best_size = size
            best_result = result
            
        trials_run += 1
        
        # If we are getting consistently worse results and time is running out, stop early?
        # No, keep going until deadline or max trials.
        
    if best_result is None:
        # Fallback: just return all numbers (shouldn't happen)
        best_result = list(range(N + 1))
    
    return best_result

# EVOLVE-BLOCK-END
