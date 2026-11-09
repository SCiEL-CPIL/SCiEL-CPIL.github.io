"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from math import isqrt, log

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
            counts = np.bincount(residues, minlength=p)
            if np.min(counts) > 0:
                min_count = int(np.min(counts))
                if min_count < min_removal_count:
                    min_removal_count = min_count
                    best_r_candidates = np.where(counts == min_count)[0]
                    best_p_val = p
                    bad_p = p
                    best_candidates = best_r_candidates
        
        if bad_p is not None:
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

def _compute_robustness(S_mask, primes):
    """Compute robustness: min over primes p<=|S| of min residue count."""
    S_indices = np.where(S_mask)[0]
    n = len(S_indices)
    if n == 0:
        return 0
    
    robustness = float('inf')
    for p in primes:
        if p > n:
            break
        residues = S_indices % p
        counts = np.bincount(residues, minlength=p)
        min_count = int(np.min(counts))
        if min_count < robustness:
            robustness = min_count
            if robustness == 0:
                break
    
    return robustness if robustness != float('inf') else n

def _simulated_annealing_refine(S_mask, primes, start_time, deadline, random_seed=42):
    """
    Multi-objective SA: primary = minimize removed, secondary = maximize robustness.
    Move: add back a removed element if it doesn't violate admissibility, 
    or remove an element from a "bad" residue to improve robustness.
    """
    rng = np.random.default_rng(random_seed)
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    current_S = S_mask.copy()
    current_removed = N + 1 - np.sum(current_S)
    current_robustness = _compute_robustness(current_S, primes)
    
    best_S = current_S.copy()
    best_removed = current_removed
    best_robustness = current_robustness
    
    # Track which elements are removed
    removed_elements = np.where(~current_S)[0]
    
    max_iterations = 50000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() - start_time > deadline - 10.0:
            break
        
        S_indices = np.where(current_S)[0]
        n = len(S_indices)
        
        if n == 0:
            break
        
        # Try to add back a removed element (improves primary objective)
        if len(removed_elements) > 0:
            candidate_idx = int(rng.integers(0, len(removed_elements)))
            candidate_val = int(removed_elements[candidate_idx])
            
            # Check if adding this element maintains admissibility
            # We need to check all primes p <= n+1
            new_n = n + 1
            valid = True
            for p in primes:
                if p > new_n:
                    break
                # Count residues including the new element
                residues = S_indices % p
                counts = np.bincount(residues, minlength=p)
                new_residue = candidate_val % p
                counts[new_residue] += 1
                
                if np.min(counts) == 0:
                    valid = False
                    break
            
            if valid:
                # Accept: improves primary objective
                current_S[candidate_val] = True
                removed_elements = removed_elements[removed_elements != candidate_val]
                current_removed -= 1
                current_robustness = _compute_robustness(current_S, primes)
                
                if current_removed < best_removed or (current_removed == best_removed and current_robustness > best_robustness):
                    best_S = current_S.copy()
                    best_removed = current_removed
                    best_robustness = current_robustness
            else:
                # Compute "cost" of adding: how many primes become violated
                # Use soft penalty
                penalty = 0
                for p in primes:
                    if p > new_n:
                        break
                    residues = S_indices % p
                    counts = np.bincount(residues, minlength=p)
                    new_residue = candidate_val % p
                    counts[new_residue] += 1
                    
                    if np.min(counts) == 0:
                        # Count how many residues are missing
                        missing = np.sum(counts == 0)
                        penalty += missing
                
                # Accept with probability exp(-penalty / T)
                if rng.random() < np.exp(-penalty / T):
                    current_S[candidate_val] = True
                    removed_elements = removed_elements[removed_elements != candidate_val]
                    current_removed -= 1
                    current_robustness = _compute_robustness(current_S, primes)
        
        # Also try to remove an element from a poorly-populated residue to improve robustness
        # This trades primary for secondary
        if n > 0 and T > T_min * 1.5:
            # Find the prime with minimum robustness
            min_rob_p = None
            min_rob_val = float('inf')
            for p in primes:
                if p > n:
                    break
                residues = S_indices % p
                counts = np.bincount(residues, minlength=p)
                min_count = int(np.min(counts))
                if min_count < min_rob_val:
                    min_rob_val = min_count
                    min_rob_p = p
            
            if min_rob_p is not None and min_rob_val < n // min_rob_p:
                # Remove one element from the least populated residue
                residues = S_indices % min_rob_p
                counts = np.bincount(residues, minlength=min_rob_p)
                min_res = int(np.argmin(counts))
                
                elements_in_res = S_indices[residues == min_res]
                if len(elements_in_res) > 0:
                    remove_idx = int(rng.integers(0, len(elements_in_res)))
                    remove_val = int(elements_in_res[remove_idx])
                    
                    # Check if removing maintains admissibility (it should, since we're removing)
                    current_S[remove_val] = False
                    removed_elements = np.append(removed_elements, remove_val)
                    current_removed += 1
                    current_robustness = _compute_robustness(current_S, primes)
        
        iteration += 1
        T = max(T * cooling_rate, T_min)
    
    return best_S

def construct():
    start_time = time.time()
    deadline = 295.0  # Safety margin
    
    primes = _primes(N)
    
    # Try multiple thresholds and strategies
    best_result = None
    best_size = 0
    best_S_mask = None
    
    # First pass: Deterministic baseline
    thresholds = [50, 75, 100, 125, 150, 200]
    strategies = [False, True]
    
    for threshold in thresholds:
        for use_least in strategies:
            if time.time() - start_time > deadline:
                return best_result if best_result else list(range(N + 1))
            
            result = _construct_with_threshold(primes, threshold, use_least, random_seed=None)
            size = len(result)
            if size > best_size:
                best_size = size
                best_result = result
                # Reconstruct mask for SA refinement
                best_S_mask = np.zeros(N + 1, dtype=bool)
                for val in result:
                    best_S_mask[val] = True
                
    # Second pass: Randomized exploration
    exploration_configs = [
        (100, True), (150, True), (200, True), 
        (75, True), (125, True),
        (100, False), (150, False)
    ]
    
    rng_master = np.random.default_rng(42)
    
    max_random_trials = 80
    trials_run = 0
    
    while trials_run < max_random_trials:
        if time.time() - start_time > deadline - 30.0:
            break
            
        config = exploration_configs[trials_run % len(exploration_configs)]
        threshold, use_least = config
        
        seed = 1000 + trials_run
        
        result = _construct_with_threshold(primes, threshold, use_least, random_seed=seed)
        size = len(result)
        
        if size > best_size:
            best_size = size
            best_result = result
            best_S_mask = np.zeros(N + 1, dtype=bool)
            for val in result:
                best_S_mask[val] = True
            
        trials_run += 1
    
    # Third pass: Simulated annealing refinement on best solution
    if best_S_mask is not None and time.time() - start_time < deadline - 15.0:
        refined_S = _simulated_annealing_refine(best_S_mask, primes, start_time, deadline, random_seed=42)
        refined_result = np.where(refined_S)[0].tolist()
        refined_size = len(refined_result)
        
        if refined_size > best_size:
            best_size = refined_size
            best_result = refined_result
    
    if best_result is None:
        best_result = list(range(N + 1))
    
    return best_result

# EVOLVE-BLOCK-END
