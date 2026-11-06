import time
import random
import numpy as np
from itertools import combinations

def _compute_multiplicities(A, n=128, lo=128, hi=256):
    """Compute the number of unordered pair representations for each sum in [lo, hi]."""
    A = sorted(set(A))
    mults = np.zeros(hi - lo + 1, dtype=np.int32)
    m = len(A)
    for i in range(m):
        a = A[i]
        for j in range(i, m):
            b = A[j]
            s = a + b
            if lo <= s <= hi:
                mults[s - lo] += 1
    return mults

def _peak(A):
    """Return the maximum representation multiplicity in [128, 256]."""
    mults = _compute_multiplicities(A)
    return int(mults.max())

def _is_valid(A, n=128, lo=128, hi=256):
    """Check that every sum in [lo, hi] is representable as a sum of two elements of A."""
    mults = _compute_multiplicities(A, n, lo, hi)
    return np.all(mults > 0)

def _try_add(A_set, candidates, deadline):
    """Try adding an element from candidates to A_set, keep if it improves or maintains validity and reduces peak."""
    A_list = sorted(A_set)
    current_peak = _peak(A_list)
    
    best_peak = current_peak
    best_set = A_set
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in A_set:
            continue
        new_set = A_set | {x}
        if not _is_valid(new_set):
            continue
        p = _peak(sorted(new_set))
        if p < best_peak:
            best_peak = p
            best_set = new_set
    
    return best_set

def _try_remove(A_set, deadline):
    """Try removing an element, keep if validity is maintained and peak improves or stays same."""
    A_list = sorted(A_set)
    current_peak = _peak(A_list)
    best_peak = current_peak
    best_set = A_set
    
    for x in A_list:
        if time.time() > deadline:
            break
        new_set = A_set - {x}
        if len(new_set) == 0:
            continue
        if not _is_valid(new_set):
            continue
        p = _peak(sorted(new_set))
        if p < best_peak:
            best_peak = p
            best_set = new_set
    
    return best_set

def _local_search(A_set, deadline, max_iter=200):
    """Iterative local search with add/remove moves."""
    best_set = set(A_set)
    best_peak = _peak(sorted(best_set))
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try removals first (to simplify)
        new_set = _try_remove(best_set, deadline)
        if new_set != best_set:
            best_set = new_set
            best_peak = _peak(sorted(best_set))
            continue
        
        # Try additions
        candidates = list(range(0, 129))
        random.shuffle(candidates)
        new_set = _try_add(best_set, candidates, deadline)
        if new_set != best_set:
            best_set = new_set
            best_peak = _peak(sorted(best_set))
            continue
        
        # No improvement, try a random perturbation
        if random.random() < 0.3:
            perturbed = set(best_set)
            # Remove a random element
            if perturbed:
                to_remove = random.choice(list(perturbed))
                perturbed.discard(to_remove)
            # Add a random element
            to_add = random.randint(0, 128)
            perturbed.add(to_add)
            if _is_valid(perturbed):
                p = _peak(sorted(perturbed))
                if p <= best_peak + 2:  # Allow slight degradation for exploration
                    best_set = perturbed
                    best_peak = p
                else:
                    best_set = perturbed  # Keep it for further search
                    best_peak = p
            else:
                # Revert
                best_set = set(A_set) if A_set else best_set
    
    return best_set, best_peak

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial seed: the interval [64, 128] which has peak 33
    # But we'll use a smarter initialization
    # Start with a set that's known to be valid
    
    # Seed 1: [64, 128]
    seed1 = set(range(64, 129))
    
    # Seed 2: Try a spread-out set
    # We need to cover [128, 256]. The minimum element needed is 0 (if we have 128), 
    # or 64 (if we only have up to 128). 
    # A good spread: every other element from 0 to 128
    seed2 = set(range(0, 129, 2))  # Even numbers 0..128
    
    # Check validity of seeds
    valid_seeds = []
    for seed in [seed1, seed2]:
        if _is_valid(seed):
            valid_seeds.append(seed)
    
    if not valid_seeds:
        # Fallback: use seed1 and hope
        valid_seeds.append(seed1)
    
    best_set = valid_seeds[0]
    best_peak = _peak(sorted(best_set))
    
    # Run local search from the best seed
    for seed in valid_seeds:
        if time.time() > deadline - 5:
            break
        result_set, result_peak = _local_search(seed, deadline - 5)
        if result_peak < best_peak:
            best_peak = result_peak
            best_set = result_set
    
    # Additional local search iterations from best
    while time.time() < deadline - 2:
        result_set, result_peak = _local_search(best_set, deadline - 2, max_iter=50)
        if result_peak < best_peak:
            best_peak = result_peak
            best_set = result_set
        else:
            break
    
    return sorted(best_set)
