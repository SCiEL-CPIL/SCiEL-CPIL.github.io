"""
Program for integer sum-product extremal sets.
Approach: Deterministic grid search over "near-geometric" integer sets (floor of real GP),
combined with structured candidates and a deterministic local search.
"""
import math
import time
import numpy as np

def _score_set(A):
    """Compute max(|A+A|, |A*A|) for a set/list of distinct positive integers."""
    n = len(A)
    if n == 0:
        return 0
    if len(set(A)) != n:
        return float('inf')
    if min(A) < 1:
        return float('inf')
    
    A = np.array(A)
    # Sumset
    # Using broadcasting for speed
    sums = A[:, None] + A[None, :]
    sset = np.unique(sums)
    
    # Productset
    prods = A[:, None] * A[None, :]
    pset = np.unique(prods)
    
    return max(len(sset), len(pset))

def construct():
    n = 20
    max_val = 10000
    best_score = float('inf')
    best_set = None
    
    def update_best(A):
        nonlocal best_score, best_set
        if len(A) != n:
            return
        if len(set(A)) != n:
            return
        if min(A) < 1 or max(A) > max_val:
            return
        s = _score_set(A)
        if s < best_score:
            best_score = s
            best_set = sorted(A)
            
    # 1. Structured candidates
    # 1..20
    update_best(list(range(1, n + 1)))
    
    # 2. Grid search over "Near-Geometric" sets: a_i = floor(a0 * r^i)
    # We search for a0 and r such that the set fits in [1, 10000]
    # r must be > 1 to be distinct.
    # a_{n-1} = floor(a0 * r^{n-1}) <= 10000
    # a0 * r^{19} <= 10000
    
    # Possible r values:
    # If r is close to 1, a0 can be large.
    # If r is large, a0 must be small.
    
    # We'll iterate over r in steps of 0.001 or so?
    # r^19 is sensitive.
    # Let's iterate over a0 from 1 to 100.
    # For each a0, find max r such that a0 * r^19 <= 10000.
    # r_max = (10000/a0)^(1/19).
    
    for a0 in range(1, 200):
        r_max = (max_val / a0) ** (1.0 / (n - 1))
        if r_max <= 1.0:
            continue
        
        # We want to try various r.
        # Smaller r (close to 1) gives more compact sumset?
        # Actually, smaller r means terms are closer, so sumset might be smaller.
        # But product set is always ~2n if r is not integer?
        # For floor(GP), product set collisions are rare unless r is integer.
        # So we should try integer ratios first?
        # But integer ratios are impossible for n=20 as shown before.
        # So we rely on the local search to fix the "near-GP" structure.
        
        # Let's sample r values.
        # Focus on r close to 1 to keep values small and close?
        # Or r such that a0 * r^i are integers?
        
        # Generate a few candidates for each a0
        num_samples = 50
        for k in range(num_samples):
            # Linearly interpolate r between 1.0 and r_max
            # Or logarithmically?
            r = 1.0 + (r_max - 1.0) * (k / (num_samples - 1))
            if r <= 1.0:
                continue
            
            A = []
            valid = True
            for i in range(n):
                val = int(math.floor(a0 * (r ** i)))
                if val < 1 or val > max_val:
                    valid = False
                    break
                A.append(val)
            
            if not valid:
                continue
            
            # Ensure distinct
            if len(set(A)) != n:
                continue
                
            update_best(A)
            
    # 3. Deterministic Local Search
    if best_set is None:
        best_set = list(range(1, n + 1))
        best_score = _score_set(best_set)
    
    best_set.sort()
    current_score = best_score
    
    start_time = time.time()
    time_limit = 55
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            old_val = best_set[i]
            candidates_for_swap = set()
            
            # Neighbors
            for d in range(1, 50):
                candidates_for_swap.add(old_val - d)
                candidates_for_swap.add(old_val + d)
            
            # Multiples/Divisors of other elements
            for j in range(n):
                if j == i: continue
                other = best_set[j]
                for d in range(1, int(math.isqrt(other)) + 1):
                    if other % d == 0:
                        candidates_for_swap.add(d)
                        candidates_for_swap.add(other // d)
                for m in range(2, 10):
                    if other * m <= max_val:
                        candidates_for_swap.add(other * m)
            
            valid_candidates = [c for c in candidates_for_swap if 1 <= c <= max_val and c != old_val and c not in best_set]
            valid_candidates.sort()
            
            best_candidate = None
            best_candidate_score = current_score
            
            for new_val in valid_candidates:
                best_set[i] = new_val
                s = _score_set(best_set)
                if s < best_candidate_score:
                    best_candidate_score = s
                    best_candidate = new_val
            
            if best_candidate is not None:
                best_set[i] = best_candidate
                current_score = best_candidate_score
                improved = True
            else:
                best_set[i] = old_val
                
    return best_set
