"""
Program for integer sum-product extremal sets.
Approach: Deterministic grid search over "rational geometric" integer sets,
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
    
    # 2. Grid search over Rational Geometric Progressions
    # A_i = round(a0 * (p/q)^i)
    # We search for small integers p, q such that p/q > 1.
    # And a0 such that the set fits in [1, 10000].
    
    # p/q roughly between 1.05 and 1.35
    # q from 1 to 20
    # p from q+1 to q+10 (ratio up to 1 + 10/q)
    
    for q in range(1, 21):
        for p in range(q + 1, q + 11):
            ratio = p / q
            if ratio <= 1.0:
                continue
            
            # Max a0 such that a0 * ratio^(n-1) <= max_val
            # a0 <= max_val / (ratio^(n-1))
            max_a0 = int(max_val / (ratio ** (n - 1)))
            if max_a0 < 1:
                continue
            
            # We should sample a0.
            # Small a0 might lead to collisions (non-distinct).
            # Large a0 keeps them distinct but might push max value too high if ratio is large.
            # We iterate a0 from 1 to max_a0.
            # To speed up, we can step. But n=20 is small.
            # max_a0 can be up to 10000 if ratio is close to 1.
            # If ratio=1.01, 1.01^19 ~ 1.2. max_a0 ~ 8000.
            # Iterating 8000 values * 400 pairs of (p,q) is 3.2M evaluations.
            # Each evaluation is O(n^2) ~ 400 ops.
            # 1.2B ops. Might be too slow in Python.
            
            # Optimization:
            # Only check a0 values that are likely to yield distinct integers.
            # Distinctness requires a0 * (ratio^i) to be well separated.
            # gap ~ a0 * ratio^i * (ratio - 1).
            # We need gap > 0.5 for rounding to be distinct?
            # Actually, we just check distinctness after generation.
            
            # Let's limit the range of a0 or step size.
            # If ratio is close to 1, we need large a0.
            # If ratio is larger, small a0 might work.
            
            # We'll sample a0 logarithmically or linearly with a step.
            # Step 1 for small max_a0, larger for large.
            
            if max_a0 <= 200:
                a0_range = range(1, max_a0 + 1)
            else:
                # Sample every 2nd or 5th value?
                # Or just cap the search?
                # Let's try a0 from 1 to min(max_a0, 500) first, then larger steps.
                # Actually, distinctness is the main constraint.
                # Let's just iterate 1 to max_a0 but break early if time is tight?
                # No, deterministic.
                # Let's restrict a0 to 1..500 and then 500..max_a0 step 10.
                a0_range = list(range(1, 501)) + list(range(510, max_a0 + 1, 10))
                
            for a0 in a0_range:
                A = []
                valid = True
                for i in range(n):
                    # Use round to get nearest integer
                    val = int(round(a0 * (ratio ** i)))
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
