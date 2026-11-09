import time
import numpy as np
from itertools import combinations

# Precompute the best set by exhaustive search over small candidate universes.
# The objective is to maximize 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
# Subject to 0 in U.

def _compute_score(U):
    if not U:
        return 0.0
    U = sorted(U)
    if 0 not in U:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    arr = np.array(U, dtype=np.int64)
    
    # Sumset: all pairwise sums
    # For a set of size n, sumset size is at most n*(n-1)/2 + n (if all distinct) but bounded by 2*max
    # We can compute the boolean array of sums
    sum_mask = np.zeros(2 * max_u + 1, dtype=bool)
    # Vectorized: for each element a, mark a + U
    for a in arr:
        sum_mask[a] = True # a + 0
        if a > 0:
            sum_mask[a:] = True # This is wrong, we need specific sums
    
    # Better vectorized approach for small arrays:
    # sums = arr[:, None] + arr[None, :]
    # np.any(sums <= 2*max_u) -> unique
    
    # For efficiency in a tight loop, let's use the fact that U is small (<= 8 or so)
    # sums = set()
    # for a in U:
    #    for b in U:
    #       sums.add(a+b)
    # diffs = set()
    # for a in U:
    #    for b in U:
    #       diffs.add(a-b)
    
    # But python sets are slow. Let's use numpy where possible or just optimized python sets for very small n.
    # Given n <= 8, n^2 <= 64. Python sets are fine.
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1.0:
        return 0.0
    return 1.0 + np.log2(ratio) / np.log2(2 * max_u + 1)

# Since construct() is called by the evaluator, and we want to be fast,
# we can precompute the best set in a module-level variable or just inside the function
# with a global cache. However, the evaluator might reset the module.
# Let's do the search inside construct() but with a hard timeout, returning a known good set if it takes too long.
# Or better: do the search once and cache it.

_best_set = None

def _find_best_set():
    global _best_set
    if _best_set is not None:
        return _best_set
    
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds for search, leave 110s for safety/overhead
    
    best_score = 1.0
    best_U = [0, 1, 3]
    
    # Search over sizes 3, 4, 5, 6
    # Candidate universe: {0, 1, ..., M}
    # If M is too large, the log(2*max+1) penalty grows.
    # Optimal max is likely small. Let's try M up to 20.
    
    max_M = 25
    
    for n in range(3, 7):
        # Choose n-1 elements from {1, ..., max_M}
        # Since 0 must be included.
        candidates = range(1, max_M + 1)
        
        # If max_M is 25 and n is 6, C(25,5) = 53130.
        # C(25,4) = 12650.
        # Total iterations ~ 100k.
        # Each iteration: O(n^2) operations. ~36 ops.
        # 100k * 36 = 3.6M ops. Very fast in Python? Maybe 1-2 seconds.
        
        for combo in combinations(candidates, n - 1):
            if time.time() > deadline:
                break
                
            U = [0] + list(combo)
            # Optional pruning: if max(U) is too large relative to score potential
            # Max possible score for this max_u is 1 + log2(max_ratio)/log2(2*max_u+1)
            # Max ratio is roughly 2n-1 / 1 = small.
            # Let's just compute.
            
            arr = U
            sums = set()
            diffs = set()
            # Optimization: only compute if it *could* be better
            # But computing is cheap.
            
            for a in arr:
                for b in arr:
                    sums.add(a + b)
                    diffs.add(a - b)
            
            n_sums = len(sums)
            n_diffs = len(diffs)
            
            if n_sums == 0:
                continue
                
            ratio = n_diffs / n_sums
            if ratio <= 1.0:
                continue
                
            max_u = arr[-1]
            score = 1.0 + np.log2(ratio) / np.log2(2 * max_u + 1)
            
            if score > best_score:
                best_score = score
                best_U = U
                # print(f"New best: {best_U}, score: {best_score:.4f}")
        
        if time.time() > deadline:
            break
            
    _best_set = best_U
    return _best_set

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    try:
        # Try to run the search if we have time
        # The search is fast enough (< 1s usually) so we can just run it.
        # But if the environment is slow, we cap it.
        
        # Check if we already have a result from a previous call in the same process
        # _best_set is global.
        
        # To be robust against timeouts, we do a quick check.
        # If _best_set is None, we run _find_best_set.
        # _find_best_set has its own internal deadline.
        
        U = _find_best_set()
        
        # Verify validity
        if 0 not in U:
            U = [0, 1, 3]
            
        return U
        
    except Exception:
        # Fallback to a known good small set
        # {0, 1, 3} gives score:
        # sums: {0,1,2,3,4,6} size 6
        # diffs: {-3,-2,-1,0,1,2,3} size 7
        # ratio 7/6 = 1.166
        # max=3, log2(7)=2.807
        # score = 1 + log2(1.166)/2.807 = 1 + 0.222/2.807 = 1.079
        # {0, 1, 3, 6} is better.
        return [0, 1, 3, 6]
