"""
Strategy: Exhaustive search over all small sets to find the global optimum.
The score function 1 + log(|U-U|/|U+U|)/log(2*max(U)+1) is maximized by sets
with high "difference-to-sum" ratio relative to their range.
For small sets (n=2..5), we can exhaustively search all combinations of elements
within a reasonable bound (e.g., max element <= 100) to find the true optimum.
This avoids the local minima traps of heuristic local search.
"""
import time
import math
from itertools import combinations, permutations

def compute_score(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2*max(U) + 1)
    """
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    sum_set = set()
    diff_set = set()
    
    for u1 in U:
        for u2 in U:
            sum_set.add(u1 + u2)
            diff_set.add(u1 - u2)
    
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    log_ratio = math.log(size_diff / size_sum)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def construct():
    """
    Entry point. Returns a list of nonnegative integers containing 0.
    Uses exhaustive search for small sets to find the global optimum.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    best_score = -1.0
    best_U = [0, 1]
    
    # We know small sets are optimal for this metric because log(2*max+1) grows
    # slowly with max, while the ratio |U-U|/|U+U| is bounded.
    # For n=2, U={0, a}. |U+U|={0,a,2a} size 3. |U-U|={-a,0,a} size 3.
    # Ratio 1. Score 1.
    # For n=3, U={0, a, b}. 
    # Let's search up to n=5 and max_val=200.
    
    max_n = 5
    max_val = 200
    
    # Precompute scores for n=2
    # U = {0, a}
    # Score is always 1.0 for a>0.
    # So any {0, a} is a baseline of 1.0.
    
    # Search n=3
    # U = {0, a, b} with 0 < a < b <= max_val
    # To save time, we can fix a=1 (scaling doesn't change ratio, only log term)
    # Wait, scaling DOES change log(2*max+1). 
    # If we scale U by k, max becomes k*max. log(2*k*max+1) increases.
    # Ratio |U-U|/|U+U| is scale invariant.
    # So we want SMALL max.
    # Thus, we should search for sets with small max values.
    
    # Let's iterate over max_val from 1 to max_val
    # For each max_val, iterate over all subsets of {1, ..., max_val-1} of size n-2
    # Add 0 and max_val.
    
    # Optimization: 
    # For a fixed set of relative positions, the score is:
    # 1 + log(ratio) / log(2*M + 1)
    # where M is the max element.
    # To maximize this, we want large ratio and small M.
    
    # Let's just brute force all sets {0} + S where S is a subset of {1..max_val}
    # containing max_val.
    
    # Actually, simpler:
    # Iterate M from 1 to max_val.
    # Iterate over all subsets S of {1, ..., M-1} of size n-2.
    # U = {0} + S + {M}.
    
    for n in range(3, max_n + 1):
        if time.time() > deadline:
            break
            
        # Number of combinations C(M-1, n-2)
        # For M=200, n=5, C(199, 3) ~ 1.3M. This is fast in Python if optimized?
        # 1.3M * compute_score cost.
        # compute_score is O(n^2) = 25 ops. 1.3M * 25 ~ 32M ops. Should be < 1s.
        
        for M in range(1, max_val + 1):
            if time.time() > deadline:
                break
            
            # We need to choose n-2 elements from {1, ..., M-1}
            if M - 1 < n - 2:
                continue
                
            # Generate combinations
            # To speed up, we can prune.
            # But let's just run it for now.
            
            # Special case: if n=3, we pick 1 element from {1..M-1}.
            # U = {0, a, M}.
            
            # Let's use itertools.combinations
            # This might be slow for M=200, n=5.
            # Let's limit M for larger n.
            if n == 3:
                max_M_search = max_val
            elif n == 4:
                max_M_search = 100
            else:
                max_M_search = 50
                
            if M > max_M_search:
                continue
                
            for comb in combinations(range(1, M), n - 2):
                U = [0] + list(comb) + [M]
                # Ensure sorted and unique (combinations are unique and sorted)
                s = compute_score(U)
                if s > best_score:
                    best_score = s
                    best_U = U
                    
    # Also check n=2 just in case (though score is 1.0)
    # And n=1 (invalid)
    
    # Refine: Local search on the best found
    current_U = list(best_U)
    current_score = best_score
    
    # Simple local search to polish
    while time.time() < deadline - 5:
        improved = False
        current_max = max(current_U) if current_U else 0
        
        # Try adding an element
        for x in range(1, current_max + 10):
            if x not in current_U:
                new_U = sorted(list(set(current_U + [x])))
                s = compute_score(new_U)
                if s > current_score:
                    current_U = new_U
                    current_score = s
                    improved = True
                    break
        
        # Try removing an element
        for x in current_U:
            if x == 0:
                continue
            new_U = [y for y in current_U if y != x]
            if len(new_U) < 2:
                continue
            s = compute_score(new_U)
            if s > current_score:
                current_U = new_U
                current_score = s
                improved = True
                break
        
        # Try swapping
        if not improved:
             for i, x in enumerate(current_U):
                 if x == 0: continue
                 for y in range(1, current_max + 5):
                     if y == x or y in current_U: continue
                     new_U = list(current_U)
                     new_U[i] = y
                     new_U = sorted(list(set(new_U)))
                     s = compute_score(new_U)
                     if s > current_score:
                         current_U = new_U
                         current_score = s
                         improved = True
                         break
                 if improved: break
                 
        if not improved:
            break
            
    if current_score > best_score:
        best_U = current_U
        best_score = current_score
        
    return best_U
