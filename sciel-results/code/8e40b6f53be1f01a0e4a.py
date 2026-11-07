import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
def _is_valid(arr):
    """Check if arr satisfies the cardinality-separated subset sum constraint."""
    n = len(arr)
    if n == 0:
        return True
    # Compute min and max subset sums for each cardinality k
    # min_sum[k] = sum of k smallest elements
    # max_sum[k] = sum of k largest elements
    sorted_arr = sorted(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + sorted_arr[i]
    
    min_sums = [prefix[k] for k in range(n+1)]
    max_sums = [prefix[n] - prefix[n-k] for k in range(n+1)]
    
    # For any k1 < k2, we need: max_sum[k1] < min_sum[k2]
    # This is equivalent to: for all k from 1 to n-1: max_sum[k] < min_sum[k+1]
    for k in range(n):
        if max_sums[k] >= min_sums[k+1]:
            return False
    return True

def construct():
    deadline = time.time() + 80  # 80s safety margin (90s limit)
    
    # Start with the known 31-element solution
    best_result = list(range(226, 257))  # 31 elements
    best_size = 31
    
    # Try to find a larger set using ILP
    # We'll try sizes 32, 33, ... up to a reasonable limit
    # For each size m, we select m integers from 1..256
    
    # The constraint: for all k < l, max_sum[k] < min_sum[l]
    # This means: sum of k largest < sum of (k+1) smallest for all k from 0 to m-1
    # Actually: max_sum[k] < min_sum[k+1] for k = 0, 1, ..., m-1
    # max_sum[k] = sum of k largest selected
    # min_sum[k] = sum of k smallest selected
    
    # Let's try a greedy + local search approach first, then ILP for verification
    
    # Greedy approach: try to build a set of size m
    # For size m, we need m distinct values from 1..256
    # The tightest constraint is usually max_sum[m-1] < min_sum[m]
    # i.e., sum of (m-1) largest < sum of m smallest
    
    # Let's try different strategies to find a set of size 32+
    
    # Strategy: For a set of size m, if we sort them as a_1 < a_2 < ... < a_m
    # Then:
    # min_sum[k] = a_1 + ... + a_k
    # max_sum[k] = a_{m-k+1} + ... + a_m
    # Constraint: a_{m-k+1} + ... + a_m < a_1 + ... + a_{k+1} for all k = 0,...,m-1
    # For k=0: 0 < a_1 (always true since a_1 >= 1)
    # For k=m-1: a_2 + ... + a_m < a_1 + ... + a_m => 0 < a_1 (always true)
    # The tightest constraints are in the middle.
    
    # Let's try to find a set of size 32 using optimization
    # We can use a simpler approach: try all possible "gaps" patterns
    
    # Actually, let's try a direct search for size 32
    # The set must have 32 distinct values from 1..256
    # Let's try to maximize the minimum gap between consecutive cardinality sum ranges
    
    # For now, let's try a few heuristic approaches to see if we can get 32
    
    # Approach 1: Try to find a 32-element set using local search
    # Start with a candidate and try to improve it
    
    # Let's try: what if we use values that are spread out?
    # For m=32, we need 32 values. Let's try evenly spaced:
    # 1, 7, 13, 19, ... or similar
    
    # Actually, the key insight is that the constraint max_sum[k] < min_sum[k+1]
    # for all k is equivalent to saying that the k-th largest sum is strictly less
    # than the (k+1)-th smallest sum.
    
    # Let me try a different approach: use the fact that if we have a set where
    # each element is greater than the sum of all smaller elements, then the
    # subset sums are all distinct. But that's a stronger condition.
    
    # For our weaker condition, we just need cardinality-separated sums.
    
    # Let's try to find a 32-element set by solving an optimization problem
    # We want to maximize the minimum "gap" between max_sum[k] and min_sum[k+1]
    
    # For a given set sorted as a_1 < a_2 < ... < a_m:
    # gap_k = min_sum[k+1] - max_sum[k] for k = 0, ..., m-1
    # We want all gap_k > 0
    
    # Let's try a simple local search for m=32
    # Start with a random set of 32 values and try to fix violations
    
    import random
    random.seed(42)
    
    def try_find_set(m, max_iters=50000, time_limit=None):
        """Try to find a valid set of size m using local search."""
        if time_limit is None:
            time_limit = deadline - time.time() - 5
        
        best_gap = -1
        best_set = None
        
        for restart in range(100):
            if time.time() > time_limit:
                break
            
            # Start with a random set of m distinct values from 1..256
            arr = sorted(random.sample(range(1, 257), m))
            
            # Try to improve by swapping elements
            improved = True
            iters = 0
            while improved and iters < max_iters:
                if time.time() > time_limit:
                    break
                improved = False
                iters += 1
                
                # Check validity
                if _is_valid(arr):
                    if len(arr) > best_size:
                        return arr
                    # Try to see if we can add more elements later
                    break
                
                # Find the worst violation
                sorted_arr = sorted(arr)
                prefix = [0] * (m + 1)
                for i in range(m):
                    prefix[i+1] = prefix[i] + sorted_arr[i]
                min_sums = [prefix[k] for k in range(m+1)]
                max_sums = [prefix[m] - prefix[m-k] for k in range(m+1)]
                
                # Find k with smallest gap (or negative gap)
                worst_k = -1
                worst_gap = float('inf')
                for k in range(m):
                    gap = min_sums[k+1] - max_sums[k]
                    if gap < worst_gap:
                        worst_gap = gap
                        worst_k = k
                
                if worst_gap >= 0:
                    if _is_valid(arr):
                        if len(arr) > best_size:
                            return arr
                        break
                
                # Try to fix by moving elements
                # If gap is negative at k, we need to increase min_sums[k+1] or decrease max_sums[k]
                # min_sums[k+1] = sum of k+1 smallest
                # max_sums[k] = sum of k largest
                
                # Try swapping a small element with a larger one, or vice versa
                # Or try adjusting specific elements
                
                # Simple heuristic: try all single-element changes
                best_change = None
                best_new_gap = worst_gap
                
                # Try removing an element from the top k and adding a smaller one
                # Or removing from bottom k+1 and adding a larger one
                
                # For simplicity, try all possible swaps
                # This is O(m^2) per iteration which might be too slow
                
                # Instead, try a simpler approach: randomly perturb
                idx = random.randint(0, m-1)
                old_val = arr[idx]
                # Try changing to a nearby value
                for delta in [1, -1, 2, -2, 3, -3]:
                    new_val = old_val + delta
                    if 1 <= new_val <= 256 and new_val not in arr:
                        arr[idx] = new_val
                        arr.sort()
                        if _is_valid(arr):
                            if len(arr) > best_size:
                                return arr
                            break
                        # Check if gap improved
                        prefix2 = [0] * (m + 1)
                        for i in range(m):
                            prefix2[i+1] = prefix2[i] + arr[i]
                        min_sums2 = [prefix2[k] for k in range(m+1)]
                        max_sums2 = [prefix2[m] - prefix2[m-k] for k in range(m+1)]
                        gap2 = min_sums2[worst_k+1] - max_sums2[worst_k]
                        if gap2 > best_new_gap:
                            best_new_gap = gap2
                            best_change = (idx, new_val)
                        else:
                            arr[idx] = old_val
                            arr.sort()
                    else:
                        continue
                else:
                    # No improvement, revert
                    arr[idx] = old_val
                    arr.sort()
                    # Try a different strategy: swap two elements
                    i, j = random.sample(range(m), 2)
                    arr[i], arr[j] = arr[j], arr[i]
                    arr.sort()
                    if not _is_valid(arr):
                        # Check if it's better
                        prefix2 = [0] * (m + 1)
                        for i2 in range(m):
                            prefix2[i2+1] = prefix2[i2] + arr[i2]
                        min_sums2 = [prefix2[k] for k in range(m+1)]
                        max_sums2 = [prefix2[m] - prefix2[m-k] for k in range(m+1)]
                        gap2 = min_sums2[worst_k+1] - max_sums2[worst_k]
                        if gap2 < worst_gap:
                            # Revert
                            arr[i], arr[j] = arr[j], arr[i]
                            arr.sort()
                            break
                    else:
                        if len(arr) > best_size:
                            return arr
                        break
            else:
                continue
        
        return best_set
    
    # Try to find sets of increasing size
    for m in range(32, 40):
        if time.time() > deadline - 10:
            break
        result = try_find_set(m)
        if result is not None and _is_valid(result):
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
                # Try to go even larger
                continue
        else:
            # Couldn't find size m, likely can't find larger sizes either
            break
    
    # Final validation
    assert len(set(best_result)) == len(best_result)
    assert all(isinstance(x, int) and 1 <= x <= 256 for x in best_result)
    assert _is_valid(best_result)
    
    return list(best_result)
# EVOLVE-BLOCK-END
