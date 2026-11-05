import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Specific to the task: q=3, n=8, radius=1
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbor lists
    # neighbors[i] is a numpy array of indices j such that dist(i, j) <= radius
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)
        
    # --- Primal-Dual Approximation Algorithm ---
    # State:
    #   dual[v]: dual variable for point v (price paid to cover v)
    #   covered[v]: boolean, is v covered?
    #   is_code[c]: boolean, is c selected as a codeword?
    #   code_list: list of selected codewords
    
    dual = np.zeros(total_words, dtype=np.float64)
    covered = np.zeros(total_words, dtype=bool)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # Precompute cost of each set (all sets have cost 1 in minimum cardinality set cover)
    # But in primal-dual, we raise duals until sum_{v in S} dual[v] >= cost(S).
    # Here cost(S) = 1 for all S.
    
    # To speed up, we can track for each set S, the sum of duals of its *uncovered* elements?
    # No, the standard primal-dual raises duals of *uncovered* elements.
    # A set S becomes tight when sum_{v in S, covered[v]==False} dual[v] >= 1?
    # Actually, the condition is sum_{v in S} dual[v] >= cost(S).
    # Since we only raise duals for uncovered v, and dual[v]=0 for covered v is not true 
    # (duals stay once raised), we must be careful.
    # Standard algorithm:
    # 1. While there are uncovered elements:
    #    a. Raise duals of all uncovered elements at equal rate epsilon.
    #    b. When some set S becomes tight (sum_{v in S} dual[v] == cost(S)), select S.
    #    c. Mark all elements in S as covered.
    #    d. Remove S and its elements from consideration? No, sets can be reused but elements are covered.
    
    # Optimization:
    # Instead of continuous raising, we can jump.
    # For each uncovered element u, it belongs to a set of sets N(u) (codewords within radius).
    # The next time a set becomes tight is determined by the minimum over all sets S of 
    # (cost(S) - current_sum_duals(S)) / count_uncovered_in_S.
    # This is complex to maintain dynamically.
    
    # Alternative: Greedy Primal-Dual (often called Greedy Set Cover with dual tracking)
    # Actually, the standard greedy set cover (max uncovered) is a 2-approx.
    # Primal-Dual is also a 2-approx but can be better in practice if implemented with "jumping".
    
    # Let's implement a version that jumps to the next tight set.
    # We maintain:
    #   uncovered_count_per_set[c]: number of uncovered elements in set c.
    #   sum_duals_per_set[c]: sum of duals of elements in set c (only uncovered ones matter for the gap? No, all duals count).
    #   But we only raise duals for uncovered elements.
    #   Let U be the set of uncovered elements.
    #   We raise dual[u] for all u in U by epsilon.
    #   For a set c, the sum of duals over U ∩ N(c) increases by epsilon * |U ∩ N(c)|.
    #   The gap is 1 - sum_{v in N(c)} dual[v].
    #   Note: sum_{v in N(c)} dual[v] = sum_{v in N(c) ∩ U} dual[v] + sum_{v in N(c) ∩ C} dual[v].
    #   The second part is constant for a given state of covered sets.
    
    # Let's precompute the "base cost" for each set: sum of duals of elements that are ALREADY covered.
    # Initially 0.
    # When we cover a set S, for every other set c, if v in S is also in c, then the base cost for c increases by dual[v].
    # This is expensive to update.
    
    # Simpler approach:
    # Iterate. At each step, find the set c that minimizes the "time" to become tight.
    # Time to tight for set c: (1 - current_sum_duals(c)) / uncovered_count_in_c(c).
    # If uncovered_count_in_c(c) == 0, it's already tight or irrelevant.
    
    # We will maintain:
    #   dual: array of length total_words
    #   covered: boolean array
    #   uncovered_count[c]: number of v in N(c) such that not covered[v]
    #   sum_duals_uncovered[c]: sum of dual[v] for v in N(c) such that not covered[v]
    #   base_sum_duals[c]: sum of dual[v] for v in N(c) such that covered[v]
    #   Total sum for c is sum_duals_uncovered[c] + base_sum_duals[c].
    #   Gap = 1 - (sum_duals_uncovered[c] + base_sum_duals[c]).
    #   If Gap <= 0, set c is tight.
    
    # Initialization
    uncovered_count = np.zeros(total_words, dtype=np.int32)
    sum_duals_uncovered = np.zeros(total_words, dtype=np.float64)
    base_sum_duals = np.zeros(total_words, dtype=np.float64)
    
    for c in range(total_words):
        # Initially all elements are uncovered
        uncovered_count[c] = len(neighbors[c])
        # duals are 0, so sums are 0
    
    remaining_uncovered = total_words
    
    while remaining_uncovered > 0 and time.time() < deadline - 5.0:
        # Find the set with the minimum time to become tight
        # time_to_tight[c] = Gap[c] / uncovered_count[c]
        # We want min time_to_tight.
        
        # We can compute this vectorized.
        # Avoid division by zero.
        with np.errstate(divide='ignore', invalid='ignore'):
            gaps = 1.0 - (sum_duals_uncovered + base_sum_duals)
            # If uncovered_count is 0, time is inf (or 0 if gap<=0, but those are already covered/tight)
            # We only care about sets with uncovered_count > 0.
            times = np.where(uncovered_count > 0, gaps / uncovered_count, np.inf)
            
            # If any gap <= 0, time is 0 (or negative, treat as 0).
            # Actually if gap <= 0, it's already tight. We should pick one.
            # The formula handles it: if gap <= 0, time <= 0.
            
            min_time_idx = np.argmin(times)
            min_time = times[min_time_idx]
            
            if not np.isfinite(min_time) or min_time < 0:
                # If min_time is inf, we are stuck? Should not happen if remaining_uncovered > 0.
                # If min_time < 0, it means a set is already tight (gap < 0).
                # We pick that set.
                pass
            
            c = min_time_idx
            if uncovered_count[c] == 0:
                # Should not happen if remaining_uncovered > 0, unless logic error.
                break
            
            # Raise duals of all uncovered elements by min_time
            # This is O(N) to update dual array.
            dual[~covered] += min_time
            
            # Update sum_duals_uncovered for all sets
            # sum_duals_uncovered[c] increases by min_time * uncovered_count[c]
            sum_duals_uncovered += min_time * uncovered_count
            
            # Select set c
            is_code[c] = True
            code_list.append(c)
            
            # Update covered status and counts
            neigh_c = neighbors[c]
            for v in neigh_c:
                if not covered[v]:
                    covered[v] = True
                    remaining_uncovered -= 1
                    
                    # For all sets s containing v, update:
                    # 1. uncovered_count[s] -= 1
                    # 2. base_sum_duals[s] += dual[v]  (v moves from uncovered to covered)
                    # 3. sum_duals_uncovered[s] -= dual[v] (v no longer in uncovered sum)
                    
                    # We need to iterate over all sets containing v.
                    # This is the "inverse neighbors".
                    # We didn't precompute inverse neighbors.
                    # Let's precompute them now? Or compute on the fly?
                    # Precomputing inverse neighbors for 6561 elements, each in ~17 sets?
                    # Total edges ~ 6561 * 17 = 111k.
                    # It's better to precompute.
                    
                    # Wait, I haven't precomputed inverse neighbors in this block.
                    # I will add that.
                    pass
        
        # The loop above is messy. Let's refactor to precompute inverse neighbors.
        
    # Refactoring: Precompute inverse neighbors
    # inv_neighbors[v] = list of sets c such that v in N(c)
    inv_neighbors = [None] * total_words
    for c in range(total_words):
        for v in neighbors[c]:
            if inv_neighbors[v] is None:
                inv_neighbors[v] = []
            inv_neighbors[v].append(c)
    # Convert to arrays for speed?
    inv_neighbors_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in inv_neighbors]
    
    # Restart the main loop with proper data structures
    dual = np.zeros(total_words, dtype=np.float64)
    covered = np.zeros(total_words, dtype=bool)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    uncovered_count = np.zeros(total_words, dtype=np.int32)
    sum_duals_uncovered = np.zeros(total_words, dtype=np.float64)
    base_sum_duals = np.zeros(total_words, dtype=np.float64)
    
    for c in range(total_words):
        uncovered_count[c] = len(neighbors[c])
        
    remaining_uncovered = total_words
    
    while remaining_uncovered > 0 and time.time() < deadline - 10.0:
        with np.errstate(divide='ignore', invalid='ignore'):
            gaps = 1.0 - (sum_duals_uncovered + base_sum_duals)
            times = np.where(uncovered_count > 0, gaps / uncovered_count, np.inf)
            
            # Handle negative gaps (already tight)
            # If gap <= 0, time should be 0 (or negative).
            # np.argmin will pick the most negative gap if any, which is correct (pick tight set).
            min_time_idx = np.argmin(times)
            min_time = times[min_time_idx]
            
            if not np.isfinite(min_time):
                # Stuck?
                break
                
            c = min_time_idx
            if uncovered_count[c] == 0:
                # Should not happen
                break
            
            # If min_time is negative, it means the set is already tight (gap < 0).
            # We raise duals by min_time (negative)? No, duals can't decrease.
            # If gap < 0, we just pick it. The "time" is effectively 0.
            # So we only raise duals if min_time > 0.
            if min_time > 0:
                dual[~covered] += min_time
                sum_duals_uncovered += min_time * uncovered_count
            
            # Select set c
            is_code[c] = True
            code_list.append(c)
            
            # Update state for all v in N(c)
            neigh_c = neighbors[c]
            for v in neigh_c:
                if not covered[v]:
                    covered[v] = True
                    remaining_uncovered -= 1
                    
                    # Update all sets s containing v
                    for s in inv_neighbors_arr[v]:
                        uncovered_count[s] -= 1
                        # Move dual[v] from uncovered sum to base sum
                        base_sum_duals[s] += dual[v]
                        sum_duals_uncovered[s] -= dual[v]
                        
    # Final Polish: Try to remove redundant codewords
    # This is a local search step to clean up the primal-dual solution.
    # Rebuild coverage counts
    cov_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cov_count[neighbors[c]] += 1
        
    final_code = code_list[:]
    final_code_set = set(final_code)
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try to remove each codeword
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            # Check if any neighbor has count 1
            if np.all(cov_count[neighbors[c]] > 1):
                cov_count[neighbors[c]] -= 1
                final_code_set.remove(c)
                final_code.pop(i)
                improved = True
                
    if not np.all(cov_count > 0):
        # Fallback to code_list if final_code is invalid (shouldn't happen)
        return [list(digits_of[i]) for i in code_list]
        
    return [list(digits_of[i]) for i in final_code]
