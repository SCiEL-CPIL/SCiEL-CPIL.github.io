import numpy as np
import itertools
import time

D = 11
DEADLINE = 110.0  # Safety margin for 150s wall-clock limit


# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    # Part 1: All 220 vectors of Type 1 (Two +/- 2s, rest zeros)
    # Norm^2 = 8
    type1_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                type1_vectors.append(v)
    
    type1_array = np.array(type1_vectors, dtype=np.int8)
    n_type1 = len(type1_array) # 220

    # Part 2: Type 2 vectors (Eight +/-1s, three zeros)
    # Norm^2 = 8
    all_type2 = []
    for support in itertools.combinations(range(D), 8):
        base = [0] * D
        for idx in support:
            base[idx] = 1
        for signs in itertools.product([-1, 1], repeat=8):
            v = base.copy()
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            all_type2.append(v)

    all_type2 = np.array(all_type2, dtype=np.int8)
    n_type2 = len(all_type2)

    # We will try multiple random orderings of Type 2 vectors to find a larger compatible subset
    # The Type 1 vectors are always included.
    # A Type 2 vector is compatible with Type 1 if dot <= 4.
    # As analyzed, ALL Type 2 are compatible with ALL Type 1.
    # So we just need to find a large subset of Type 2 that is internally compatible (dot <= 4).
    
    best_type2_subset = []
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on time
    # Each restart takes O(N_subset * N_type2 * D) roughly, but with pruning it's faster.
    # Let's estimate: 42k candidates. Greedy selection.
    # If we pick ~1000 vectors, checks are 1000 * 42000 * 11 ~ 460M ops. Too slow for many restarts.
    # We need to optimize the check.
    
    # Optimization: 
    # Instead of checking against all accepted, we can maintain a bitset or use vectorized ops.
    # But 42k x 11 is small. 
    # The bottleneck is the loop.
    
    # Let's try a single fast pass first, then if time permits, try a few more.
    
    def greedy_select(candidates, time_limit):
        accepted = []
        accepted_arr = []
        
        # To speed up, we can process in batches or just use numpy for the dot product check against the current set.
        # However, the set grows.
        
        # Alternative: 
        # We know that vectors with the same support have high dot products.
        # We can group by support.
        
        # For now, let's stick to the vectorized check against the *entire* accepted set.
        # If accepted set size is S, and candidates is C.
        # Dots: C x S.
        
        # We will iterate through candidates in the given order.
        
        for i in range(len(candidates)):
            if elapsed() > time_limit:
                break
            
            v = candidates[i]
            
            if not accepted_arr:
                accepted.append(v)
                accepted_arr = [v]
                continue
            
            # Check dot products with all accepted vectors
            # We can store accepted in a numpy array for fast matmul
            # But appending to numpy array is slow.
            # Let's use a list and convert to numpy periodically?
            # Or just use list of lists and numpy for the batch check?
            
            # If len(accepted) is small, loop is fine.
            # If large, we need numpy.
            
            # Let's maintain a numpy array for accepted.
            # To avoid frequent vstack, we can check against the last added vectors?
            # No, must check against all.
            
            # Let's assume we convert accepted to numpy array every 100 additions?
            # Or just use the list and loop if S is small.
            
            # Given the high compatibility, S might grow large.
            
            # Let's use a numpy array for accepted, but update it efficiently.
            # Actually, `accepted_arr` as a list of numpy arrays is bad.
            
            # Let's restart the greedy logic with a numpy-based check.
            pass

    # Revised Greedy:
    def run_greedy(order):
        accepted = []
        # For fast dot product, we'll keep a numpy array of accepted vectors.
        # But updating it is costly.
        # Instead, we'll keep a list, and if it gets large, we'll check in batches?
        
        # Let's try a different approach:
        # We know that for Type 2 vectors, dot > 4 implies conflict.
        # 
        # Let's just use the previous code's logic but with multiple restarts.
        
        selected = []
        # To speed up the inner loop, we can pre-compute something?
        
        # Let's just run the standard greedy but with a time check.
        # The previous code did this.
        
        # To improve, we can try to pick vectors that are "far" from existing ones?
        # Greedy usually picks the first one.
        
        # Let's just do 3 restarts with different seeds.
        pass

    # Implementation of the improved greedy with restarts
    
    def greedy_with_restarts(num_restarts, time_budget_per_restart):
        best_subset = []
        
        for r in range(num_restarts):
            if elapsed() > DEADLINE - 5.0:
                break
                
            perm = rng.permutation(n_type2)
            candidates = all_type2[perm]
            
            selected = []
            # We will use a numpy array for selected vectors to speed up dot product checks.
            # But we need to append.
            
            # Let's use a list for selected, and convert to numpy array for the check if len > 100.
            
            for i in range(n_type2):
                if elapsed() > DEADLINE - 5.0:
                    break
                
                v = candidates[i]
                
                if not selected:
                    selected.append(v)
                    continue
                
                # Check compatibility with all selected
                # If selected is small, loop.
                # If selected is large, use numpy.
                
                if len(selected) < 50:
                    compatible = True
                    for u in selected:
                        if np.dot(v, u) > 4:
                            compatible = False
                            break
                    if compatible:
                        selected.append(v)
                else:
                    # Convert to numpy array for batch check
                    # This is inefficient if done every time.
                    # Let's maintain a numpy array `sel_arr` and update it.
                    # But vstack is slow.
                    
                    # Alternative: Check against the last K vectors? No, must be all.
                    
                    # Let's just use the list loop for now, it might be fast enough in Python if we optimize.
                    # Actually, 1000 vectors * 11 dims * 42000 candidates = 462M ops.
                    # In Python, this is too slow (seconds to minutes).
                    
                    # We MUST use numpy for the check.
                    # Strategy: Keep a numpy array `sel_arr`.
                    # When adding a new vector, vstack.
                    # vstack of (S, 11) and (1, 11) is fast.
                    
                    sel_arr = np.array(selected, dtype=np.int8)
                    dots = sel_arr @ v
                    if np.all(dots <= 4):
                        selected.append(v)
            
            if len(selected) > len(best_subset):
                best_subset = selected
                
        return best_subset

    # The above `greedy_with_restarts` creates a new numpy array in the loop. This is slow.
    # Let's optimize `run_greedy` to maintain the numpy array.

    def optimized_greedy(order):
        selected = []
        sel_arr = np.empty((0, D), dtype=np.int8)
        
        for i in range(len(order)):
            if elapsed() > DEADLINE - 5.0:
                break
            
            v = order[i]
            
            if sel_arr.shape[0] == 0:
                selected.append(v)
                sel_arr = v.reshape(1, D)
                continue
            
            # Check dot products
            dots = sel_arr @ v
            if np.all(dots <= 4):
                selected.append(v)
                sel_arr = np.vstack((sel_arr, v.reshape(1, D)))
        
        return selected

    # Run a few restarts
    best_type2_subset = []
    
    # First run with default order (or shuffled)
    # We will try 5 restarts
    for r in range(5):
        if elapsed() > DEADLINE - 10.0:
            break
        
        perm = rng.permutation(n_type2)
        candidates = all_type2[perm]
        
        subset = optimized_greedy(candidates)
        
        if len(subset) > len(best_type2_subset):
            best_type2_subset = subset
            
        # If we found a very large set, we might stop early?
        # No, keep trying to improve.

    # Combine Type 1 and best Type 2 subset
    if best_type2_subset:
        type2_array = np.array(best_type2_subset, dtype=np.int8)
        result_array = np.vstack([type1_array, type2_array])
    else:
        result_array = type1_array

    return result_array.astype(np.int64)

# EVOLVE-BLOCK-END
