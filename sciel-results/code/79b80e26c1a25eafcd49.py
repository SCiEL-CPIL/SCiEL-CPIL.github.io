import numpy as np
import itertools
import time

def construct():
    D = 11
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # 1. Generate candidate vectors with squared norm 4
    # Type 1: +/- 2e_i (11 * 2 = 22 vectors)
    # Type 2: 4 entries of +/-1, rest 0 (C(11,4) * 2^4 = 330 * 16 = 5280 vectors)
    
    type1_vectors = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            type1_vectors.append(v)
    type1_array = np.array(type1_vectors, dtype=np.int64)
    
    type2_list = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx in enumerate(indices):
                v[idx] = signs[k]
            type2_list.append(v)
            
    type2_array = np.array(type2_list, dtype=np.int32)
    N = len(type2_array)
    
    # 2. Compute Conflict Matrix for Type 2 vectors
    # Two vectors conflict if their distance is < max_norm = 2.
    # dist^2 = 8 - 2a.b >= 4  =>  a.b <= 2.
    # So conflict if a.b > 2.
    
    M = type2_array @ type2_array.T
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Precompute neighbor lists for efficiency
    neighbors = [np.where(Conflict[i] == 1)[0] for i in range(N)]
    
    # 3. Greedy Maximum Independent Set (MIS) on Type 2 graph
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline - 5:
                break
                
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors from active set
            neighbors_mask = Conflict[idx].astype(bool)
            neighbors_mask[idx] = True
            removed_indices = np.where(neighbors_mask & active)[0]
            
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
            
            active[removed_indices] = False
            
        return selected
    
    # 4. Local search: try to swap one selected vector for two compatible non-selected vectors
    def local_search(selected_indices, max_iterations=500):
        if time.time() > deadline - 10:
            return selected_indices
            
        selected_set = set(selected_indices)
        rng_ls = np.random.default_rng(42)
        
        # Precompute conflict count with selected set for each non-selected node
        # For efficiency, we'll compute on the fly but cache where possible
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 10:
            improved = False
            iteration += 1
            
            # Shuffle order to break symmetry
            indices_list = list(selected_set)
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline - 10:
                    return list(selected_set)
                
                # Find candidates j not in selected_set that conflict ONLY with i
                # and do NOT conflict with any other element in selected_set
                neighbors_of_i = neighbors[i]
                
                # Filter to those not in selected_set
                valid_js = [j for j in neighbors_of_i if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                # For each candidate j, check if it's compatible with S\{i}
                compatible_with_rest = []
                for j in valid_js:
                    # Check conflicts with S \ {i}
                    conflict_found = False
                    for k in selected_set:
                        if k == i:
                            continue
                        if Conflict[j, k]:
                            conflict_found = True
                            break
                    if not conflict_found:
                        compatible_with_rest.append(j)
                
                if len(compatible_with_rest) < 2:
                    continue
                
                # Find a pair (j1, j2) in compatible_with_rest that are compatible with each other
                found_pair = False
                for idx1 in range(len(compatible_with_rest)):
                    if time.time() > deadline - 10:
                        return list(selected_set)
                    j1 = compatible_with_rest[idx1]
                    for idx2 in range(idx1 + 1, len(compatible_with_rest)):
                        j2 = compatible_with_rest[idx2]
                        if not Conflict[j1, j2]:
                            # Swap i for j1, j2
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if found_pair:
                    # Restart the while loop with the new set
                    break
            
            if not improved:
                break
        
        return list(selected_set)
    
    # 5. Run multiple greedy restarts, then local search on the best
    best_size = 0
    best_set = []
    
    # Use first 60% of time for greedy restarts
    greedy_deadline = start_time + (deadline - start_time) * 0.6
    
    num_restarts = 0
    while time.time() < greedy_deadline:
        if time.time() > deadline - 10:
            break
        selected = run_greedy(num_restarts)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
        num_restarts += 1
    
    # Use remaining time for local search on the best set
    if best_size > 0 and time.time() < deadline - 15:
        refined_set = local_search(best_set)
        if len(refined_set) > best_size:
            best_size = len(refined_set)
            best_set = refined_set
    
    # 6. Combine results
    if best_size > 0:
        selected_type2 = type2_array[best_set]
        final_vectors = np.vstack([type1_array, selected_type2])
    else:
        final_vectors = type1_array
        
    return final_vectors

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} vectors")
