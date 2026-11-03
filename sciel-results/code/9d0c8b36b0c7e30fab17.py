import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Generate candidates
    candidates = []
    
    # Type 1: One 2, rest 0 (Norm squared = 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s/-1s, rest 0 (Norm squared = 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # Norm squared is 4 for all.
    # Distance squared d^2 = 4 + 4 - 2 dot = 8 - 2 dot.
    # We require d^2 >= 4 (max norm squared).
    # 8 - 2 dot >= 4 => 4 >= 2 dot => dot <= 2.
    # So conflict if dot > 2.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Type 1 vectors (first 2*D = 22 vectors) never conflict with anyone else?
    # Type 1 dot Type 1: 4 if same, 0 if diff.
    # Type 1 dot Type 2: 2 or -2 or 0.
    # So Type 1 vectors only conflict with themselves (diagonal).
    # We can fix all Type 1 vectors in the solution.
    n_type1 = 2 * D
    type1_indices = list(range(n_type1))
    
    # We need to find an independent set in the subgraph of Type 2 vectors.
    # Type 2 indices are n_type1 to N-1.
    n_type2 = N - n_type1
    
    # Extract sub-conflict matrix for Type 2
    Conflict_2 = Conflict[n_type1:, n_type1:].copy()
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(n_type2, dtype=bool)
        # Use int64 for degrees to avoid casting issues
        deg = Conflict_2.sum(axis=1).astype(np.int64)
        selected_local = []
        
        while active.any():
            if time.time() > deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected_local.append(idx)
            
            # Remove idx and neighbors
            neighbors_mask = Conflict_2[idx].astype(bool)
            removed_mask = np.zeros(n_type2, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            if len(removed_indices) > 0:
                removal_counts = Conflict_2[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
            
            active[removed_mask] = False
            
        return selected_local

    def local_search(selected_local_indices):
        if time.time() > deadline:
            return list(selected_local_indices)
        
        selected_set = set(selected_local_indices)
        improved = True
        iters = 0
        max_iters = 50
        
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng(iters * 12345 + 7)
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline - 5:
                    return list(selected_set)
                
                # Find j not in selected_set that conflicts ONLY with i
                neighbors_i = np.where(Conflict_2[i])[0]
                valid_js = [j for j in neighbors_i if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                # For each j in valid_js, check if it conflicts with any k in selected_set \ {i}
                compatible_with_rest = []
                for j in valid_js:
                    # Check conflicts with S \ {i}
                    conflict_count = 0
                    for k in selected_set:
                        if k == i:
                            continue
                        if Conflict_2[j, k]:
                            conflict_count += 1
                            break
                    if conflict_count == 0:
                        compatible_with_rest.append(j)
                
                if len(compatible_with_rest) < 2:
                    continue
                
                # Find a pair (j1, j2) in compatible_with_rest such that Conflict_2[j1, j2] == 0
                found = False
                for a_idx in range(len(compatible_with_rest)):
                    if found:
                        break
                    ja = compatible_with_rest[a_idx]
                    for b_idx in range(a_idx + 1, len(compatible_with_rest)):
                        jb = compatible_with_rest[b_idx]
                        if Conflict_2[ja, jb] == 0:
                            selected_set.remove(i)
                            selected_set.add(ja)
                            selected_set.add(jb)
                            improved = True
                            found = True
                            break
                            
                if improved:
                    break
        
        return list(selected_set)

    def simulated_annealing(initial_set, max_steps=20000):
        """
        Simulated Annealing on the independent set.
        Moves: 
        1. Remove one vertex from set.
        2. Add one vertex to set (if compatible).
        3. Swap one vertex in set with one outside (if compatible).
        Objective: Maximize size of set.
        """
        if time.time() > deadline:
            return list(initial_set)
        
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(current_set)
        best_size = current_size
        
        T = 10.0
        cooling_rate = 0.9999
        rng = np.random.default_rng(42)
        
        # Precompute list of all vertices for random selection
        all_vertices = np.arange(n_type2)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Randomly choose a move type
            move_type = rng.integers(0, 3)
            new_set = set(current_set)
            
            if move_type == 0: # Remove
                if not current_set:
                    continue
                v_to_remove = rng.choice(list(current_set))
                new_set.remove(v_to_remove)
                delta_size = -1
                delta_conflict = 0 # Removing always reduces or keeps conflict count same (actually reduces)
                # Actually, removing a vertex can only decrease the number of conflicts in the set.
                # Since we are maintaining a valid independent set, removing a vertex always results in a valid set.
                # The "energy" we care about is just size.
                
            elif move_type == 1: # Add
                # Pick a random vertex not in current_set
                # To speed up, we can just pick random from all and check if in set
                # Or maintain a list of outside vertices? That's expensive to update.
                # Let's just pick random from all_vertices
                v_to_add = rng.choice(all_vertices)
                if v_to_add in current_set:
                    continue
                # Check if v_to_add is compatible with all in current_set
                is_compatible = True
                for v in current_set:
                    if Conflict_2[v_to_add, v]:
                        is_compatible = False
                        break
                if not is_compatible:
                    continue
                new_set.add(v_to_add)
                delta_size = 1
                
            else: # Swap
                if not current_set:
                    continue
                v_in = rng.choice(list(current_set))
                v_out = rng.choice(all_vertices)
                if v_out in current_set:
                    continue
                # Check if v_out is compatible with all in current_set \ {v_in}
                is_compatible = True
                for v in current_set:
                    if v == v_in:
                        continue
                    if Conflict_2[v_out, v]:
                        is_compatible = False
                        break
                if not is_compatible:
                    continue
                new_set.remove(v_in)
                new_set.add(v_out)
                delta_size = 0 # Size unchanged, but structure changes, might allow future growth
                
            new_size = len(new_set)
            
            # Acceptance criteria
            if new_size >= best_size:
                current_set = new_set
                current_size = new_size
                if new_size > best_size:
                    best_set = set(new_set)
                    best_size = new_size
            else:
                # Accept with probability exp(-delta / T) where delta is change in "cost"
                # Here cost = -size. So delta_cost = -delta_size.
                # If delta_size < 0, delta_cost > 0.
                delta = -delta_size # This is the increase in cost (decrease in size)
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_set = new_set
                        current_size = new_size
            
            T *= cooling_rate
            if T < 1e-4:
                break
                
        return list(best_set)

    best_set_local = []
    best_size = 0
    
    # Phase 1: Greedy restarts to find a good starting point
    n_restarts_greedy = 20
    for seed in range(n_restarts_greedy):
        if time.time() > deadline - 30:
            break
        result = run_greedy(seed)
        if len(result) > best_size:
            best_size = len(result)
            best_set_local = result
            
    # Phase 2: Local search on the best greedy result
    if time.time() < deadline - 20 and best_set_local:
        improved_set = local_search(best_set_local)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set_local = improved_set

    # Phase 3: Simulated Annealing starting from the best found so far
    if time.time() < deadline - 10 and best_set_local:
        sa_result = simulated_annealing(best_set_local, max_steps=5000)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set_local = sa_result
            
        # If SA improved, do another round of local search
        if time.time() < deadline - 5:
            improved_set = local_search(best_set_local)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set_local = improved_set

    # Phase 4: More Greedy restarts if time permits, combined with SA
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        for seed in range(n_restarts_greedy, n_restarts_greedy + 30):
            if time.time() > deadline - 10:
                break
            result = run_greedy(seed)
            # Quick local search
            if time.time() < deadline - 10:
                result = local_search(result)
            
            if len(result) > best_size:
                best_size = len(result)
                best_set_local = result
                
                # Run SA on this new best
                if time.time() < deadline - 5:
                    sa_result = simulated_annealing(best_set_local, max_steps=2000)
                    if len(sa_result) > best_size:
                        best_size = len(sa_result)
                        best_set_local = sa_result

    # Construct final vector list
    final_indices = type1_indices + [n_type1 + idx for idx in best_set_local]
    
    vectors = candidates[final_indices]
    
    if len(vectors) == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return vectors.astype(np.int64)

if __name__ == "__main__":
    v = construct()
    print(f"Found {len(v)} vectors")
