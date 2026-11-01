"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate all integer vectors in R^11 with norm^2 = 4
    # Pattern 1: weight 4, entries in {-1, 0, 1}
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Pattern 2: weight 1, entries in {-2, 0, 2}
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    
    all_candidates = v4_candidates + v1_candidates
    vectors = np.array(all_candidates, dtype=np.int64)
    N = len(vectors)
    
    if N == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors @ vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]
            
            if len(cands) > 1:
                # Tie-break by sum of degrees of alive neighbors
                scores = np.zeros(len(cands))
                for idx, c in enumerate(cands):
                    nbrs_c = neighbors[c]
                    alive_nbrs = nbrs_c[alive[nbrs_c]]
                    if len(alive_nbrs) > 0:
                        scores[idx] = np.sum(degrees[alive_nbrs])
                    else:
                        scores[idx] = 0
                chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def fast_greedy_add(in_set_init, max_time=3.0):
        """Fast polish: repeatedly add all vertices that have zero conflicts with current set."""
        polish_set = in_set_init.copy()
        polish_size = int(polish_set.sum())
        polish_deadline = time.time() + max_time
        
        while time.time() < polish_deadline:
            not_in_mask = ~polish_set
            if not np.any(not_in_mask):
                break
            not_in_indices = np.where(not_in_mask)[0]
            counts = conflicts[not_in_indices] @ polish_set
            addable_mask = counts == 0
            if not np.any(addable_mask):
                break
            add_indices = not_in_indices[addable_mask]
            polish_set[add_indices] = True
            polish_size += len(add_indices)
            
        return polish_set, polish_size
    
    def tabu_search(in_set_init, max_time=15.0):
        """Tabu search with diversification for maximum independent set."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # Tabu list: (vertex, iteration) - vertex is tabu until this iteration
        tabu_list = {}
        tabu_tenure = 30
        max_tabu_size = 1000
        
        # Count conflicts for each vertex
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        ts_deadline = time.time() + max_time
        iteration = 0
        no_improve = 0
        
        while time.time() < ts_deadline:
            iteration += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0:
                break
            
            # Find best candidate to add (min conflicts, not tabu)
            counts = count_conflicts[not_in_set]
            
            # Filter out tabu vertices
            tabu_vertices = set(tabu_list.keys())
            non_tabu_mask = np.array([v not in tabu_vertices for v in not_in_set])
            
            if np.any(non_tabu_mask):
                candidates = not_in_set[non_tabu_mask]
                cand_counts = counts[non_tabu_mask]
            else:
                candidates = not_in_set
                cand_counts = counts
            
            if len(candidates) == 0:
                break
            
            # Choose vertex with minimum conflicts among non-tabu
            min_count = np.min(cand_counts)
            min_cands = candidates[cand_counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]
            
            k = counts[v]
            delta = 1 - k
            
            # Aspiration criterion: accept if improves best
            is_aspiration = (current_size + delta > local_best_size)
            
            if delta > 0 or is_aspiration or (delta == 0 and rng.random() < 0.3) or (delta < 0 and rng.random() < 0.1):
                # Execute move: add v, remove conflicting vertices
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                        # Add removed vertices to tabu list
                        tabu_list[u] = iteration + tabu_tenure
                
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                
                # Add v to tabu list
                tabu_list[v] = iteration + tabu_tenure
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            # Clean up expired tabu entries
            if len(tabu_list) > max_tabu_size:
                tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
            
            # Diversification: if stuck, do a random restart
            if no_improve > 500:
                # Remove a random subset and re-add
                if len(in_set_indices) > 5:
                    remove_count = max(3, len(in_set_indices) // 5)
                    to_remove = rng.choice(in_set_indices, size=remove_count, replace=False)
                    for u in to_remove:
                        in_set[u] = False
                        count_conflicts[neighbors[u]] -= 1
                    current_size -= remove_count
                    no_improve = 0
                    tabu_list.clear()
            
        return local_best_in_set, local_best_size
    
    # Multiple restarts with different strategies
    n_restarts = 0
    max_restarts = 80
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. Fast greedy add polish
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        polish_time = min(remaining_time * 0.1, 2.0)
        if polish_time > 0.3:
            polish_result, polish_size = fast_greedy_add(in_set, max_time=polish_time)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
                current_size = polish_size
                in_set = polish_result
        
        # 3. Tabu Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        ts_time = min(remaining_time * 0.7, 12.0)
        if ts_time < 2:
            ts_time = 2.0
        
        ts_result, ts_size = tabu_search(in_set, max_time=ts_time)
        
        if ts_size > best_size:
            best_size = ts_size
            best_in_set = ts_result.copy()
        
        # 4. Final polish
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            final_polish, final_size = fast_greedy_add(ts_result, max_time=min(remaining_time * 0.3, 3.0))
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_polish.copy()
    
    # Final result
    final_indices = np.where(best_in_set)[0]
    final_vectors = vectors[final_indices]
    
    return final_vectors

# EVOLVE-BLOCK-END
