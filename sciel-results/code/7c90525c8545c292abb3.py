"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic, improved tie-breaking."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]
            
            if len(cands) > 1:
                # Tie-break: prefer vertex whose neighbors have lower total degree
                # (removing it frees up less "valuable" vertices)
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
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Enhanced simulated annealing with diverse moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 4.0
        cooling = 0.9993
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0
        
        while time.time() < sa_deadline:
            step += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.4 and len(not_in_set) > 0:
                # ADD move: try to add a vertex with minimum conflicts
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.8 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SMART SWAP move: Remove a vertex with high conflict count, add one with low
                # This tends to improve density/size
                # Find in-set vertex with max conflicts
                in_conf_counts = count_conflicts[in_set_indices]
                # We want to remove a vertex that is "bad" (high conflicts) or just random?
                # Let's pick a random in-set vertex to remove, but bias towards high conflict if we want to clear space
                # Or just random swap is fine if we have good add logic.
                
                # Let's try: Pick v_add from not_in_set with min conflicts.
                # Pick v_remove from in_set.
                # If v_add conflicts with v_remove, it's a clean swap if no other conflicts.
                
                # Select v_add
                out_conf_counts = count_conflicts[not_in_set]
                min_out_count = np.min(out_conf_counts)
                min_out_cands = not_in_set[out_conf_counts == min_out_count]
                v_add = min_out_cands[rng.integers(len(min_out_cands))]
                
                # Select v_remove: Random from in_set
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Check if v_add conflicts with any in-set vertex other than v_remove
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                
                if len(other_conflicts) == 0:
                    # Clean swap
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    # Note: count_conflicts for neighbors of v_remove decreases by 1
                    # count_conflicts for neighbors of v_add increases by 1
                    # This is handled above.
                else:
                    # If v_add conflicts with others, we must remove them too.
                    # This is a "kick-out" move.
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        stuck_counter += 1
                        
            else:
                # REMOVE move or RESTART-ish move
                # If we are stuck, try removing a vertex to escape local optimum
                if len(in_set_indices) > 0:
                    v = in_set_indices[rng.integers(len(in_set_indices))]
                    delta = -1
                    # Accept removal with some probability, higher when T is high
                    accept = rng.random() < np.exp(delta / T)
                    
                    if accept:
                        in_set[v] = False
                        current_size += delta
                        count_conflicts[neighbors[v]] -= 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 1500:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def polish_greedy_add(in_set_init, max_time=5.0):
        """Polish by greedily adding non-conflicting vertices using vectorized operations."""
        polish_set = in_set_init.copy()
        polish_size = int(polish_set.sum())
        polish_deadline = time.time() + max_time
        
        # Vectorized polish: Find all vertices that do not conflict with current set
        # and add them all at once. Repeat until no more can be added.
        # This is equivalent to finding a maximal independent set in the subgraph induced by non-set vertices,
        # but we just do iterative "add all compatible" which is fast and effective for dense graphs.
        
        while time.time() < polish_deadline:
            # Calculate for each vertex if it conflicts with any vertex in polish_set
            # conflicts is N4xN4 boolean matrix
            # If we do conflicts @ polish_set, we get a vector of counts of conflicts with set.
            # But we need boolean: does it conflict with ANY?
            # np.any(conflicts[:, polish_set_indices], axis=1) is slow if done naively.
            # Better: 
            # Let S be indices in polish_set.
            # We want to know for each v in ~polish_set if conflicts[v, S].any() is False.
            
            not_in_mask = ~polish_set
            if not np.any(not_in_mask):
                break
                
            # Get indices of vertices not in set
            not_in_indices = np.where(not_in_mask)[0]
            
            # Check conflicts of these candidates against the current set
            # We can compute the number of conflicts for each candidate with the current set
            # conflict_counts = conflicts[not_in_indices] @ polish_set
            # If conflict_counts == 0, the vertex can be added.
            
            # To save memory/time, we can just check:
            # compatible = np.all(~conflicts[not_in_indices][:, polish_set], axis=1)
            # This is O(N^2) per step but vectorized.
            
            # Alternative: 
            # We want to add all v such that no u in polish_set has conflicts[v,u]==True.
            # This is equivalent to: for all u in polish_set, conflicts[v,u] == False.
            
            # Let's try the matrix multiplication approach for speed if N4 is large enough, 
            # but boolean logic might be faster for sparse/dense mix.
            # Actually, `conflicts` is boolean. `polish_set` is boolean.
            # `conflicts @ polish_set` gives integer counts.
            
            counts = conflicts[not_in_indices] @ polish_set
            
            # Vertices with 0 conflicts can be added
            addable_mask = counts == 0
            if not np.any(addable_mask):
                break
                
            # Add them
            add_indices = not_in_indices[addable_mask]
            polish_set[add_indices] = True
            polish_size += len(add_indices)
            
            # If we added nothing, break (should be caught by addable_mask check)
            if len(add_indices) == 0:
                break
                
        return polish_set, polish_size
    
    # Multiple restarts
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
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Polish: greedy add on the best found
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            polish_time = min(remaining_time * 0.4, 6.0)
            if polish_time > 1:
                polish_result, polish_size = polish_greedy_add(sa_result if sa_size >= current_size else in_set, max_time=polish_time)
                
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
                
                # 4. Second SA on polished result
                remaining_time2 = deadline - 3 - time.time()
                if remaining_time2 > 4 and polish_size >= current_size:
                    sa2_time = min(remaining_time2 * 0.6, 8.0)
                    if sa2_time > 2:
                        sa2_result, sa2_size = sa_local_search(polish_result, max_time=sa2_time)
                        if sa2_size > best_size:
                            best_size = sa2_size
                            best_in_set = sa2_result.copy()
                        
                        # Final polish
                        remaining_time3 = deadline - 2 - time.time()
                        if remaining_time3 > 2:
                            final_polish, final_size = polish_greedy_add(sa2_result, max_time=min(remaining_time3 * 0.5, 4.0))
                            if final_size > best_size:
                                best_size = final_size
                                best_in_set = final_polish.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
