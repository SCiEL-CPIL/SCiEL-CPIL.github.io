"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Generate W1 vectors (weight 1, entries in {-2, 0, 2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # 2. Generate W4 vectors (weight 4, entries in {-1, 0, 1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1

    # 3. Compute conflict matrix for W4 vectors
    # Conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays for fast SA updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Find min degree among alive
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            
            # Remove chosen and its neighbors from alive
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Subtract 1 for each neighbor removed
                # conflicts[rem][:, to_remove] is a boolean matrix (len(rem) x len(to_remove))
                # Summing along axis 1 gives count of removed neighbors for each rem node
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing with 2-for-1 swaps."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.80:
                # 1-for-0: Add a vertex, remove its conflicts
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                    
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
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
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.95:
                # 2-for-1: Remove one, add two non-conflicting
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) < 1:
                    continue
                    
                remove_v = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Temporarily remove remove_v to find candidates
                # Candidates must not conflict with (in_set \ {remove_v})
                # and not conflict with each other.
                
                # Mask of vertices that conflict with current in_set (excluding remove_v)
                # This is expensive if done from scratch.
                # Approximation: Use count_conflicts.
                # A vertex u is a candidate if count_conflicts[u] == (1 if u conflicts with remove_v else 0)
                # AND u is not in set.
                
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                
                # Filter candidates that don't conflict with in_set \ {remove_v}
                # count_conflicts[u] includes remove_v if u conflicts with it.
                # We want count_conflicts[u] - (1 if conflicts[u, remove_v] else 0) == 0
                
                conflicts_with_remove = conflicts[not_in_set, remove_v]
                effective_conflicts = count_conflicts[not_in_set] - conflicts_with_remove.astype(np.int32)
                
                candidates = not_in_set[effective_conflicts == 0]
                
                if len(candidates) < 2:
                    stuck_counter += 1
                    continue
                
                # Try to find a pair of non-conflicting candidates
                # Sample pairs
                found_pair = None
                for _ in range(10):
                    i, j = rng.integers(len(candidates), size=2)
                    if i == j:
                        continue
                    ci, cj = candidates[i], candidates[j]
                    if not conflicts[ci, cj]:
                        found_pair = (ci, cj)
                        break
                
                if found_pair is None:
                    stuck_counter += 1
                    continue
                
                ci, cj = found_pair
                
                # Execute move: remove remove_v, add ci, cj
                # Delta is +1
                accept = rng.random() < np.exp(1.0 / T)
                
                if accept:
                    # Remove remove_v
                    in_set[remove_v] = False
                    for u in neighbors[remove_v]:
                        count_conflicts[u] -= 1
                    
                    # Add ci
                    to_remove_ci = np.where(conflicts[ci] & in_set)[0]
                    if len(to_remove_ci) > 0:
                         # Should be empty if our filter was correct, but safety check
                         in_set[to_remove_ci] = False
                         for u in to_remove_ci:
                             count_conflicts[neighbors[u]] -= 1
                             
                    in_set[ci] = True
                    count_conflicts[neighbors[ci]] += 1
                    
                    # Add cj
                    to_remove_cj = np.where(conflicts[cj] & in_set)[0]
                    if len(to_remove_cj) > 0:
                         in_set[to_remove_cj] = False
                         for u in to_remove_cj:
                             count_conflicts[neighbors[u]] -= 1
                             
                    in_set[cj] = True
                    count_conflicts[neighbors[cj]] += 1
                    
                    current_size = in_set.sum()
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # Perturbation: Remove 3-5 random in-set vertices
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) < 3:
                    continue
                    
                n_remove = min(rng.integers(3, 6), len(in_set_indices))
                to_remove = in_set_indices[rng.choice(len(in_set_indices), size=n_remove, replace=False)]
                
                for v in to_remove:
                    in_set[v] = False
                    for u in neighbors[v]:
                        count_conflicts[u] -= 1
                
                current_size = in_set.sum()
                stuck_counter = 0
                T = max(T, 2.5)
            
            if stuck_counter > 1000:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size

    # Main Loop: Multiple restarts
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        # 2. SA Polish
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 3:
            break
            
        sa_time = min(remaining_time * 0.5, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
            
        # 3. Second Polish if improved
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.4, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    # Final Polish
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 5 and best_size > 0:
        final_polish_time = min(remaining_time * 0.8, 15.0)
        final_result, final_size = sa_local_search(best_in_set, max_time=final_polish_time)
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_result

    # Construct Final Output
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Combine W4 and W1
    if len(result_v4) > 0:
        final_vectors = np.vstack([result_v4, vectors_v1])
    else:
        final_vectors = vectors_v1

    return final_vectors

# EVOLVE-BLOCK-END
