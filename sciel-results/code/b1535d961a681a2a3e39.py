"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill

    # Generate all candidate vectors with norm^2 in {2, 4}
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    all_candidates = v1_candidates + v2_candidates + v4_candidates
    vectors_all = np.array(all_candidates, dtype=np.int64)
    
    N = len(vectors_all)
    
    # Compute conflict matrix: conflict if dist^2 < max_norm^2 = 4
    # dist^2 = n1^2 + n2^2 - 2*dot, need >= 4
    # So conflict when dot > (n1^2 + n2^2 - 4) / 2
    norms_sq = np.sum(vectors_all ** 2, axis=1)
    dots = vectors_all @ vectors_all.T
    
    threshold = (norms_sq[:, None] + norms_sq[None, :] - 4) / 2.0
    conflicts = (dots > threshold).astype(np.uint8)
    np.fill_diagonal(conflicts, 0)
    
    # Precompute neighbor lists and conflict masks
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(12345)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def greedy_max_weight():
        """Greedy: pick vertex with max (norm contribution) / degree ratio."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Prefer lower degree, break ties by norm
            deg = degrees[alive_indices]
            min_deg = np.min(deg)
            cands = alive_indices[deg == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Simulated annealing with add and swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 2.0
        cooling = 0.9995
        min_T = 0.01
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            # Choose move type: add (60%), swap (40%)
            use_swap = rng.random() < 0.4
            
            if use_swap and current_size > 0:
                # Swap: pick v_in (not in set) and v_out (in set, neighbor of v_in)
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    use_swap = False
                else:
                    counts = count_conflicts[not_in_set]
                    min_count = np.min(counts)
                    threshold_count = min_count + 2
                    cands = not_in_set[counts <= threshold_count]
                    v_in = cands[rng.integers(len(cands))]
                    
                    # v_in conflicts with some in-set vertices; pick one to remove
                    in_set_nbrs = np.where(conflicts[v_in] & in_set)[0]
                    if len(in_set_nbrs) == 0:
                        # No conflict, just add
                        use_swap = False
                    else:
                        v_out = in_set_nbrs[rng.integers(len(in_set_nbrs))]
                        
                        # Delta: add v_in, remove v_out
                        # Gain from adding v_in: 1 - count_conflicts[v_in] (but we're removing v_out which is a conflict)
                        # After removing v_out, v_in's conflicts decrease by 1 (v_out was conflicting)
                        # But removing v_out may free up other vertices
                        # Net: +1 (add v_in) - 1 (remove v_out) + (new vertices that can be added? no, single swap)
                        # Actually: we add v_in and remove v_out. Size change = 0.
                        # But this might enable future improvements.
                        # For SA, accept based on whether it "helps": v_in has fewer conflicts after removal
                        new_conflicts_v_in = count_conflicts[v_in] - 1  # v_out removed
                        # v_out's neighbors lose one conflict
                        # Accept if v_in is "better" than v_out in some sense
                        # Use: accept if new_conflicts_v_in <= count_conflicts[v_out]
                        accept_prob = 1.0 if new_conflicts_v_in < count_conflicts[v_out] else \
                                      (0.5 if new_conflicts_v_in == count_conflicts[v_out] else \
                                       np.exp(-1.0 / T))
                        accept = rng.random() < accept_prob
                        
                        if accept:
                            # Remove v_out
                            in_set[v_out] = False
                            for u in neighbors[v_out]:
                                count_conflicts[u] -= 1
                            # Add v_in
                            in_set[v_in] = True
                            for u in neighbors[v_in]:
                                count_conflicts[u] += 1
                            # Size unchanged
                            stuck_counter += 1
                        else:
                            stuck_counter += 1
            else:
                # Add move
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                    
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                
                threshold_count = min_count + 1
                cands = not_in_set[counts <= threshold_count]
                v = cands[rng.integers(len(cands))]
                
                k = count_conflicts[v]
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
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
            
            if stuck_counter > 600:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def local_polish(in_set_init, max_iters=20000):
        """Exact local search: try all single add moves, accept improving ones."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # Precompute count_conflicts
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3:
            improved = False
            iters += 1
            
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            # Find best gain vertex
            counts = count_conflicts[not_in_set]
            gains = 1 - counts
            max_gain = np.max(gains)
            
            if max_gain > 0:
                best_cands = not_in_set[gains == max_gain]
                best_v = best_cands[rng.integers(len(best_cands))]
                
                to_remove = np.where(conflicts[best_v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[best_v] = True
                current_size += max_gain
                count_conflicts[neighbors[best_v]] += 1
                improved = True
        
        return in_set, current_size
    
    # Multiple restarts with varied initializations
    n_restarts = 0
    max_restarts = 120
    
    while time.time() < deadline - 12 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Alternate between initialization strategies
        if n_restarts % 2 == 0:
            in_set = greedy_min_degree()
        else:
            in_set = greedy_max_weight()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # SA Local Search
        remaining_time = deadline - 12 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.4, 10.0)
        if sa_time < 1.5:
            sa_time = 1.5
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # Exact local polish on best found so far
        remaining_time = deadline - 6 - time.time()
        if remaining_time > 2:
            polish_result, polish_size = local_polish(sa_result, max_iters=10000)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
    
    # Final polish on the best solution
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1:
        final_result, final_size = local_polish(best_in_set, max_iters=10000)
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_result
    
    # Final result
    final_indices = np.where(best_in_set)[0]
    result_vectors = vectors_all[final_indices]
    
    return result_vectors

# EVOLVE-BLOCK-END
