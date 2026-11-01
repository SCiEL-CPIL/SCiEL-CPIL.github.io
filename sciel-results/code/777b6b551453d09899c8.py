"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin before 150s kill

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
    
    # Precompute neighbor lists and degrees
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Compute "expansion" score: number of edges among neighbors (lower = better to pick first)
    # This helps greedy choose vertices that don't block many others
    expansion_scores = np.zeros(N4, dtype=np.int32)
    for i in range(N4):
        nbrs = neighbors[i]
        if len(nbrs) > 1:
            # Count edges among neighbors
            sub_conf = conflicts[np.ix_(nbrs, nbrs)]
            expansion_scores[i] = sub_conf.sum() // 2
    
    rng = np.random.default_rng(12345)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_weighted_init(seed=None):
        """Greedy independent set with expansion-aware heuristic."""
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = rng
            
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        expansions = expansion_scores.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Score: primarily degree, secondarily expansion
            degs = degrees[alive_indices]
            mins = np.min(degs)
            cands = alive_indices[degs == mins]
            
            if len(cands) > 1:
                exps = expansions[cands]
                min_exp = np.min(exps)
                final_cands = cands[exps == min_exp]
                chosen = final_cands[local_rng.integers(len(final_cands))]
            else:
                chosen = cands[0]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Update degrees
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                # Update expansions (approximate: recompute for remaining)
                # Skip full recomputation for speed; use stale values
                # degrees[rem] already updated, which is the main signal
                
        return in_set
    
    def sa_local_search(in_set_init, max_time=8.0, T_start=2.0):
        """Simulated annealing with add/remove/swap/2-swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = T_start
        cooling = 0.9997
        min_T = 0.02
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0
        
        while time.time() < sa_deadline:
            step += 1
            in_set_indices = np.where(in_set)[0]
            not_in_set = np.where(~in_set)[0]
            
            if len(not_in_set) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.55 and len(not_in_set) > 0:
                # ADD move - optimized
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                
                # Prefer vertices with lower degree among candidates
                if len(min_cands) > 1:
                    degs = neighbor_counts[min_cands]
                    min_deg = np.min(degs)
                    v = min_cands[degs == min_deg][rng.integers(np.sum(degs == min_deg))]
                else:
                    v = min_cands[0]
                
                k = int(min_count)
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
                    
            elif move_type < 0.70 and len(in_set_indices) > 0:
                # REMOVE move
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move - optimized
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                # Sample candidates, prefer those with low conflict count
                n_sample = min(15, len(not_in_set))
                sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                sample_cands = not_in_set[sample_idx]
                
                # Use count_conflicts to quickly filter
                # After removing v_remove, conflicts decrease by 1 for v_remove's neighbors
                temp_counts = count_conflicts.copy()
                temp_counts[neighbors[v_remove]] -= 1
                temp_counts[v_remove] = 0  # v_remove is being removed
                
                # Find candidates with minimal conflicts after v_remove is gone
                cands_conf = temp_counts[sample_cands]
                min_conf = np.min(cands_conf)
                good_cands = sample_cands[cands_conf == min_conf]
                
                if len(good_cands) > 0:
                    v_add = good_cands[rng.integers(len(good_cands))]
                    delta = 1 - int(min_conf)
                    
                    if delta >= 0 or rng.random() < np.exp(delta / T):
                        # Execute swap
                        to_remove = np.where(conflicts[v_add] & in_set)[0]
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                else:
                    stuck_counter += 1
                    
            else:
                # 2-SWAP: remove 2 in-set, add 2 not-in-set
                if len(in_set_indices) >= 2 and len(not_in_set) >= 2:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    v2 = in_set_indices[rng.integers(len(in_set_indices))]
                    if v1 == v2:
                        v2 = in_set_indices[rng.integers(len(in_set_indices))]
                        if v1 == v2:
                            stuck_counter += 1
                        else:
                            # Try to find 2 non-conflicting additions
                            n_sample = min(10, len(not_in_set))
                            sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                            sample_cands = not_in_set[sample_idx]
                            
                            found = []
                            for v_add in sample_cands:
                                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                                other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]
                                if len(other_conflicts) == 0:
                                    found.append(v_add)
                                    if len(found) == 2:
                                        break
                            
                            if len(found) == 2:
                                # Clean 2-swap
                                to_remove = np.array([v1, v2], dtype=np.int64)
                                in_set[to_remove] = False
                                for u in to_remove:
                                    count_conflicts[neighbors[u]] -= 1
                                for v_ad in found:
                                    in_set[v_ad] = True
                                    count_conflicts[neighbors[v_ad]] += 1
                                # delta = 0
                            else:
                                stuck_counter += 1
                    else:
                        n_sample = min(10, len(not_in_set))
                        sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                        sample_cands = not_in_set[sample_idx]
                        
                        found = []
                        for v_add in sample_cands:
                            conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                            other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]
                            if len(other_conflicts) == 0:
                                found.append(v_add)
                                if len(found) == 2:
                                    break
                        
                        if len(found) == 2:
                            to_remove = np.array([v1, v2], dtype=np.int64)
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                            for v_ad in found:
                                in_set[v_ad] = True
                                count_conflicts[neighbors[v_ad]] += 1
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            
            if stuck_counter > 1500:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def greedy_polish(in_set_init):
        """Fast greedy: add any vertex that doesn't conflict with current set."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # Use count_conflicts array for O(1) conflict checks
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            
            # Find vertices with zero conflicts using precomputed counts
            addable = not_in[count_conflicts[not_in] == 0]
            
            if len(addable) > 0:
                # Add one vertex at a time, updating counts
                v_idx = addable[0]
                in_set[v_idx] = True
                current_size += 1
                count_conflicts[neighbors[v_idx]] += 1
                changed = True
        
        return in_set, current_size
    
    # Luby sequence for restart lengths
    def luby_sequence():
        """Generate Luby's restart sequence."""
        n = 1
        while True:
            # 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
            # Standard Luby: for k = 0,1,2,...: for j in 0..k: yield 2^(k - floor(log2(k)))
            # Simpler: use the pattern 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,...
            pass
    
    # Use a simpler adaptive restart strategy
    # Luby-like: short, short, medium, short, short, medium, long, ...
    luby_pattern = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 8, 16]
    
    n_restarts = 0
    max_restarts = 100
    luby_idx = 0
    
    while time.time() < deadline - 6 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Get restart length from Luby pattern
        luby_len = luby_pattern[luby_idx % len(luby_pattern)]
        luby_idx += 1
        
        # 1. Greedy Init with different seeds for diversity
        seed = n_restarts * 7 + 13
        in_set = greedy_weighted_init(seed=seed)
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search - time based on Luby length
        remaining_time = deadline - 6 - time.time()
        if remaining_time < 3:
            break
        
        # Scale SA time by Luby factor
        base_sa_time = min(remaining_time * 0.5, 10.0)
        sa_time = base_sa_time * (luby_len / 4.0)  # Normalize: 4 is median
        sa_time = max(sa_time, 1.5)
        sa_time = min(sa_time, remaining_time * 0.7)
        
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Greedy polish
        remaining_time = deadline - 4 - time.time()
        if remaining_time > 2:
            polish_set, polish_size = greedy_polish(sa_result if sa_size >= current_size else in_set)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_set.copy()
        
        # 4. If we have time, do a second SA on the best found
        remaining_time = deadline - 3 - time.time()
        if remaining_time > 5 and n_restarts % 3 == 0:
            polish_time = min(remaining_time * 0.6, 8.0)
            if polish_time > 2:
                polish_result, polish_size = sa_local_search(best_in_set, max_time=polish_time, T_start=1.0)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final greedy polish on best result
    if time.time() < deadline - 2:
        final_polish, final_size = greedy_polish(best_in_set)
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_polish
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
