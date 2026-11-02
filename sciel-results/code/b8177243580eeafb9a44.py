"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate all candidate vectors with norm^2 = 4
    # Type A: weight 4, entries in {-1, 0, 1}
    # Type B: weight 1, entries in {-2, 0, 2}
    # Type C: weight 2, entries in {-1, 1, -2, 2} with norm^2 = 4
    #   e.g., (±1, ±1, ...) -> norm^2 = 2 (not 4)
    #   e.g., (±2, 0, ...) -> already Type B
    #   e.g., (±1, ±1) with one entry being 2? No, entries must be integers.
    #   Actually for weight 2: entries a,b with a^2 + b^2 = 4
    #   (2,0) -> weight 1, (0,2) -> weight 1
    #   (1, sqrt(3)) -> not integer
    #   So only weight 1 and weight 4 vectors have integer entries with norm^2=4.
    
    candidates = []
    
    # Type A: weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    # Type B: weight 1
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    
    all_vectors = np.array(candidates, dtype=np.int64)
    N = len(all_vectors)
    
    # Compute conflict matrix: conflict if |dot| > 2 (i.e., dot > 2 or dot < -2)
    # Wait: the condition is minimum pairwise distance >= max norm.
    # For vectors u, v with norm^2 = 4: dist^2 = |u-v|^2 = |u|^2 + |v|^2 - 2*u.v = 8 - 2*u.v
    # We need dist^2 >= 4 (max norm squared = 4), so 8 - 2*u.v >= 4, i.e., u.v <= 2.
    # Also need dist^2 >= 4, which is always true if u.v <= 2.
    # But we also need the distance to be at least the maximum norm, which is 2.
    # So dist >= 2, i.e., dist^2 >= 4.
    # 8 - 2*u.v >= 4 => u.v <= 2.
    # Since vectors are nonzero and we're looking at distinct vectors, u.v can be negative.
    # The constraint is simply: for all pairs, u.v <= 2.
    # Conflict if u.v > 2.
    
    dots = all_vectors @ all_vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(12345)
    
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
                # For each candidate, count alive neighbors with high degree
                scores = []
                for c in cands:
                    nbrs = neighbors[c]
                    alive_nbrs = nbrs[alive[nbrs]]
                    if len(alive_nbrs) > 0:
                        scores.append(np.sum(degrees[alive_nbrs]))
                    else:
                        scores.append(0)
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
    
    def fast_greedy_add(in_set_init, max_time=5.0):
        """Fast polish: repeatedly add all vertices with zero conflicts."""
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
    
    def local_search_2for2(in_set_init, max_time=10.0):
        """Local search with 1-for-1 swaps AND 2-for-2 swaps."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 3.0
        cooling = 0.9999
        min_T = 0.005
        
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        ls_deadline = time.time() + max_time
        no_improve = 0
        
        while time.time() < ls_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.35 and len(not_in_set) > 0:
                # ADD move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = min_count
                delta = 1 - k
                
                accept = False
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
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            elif move_type < 0.70 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # 1-for-1 SWAP
                out_counts = count_conflicts[not_in_set]
                min_out = np.min(out_counts)
                min_out_cands = not_in_set[out_counts == min_out]
                v_add = min_out_cands[rng.integers(len(min_out_cands))]
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                
                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = False
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = rng.random() < 0.4
                    else:
                        accept = rng.random() < np.exp(delta / T)
                    
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                        
            elif move_type < 0.90 and len(in_set_indices) >= 2 and len(not_in_set) >= 2:
                # 2-for-2 SWAP: remove 2 from in_set, add 2 from not_in_set
                # Choose v_add1, v_add2 from not_in_set with low conflict counts
                out_counts = count_conflicts[not_in_set]
                # Pick two with lowest counts
                sorted_idx = np.argsort(out_counts)
                v_add1 = not_in_set[sorted_idx[0]]
                v_add2 = not_in_set[sorted_idx[1]]
                
                # Choose v_remove1, v_remove2 from in_set
                v_remove1 = in_set_indices[rng.integers(len(in_set_indices))]
                remaining = np.delete(in_set_indices, np.where(in_set_indices == v_remove1)[0][0])
                v_remove2 = remaining[rng.integers(len(remaining))]
                
                # Check if v_add1 and v_add2 conflict with each other
                if conflicts[v_add1, v_add2]:
                    no_improve += 1
                    continue
                
                # Find all in_set vertices that conflict with v_add1 or v_add2
                # but excluding v_remove1 and v_remove2
                conf1 = np.where(conflicts[v_add1] & in_set)[0]
                conf2 = np.where(conflicts[v_add2] & in_set)[0]
                all_conf = np.unique(np.concatenate([conf1, conf2]))
                to_remove = all_conf[~np.isin(all_conf, [v_remove1, v_remove2])]
                
                # After removing v_remove1 and v_remove2 and to_remove, 
                # we need to check that v_add1 and v_add2 don't conflict with remaining
                delta = 2 - (2 + len(to_remove))  # add 2, remove 2 + to_remove
                
                accept = False
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    all_to_remove = np.unique(np.concatenate(([v_remove1], [v_remove2], to_remove)))
                    in_set[all_to_remove] = False
                    for u in all_to_remove:
                        count_conflicts[neighbors[u]] -= 1
                    in_set[v_add1] = True
                    in_set[v_add2] = True
                    count_conflicts[neighbors[v_add1]] += 1
                    count_conflicts[neighbors[v_add2]] += 1
                    current_size += delta
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            else:
                # REMOVE move
                if len(in_set_indices) > 0:
                    v = in_set_indices[rng.integers(len(in_set_indices))]
                    if rng.random() < np.exp(-1.0 / T):
                        in_set[v] = False
                        current_size -= 1
                        count_conflicts[neighbors[v]] -= 1
                        no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if no_improve > 2000:
                # Perturbation: remove random subset, re-add greedily
                if len(in_set_indices) > 5:
                    n_perturb = max(4, len(in_set_indices) // 8)
                    perturb_victims = in_set_indices[rng.choice(len(in_set_indices), size=n_perturb, replace=False)]
                    in_set[perturb_victims] = False
                    current_size -= len(perturb_victims)
                    for u in perturb_victims:
                        count_conflicts[neighbors[u]] -= 1
                    # Re-add greedily
                    for v in perturb_victims:
                        if not np.any(conflicts[v] & in_set):
                            in_set[v] = True
                            current_size += 1
                            count_conflicts[neighbors[v]] += 1
                T = max(T, 2.5)
                no_improve = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return local_best_in_set, local_best_size
    
    # Multiple restarts with diverse initialization strategies
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Alternate between different initialization strategies
        if n_restarts % 3 == 0:
            # Strategy 1: Greedy min-degree
            in_set = greedy_min_degree()
        elif n_restarts % 3 == 1:
            # Strategy 2: Random + greedy polish
            in_set = np.zeros(N, dtype=bool)
            alive = np.ones(N, dtype=bool)
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                chosen = alive_indices[rng.integers(len(alive_indices))]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
        else:
            # Strategy 3: Weighted random based on inverse degree
            in_set = np.zeros(N, dtype=bool)
            alive = np.ones(N, dtype=bool)
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                weights = 1.0 / (neighbor_counts[alive_indices] + 1)
                weights /= weights.sum()
                chosen = alive_indices[rng.choice(len(alive_indices), p=weights)]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
        
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # Fast greedy add polish
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        polish_time = min(remaining_time * 0.10, 2.0)
        if polish_time > 0.3:
            polish_result, polish_size = fast_greedy_add(in_set, max_time=polish_time)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
                current_size = polish_size
                in_set = polish_result
        
        # Local search with 2-for-2 swaps
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        ls_time = min(remaining_time * 0.70, 10.0)
        if ls_time < 1.5:
            ls_time = 1.5
        
        ls_result, ls_size = local_search_2for2(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
        
        # Final polish
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            final_polish, final_size = fast_greedy_add(ls_result, max_time=min(remaining_time * 0.4, 4.0))
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_polish.copy()
        
        # Extra SA on best if time permits
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 4 and n_restarts >= 5:
            extra_ls, extra_size = local_search_2for2(best_in_set, max_time=min(remaining_time * 0.35, 6.0))
            if extra_size > best_size:
                best_size = extra_size
                best_in_set = extra_ls.copy()
    
    # Final result
    final_indices = np.where(best_in_set)[0]
    final_vectors = all_vectors[final_indices]
    
    return final_vectors

# EVOLVE-BLOCK-END
