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
    
    # Precompute neighbor lists as arrays for fast access
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Convert conflicts to int8 for faster arithmetic
    conflicts_int = conflicts.astype(np.int8)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree(seed_offset=0):
        """Greedy independent set with min-degree heuristic, randomized tie-breaking."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.int32)
        local_rng = np.random.default_rng(42 + seed_offset)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1]
            
            if len(cands) > 1:
                # Vectorized tie-break: prefer vertex whose alive-neighbors have lower total degree
                # For each candidate, sum degrees of its alive neighbors
                scores = np.zeros(len(cands), dtype=np.float64)
                for idx, c in enumerate(cands):
                    nbrs_c = neighbors[c]
                    alive_mask = alive[nbrs_c]
                    if np.any(alive_mask):
                        scores[idx] = np.sum(degrees[nbrs_c[alive_mask]])
                # Add small random noise
                noise = local_rng.random(len(cands)) * 0.1
                scores += noise
                chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Vectorized degree update
                degrees[rem] -= conflicts_int[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Enhanced simulated annealing with vectorized conflict counting."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 4.0
        cooling = 0.9993
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        # Vectorized computation using matrix multiplication
        count_conflicts = (conflicts_int @ in_set.astype(np.int8)).astype(np.int32)
        
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
            
            # Kick mechanism
            if stuck_counter > 3000:
                n_remove = max(1, int(0.15 * len(in_set_indices)))
                if n_remove < len(in_set_indices):
                    remove_idx = rng.choice(len(in_set_indices), size=n_remove, replace=False)
                    to_remove = in_set_indices[remove_idx]
                    in_set[to_remove] = False
                    # Vectorized update
                    count_conflicts -= conflicts_int[to_remove].sum(axis=0).astype(np.int32)
                    current_size -= n_remove
                    stuck_counter = 0
                    # Re-add greedily: add all with 0 conflicts
                    zero_conf = not_in_set[count_conflicts[not_in_set] == 0]
                    if len(zero_conf) > 0:
                        in_set[zero_conf] = True
                        count_conflicts += conflicts_int[zero_conf].sum(axis=0).astype(np.int32)
                        current_size += len(zero_conf)
                    not_in_set = np.where(~in_set)[0]
                    in_set_indices = np.where(in_set)[0]
                    continue
            
            move_type = rng.random()
            
            if move_type < 0.45 and len(not_in_set) > 0:
                # ADD move
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
                        count_conflicts -= conflicts_int[to_remove].sum(axis=0).astype(np.int32)
                    in_set[v] = True
                    current_size += delta
                    count_conflicts += conflicts_int[v]
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.70 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move
                in_degrees = count_conflicts[in_set_indices]
                weights = 1.0 / (1.0 + in_degrees.astype(np.float64))
                weights /= weights.sum()
                v_remove = in_set_indices[rng.choice(len(in_set_indices), p=weights)]
                
                add_counts = count_conflicts[not_in_set]
                add_weights = 1.0 / (1.0 + add_counts.astype(np.float64))
                add_weights /= add_weights.sum()
                v_add = not_in_set[rng.choice(len(not_in_set), p=add_weights)]
                
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                
                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts += conflicts_int[v_add] - conflicts_int[v_remove]
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        in_set[v_add] = True
                        count_conflicts += conflicts_int[v_add] - conflicts_int[to_remove].sum(axis=0).astype(np.int32)
                        current_size += delta
                    else:
                        stuck_counter += 1
                        
            elif move_type < 0.85 and len(in_set_indices) >= 2 and len(not_in_set) > 0:
                # 2-REMOVE, 1-ADD
                idx1, idx2 = rng.choice(len(in_set_indices), size=2, replace=False)
                v_rem1 = in_set_indices[idx1]
                v_rem2 = in_set_indices[idx2]
                
                add_counts = count_conflicts[not_in_set]
                add_weights = 1.0 / (1.0 + add_counts.astype(np.float64))
                add_weights /= add_weights.sum()
                v_add = not_in_set[rng.choice(len(not_in_set), p=add_weights)]
                
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[~np.isin(conflicts_with, [v_rem1, v_rem2])]
                
                if len(other_conflicts) == 0:
                    delta = -1
                    accept = rng.random() < np.exp(delta / T)
                    if accept:
                        in_set[v_rem1] = False
                        in_set[v_rem2] = False
                        in_set[v_add] = True
                        count_conflicts += conflicts_int[v_add] - conflicts_int[v_rem1] - conflicts_int[v_rem2]
                        current_size += delta
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            else:
                # REMOVE move
                if len(in_set_indices) > 0:
                    v = in_set_indices[rng.integers(len(in_set_indices))]
                    delta = -1
                    accept = rng.random() < np.exp(delta / T)
                    if accept:
                        in_set[v] = False
                        current_size += delta
                        count_conflicts -= conflicts_int[v]
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
        
        for trial in range(5):
            if time.time() >= polish_deadline:
                break
            
            not_in = np.where(~polish_set)[0]
            if len(not_in) == 0:
                break
            
            # Vectorized: compute conflict counts for all candidates at once
            # conflicts_int[not_in] has shape (len(not_in), N4)
            # We need for each candidate v in not_in: count of conflicts with polish_set
            conflict_counts = (conflicts_int[not_in] @ polish_set.astype(np.int8)).astype(np.int32)
            
            added_any = True
            while added_any and time.time() < polish_deadline:
                added_any = False
                zero_conflict = not_in[conflict_counts == 0]
                if len(zero_conflict) > 0:
                    polish_set[zero_conflict] = True
                    polish_size += len(zero_conflict)
                    # Recompute for remaining
                    remaining = np.where(~polish_set)[0]
                    if len(remaining) > 0:
                        conflict_counts = (conflicts_int[remaining] @ polish_set.astype(np.int8)).astype(np.int32)
                        not_in = remaining
                    else:
                        break
                    added_any = True
        
        return polish_set, polish_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 100
    
    greedy_cache = None
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        if n_restarts == 1:
            greedy_cache = greedy_min_degree(0)
            in_set = greedy_cache
        else:
            in_set = greedy_min_degree(n_restarts)
        
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
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
        
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            polish_time = min(remaining_time * 0.4, 6.0)
            if polish_time > 1:
                polish_result, polish_size = polish_greedy_add(sa_result if sa_size >= current_size else in_set, max_time=polish_time)
                
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
                
                remaining_time2 = deadline - 3 - time.time()
                if remaining_time2 > 4 and polish_size >= current_size:
                    sa2_time = min(remaining_time2 * 0.6, 8.0)
                    if sa2_time > 2:
                        sa2_result, sa2_size = sa_local_search(polish_result, max_time=sa2_time)
                        if sa2_size > best_size:
                            best_size = sa2_size
                            best_in_set = sa2_result.copy()
                        
                        remaining_time3 = deadline - 2 - time.time()
                        if remaining_time3 > 2:
                            final_polish, final_size = polish_greedy_add(sa2_result, max_time=min(remaining_time3 * 0.5, 4.0))
                            if final_size > best_size:
                                best_size = final_size
                                best_in_set = final_polish.copy()
    
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
