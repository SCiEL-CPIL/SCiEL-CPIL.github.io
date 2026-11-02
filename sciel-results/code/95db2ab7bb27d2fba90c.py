"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin

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
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
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

    def sa_local_search(in_set_init, max_time=10.0, seed_offset=0):
        """Simulated Annealing local search with multi-start support."""
        rng_local = np.random.default_rng(42 + seed_offset)
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 5.0
        cooling = 0.9999
        min_T = 0.01
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
                
            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            
            min_cands = not_in_set[counts == min_count]
            v = min_cands[rng_local.integers(len(min_cands))]
            
            k = min_count
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng_local.random() < 0.5
            else:
                accept = rng_local.random() < np.exp(delta / T)
                
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
            
            if stuck_counter > 2000:
                # Perturbation: remove random nodes and re-add
                num_remove = rng_local.integers(10, 50)
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > num_remove:
                    remove_cands = in_set_indices[rng_local.choice(len(in_set_indices), num_remove, replace=False)]
                    for u in remove_cands:
                        in_set[u] = False
                        count_conflicts[neighbors[u]] -= 1
                    current_size -= num_remove
                    
                    # Greedy re-add
                    alive = ~in_set
                    while np.any(alive):
                        alive_indices = np.where(alive)[0]
                        if not len(alive_indices): break
                        min_deg = np.min(count_conflicts[alive_indices])
                        cands = alive_indices[count_conflicts[alive_indices] == min_deg]
                        chosen = cands[rng_local.integers(len(cands))]
                        if count_conflicts[chosen] == 0:
                            in_set[chosen] = True
                            current_size += 1
                            count_conflicts[neighbors[chosen]] += 1
                            alive[neighbors[chosen]] = False
                            alive[chosen] = False
                        else:
                            # If we can't add anyone, break
                            break
                            
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size

    # Initial Greedy
    in_set = greedy_min_degree()
    current_size = in_set.sum()
    best_size = current_size
    best_in_set = in_set.copy()
    
    # Multi-Start SA
    remaining_time = deadline - 5 - time.time()
    num_starts = 5
    time_per_start = remaining_time / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 2:
            break
        # Random initialization for some starts
        if i == 0:
            init_set = in_set
        else:
            # Random subset of size roughly half of best
            target_size = max(10, best_size // 2)
            init_indices = rng.choice(N4, target_size, replace=False)
            init_set = np.zeros(N4, dtype=bool)
            init_set[init_indices] = True
            # Quick fix: remove conflicts from init_set
            # This is a heuristic cleanup
            to_remove = []
            for idx in init_indices:
                if init_set[idx]:
                    for n in neighbors[idx]:
                        if init_set[n]:
                            to_remove.append(n)
                            break
            for idx in to_remove:
                init_set[idx] = False

        sa_result, sa_size = sa_local_search(init_set, max_time=time_per_start, seed_offset=i)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # W1 vectors are compatible with ALL W4 vectors (dot product <= 2)
    # So we can simply concatenate them.
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
