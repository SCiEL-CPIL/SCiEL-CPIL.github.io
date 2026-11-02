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
    
    def greedy_random_order(seed_offset=0):
        """Greedy independent set with random ordering."""
        rng_local = np.random.default_rng(42 + seed_offset)
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        order = rng_local.permutation(N4)
        
        for v in order:
            if not alive[v]:
                continue
            # Check if v conflicts with any in_set
            if np.any(conflicts[v] & in_set):
                continue
            in_set[v] = True
            alive[neighbors[v]] = False
            alive[v] = False
            
        return in_set

    def itls_local_search(in_set_init, max_time=10.0, seed_offset=0):
        """Iterated Local Search with swap-based moves."""
        rng_local = np.random.default_rng(42 + seed_offset)
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        itls_deadline = time.time() + max_time
        
        # Precompute conflict counts
        def compute_conflict_counts(in_set_arr):
            cc = np.zeros(N4, dtype=np.int32)
            for i in np.where(in_set_arr)[0]:
                cc[neighbors[i]] += 1
            return cc
        
        count_conflicts = compute_conflict_counts(in_set)
        
        while time.time() < itls_deadline:
            # Local search: try to improve by swapping
            improved = True
            while improved and time.time() < itls_deadline:
                improved = False
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                
                # Try adding each node that has 0 conflicts
                zero_conflict = not_in_set[count_conflicts[not_in_set] == 0]
                if len(zero_conflict) > 0:
                    v = zero_conflict[rng_local.integers(len(zero_conflict))]
                    in_set[v] = True
                    current_size += 1
                    count_conflicts[neighbors[v]] += 1
                    improved = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                    continue
                
                # Try swap: remove one, add another
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    break
                
                # Find nodes with high conflict count to potentially remove
                high_conflict = in_set_indices[count_conflicts[in_set_indices] >= 2]
                if len(high_conflict) == 0:
                    break
                    
                v_remove = high_conflict[rng_local.integers(len(high_conflict))]
                # Temporarily remove
                in_set[v_remove] = False
                current_size -= 1
                count_conflicts[neighbors[v_remove]] -= 1
                
                # Try to add a node
                not_in_set_now = np.where(~in_set)[0]
                if len(not_in_set_now) > 0:
                    zero_now = not_in_set_now[count_conflicts[not_in_set_now] == 0]
                    if len(zero_now) > 0:
                        v_add = zero_now[rng_local.integers(len(zero_now))]
                        in_set[v_add] = True
                        current_size += 1
                        count_conflicts[neighbors[v_add]] += 1
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                    else:
                        # Put back
                        in_set[v_remove] = True
                        current_size += 1
                        count_conflicts[neighbors[v_remove]] += 1
                
            # Perturbation: random restart with small modification
            if time.time() < itls_deadline:
                num_perturb = rng_local.integers(3, 10)
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > num_perturb:
                    remove_cands = in_set_indices[rng_local.choice(len(in_set_indices), num_perturb, replace=False)]
                    for u in remove_cands:
                        in_set[u] = False
                        count_conflicts[neighbors[u]] -= 1
                    current_size -= num_perturb
                    
                    # Greedy re-add
                    alive = ~in_set
                    while np.any(alive):
                        alive_indices = np.where(alive)[0]
                        if not len(alive_indices): break
                        zero_alive = alive_indices[count_conflicts[alive_indices] == 0]
                        if len(zero_alive) == 0:
                            break
                        chosen = zero_alive[rng_local.integers(len(zero_alive))]
                        in_set[chosen] = True
                        current_size += 1
                        count_conflicts[neighbors[chosen]] += 1
                        alive[neighbors[chosen]] = False
                        alive[chosen] = False
                        
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    
        return local_best_in_set, local_best_size

    # Multiple greedy initializations with different seeds
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        in_set = greedy_random_order(seed)
        sz = in_set.sum()
        if sz > best_size:
            best_size = sz
            best_in_set = in_set.copy()
    
    # Iterated Local Search from best greedy solution
    remaining_time = deadline - 5 - time.time()
    num_starts = max(1, int(remaining_time / 8))
    time_per_start = remaining_time / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 2:
            break
        if i == 0:
            init_set = best_in_set
        else:
            # Perturb best solution
            init_set = best_in_set.copy()
            num_perturb = rng.integers(5, 20)
            in_set_indices = np.where(init_set)[0]
            if len(in_set_indices) > num_perturb:
                remove_cands = in_set_indices[rng.choice(len(in_set_indices), num_perturb, replace=False)]
                init_set[remove_cands] = False

        itls_result, itls_size = itls_local_search(init_set, max_time=time_per_start, seed_offset=i)
        if itls_size > best_size:
            best_size = itls_size
            best_in_set = itls_result
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # W1 vectors are compatible with ALL W4 vectors (dot product <= 2)
    # So we can simply concatenate them.
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
