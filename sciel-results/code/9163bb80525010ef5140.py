"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    # Generate all D11 minimal vectors (squared norm 4)
    # Type 1: two coordinates ±2, rest 0 -> C(11,2)*4 = 220 vectors
    # Type 2: four coordinates ±1, rest 0 -> C(11,4)*16 = 5280 vectors
    # All have even coordinate sum (valid D11 vectors)
    
    v1_list = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    v1_list.append(v)
    
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    
    all_vectors = np.array(v1_list + v4_list, dtype=np.int64)
    N = len(all_vectors)
    
    # Check if all vectors are mutually at distance >= 2 (i.e., dot <= 2)
    # |vi - vj|^2 = 4 + 4 - 2*dot(vi,vj) >= 4  =>  dot(vi,vj) <= 2
    dots = all_vectors @ all_vectors.T
    np.fill_diagonal(dots, 0)
    
    if np.max(dots) <= 2:
        # All vectors are compatible - return the full kissing configuration
        return all_vectors
    
    # Fallback: need to find maximum independent set
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    start_time = time.time()
    deadline = start_time + 115
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def greedy_min_degree():
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
    
    def sa_local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        count_conflicts = np.zeros(N, dtype=np.int32)
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
            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0
            T *= cooling
            if T < min_T:
                T = min_T
        return local_best_in_set, local_best_size
    
    n_restarts = 0
    max_restarts = 50
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        sa_time = min(remaining_time * 0.6, 15.0)
        if sa_time < 2:
            sa_time = 2.0
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    final_indices = np.where(best_in_set)[0]
    return all_vectors[final_indices]

# EVOLVE-BLOCK-END
