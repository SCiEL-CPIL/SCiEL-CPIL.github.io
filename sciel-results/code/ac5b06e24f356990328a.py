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
            
    # Generate W2 vectors (weight 2, entries in {-1,0,1}, norm^2=2)
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    vectors_v2 = np.array(v2_candidates, dtype=np.int64)
    
    # Combine all vectors
    all_vectors = np.vstack([vectors_v4, vectors_v1, vectors_v2])
    N_total = len(all_vectors)
    
    # Compute norms squared
    norms_sq = np.sum(all_vectors**2, axis=1)
    
    # Compute Gram matrix
    G = all_vectors @ all_vectors.T
    
    # Conflict matrix: C[i,j] = True if dist^2 < 4
    # dist^2 = norms_sq[i] + norms_sq[j] - 2*G[i,j]
    # Conflict if norms_sq[i] + norms_sq[j] - 2*G[i,j] < 4
    conflict_matrix = np.zeros((N_total, N_total), dtype=bool)
    
    # Vectorize conflict calculation
    # dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G
    conflict_matrix = dist_sq < 4
    
    # Diagonal is self, no conflict
    np.fill_diagonal(conflict_matrix, False)
    
    # Get neighbors for each node
    neighbors = [np.where(conflict_matrix[i])[0] for i in range(N_total)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N_total, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N_total, dtype=bool)
        alive = np.ones(N_total, dtype=bool)
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
                degrees[rem] -= conflict_matrix[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        count_conflicts = np.zeros(N_total, dtype=np.int32)
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
            s = min_cands[rng.integers(len(min_cands))]
            
            k = min_count
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(delta / T)
                
            if accept:
                to_remove = np.where(conflict_matrix[s] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[s] = True
                current_size += delta
                count_conflicts[neighbors[s]] += 1
                
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
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 200  # Reduced restarts due to larger graph
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
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
        
        # 3. Polish: try to add any free node
        polish_deadline = deadline - 2
        if time.time() < polish_deadline:
             polish_in_set = best_in_set.copy()
             count_conflicts = np.zeros(N_total, dtype=np.int32)
             for i in np.where(polish_in_set)[0]:
                 count_conflicts[neighbors[i]] += 1
             
             improved = True
             while improved and time.time() < polish_deadline:
                 improved = False
                 not_in_set = np.where(~polish_in_set)[0]
                 for v in not_in_set:
                     if time.time() > polish_deadline:
                         break
                     if count_conflicts[v] == 0:
                         polish_in_set[v] = True
                         count_conflicts[neighbors[v]] += 1
                         improved = True
                         
             if polish_in_set.sum() > best_size:
                 best_size = polish_in_set.sum()
                 best_in_set = polish_in_set.copy()

    # Final result construction
    result_vectors = all_vectors[best_in_set]
    
    return result_vectors

# EVOLVE-BLOCK-END
