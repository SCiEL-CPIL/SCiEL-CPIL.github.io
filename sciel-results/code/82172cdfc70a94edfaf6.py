import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit (or 150s if specified, but 115 is safe)
    
    # 1. Generate all supports of size 4 from 11 indices
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # 2. Build conflict graph
    # Conflict if intersection size >= 3
    # Represent graph as adjacency list or matrix
    # 330 nodes is small enough for a matrix
    conflict_matrix = np.zeros((n_supports, n_supports), dtype=bool)
    
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                conflict_matrix[i, j] = True
                conflict_matrix[j, i] = True
                
    # Precompute neighbors for speed
    neighbors = [np.where(conflict_matrix[i])[0] for i in range(n_supports)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_support_set = np.zeros(n_supports, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive):
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

    def local_search(in_set_init, max_time=10.0):
        """Simulated Annealing / Local Search on the support graph."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 5.0
        cooling = 0.999
        min_T = 0.1
        
        # count_conflicts[s] = number of neighbors of s that are in in_set
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
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
            delta = 1 - k  # Change in size
            
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
            
            if stuck_counter > 1000:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size

    # Multiple restarts
    n_restarts = 0
    max_restarts = 1000
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        # 2. Local Search
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
        
        sa_time = min(remaining_time * 0.7, 15.0)
        if sa_time < 1:
            sa_time = 1.0
            
        sa_result, sa_size = local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()
            
    # Construct the final vectors
    vectors = []
    
    # Add all 22 Type 1 vectors (weight 1, entries +/- 2)
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            vectors.append(v)
            
    # Add vectors for selected supports
    for i in np.where(best_support_set)[0]:
        support = supports[i]
        # Generate all 16 sign variants
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(support, signs):
                v[pos] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
