import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Generate W4 vectors
    # Weight 4, entries in {-1, 0, 1}, norm^2 = 4
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors
    # Weight 1, entries in {-2, 0, 2}, norm^2 = 4
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

    # Compute conflict matrix for W4 vectors
    # Conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays for efficient updates
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
            
            # Find alive nodes with minimum degree
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            min_cands = alive_indices[alive_degrees == min_deg]
            
            # Pick one randomly
            chosen = min_cands[rng.integers(len(min_cands))]
            
            in_set[chosen] = True
            
            # Remove chosen and its neighbors from alive
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # For each remaining node u, count how many of its neighbors are in to_remove
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def sa_local_search(in_set_init, max_time=10.0):
        """Simulated Annealing local search using neighbor lists."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        stuck_counter = 0
        
        while time.time() < sa_deadline:
            # Find nodes not in set
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            # Prioritize nodes with minimum conflicts
            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            min_cands = not_in_set[counts == min_count]
            
            v = min_cands[rng.integers(len(min_cands))]
            
            k = count_conflicts[v]
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                if delta < -20 * T:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
            if accept:
                # Add v to set
                in_set[v] = True
                current_size += 1
                
                # Find neighbors of v that are currently in the set
                # Using neighbor list for efficiency
                nbrs_v = neighbors[v]
                if len(nbrs_v) > 0:
                    nbrs_in_set = nbrs_v[in_set[nbrs_v]]
                    
                    if len(nbrs_in_set) > 0:
                        in_set[nbrs_in_set] = False
                        current_size -= len(nbrs_in_set)
                        
                        # Update count_conflicts
                        # 1. Add v's effect: neighbors of v gain 1 conflict
                        count_conflicts[nbrs_v] += 1
                        
                        # 2. Remove r's effect: neighbors of r lose 1 conflict
                        for r in nbrs_in_set:
                            count_conflicts[neighbors[r]] -= 1
                        
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
            else:
                T *= cooling
                if T < min_T:
                    T = min_T
                    
        return local_best_in_set, local_best_size

    # Main Loop
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
            
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        sa_time = min(remaining_time * 0.7, 10.0)
        if sa_time < 1:
            sa_time = 1.0
            
        sa_result, sa_size = sa_local_search(in_set, sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
            
        # 3. Polish (Second pass on best)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 5.0)
            if polish_time > 1:
                polish_result, polish_size = sa_local_search(sa_result, polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    
    # Combine with all W1 vectors
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
