"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # Generate all candidate vectors in D_11 with norm squared 4
    # Type 1: Weight 1, entries +/-2. (22 vectors)
    # Type 2: Weight 4, entries +/-1. (5280 vectors)
    
    candidates = []
    
    # Type 1
    for pos in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[pos] = 2 * sign
            candidates.append(v)
            
    # Type 2
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            # Sum of signs must be even for D_lattice.
            # Any combination of 4 signs has an even sum.
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    vectors = np.array(candidates, dtype=np.int64)
    N = len(vectors)
    
    # Compute conflict matrix
    # Conflict if dot product > 2
    dots = vectors @ vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors
    # Using a list of lists of indices for fast access
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
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
                # Update degrees: subtract number of neighbors removed
                # conflicts[rem][:, to_remove] is a boolean matrix
                # Sum along axis 1 gives count of removed neighbors for each remaining node
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search(in_set_init, max_time=10.0):
        """Simulated Annealing / Hill Climbing local search."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 2.0
        cooling = 0.999
        min_T = 0.1
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        ls_deadline = time.time() + max_time
        
        while time.time() < ls_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
                
            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            
            # Prioritize adding nodes with fewest conflicts
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
            
            if stuck_counter > 1000:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size

    def exact_polish(in_set_init, max_iters=10000):
        """Greedy improvement pass."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            best_gain = 0
            best_v = -1
            
            for v in not_in_set:
                k = count_conflicts[v]
                gain = 1 - k
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v >= 0 and best_gain > 0:
                to_remove = np.where(conflicts[best_v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[best_v] = True
                current_size += best_gain
                count_conflicts[neighbors[best_v]] += 1
                improved = True
        
        return in_set, current_size

    # Main loop with multiple restarts
    n_restarts = 0
    max_restarts = 100 # High number, but limited by time
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
        
        ls_time = min(remaining_time * 0.6, 15.0)
        if ls_time < 1.0:
            ls_time = 1.0
            
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
        
        # 3. Polish
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 1:
            polish_result, polish_size = exact_polish(ls_result, max_iters=5000)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
                
    final_indices = np.where(best_in_set)[0]
    result_vectors = vectors[final_indices]
    
    return result_vectors

# EVOLVE-BLOCK-END
