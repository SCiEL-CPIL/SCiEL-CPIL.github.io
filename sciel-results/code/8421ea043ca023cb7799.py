"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # 1. Generate Type A vectors (Weight 1, value +/-2, Norm^2=4)
    # These are mutually compatible and compatible with all Type B and Type C vectors.
    type_a_vectors = []
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * sign
            type_a_vectors.append(v)
    
    type_a_array = np.array(type_a_vectors)
    n_a = len(type_a_array) # 22

    # 2. Generate Type B vectors (Weight 2, values +/-1, Norm^2=2)
    type_b_vectors = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_b_vectors.append(v)
            
    type_b_array = np.array(type_b_vectors)
    n_b = len(type_b_array) # 220

    # 3. Generate Type C vectors (Weight 4, values +/-1, Norm^2=4)
    type_c_vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_c_vectors.append(v)
            
    type_c_array = np.array(type_c_vectors)
    n_c = len(type_c_array) # 5280

    # Combine B and C into a single pool for optimization
    # Total nodes N = n_b + n_c
    N = n_b + n_c
    pool_vectors = np.vstack([type_b_array, type_c_array])
    
    # 4. Compute Conflict Matrix for the pool (B and C)
    # Norms: B has norm^2=2, C has norm^2=4.
    # Constraint: ||v_i - v_j||^2 >= 4  =>  ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j >= 4
    # v_i.v_j <= (||v_i||^2 + ||v_j||^2 - 4) / 2
    
    # Precompute norms squared for the pool
    norms_sq = np.sum(pool_vectors * pool_vectors, axis=1)
    
    # Compute dot products
    dots = pool_vectors @ pool_vectors.T
    
    # Calculate threshold for each pair (i, j)
    # threshold[i, j] = (norms_sq[i] + norms_sq[j] - 4) / 2
    # We need dots[i, j] <= threshold[i, j]
    # Conflict if dots[i, j] > threshold[i, j]
    
    # Vectorized threshold calculation
    # norms_sq is shape (N,). 
    # We need a (N, N) matrix where entry (i, j) is (norms_sq[i] + norms_sq[j] - 4) / 2.
    # norms_sq[:, None] + norms_sq[None, :]
    threshold_matrix = (norms_sq[:, None] + norms_sq[None, :] - 4) / 2.0
    
    conflicts = dots > threshold_matrix
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists (indices of conflicting nodes)
    # Using a list of arrays for fast access
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def run_greedy_sa(max_time):
        in_set = np.zeros(N, dtype=bool)
        conf_counts = np.zeros(N, dtype=np.int32)
        
        # Greedy Initialization: Random Order
        order = rng.permutation(N)
        for v in order:
            if conf_counts[v] == 0:
                in_set[v] = True
                conf_counts[neighbors[v]] += 1
                
        current_size = in_set.sum()
        
        # Local Search with Simulated Annealing
        sa_deadline = time.time() + max_time
        T = 1.0
        cooling = 0.999
        
        while time.time() < sa_deadline:
            # 1. Try to add a free node
            free_nodes = np.where((~in_set) & (conf_counts == 0))[0]
            if len(free_nodes) > 0:
                v = free_nodes[rng.integers(len(free_nodes))]
                in_set[v] = True
                conf_counts[neighbors[v]] += 1
                current_size += 1
            else:
                # 2. Try a swap: Remove a neighbor of a blocked node, then add blocked node
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                
                # Pick a random blocked node
                v = not_in_set[rng.integers(len(not_in_set))]
                
                # Get neighbors of v that are in the set
                nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                
                if len(nbrs_in_set) == 0:
                    # Should be free
                    if conf_counts[v] == 0:
                        in_set[v] = True
                        conf_counts[neighbors[v]] += 1
                        current_size += 1
                    continue
                
                # Pick a random neighbor to remove
                u = nbrs_in_set[rng.integers(len(nbrs_in_set))]
                
                # Simulate removal of u
                in_set[u] = False
                conf_counts[neighbors[u]] -= 1
                current_size -= 1
                
                # Now try to add v
                if conf_counts[v] == 0:
                    in_set[v] = True
                    conf_counts[neighbors[v]] += 1
                    current_size += 1
                else:
                    # Revert removal of u if we couldn't add v
                    in_set[u] = True
                    conf_counts[neighbors[u]] += 1
                    current_size += 1
                    
                # Cooling
                T *= cooling
                if T < 0.1:
                    T = 0.1
                    
        return in_set, current_size

    # Run multiple restarts
    n_restarts = 0
    while time.time() < deadline - 5:
        n_restarts += 1
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
            
        # Allocate time for this restart
        t_alloc = min(remaining * 0.8, 10.0)
        
        in_set, size = run_greedy_sa(t_alloc)
        
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
    # Construct final result
    # best_in_set corresponds to pool_vectors (B and C)
    selected_bc = pool_vectors[best_in_set]
    
    # Combine Type A and selected B/C
    if len(selected_bc) > 0:
        final_vectors = np.vstack([type_a_array, selected_bc])
    else:
        final_vectors = type_a_array
        
    return final_vectors

# EVOLVE-BLOCK-END
