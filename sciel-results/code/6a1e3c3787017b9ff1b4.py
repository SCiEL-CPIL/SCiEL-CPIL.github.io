import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Generate Pool of Weight-4 Vectors
    # Norm squared = 4. Norm = 2.
    # Constraint for equal norm set: dist^2 >= 4 => 4+4 - 2*u.v >= 4 => u.v <= 2.
    # Conflict if u.v > 2.
    
    candidates = []
    indices = range(D)
    # Weight 4: 4 non-zero entries, each +/-1
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    if time.time() > deadline - 10:
        return candidates[:10] # Fallback
    
    # 2. Compute Dot Products
    dots = candidates @ candidates.T
    
    # 3. Build Conflict Graph
    # Conflict if dot > 2
    conflict = (dots > 2).astype(np.int8)
    # Zero out diagonal (self-conflict is irrelevant for independent set logic, or handle separately)
    np.fill_diagonal(conflict, 0)
    
    # 4. Spectral Greedy for Maximum Independent Set
    # We want a set of vertices with no edges between them.
    # Heuristic: 
    # a) Compute principal eigenvector of the adjacency matrix of the conflict graph.
    # b) Vertices with small values in this eigenvector are "peripheral" or "less central" in the conflict structure.
    # c) Sort by this value and greedily pick.
    
    # For large sparse graphs, eigenvector calculation can be expensive.
    # n_cands = 5280. 5280x5280 matrix is ~22MB. Eigen decomposition is O(N^3) ~ 1.5e11 ops. Too slow.
    # Use Power Iteration for principal eigenvector?
    # Or use a simpler heuristic: Degree-based greedy with randomization?
    # The prompt mentions "Spectral guidance... Measured: 2.282 -> 2.318".
    # Let's try a faster spectral approximation or just Degree-based with a twist.
    
    # Alternative: Use the fact that the graph is regular-ish?
    # Let's try a multi-start greedy with different orderings.
    # Ordering 1: Random
    # Ordering 2: By Degree (ascending)
    # Ordering 3: By "Spectral" approximation (e.g., PageRank or just random walk potential)
    
    # Given time constraints, let's use a robust **Multi-Start Greedy with Degree Heuristic** 
    # but optimized with NumPy vectorization for the selection step.
    
    best_set = []
    best_size = 0
    
    # Precompute degrees
    degrees = np.sum(conflict, axis=1)
    
    # We will run several rounds of greedy selection with different orderings.
    # To stay within time, we limit the number of restarts.
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Create an ordering
        if restart == 0:
            # Ascending degree
            order = np.argsort(degrees)
        elif restart == 1:
            # Random
            order = np.random.permutation(n_cands)
        else:
            # Perturbed degree (add noise to break ties and explore)
            noise = np.random.rand(n_cands) * 0.1
            order = np.argsort(degrees + noise)
            
        # Greedy Selection
        selected = set()
        available = np.ones(n_cands, dtype=bool)
        
        # Optimization: Instead of checking all pairs for each new addition,
        # we can maintain a bitmask or boolean array of "forbidden" vertices.
        
        current_set_indices = []
        
        for idx in order:
            if not available[idx]:
                continue
            
            # Check if idx conflicts with any already selected vertex
            # We can check: conflict[idx, selected_indices].any()
            # But selected_indices grows.
            
            # Faster: Maintain a "blocked" array.
            # When we select idx, we block idx and all its neighbors.
            
            # To do this efficiently in NumPy:
            # blocked = conflict[idx, :] == 1
            # available[blocked] = False
            
            # Let's implement this loop.
            # Initial available is all True.
            
            # Check if idx is still available (it is, by loop condition)
            # Select idx
            current_set_indices.append(idx)
            
            # Block idx and its neighbors
            # conflict[idx, :] is a vector of length n_cands
            # We want to set available[j] = False if conflict[idx, j] == 1
            available[conflict[idx, :] == 1] = False
            available[idx] = False # Ensure self is blocked
            
        if len(current_set_indices) > best_size:
            best_size = len(current_set_indices)
            best_set = current_set_indices[:]
            
    # Return the best set found
    if not best_set:
        # Fallback to a single vector
        return candidates[:1]
        
    result_vectors = candidates[best_set]
    return result_vectors

# EVOLVE-BLOCK-END
