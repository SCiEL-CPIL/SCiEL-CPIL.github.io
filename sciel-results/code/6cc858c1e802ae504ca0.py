"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START

def generate_vectors(weight):
    vectors = []
    for support in itertools.combinations(range(D), weight):
        for signs in itertools.product([1, -1], repeat=weight):
            v = np.zeros(D, dtype=np.int8)
            for idx, val in zip(support, signs):
                v[idx] = val
            vectors.append(v)
    return np.array(vectors)

def find_large_code_greedy(V, limit, deadline, num_restarts=2000):
    N = V.shape[0]
    if N == 0:
        return np.array([], dtype=int).reshape(0, D)
    
    # Compute dot products
    # Use int16 or int32 to be safe, but int8 might overflow?
    # Max dot product is weight. Weight is small (<=5). So int8 is fine.
    dots = V @ V.T
    
    best_indices = None
    best_count = 0
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random permutation of indices
        perm = np.random.permutation(N)
        
        active = np.ones(N, dtype=bool)
        selected = []
        
        # We iterate through the permuted list
        # Instead of while active.any(), we can just go through the list
        # and check if the node is still active.
        # This is O(N) per restart, which is very fast.
        
        # Optimization: 
        # active &= (dots[idx] <= limit)
        # This is O(N) per selection.
        # If we select K nodes, it is O(K*N).
        
        # Alternative:
        # Maintain active set.
        # For each idx in perm:
        #   if active[idx]:
        #      selected.append(idx)
        #      active &= (dots[idx] <= limit)
        #      active[idx] = False # optional, since we only visit each idx once in perm
        
        for idx in perm:
            if active[idx]:
                selected.append(idx)
                # Update active
                # A vector j is compatible if dots[idx, j] <= limit
                active &= (dots[idx] <= limit)
                active[idx] = False # Mark as used/inactive to be safe, though perm ensures unique visit
                
        if len(selected) > best_count:
            best_count = len(selected)
            best_indices = np.array(selected, dtype=int)
            
    if best_indices is None:
        return np.array([], dtype=int).reshape(0, D)
        
    return best_indices

def construct():
    start_time = time.time()
    # Safety margin: 10 seconds
    deadline = start_time + 130.0
    
    best_vectors = None
    best_count = 0
    
    # Try weights 4, 3, 2
    # W=4 is the most promising for large k
    
    for weight in [4, 3, 2]:
        if time.time() > deadline - 5:
            break
            
        # Generate vectors
        V = generate_vectors(weight)
        N = V.shape[0]
        
        if N == 0:
            continue
            
        # Calculate limit
        # Condition: u.v <= w/2
        # If w is even, w/2 is int.
        # If w is odd, w/2 is x.5, so floor(w/2).
        limit = weight // 2
        
        # Run search
        # Adjust restarts based on N and time
        # N=5280 for w=4.
        # Each restart is O(N*K). K is likely ~1000-3000.
        # 5280 * 2000 = 10M ops.
        # 1000 restarts = 10B ops? No, NumPy is fast.
        # Let's try 500 restarts for w=4.
        
        num_restarts = 500 if weight == 4 else 1000
        
        indices = find_large_code_greedy(V, limit, deadline, num_restarts=num_restarts)
        
        if indices is not None and len(indices) > best_count:
            best_count = len(indices)
            best_vectors = V[indices]
            
    # Fallback
    if best_vectors is None or len(best_vectors) == 0:
        V2 = generate_vectors(2)
        if len(V2) > 0:
            best_vectors = V2
            best_count = len(V2)

    return best_vectors

# EVOLVE-BLOCK-END
