"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    # 1. Generate all candidate vectors
    # We consider supports of size 4 (norm^2 = 4) as a strong baseline.
    # We can also consider other support sizes if beneficial, but size 4 is standard for kissing number problems in low dimensions.
    # Let's stick to supports of size 4 for now, as it yields the highest density of vectors.
    # Total vectors: C(11, 4) * 2^4 = 330 * 16 = 5280.
    
    supports = list(combinations(range(D), 4))
    all_vectors = []
    for s in supports:
        # Generate all 16 sign variants for this support
        # Using a more efficient generation method
        indices = list(s)
        for mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(indices):
                v[idx] = 1 if (mask & (1 << j)) else -1
            all_vectors.append(v)
    
    V = np.array(all_vectors, dtype=np.int64)
    N = V.shape[0]
    
    # 2. Precompute norms squared
    # All vectors have norm^2 = 4, so M_sq = 4.
    # Condition: ||u - v||^2 >= 4
    # ||u||^2 + ||v||^2 - 2 u.v >= 4
    # 4 + 4 - 2 u.v >= 4  =>  4 - 2 u.v >= 0  =>  u.v <= 2
    
    # We need to find the largest subset of V such that for all distinct i, j in subset, V[i].V[j] <= 2.
    # This is the Maximum Independent Set (MIS) problem on a graph G where (i, j) is an edge if V[i].V[j] > 2.
    
    # 3. Build the conflict graph (adjacency matrix)
    # N = 5280. A 5280x5280 boolean matrix is ~25MB, which is manageable.
    # Computing dot products: V @ V.T is 5280x5280. 
    # 5280^2 * 11 flops ~ 300 million flops. Numpy can do this in < 1 second.
    
    dot_products = V @ V.T
    # Conflict if dot > 2. Note that diagonal is 4, so we ignore self-conflicts.
    conflict = (dot_products > 2)
    # Zero out diagonal to avoid self-conflicts in graph algorithms
    np.fill_diagonal(conflict, False)
    
    # 4. Solve for MIS using a fast heuristic: Local Search / Max-Cut style
    # Since N is 5280, we can't do exact MIS. We use a randomized greedy initialization followed by local search.
    
    # Heuristic: 
    # Start with a random subset or a greedy maximal independent set.
    # Local Search: Try to swap nodes or add nodes if possible.
    # A more effective approach for dense-ish graphs is "Greedy with Random Restart + Local Improvement".
    
    # Let's use a specific efficient heuristic: 
    # 1. Greedy construction: Order nodes randomly. Add a node if it doesn't conflict with any already added node.
    # 2. Local Improvement: For each node in the set, try to remove it and see if we can add more nodes that were previously blocked.
    #    Actually, a simpler and often effective method is "Min-Conflicts" or just repeated greedy passes with different orderings.
    
    # Let's try a "Randomized Greedy" with many restarts, but optimized with NumPy.
    # For each restart:
    #   - Shuffle node indices
    #   - Iterate through shuffled indices
    #   - If node is not in current set AND (current_set is empty OR no conflict with current_set), add it.
    #   - To check conflict efficiently, we maintain a boolean array `in_set` and use the precomputed conflict matrix.
    #   - Conflict check: conflict[node, in_set].any()
    
    best_size = 0
    best_set_indices = np.array([], dtype=int)
    
    # Number of restarts depends on time.
    # Each restart is O(N * K) where K is the size of the set found.
    # K is likely around 100-200. N=5280.
    # 5280 * 200 = 1 million ops per restart.
    # We can do ~100-1000 restarts in 110 seconds.
    
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Shuffle nodes
        order = np.random.permutation(N)
        
        in_set = np.zeros(N, dtype=bool)
        set_indices = []
        
        # We can speed this up by maintaining a list of "active" candidates or just iterating.
        # Iterating 5280 times and doing a boolean check on a subset of 5280 is fast in NumPy.
        
        for node in order:
            if in_set[node]:
                continue
            
            # Check if node conflicts with any node in set
            # conflict[node] is a boolean array of length N.
            # We need to check if any True in conflict[node] corresponds to an index where in_set is True.
            # np.any(conflict[node] & in_set)
            
            if not np.any(conflict[node] & in_set):
                in_set[node] = True
                set_indices.append(node)
        
        current_size = len(set_indices)
        if current_size > best_size:
            best_size = current_size
            best_set_indices = np.array(set_indices, dtype=int)
            
    # 5. Construct the result array
    if best_size == 0:
        # Fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    result = V[best_set_indices]
    return result

# EVOLVE-BLOCK-END
