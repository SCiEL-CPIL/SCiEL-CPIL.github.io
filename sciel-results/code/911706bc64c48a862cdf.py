import numpy as np
import time

def compute_k_core(adjacency, n):
    """
    Compute k-core numbers for each vertex in an undirected graph.
    adjacency: NxN boolean matrix.
    Returns: array of k-core numbers.
    """
    if n == 0:
        return np.array([], dtype=int)
    
    # Calculate initial degrees
    deg = np.sum(adjacency, axis=1).astype(int)
    
    # k_core will store the final k-core number for each node
    k_core = np.zeros(n, dtype=int)
    
    # We use a peeling algorithm.
    # We maintain a list of active nodes.
    # We iterate k from 1 to max_degree.
    # For each k, we remove all nodes with degree < k.
    # The degree of remaining nodes decreases as their neighbors are removed.
    
    # To do this efficiently for N=200, we can just simulate.
    # However, standard peeling is O(E).
    
    # Let's use the standard bucket-based peeling or just iterative removal.
    # Since N is small (200), O(N^2) is perfectly fine.
    
    active = np.ones(n, dtype=bool)
    current_deg = deg.copy()
    max_k = int(deg.max()) if n > 0 else 0
    
    # We need to assign k_core[v] = max k such that v is in the k-core.
    # This is equivalent to the order in which they are removed if we remove
    # nodes with degree < k for increasing k.
    # Actually, k_core[v] is the largest k such that v remains after peeling with threshold k.
    
    # Algorithm:
    # 1. Put all nodes in a bucket sorted by degree.
    # 2. Let k = 0.
    # 3. While there are nodes:
    #    a. Find min degree d_min among active nodes.
    #    b. If d_min < k: 
    #         # This shouldn't happen if we process in order, but let's be safe.
    #         pass
    #    c. Actually, the standard algorithm:
    #       k = 1
    #       While there are nodes with degree < k:
    #           Remove such a node v.
    #           k_core[v] = k-1 (or the current k-1? No, k_core[v] is the k it survived to).
    #           Actually, k_core[v] is the value of k when it was removed?
    #           If a node is removed at step k (because its degree dropped below k),
    #           its k-core number is k-1? 
    #           Let's check definition: The k-core is the maximal subgraph where min degree >= k.
    #           If v is removed when we try to find the k-core, it means it wasn't in the k-core.
    #           So k_core[v] < k.
    #           
    #       Correct Algorithm:
    #       1. Initialize current_deg = deg.
    #       2. active = all True.
    #       3. k = 0.
    #       4. While active nodes exist:
    #           k += 1
    #           # Find all active nodes with current_deg < k
    #           # Remove them. Their k_core is k-1.
    #           # Update current_deg of their neighbors.
    
    # Let's implement this.
    
    # To speed up finding nodes with deg < k, we can just scan. N=200.
    
    remaining_deg = deg.copy()
    is_active = np.ones(n, dtype=bool)
    k_core = np.zeros(n, dtype=int)
    
    k = 0
    while np.any(is_active):
        k += 1
        # Find nodes to remove: active and remaining_deg < k
        to_remove = np.where(is_active & (remaining_deg < k))[0]
        
        if len(to_remove) == 0:
            # No nodes to remove for this k, increment k and try again?
            # If no node has deg < k, then all active nodes are in the k-core.
            # We should continue to increase k until we find a node to remove?
            # Yes.
            continue
        
        # Assign k_core for these nodes.
        # They were active until we tried to enforce k-core k.
        # So they are in the (k-1)-core but not the k-core.
        # Thus k_core[v] = k - 1.
        k_core[to_remove] = k - 1
        
        # Remove them
        is_active[to_remove] = False
        
        # Update degrees of neighbors
        # For each removed node v, for each neighbor u, if u is active, remaining_deg[u] -= 1
        for v in to_remove:
            neighbors = np.where(adjacency[v])[0]
            active_neighbors = neighbors[is_active[neighbors]]
            remaining_deg[active_neighbors] -= 1
            
    # Any nodes still active are in the max_k core?
    # The loop ends when no active nodes have deg < k.
    # This means all active nodes have deg >= k.
    # We should assign them the final k?
    # Actually, the loop condition `while np.any(is_active)` ensures we process everyone.
    # If we exit because `to_remove` was empty, we incremented k and found no one.
    # We should keep incrementing k until we remove everyone?
    # If no one is removed, it means min_degree >= k.
    # We can just set the remaining k_core to the current k?
    # No, if min_degree >= k, they are in the k-core.
    # They might be in the (k+1)-core?
    # We should continue increasing k until we remove them.
    
    # My loop above:
    # if len(to_remove) == 0: continue
    # This will increment k and check again.
    # Eventually, k will exceed max_degree, and remaining_deg (which is <= max_degree) will be < k.
    # So they will be removed.
    # So the logic holds.
    
    return k_core

def priority(adjacency):
    start_time = time.time()
    
    # Ensure adjacency is a numpy array for efficient operations
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    # 1. Degrees
    deg = np.sum(adjacency, axis=1).astype(float)
    
    # 2. K-core
    k_core = compute_k_core(adjacency, n).astype(float)
    
    # 3. Mean neighbor degree
    # Sum of degrees of neighbors
    # S_v = sum(d_u for u in N(v))
    # We can compute this using matrix multiplication: deg @ adjacency
    # Since adjacency is symmetric, adj @ deg gives sum of degrees of neighbors for each node.
    # Note: adj is bool, deg is float.
    neighbor_deg_sum = adjacency @ deg
    
    # Mean neighbor degree (avoid div by zero)
    # If degree is 0, mean is 0.
    mean_neighbor_deg = np.divide(neighbor_deg_sum, deg, out=np.zeros_like(deg), where=deg>0)
    
    # 4. Combine scores
    # We want to prioritize low degree, low k-core, low mean neighbor degree.
    # Weighting:
    # Degree is the strongest signal.
    # K-core is a good tie-breaker/structural signal.
    # Mean neighbor degree refines the local structure.
    
    # Let's try: score = - (deg + 0.5 * k_core + 0.1 * mean_neighbor_deg)
    score = -(deg + 0.5 * k_core + 0.1 * mean_neighbor_deg)
    
    # Safety check: if time is running out, return current best (which is just this calculation)
    # Calculation is < 1ms for N=200.
    
    return score
