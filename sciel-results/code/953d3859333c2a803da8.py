"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    Strategy: Prefer vertices with low degree, low average neighbor degree, 
    low eigenvector centrality, and low k-core number.
    """
    start_time = time.time()
    
    # Ensure float matrix for linear algebra
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
        
    # 1. Degree
    deg = A.sum(axis=1)
    
    # 2. Average Neighbor Degree
    sum_nei_deg = A @ deg
    avg_nei_deg = np.zeros(N)
    mask = deg > 0
    avg_nei_deg[mask] = sum_nei_deg[mask] / deg[mask]
    
    # 3. Eigenvector Centrality
    # For small N, dense eig is sufficient and robust
    v = np.zeros(N)
    try:
        # A is symmetric, so eigenvectors are real
        eigvals, eigvecs = np.linalg.eig(A)
        # Find the eigenvalue with the largest magnitude (principal)
        # For connected graphs, this is the largest real eigenvalue
        idx = np.argmax(eigvals.real)
        v = eigvecs[:, idx].real
        # Standardize orientation to be positive sum
        if v.sum() < 0:
            v = -v
    except Exception:
        v = np.zeros(N)
        
    # 4. K-Core Number
    # Iterative peeling
    k_core = np.zeros(N, dtype=int)
    active = np.ones(N, dtype=bool)
    current_deg = deg.copy()
    
    # Simple O(N^2) k-core peeling is fine for N=200
    # We can do a simpler approximation or exact peeling
    # Exact peeling:
    # Sort by degree, peel lowest degree nodes, assign core number, update neighbors
    # For N=200, even a naive loop is fine.
    
    # Let's use a simple iterative removal approach
    # k is the core number
    # A node is in k-core if it has at least k neighbors in the k-core.
    # We can approximate this by the "onion" layers.
    
    # Efficient enough for N=200:
    # 1. Create a list of nodes
    # 2. Repeatedly remove nodes with degree < current_k
    # This is a bit complex to implement correctly in a few lines.
    # Alternative: Use the fact that for ER graphs, degree is a strong proxy for core number.
    # However, k-core captures "nested" structure better.
    
    # Let's try a simple k-core calculation:
    # Max degree is D.
    # For k from 1 to D:
    #   Find all nodes with degree >= k.
    #   If a node's degree in the subgraph induced by nodes with degree >= k is < k, remove it.
    # This is O(D * (N+M)). D is usually small for ER(200, p).
    
    # Given time constraints and N=200, we can use a simpler metric if k-core is too slow to code correctly.
    # But k-core is a known good heuristic.
    
    # Implement K-Core Number via peeling
    # This is a strong structural signal for independent set (low core = peripheral)
    k_core = np.zeros(N, dtype=int)
    active = np.ones(N, dtype=bool)
    current_deg = deg.copy()
    
    # Simple O(N^2) peeling is fine for N=200
    # We peel nodes with degree < k, incrementing k as we go
    # Actually, standard algorithm:
    # 1. Start with all nodes active.
    # 2. Find node with min degree in active subgraph.
    # 3. If min_deg < k, remove it, update neighbors, repeat.
    # 4. Else, all remaining are in k-core. Increment k? 
    #    Actually, the standard peeling assigns core number = k when node is removed.
    #    Let's use the standard "onion" peeling.
    
    # Efficient implementation for N=200:
    # We can just iterate k from 1 to max_deg.
    # For each k, remove nodes with current_degree < k.
    # But we need to be careful with the order.
    
    # Let's use a queue-based approach for efficiency, though N=200 is small.
    # Or just simple loops.
    
    max_k = int(deg.max()) if N > 0 else 0
    active = np.ones(N, dtype=bool)
    cur_deg = deg.copy()
    
    for k in range(1, max_k + 1):
        # Find all active nodes with degree < k
        # Remove them
        # Update degrees of their neighbors
        # Repeat until no more nodes can be removed for this k?
        # No, the standard definition:
        # A node is in the k-core if it has at least k neighbors in the k-core.
        # Peeling:
        # While there exists a node v with degree < k:
        #   Remove v, assign core_number[v] = k-1 (or current k?)
        #   Update degrees of neighbors.
        
        # Actually, the core number of a node is the largest k such that it is in the k-core.
        # So we peel from k=1 upwards.
        # When we remove a node at step k, its core number is k-1?
        # Let's check:
        # If a node is removed when looking for 1-core, it has degree 0. Core number 0.
        # If removed when looking for 2-core, it had degree < 2 in the remaining graph. 
        # It had at least 1 neighbor in the 1-core (otherwise it would have been removed in 1-core).
        # So core number is 1.
        
        # So:
        # k=1: remove nodes with deg < 1. Assign core=0.
        # k=2: remove nodes with deg < 2. Assign core=1.
        # ...
        # k=K: remove nodes with deg < K. Assign core=K-1.
        # Remaining nodes have core >= K.
        
        # Let's implement this.
        
        # To make it efficient, we can just loop.
        # Since N=200, even O(N^3) is fine.
        
        changed = True
        while changed:
            changed = False
            # Find active nodes with cur_deg < k
            to_remove = np.where(active & (cur_deg < k))[0]
            if len(to_remove) == 0:
                break
            for v in to_remove:
                if not active[v]:
                    continue
                active[v] = False
                k_core[v] = k - 1
                changed = True
                # Update neighbors
                neighbors = np.where(A[v, :] == 1)[0]
                for u in neighbors:
                    if active[u]:
                        cur_deg[u] -= 1
                        
    # Assign core numbers to remaining active nodes
    # They are in the max_k core? 
    # Actually, the loop ends when no node has deg < max_k.
    # So all remaining have deg >= max_k.
    # Their core number is at least max_k.
    # Since max_k is the max degree, they can't be higher than max_k.
    # So they are max_k.
    remaining = np.where(active)[0]
    k_core[remaining] = max_k
    
    # 5. Second-Order Neighbor Degree
    # Average degree of neighbors' neighbors
    # A^2 @ deg / (A^2 @ ones)
    A2 = A @ A
    sum_nei2_deg = A2 @ deg
    avg_nei2_deg = np.zeros(N)
    mask2 = (A2 @ np.ones(N)) > 0
    avg_nei2_deg[mask2] = sum_nei2_deg[mask2] / (A2 @ np.ones(N))[mask2]
    
    # Z-Score Normalization
    def zscore(x):
        m = x.mean()
        s = x.std()
        if s < 1e-9:
            return np.zeros_like(x)
        return (x - m) / s
        
    z_deg = zscore(deg)
    z_avg = zscore(avg_nei_deg)
    z_eig = zscore(v)
    z_core = zscore(k_core.astype(float))
    z_avg2 = zscore(avg_nei2_deg)
    
    # Composite Score
    # We want to minimize centrality measures.
    # Degree is primary.
    # K-core is a strong structural proxy for "coreness".
    # Avg neighbor degree captures local density.
    # Eigenvector centrality captures global influence.
    
    # Tuning weights:
    # Degree and Core are highly correlated but distinct.
    # Let's try: 1.0 * deg + 0.8 * core + 0.5 * avg + 0.5 * eig + 0.2 * avg2
    
    score = -(1.0 * z_deg + 0.8 * z_core + 0.5 * z_avg + 0.5 * z_eig + 0.2 * z_avg2)
    
    # Safety check
    if time.time() - start_time > 5.0:
         # Fallback to simple degree
         return -deg.astype(float)
         
    return score
# EVOLVE-BLOCK-END
