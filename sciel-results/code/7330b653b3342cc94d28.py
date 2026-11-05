import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin

    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])

    # 1. Compute Degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # 2. Compute Average Neighbor Degree
    # Sum of degrees of neighbors
    nbr_deg_sums = adj @ degrees
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degrees > 0, nbr_deg_sums / degrees, 0.0)
        
    # 3. Compute Spectral Component (Fiedler Vector)
    # We want the eigenvector corresponding to the 2nd largest eigenvalue.
    # For connected graphs, this is the Fiedler vector, which helps identify
    # bipartite-like structures where picking one side of the "split" is good.
    v2 = np.zeros(n)
    try:
        if time.time() < deadline - 10:
            # Use eigh for symmetric matrices
            eigvals, eigvecs = np.linalg.eigh(adj)
            # eigh returns eigenvalues in ascending order.
            # Largest is at index -1 (Perron).
            # 2nd largest is at index -2 (Fiedler).
            # If the graph is disconnected, there might be multiple eigenvalues close to the max,
            # but -2 is still a reasonable heuristic for the "next" principal mode.
            
            idx = -2
            v2 = eigvecs[:, idx]
            # Normalize to have max abs value 1
            max_val = np.max(np.abs(v2))
            if max_val > 0:
                v2 = v2 / max_val
            else:
                v2 = np.zeros(n)
    except Exception:
        v2 = np.zeros(n)

    # 4. Compute Core Number (Approximate or Exact if fast)
    # Core number is a strong indicator of centrality.
    # Vertices with low core numbers are on the periphery.
    # We want to pick periphery vertices first? 
    # Actually, greedy MIS often works well by picking low-degree vertices.
    # Core number is related to degree but captures structure better.
    # Let's compute it exactly as N=200 is small.
    
    core_numbers = np.zeros(n, dtype=int)
    try:
        if time.time() < deadline - 5:
            # Simple O(N^2) or O(N+E) core decomposition
            # Since N=200, even O(N^2) is trivial.
            # We can use a simple peeling algorithm.
            
            # Copy degrees and adjacency for peeling
            deg = degrees.copy()
            # We need to track active vertices
            active = np.ones(n, dtype=bool)
            # We need to simulate removal. 
            # Standard algorithm:
            # 1. Sort vertices by degree.
            # 2. Repeatedly remove vertex with min degree, update neighbors.
            
            # Efficient implementation for small N:
            # Use a bucket queue or just scan for min each time (O(N^2) total).
            # N=200, N^2 = 40,000 ops. Very fast.
            
            remaining_deg = degrees.copy()
            remaining_active = np.ones(n, dtype=bool)
            
            for _ in range(n):
                # Find active vertex with min degree
                min_deg = np.inf
                min_idx = -1
                for i in range(n):
                    if remaining_active[i]:
                        if remaining_deg[i] < min_deg:
                            min_deg = remaining_deg[i]
                            min_idx = i
                            
                if min_idx == -1:
                    break
                
                core_numbers[min_idx] = min_deg
                remaining_active[min_idx] = False
                
                # Update neighbors of min_idx
                neighbors = np.where(adj[min_idx] > 0)[0]
                for nb in neighbors:
                    if remaining_active[nb]:
                        remaining_deg[nb] -= 1

    except Exception:
        core_numbers = degrees.astype(int) # Fallback to degree

    # 5. Combine Scores
    # We want to pick vertices that are "easy" to include.
    # Low degree is good.
    # Low core number is good (periphery).
    # High average neighbor degree is good (neighbors are hubs, so they are less likely to be picked early? 
    # Wait, if neighbors are hubs, they have high degree, so they have low priority in a "low degree first" scheme.
    # So if my neighbors are low priority, I am "safe" to pick.
    # So High Avg Nbr Degree -> Good.
    
    # Spectral: 
    # The Fiedler vector splits the graph into two communities.
    # In a bipartite graph, the independent set is one of the partitions.
    # The Fiedler vector has positive and negative values.
    # If we pick vertices with high absolute value? Or align with sign?
    # Greedy doesn't know the sign. 
    # However, often the "periphery" aligns with one side.
    # Let's try to add a small spectral component. 
    # Vertices with large |v2| are often central to the split.
    # Vertices with small |v2| are more "internal" to the communities?
    # Actually, for a path, Fiedler vector is sinusoidal. Ends are large, middle is small.
    # Ends are low degree. Middle is high degree.
    # So large |v2| correlates with low degree.
    # So we might want to *penalize* large |v2| if we are already penalizing degree?
    # Or maybe the spectral info adds value in dense graphs where degree is uniform.
    
    # Let's normalize the features to combine them.
    # Standardize features to mean 0, std 1 to avoid scale issues.
    
    def normalize(x):
        x = x - np.mean(x)
        std = np.std(x)
        if std > 1e-9:
            x = x / std
        return x

    deg_norm = normalize(degrees)
    nbr_norm = normalize(avg_nbr_deg)
    core_norm = normalize(core_numbers.astype(float))
    spec_norm = normalize(v2)

    # Weights:
    # Degree is the strongest predictor for greedy MIS.
    # Core number is similar but structural.
    # Nbr degree is secondary.
    # Spectral is tertiary.
    
    w_deg = 1.0
    w_core = 0.5
    w_nbr = 0.2
    w_spec = 0.1

    # Score = -w_deg * deg - w_core * core + w_nbr * nbr + w_spec * spec
    # Note: We want low degree/core, so negative weight.
    # We want high nbr degree, so positive weight.
    # Spectral: Let's assume vertices with higher v2 value are "better" to pick? 
    # In many cases, the "periphery" has consistent sign in one part.
    # Let's just add it with a small weight. If it hurts, it's small.
    
    scores = -w_deg * deg_norm - w_core * core_norm + w_nbr * nbr_norm + w_spec * spec_norm

    return scores
