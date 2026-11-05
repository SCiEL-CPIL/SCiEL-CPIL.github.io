import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Compute eigenvector centrality using power iteration on the adjacency matrix.
       This provides a global measure of a vertex's importance/centrality in the network.
    2. Prioritize vertices with LOW eigenvector centrality.
       Rationale: Vertices with low centrality are typically peripheral or "leaves" 
       relative to the core structure. Selecting them first allows the greedy decoder 
       to "peel off" the periphery, potentially preserving high-centrality vertices 
       (which are often hubs that block many others) to be handled later or effectively 
       blocking them if they become neighbors of selected peripheral nodes.
       
       Alternatively, one might think high centrality should be prioritized to "block" 
       hubs early. However, in MIS, picking a hub removes it AND its neighbors. 
       If a hub is picked, we lose the hub (good, it blocks many) but we also lose 
       all its neighbors (bad, they might have been part of the IS).
       
       Actually, the standard heuristic for MIS is often "pick the vertex with minimum 
       degree" (LDF - Lowest Degree First). Why? Because picking a low-degree vertex 
       removes few neighbors, preserving the rest of the graph structure for further 
       selection. Picking a high-degree vertex (hub) removes many neighbors, which 
       likely reduces the size of the remaining independent set significantly.
       
       Eigenvector centrality is a smoothed version of degree. Low eigenvector 
       centrality corresponds to vertices that are not well-connected to other 
       well-connected vertices. These are effectively "low structural impact" vertices.
       
       Therefore, we assign a score that is inversely related to eigenvector centrality.
       To make it compatible with the greedy "descending score" decoder, we return 
       -centrality.
       
       We also add a small component of local degree to break ties or refine the 
       ordering, as eigenvector centrality can be slow to converge or have ties.
       
       Score = -alpha * centrality - beta * degree
       
       Since we want to minimize both, and the decoder picks highest score, 
       we use negative values.
       
       Let's use:
       score = -centrality - 0.1 * degree
       
       Wait, the current best uses -deg + 0.6 * avg_neighbor_deg.
       If centrality correlates with degree, -centrality is similar to -deg.
       But centrality captures global structure.
       
       Let's try:
       score = -centrality
       
       And to be safe, combine with a small degree penalty if centrality is not 
       strictly monotonic with degree.
       
       Actually, let's look at the mechanics.
       If u has low centrality, it's a "leaf" or in a sparse part.
       Picking u removes u and its neighbors.
       If u is a leaf (deg 1), it removes itself and its neighbor.
       If the neighbor is a hub, we've removed the hub. That's good?
       No, if we remove the hub, we can't pick the hub later.
       But if we don't remove the hub now, we might pick the hub later?
       If we pick the hub later, we remove ALL its neighbors.
       So it is better to remove the hub's neighbors one by one?
       No, if we pick a leaf u adjacent to hub h, we remove u and h.
       Then h is gone. We lost h, but we also lost u.
       If we didn't pick u, we might pick another leaf v adjacent to h?
       No, h is adjacent to many leaves.
       
       The key insight of LDF is: picking a low-degree node is "cheap" because it 
       sacrifices few other nodes. Picking a high-degree node is "expensive" because 
       it sacrifices many other nodes.
       
       Eigenvector centrality generalizes this. A node with high centrality is 
       "expensive" to pick because it is connected to other important nodes.
       
       So, minimizing centrality is the correct direction.
       
       We will compute the principal eigenvector.
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Power iteration for principal eigenvector
    # Initialize with random or ones
    # Using ones is stable and fast for ER graphs
    x = np.ones(n)
    x /= np.linalg.norm(x)
    
    # Number of iterations: 50 is usually enough for convergence in ER(200)
    # We check time every 10 iterations
    max_iter = 100
    for i in range(max_iter):
        x_new = adj @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-10:
            # Graph might be disconnected or zero, fallback to degree
            break
        x_new /= norm
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-8:
            x = x_new
            break
        x = x_new
        
        # Time check
        if time.time() - start_time > 100:
            break
            
    # Fallback if power iteration failed or converged to zero
    if np.linalg.norm(x) < 1e-10:
        deg = adj.sum(axis=1)
        return -deg
        
    # Ensure x is non-negative (principal eigenvector of symmetric matrix can be 
    # flipped, but we want centrality to be positive)
    if x.min() < 0:
        x = -x
        
    # Normalize x to [0, 1] range for consistent scaling with degree
    # Or just use it directly. 
    # Degree is ~10-20. Eigenvector components might be ~0.1-0.5.
    # Let's scale x to be comparable to degree.
    # Max degree is usually around 2*mean_deg.
    # Let's just use x directly and tune weights.
    
    # Strategy: Minimize centrality. 
    # Add a small degree term to break ties and handle local variations.
    deg = adj.sum(axis=1)
    
    # The current best score was around 1.015.
    # If we just use -x, is it better?
    # Let's try a combination.
    # score = -x - 0.01 * deg
    
    # Actually, let's stick to the pure spectral signal first, as it's the "different" strategy.
    # But to be robust, let's normalize x.
    x_norm = x / (x.max() + 1e-9)
    
    # Score: Lower centrality is better.
    # We return negative centrality so that lower centrality -> higher score (less negative).
    # Wait, if centrality is 0.1 and 0.5.
    # -0.1 > -0.5. So 0.1 is picked first. Correct.
    
    # Let's add a tiny bit of degree to ensure local structure is respected if 
    # eigenvectors are flat.
    # score = -x_norm - 0.01 * deg / (deg.max() + 1e-9)
    
    # Let's try just -x_norm first.
    score = -x_norm
    
    # Safety deadline check
    if time.time() - start_time > 140:
        return -deg
        
    return score
