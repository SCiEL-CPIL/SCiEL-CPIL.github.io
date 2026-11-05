import numpy as np
import time

def _greedy_decode(adjacency, scores):
    """Fixed greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    picked = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if forbidden[v]:
            continue
        picked[v] = True
        count += 1
        # Mark neighbours as forbidden
        neighbors = np.where(adjacency[v])[0]
        forbidden[neighbors] = True
    return count

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # 1. Compute Degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # 2. Compute Average Neighbor Degree
    # Sum of degrees of neighbors
    nbr_deg_sums = adj @ degrees
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degrees > 0, nbr_deg_sums / degrees, 0.0)
        
    # 3. Compute Principal Eigenvector (Perron Vector) for Bipartite-like Coloring
    # The principal eigenvector of the adjacency matrix often exhibits a sign
    # pattern that approximates a bipartition in graphs with strong community
    # structure or near-bipartite components. Vertices with the same sign
    # tend to be less connected to each other than to vertices with the opposite
    # sign. We use the magnitude of this component to boost scores for vertices
    # in the dominant color class (or simply use the signed value to create
    # a smooth separation).
    
    v1 = np.zeros(n)
    try:
        if time.time() < deadline - 10:
            # Compute only the largest eigenvalue and its vector for speed
            # Using power iteration or sparse methods if available, but for N=200
            # dense eigh is fine. We specifically want the LARGEST eigenvalue.
            eigvals, eigvecs = np.linalg.eigh(adj.astype(float))
            # Largest eigenvalue is the last one in ascending order
            v1 = eigvecs[:, -1]
            # Ensure consistent sign convention (optional, but helps if we used signed score)
            # For now, we will use the absolute value or signed value depending on strategy.
            # Strategy: Use signed value. If we pick high scores, we pick one side.
            # But greedy picks ALL high scores. If we want an independent set, we want
            # to pick one side of the bipartition.
            # If v1 has mixed signs, the "positive" side is an independent set approximation?
            # Not exactly, but vertices with high |v1| are often peripheral or part of the main structure.
            # Let's try using the signed value directly. High positive -> pick.
            # This effectively biases the greedy selection towards one "color" class.
            
            # Normalize for stability
            max_val = np.max(np.abs(v1))
            if max_val > 0:
                v1 = v1 / max_val
            else:
                v1 = np.zeros(n)
        else:
            v1 = np.zeros(n)
    except Exception:
        v1 = np.zeros(n)
        
    # 4. Combine Scores
    # Base: Low degree is good (standard greedy heuristic)
    # Bonus: High avg neighbor degree is good (removes hubs by prioritizing their low-degree neighbors? 
    #        Actually, high avg_nbr_deg means the vertex is connected to hubs. 
    #        In independent set, we usually want to avoid hubs. 
    #        If we pick a vertex with high avg_nbr_deg, we block many hubs. This is good.)
    # Spectral: Align with Principal Eigenvector (Bipartition heuristic)
    
    # Weights
    w_deg = 1.0
    w_nbr = 1.0
    w_spec = 0.5  # Increased weight for spectral component to leverage bipartition
    
    scores = -w_deg * degrees + w_nbr * avg_nbr_deg + w_spec * v1
    
    # 5. Local Polish: try swapping a few vertices to improve the greedy result
    # We can't change the decoder, but we can adjust scores to change the order.
    # Let's evaluate the current greedy result.
    best_count = _greedy_decode(adj, scores)
    best_scores = scores.copy()
    
    # Try small perturbations: for each vertex, try increasing/decreasing its score
    # slightly to see if it improves the greedy result.
    # This is expensive for 200 vertices, so do a limited number of iterations.
    max_polish_iters = 20
    for _ in range(max_polish_iters):
        if time.time() > deadline:
            break
        improved = False
        # Try swapping pairs of vertices in the score order
        # This is O(n^2) per iteration, which is 40000 for n=200. Acceptable.
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                # Swap scores of i and j
                scores[i], scores[j] = scores[j], scores[i]
                count = _greedy_decode(adj, scores)
                if count > best_count:
                    best_count = count
                    best_scores = scores.copy()
                    improved = True
                else:
                    # Revert
                    scores[i], scores[j] = scores[j], scores[i]
        if not improved:
            break
    
    return best_scores
