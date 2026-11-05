"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Compute a 2-coloring heuristic via the principal eigenvector of the adjacency matrix.
       Vertices with large positive eigenvector components are preferred (they form a 
       large independent set in many sparse graphs, approximating a bipartition).
    2. Combine this with degree-based scoring: lower degree vertices are generally 
       "safer" to pick (fewer conflicts), but we want to pick from the dominant color class.
    3. Use a composite score: prioritize vertices from the dominant color class, 
       and within that, prefer lower-degree vertices.
    
    The greedy decoder picks vertices in descending score order, skipping neighbors 
    of already-picked vertices.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adjacency = np.asarray(adjacency, dtype=float)
    
    # Compute degrees
    degrees = adjacency.sum(axis=1)
    
    # Try spectral approach: principal eigenvector via power iteration
    # This is O(n^2 * iterations), which for n=200 is fast
    try:
        # Power iteration for principal eigenvector
        v = np.ones(n) / np.sqrt(n)
        for _ in range(100):
            if time.time() > deadline:
                break
            new_v = adjacency @ v
            norm = np.linalg.norm(new_v)
            if norm < 1e-12:
                break
            new_v /= norm
            # Check convergence
            if np.linalg.norm(new_v - v) < 1e-10:
                v = new_v
                break
            v = new_v
        
        # The principal eigenvector gives a soft 2-coloring
        # Vertices with positive v[i] are in one "color class", negative in the other
        # We want to prefer vertices from the larger/more connected color class
        
        # Determine the dominant color class
        pos_sum = v[v > 0].sum()
        neg_sum = v[v < 0].sum()
        
        # If the graph is roughly bipartite, the principal eigenvector will have
        # a clear sign pattern. Prefer vertices from the color class with larger 
        # total magnitude (more "central" vertices).
        
        # Score: combine eigenvector magnitude (prefer dominant class) with 
        # inverse degree (prefer low-degree vertices within a class)
        
        # Normalize eigenvector components to [0, 1] range
        v_min = v.min()
        v_max = v.max()
        v_range = v_max - v_min
        if v_range < 1e-12:
            v_norm = np.zeros(n)
        else:
            v_norm = (v - v_min) / v_range
        
        # Degree normalization
        deg_min = degrees.min()
        deg_max = degrees.max()
        deg_range = deg_max - deg_min
        if deg_range < 1e-12:
            deg_norm = np.zeros(n)
        else:
            deg_norm = (degrees - deg_min) / deg_range
        
        # Composite score: 
        # - Primary: eigenvector-based color preference (higher v_norm = more preferred)
        # - Secondary: lower degree is better (so use 1 - deg_norm)
        # Weight the eigenvector component more heavily
        scores = 0.7 * v_norm + 0.3 * (1.0 - deg_norm)
        
        # Add a small random tiebreaker to avoid symmetry issues
        np.random.seed(42)
        scores += np.random.uniform(0, 0.01, size=n)
        
        return scores
    
    except Exception:
        # Fallback to degree-based scoring
        return -degrees.astype(float)
# EVOLVE-BLOCK-END
