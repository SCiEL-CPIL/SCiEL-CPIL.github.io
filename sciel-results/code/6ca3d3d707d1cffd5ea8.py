"""Priority function using refined Fiedler vector for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def priority(adjacency):
    """
    Score vertices using the Fiedler vector of the graph Laplacian.
    Vertices with large |Fiedler component| are at the periphery of the
    graph's main connected structure and are safer to pick for an
    independent set.
    
    Improvements:
    1. Use dense linear algebra for robustness and speed on N=200.
    2. Increased weight on degree penalty to avoid picking high-degree hubs.
    3. Safety check for disconnected graphs (fall back to degree-based).
    """
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]

    if n == 0:
        return np.array([], dtype=np.float64)
    
    if n == 1:
        return np.array([0.0], dtype=np.float64)

    # Ensure symmetry
    A = (A + A.T) / 2.0

    # Compute degree vector
    deg = A.sum(axis=1)

    # Build Laplacian L = D - A
    L = np.diag(deg) - A

    try:
        # Dense eigendecomposition is fast and robust for N=200
        vals, vecs = np.linalg.eigh(L)
        
        # Check for disconnected graph
        # If the second smallest eigenvalue is very close to 0, the graph is disconnected
        # In that case, the Fiedler vector is not well-defined as a global periphery indicator
        if vals[1] < 1e-5:
            # Fallback: Degree-based score (lower degree is better)
            # Normalize degree to [0, 1] for consistent scaling
            max_deg = deg.max()
            if max_deg > 0:
                deg_norm = deg / max_deg
            else:
                deg_norm = deg
            return -deg_norm.astype(np.float64)

        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = vecs[:, 1]
        
        # Use magnitude
        scores = np.abs(fiedler)
        
        # Normalize Fiedler scores to [0, 1] range for better combination with degree
        fiedler_min = scores.min()
        fiedler_max = scores.max()
        if fiedler_max > fiedler_min:
            scores = (scores - fiedler_min) / (fiedler_max - fiedler_min)
        else:
            scores = np.zeros(n, dtype=np.float64)

        # Degree penalty: We want to avoid high-degree vertices.
        # Normalize degree to [0, 1]
        max_deg = deg.max()
        if max_deg > 0:
            deg_norm = deg / max_deg
        else:
            deg_norm = deg

        # Combine: High Fiedler magnitude (periphery) and Low Degree are good.
        # Current best was 0.1 weight. Let's try 0.5 to strongly penalize hubs.
        weight_deg = 0.5
        final_scores = scores - weight_deg * deg_norm

        return final_scores.astype(np.float64)

    except Exception:
        # Ultimate fallback: degree-based
        return -deg.astype(np.float64)
# EVOLVE-BLOCK-END
