"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign a priority score to each vertex for greedy independent set selection.
    
    Approach:
    1. Formulate the priority as a spectral relaxation of the Maximum Weight Independent Set.
    2. Use the principal eigenvector of the symmetric normalized adjacency matrix to identify
       the "core" of the graph.
    3. Vertices in the core (high eigenvector value) are less desirable for the independent set
       because they have high connectivity and block many other vertices.
    4. Therefore, we invert the spectral score to favor peripheral (low coreness) vertices.
    5. Combine this inverted spectral score with a degree-based score (lower degree is better).
    """
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # Handle empty graph
    if n == 0:
        return np.array([], dtype=float)
    
    # Degrees
    degrees = adj.sum(axis=1)
    
    # Handle isolated vertices or empty edges
    if np.all(degrees == 0):
        return np.ones(n, dtype=float)
    
    # Symmetric Normalized Adjacency Matrix: A_sym = D^{-1/2} A D^{-1/2}
    # This ensures the matrix is symmetric, allowing for real eigenvectors.
    # It also normalizes the influence of high-degree nodes.
    
    # Avoid division by zero for isolated nodes
    sqrt_deg = np.sqrt(degrees)
    sqrt_deg_safe = np.where(sqrt_deg > 0, sqrt_deg, 1.0)
    
    D_inv_sqrt = np.diag(1.0 / sqrt_deg_safe)
    A_sym = D_inv_sqrt @ adj @ D_inv_sqrt
    
    # Set diagonal to 0 explicitly (should already be, but for safety)
    np.fill_diagonal(A_sym, 0.0)
    
    # Power iteration to find the principal eigenvector of A_sym
    # The principal eigenvector indicates the "core" of the graph.
    # High values = high connectivity/coreness.
    
    # Initialize with random vector, but ensure it's not orthogonal to the principal eigenvector
    # Using ones is a good start for connected graphs.
    x = np.ones(n, dtype=float)
    # Add small noise to break symmetry if the graph is highly regular
    x += np.random.uniform(-0.01, 0.01, n)
    
    max_iter = 100
    tol = 1e-8
    
    for _ in range(max_iter):
        x_new = A_sym @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new = x_new / norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    
    # x is the principal eigenvector (coreness).
    # We want LOW coreness for independent set.
    # So we invert the score.
    
    def normalize(v):
        v_min, v_max = v.min(), v.max()
        if v_max - v_min < 1e-12:
            return np.zeros_like(v)
        return (v - v_min) / (v_max - v_min)
    
    # Normalize coreness to [0, 1]
    coreness = normalize(x)
    
    # Invert coreness: high score for low coreness
    spectral_score = 1.0 - coreness
    
    # Degree score: high score for low degree
    degree_score = normalize(-degrees)
    
    # Combine scores
    # Weight spectral score more heavily as it captures global structure
    # which helps avoid picking hubs that block many others.
    combined = 0.7 * spectral_score + 0.3 * degree_score
    
    return combined
# EVOLVE-BLOCK-END
