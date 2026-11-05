"""Program using spectral centrality for Maximum Independent Set priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign scores to vertices based on a combination of spectral centrality and degree.
    Vertices with lower combined centrality get higher priority.
    
    We compute:
    1. Principal eigenvector centrality via power iteration
    2. Degree centrality
    
    Then combine them to identify vertices that are both spectrally peripheral and
    have low degree, which are less likely to conflict with many other vertices.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    
    # Handle edge case of empty graph
    if n == 0:
        return np.array([])
    
    # Compute degree
    degree = adjacency.sum(axis=1)
    
    # Power iteration to find principal eigenvector (eigenvector centrality)
    # Initialize with random positive vector
    rng = np.random.RandomState(42)
    x = rng.rand(n)
    x = x / np.linalg.norm(x)
    
    # Iterate to convergence
    for _ in range(300):
        x_new = adjacency @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            # Graph has no edges or is disconnected in a way that gives zero vector
            x_new = np.ones(n) / np.sqrt(n)
            break
        x_new = x_new / norm
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-12:
            x = x_new
            break
        x = x_new
    
    # Ensure non-negative (eigenvector centrality should be non-negative for connected graphs)
    x = np.abs(x)
    
    # Normalize eigenvector centrality to [0, 1] range
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min > 1e-12:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = np.zeros(n)
    
    # Normalize degree to [0, 1] range
    deg_min = degree.min()
    deg_max = degree.max()
    if deg_max - deg_min > 1e-12:
        deg_norm = (degree - deg_min) / (deg_max - deg_min)
    else:
        deg_norm = np.zeros(n)
    
    # Combine: we want vertices with LOW eigenvector centrality AND LOW degree to be selected first
    # Use a weighted combination. Weight eigenvector more heavily as it captures global structure.
    # score = -(w1 * x_norm + w2 * deg_norm)
    # Higher score = lower combined centrality = selected first
    w1 = 0.7
    w2 = 0.3
    combined = w1 * x_norm + w2 * deg_norm
    scores = -combined
    
    return scores
# EVOLVE-BLOCK-END
