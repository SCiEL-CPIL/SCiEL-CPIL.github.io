"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Return a score per vertex for the greedy independent set decoder.
    
    Strategy: Simulate the greedy selection process to compute marginal contributions.
    We perform a single forward pass that mimics the greedy decoder.
    For each vertex, we estimate its "value" based on:
    1. Its degree (lower is better, as it blocks fewer vertices).
    2. The degree of its neighbors (higher is better, as blocking high-degree vertices
       is more efficient in terms of opportunity cost).
    3. A dynamic "survival" probability or weight derived from a simulation.
    
    To keep it efficient (O(n^2) or O(n^3) worst case but fast in practice for n=200),
    we compute a static score that correlates with good greedy ordering.
    
    Key Insight from inspirations:
    - Low degree is primary.
    - Mean neighbor degree is secondary (blocking "expensive" neighbors is good).
    - We can refine this by simulating a few steps or using a more sophisticated
      local structure metric.
      
    Let's try a score based on:
    score(v) = -degree(v) + alpha * mean_neighbor_degree(v) + beta * clustering(v)
    
    Actually, the directive suggests a "dynamic process" model.
    Let's simulate the greedy process but assign scores based on the "order" 
    a vertex would be picked if we used a specific heuristic, then invert or use 
    that order? No, the decoder uses the scores we return.
    
    Alternative: 
    The greedy decoder picks v if score(v) is highest among available.
    We want v to be picked early if it has low degree and high neighbor degrees.
    
    Let's compute:
    1. degrees
    2. neighbor_degrees_sum = A @ degrees
    3. mean_neighbor_degree = neighbor_degrees_sum / (degrees + eps)
    
    Score = -degrees + w1 * mean_neighbor_degree
    
    Let's also add a term for "independence" of neighbors.
    If neighbors are not connected to each other, picking v is "wasteful" in terms of 
    blocking potential independent sets among them? No, if neighbors are connected, 
    they can't be picked together anyway, so blocking them is less of a loss.
    So high clustering among neighbors is GOOD.
    
    Let's use:
    score = -degrees + w1 * mean_neighbor_degree + w2 * clustering_coeff
    
    We will tune w1 and w2. Based on the lesson, reducing degree dominance helped.
    So w1 should be significant.
    
    Let's also consider a "simulated" score.
    We can run a quick greedy simulation with a random or simple heuristic to see 
    which vertices get picked early? No, that's circular.
    
    Let's stick to the structural features but optimize the weights.
    
    Implementation:
    1. degrees = A.sum(axis=1)
    2. nbr_deg_sum = A @ degrees
    3. mean_nbr_deg = nbr_deg_sum / (degrees + 1e-9)
    4. triangles = 0.5 * (A @ A * A).sum(axis=1)
    5. max_tri = 0.5 * degrees * (degrees - 1)
    6. clustering = triangles / (max_tri + 1e-9)
    
    score = -degrees + 0.5 * mean_nbr_deg + 0.5 * clustering * degrees.mean()
    
    Wait, the scale of mean_nbr_deg is similar to degrees.
    The scale of clustering is 0-1.
    
    Let's try:
    score = -degrees + 0.8 * mean_nbr_deg
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
        
    # 1. Degrees
    degrees = adjacency.sum(axis=1).astype(float)
    
    # 2. Mean Neighbor Degree
    nbr_deg_sum = adjacency @ degrees
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_nbr_deg = np.where(degrees > 0, nbr_deg_sum / degrees, 0.0)
        
    # 3. Clustering Coefficient
    # Triangles containing v
    # (A @ A) gives number of common neighbors.
    # (A @ A * A).sum(axis=1) counts each triangle twice.
    common_neighbors = adjacency @ adjacency
    triangles_v = 0.5 * (common_neighbors * adjacency).sum(axis=1)
    max_triangles = 0.5 * degrees * (degrees - 1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        clustering_coeff = np.where(max_triangles > 0, triangles_v / max_triangles, 0.0)
        
    # 4. Combine Scores
    # We want to prioritize low degree, but also high mean neighbor degree.
    # The lesson says: "Reducing the dominance of the degree penalty allows the 
    # mean neighbor degree term to influence vertex selection... Measured: 1.008 -> 1.026."
    # This suggests a weight > 0 for mean_nbr_deg is good.
    
    # Let's normalize features to be comparable?
    # degrees range ~0 to n.
    # mean_nbr_deg range ~0 to n.
    # clustering range 0 to 1.
    
    # Let's try a linear combination.
    # score = -degrees + w1 * mean_nbr_deg + w2 * clustering_coeff * scale
    
    # If we just use -degrees + mean_nbr_deg, vertices with low degree but high 
    # neighbor degree might beat vertices with slightly higher degree but low 
    # neighbor degree. This is desirable.
    
    # Let's try w1 = 1.0.
    # This effectively scores by: -(degree - mean_neighbor_degree)
    # i.e., we prefer vertices where degree < mean_neighbor_degree.
    # This is a "local independence" measure.
    
    scores = -degrees + 1.0 * mean_nbr_deg
    
    # Add a small clustering bonus?
    # If neighbors are clustered, they are less likely to be picked anyway, 
    # so blocking them is cheaper.
    # Let's add a small term.
    scale_clust = degrees.mean() * 0.1
    scores += 0.5 * clustering_coeff * scale_clust
    
    return scores
# EVOLVE-BLOCK-END
