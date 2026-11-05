"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Score vertices for greedy independent set selection.
    
    Strategy: Use a multi-scale heuristic that considers:
    1. Degree (lower is better for independent set)
    2. Number of neighbors that are high-degree (avoid being adjacent to hubs)
    3. Local density / clustering (lower local density is better)
    4. A soft approximation of 'independent set potential' using iterative refinement
    
    We use a time-bounded iterative refinement approach:
    - Start with degree-based scores
    - Iteratively adjust scores based on neighbor interactions
    - Use soft-min/soft-max style smoothing to handle the combinatorial structure
    """
    adjacency = np.asarray(adjacency, dtype=np.float64)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Track deadline with safety margin (150s wall clock, return by 140s)
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Initial: degree
    degree = adjacency.sum(axis=1)
    
    # Compute secondary features
    # For each vertex, count how many of its neighbors have high degree
    # This helps avoid selecting vertices that are adjacent to hubs
    
    # Normalize degree to [0, 1]
    if degree.max() > degree.min():
        deg_norm = (degree - degree.min()) / (degree.max() - degree.min())
    else:
        deg_norm = np.zeros(n)
    
    # For each vertex v, compute the average degree of its neighbors
    # neighbor_avg_deg[v] = sum(adjacency[v] * degree) / degree[v]
    neighbor_avg_deg = np.zeros(n)
    nonzero_deg = degree > 0
    if np.any(nonzero_deg):
        neighbor_avg_deg[nonzero_deg] = (adjacency * degree[:, None]).sum(axis=1)[nonzero_deg] / degree[nonzero_deg]
        # Actually let's do it properly
        neighbor_avg_deg = np.zeros(n)
        for v in range(n):
            if degree[v] > 0:
                neighbor_avg_deg[v] = (adjacency[v] * degree).sum() / degree[v]
    
    # Local clustering coefficient approximation: 
    # For vertex v, the fraction of edges among its neighbors
    # This is expensive, so we use a simpler proxy:
    # number of edges within the neighborhood / possible edges
    # We'll use a vectorized approximation
    
    # Simpler: use the number of common neighbors between v and each of its neighbors
    # This is O(n^3) in worst case, too expensive. Use a proxy.
    
    # Alternative: Use a soft independent set score via iterative message passing
    # Initialize scores as negative degree (lower degree = higher priority)
    scores = -degree.copy()
    
    # Iterative refinement: adjust scores based on neighbor scores
    # If a vertex has many high-scoring neighbors, its effective score should decrease
    # (because selecting it would block many good candidates)
    
    num_iterations = 0
    while time.time() < deadline:
        num_iterations += 1
        
        # Compute for each vertex the sum of scores of its neighbors
        neighbor_score_sum = (adjacency * scores[:, None]).sum(axis=1)
        
        # Average neighbor score
        avg_neighbor_score = np.zeros(n)
        if np.any(nonzero_deg):
            avg_neighbor_score[nonzero_deg] = neighbor_score_sum[nonzero_deg] / degree[nonzero_deg]
        
        # Adjust score: penalize vertices whose neighbors have high scores
        # This encourages selecting vertices that are not adjacent to other high-priority vertices
        # Use a soft penalty
        penalty = avg_neighbor_score
        
        # Combine: base score (negative degree) + interaction term
        # The interaction term should reduce the score of vertices adjacent to high-scoring vertices
        new_scores = scores - 0.1 * penalty
        
        # Check convergence
        if num_iterations > 1:
            change = np.max(np.abs(new_scores - scores))
            if change < 1e-8:
                break
        
        scores = new_scores
        
        # Add a small regularization to prevent all scores from converging to same value
        # Keep some spread based on original degree
        scores = scores - 0.01 * degree
    
    # Final scoring: combine multiple signals
    # 1. Refined score from iterative process
    # 2. Negative degree (original)
    # 3. Negative neighbor average degree (avoid hubs)
    
    # Normalize each component to [0, 1] range for combining
    def normalize(arr):
        if arr.max() > arr.min():
            return (arr - arr.min()) / (arr.max() - arr.min())
        return np.zeros_like(arr)
    
    score_norm = normalize(scores)
    neg_deg_norm = normalize(-degree)
    neg_neighbor_avg_norm = normalize(-neighbor_avg_deg)
    
    # Weighted combination
    final_scores = (0.5 * score_norm + 0.3 * neg_deg_norm + 0.2 * neg_neighbor_avg_norm)
    
    # Return as priority (higher = selected first)
    # The greedy decoder picks in descending order, so higher score = picked first
    return final_scores.astype(float)
# EVOLVE-BLOCK-END
