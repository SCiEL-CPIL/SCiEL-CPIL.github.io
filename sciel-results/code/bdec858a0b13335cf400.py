import numpy as np

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    - Prioritize vertices with low degree (they block fewer neighbors).
    - Among low-degree vertices, prioritize those whose neighbors are high-degree (hubs).
      This is because picking a low-degree vertex that is a neighbor of a hub effectively
      "sacrifices" the hub (which would be a bad pick due to its high blocking power)
      while preserving other low-degree vertices.
    - Vertices whose neighbors are low-degree are penalized, as picking them would
      "waste" other good (low-degree) candidates.
    
    Score Formula:
    S(v) = -deg(v) - sum_{u in N(v)} (1 / (deg(u) + 1))
    
    This formula yields higher scores for:
    1. Low degree v.
    2. Neighbors u with high degree (since 1/(deg(u)+1) is small).
    """
    # Convert to float for arithmetic
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Calculate weights for each vertex based on its degree
    # High degree -> Low weight. Low degree -> High weight.
    # +1 avoids division by zero for isolated nodes.
    weights = 1.0 / (degrees + 1.0)
    
    # Calculate the penalty for each vertex:
    # penalty[v] = sum of weights of its neighbors
    # This is the matrix-vector product A @ weights
    neighbor_penalty = A @ weights
    
    # Final Score
    # We want to maximize this score.
    # -degrees favors low degree.
    # -neighbor_penalty favors vertices whose neighbors have high degrees (low weights).
    scores = -degrees - neighbor_penalty
    
    # Add a tiny deterministic noise to break ties based on index
    # This ensures a stable order if scores are identical, favoring lower indices
    # which can sometimes be slightly better in symmetric random graphs.
    tie_breaker = np.arange(n) * 1e-9
    
    return scores + tie_breaker
