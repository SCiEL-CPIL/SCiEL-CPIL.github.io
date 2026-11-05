import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Primary: Lowest Degree (vertices with fewer neighbors are better candidates).
    2. Secondary: Prefer vertices whose neighbors have high average degree.
       Rationale: Selecting a vertex that is adjacent to hubs is beneficial because
       it "blocks" the hub. Using average neighbor degree normalizes the secondary
       term so it's comparable in scale to degree.
       
    Score = -deg(u) + beta * (sum(deg(N(u))) / (deg(u) + 1))
    
    The +1 in denominator avoids division by zero and slightly dampens the bonus
    for very low-degree vertices.
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Calculate degrees
    deg = adj.sum(axis=1)
    
    # 2. Calculate sum of degrees of neighbors
    # neighbor_deg_sum[i] = sum_{j in N(i)} deg[j]
    neighbor_deg_sum = adj @ deg
    
    # 3. Calculate average neighbor degree (normalized secondary term)
    # Use deg + 1 to avoid division by zero and to slightly penalize isolated vertices
    # less aggressively in the secondary term
    safe_deg = deg + 1.0
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # 4. Combine scores
    # Primary: minimize degree
    # Secondary: maximize average neighbor degree (to block hubs efficiently)
    # Beta controls the weight of the secondary term
    beta = 0.5
    
    score = -deg + beta * avg_neighbor_deg
    
    # Safety deadline check
    if time.time() - start_time > 140:
        # Fallback to simple LDF
        return -deg
        
    return score
