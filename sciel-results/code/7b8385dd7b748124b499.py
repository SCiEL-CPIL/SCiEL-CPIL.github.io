import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin out of 120s
    
    # Convert to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=np.int8)
        
    n = adjacency.shape[0]
    
    # Compute degrees
    degrees = adjacency.sum(axis=1)
    
    # Compute sum of neighbor degrees
    # sum_neighbor_degrees[i] = sum(degrees[j] for j where adjacency[i][j] == 1)
    # This is matrix-vector product: adjacency @ degrees
    sum_neighbor_degrees = adjacency @ degrees
    
    # Compute average neighbor degree
    # Avoid division by zero for isolated nodes
    # For isolated nodes, avg is 0.
    avg_neighbor_degrees = np.zeros(n, dtype=float)
    mask = degrees > 0
    avg_neighbor_degrees[mask] = sum_neighbor_degrees[mask] / degrees[mask]
    
    # Second-order: sum of neighbor degrees squared (captures hub adjacency more strongly)
    # Also compute: for each vertex, the number of neighbors that are "low-degree" 
    # (degree <= median). Picking a vertex adjacent to many low-degree nodes is costly
    # because those low-degree nodes would have been good picks themselves.
    
    # Alternative: use the variance of neighbor degrees as a feature
    # Vertices connected to a mix of high and low degree nodes may be less useful.
    # Simpler: count of neighbors with degree > mean_degree (hub-adjacency count)
    mean_degree = degrees.mean()
    hub_neighbors = (adjacency @ (degrees > mean_degree).astype(np.int8))
    
    # Heuristic:
    # Primary: Low degree is better (negative degree)
    # Secondary: Connected to high-degree nodes is better (positive avg_neighbor_degree)
    # Tertiary: Connected to many hub nodes is good (positive hub_neighbors)
    # This helps "sacrifice" hubs by picking their neighbors first.
    
    # Tuning coefficients
    alpha = 1.0   # weight on -degree
    beta = 1.2    # weight on avg_neighbor_degree
    gamma = 0.3   # weight on hub_neighbors count
    
    scores = -alpha * degrees.astype(float) + beta * avg_neighbor_degrees + gamma * hub_neighbors.astype(float)
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple degree if calculation took too long (unlikely)
        return -degrees.astype(float)
        
    return scores
