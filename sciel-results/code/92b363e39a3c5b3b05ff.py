import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    Use a static heuristic score based on local structural properties.
    Score(v) = average_degree_of_neighbors(v) - degree(v)
    
    Rationale:
    - We want to pick vertices that "block" high-degree hubs efficiently.
    - A low-degree vertex connected to high-degree hubs, when picked, removes the hubs 
      from the graph without removing many other low-degree candidates that are 
      connected to those hubs but not to the picked vertex.
    - High-degree vertices should have lower priority because picking them removes 
      many neighbors, potentially many of which are low-degree and could have formed 
      a larger independent set among themselves.
    - The term (avg_neighbor_degree - own_degree) rewards low-degree vertices with 
      high-degree neighbors.
    """
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Compute degrees
    deg = adj.sum(axis=1).astype(float)
    
    # Compute sum of degrees of neighbors for each vertex
    # sum_deg_neighbors[v] = sum(deg[u] for u in neighbors(v))
    # This can be computed efficiently using matrix multiplication or dot product
    # adj @ deg gives the sum of degrees of neighbors for each vertex
    sum_deg_neighbors = adj.dot(deg)
    
    # Average degree of neighbors
    # Avoid division by zero for isolated nodes
    # If deg[v] == 0, avg is 0 (or undefined, but contribution to score should be neutral or based on deg only)
    # For isolated nodes, deg=0, sum_deg_neighbors=0. Score = 0 - 0 = 0.
    # We can handle division safely.
    avg_deg_neighbors = np.zeros(n)
    mask = deg > 0
    avg_deg_neighbors[mask] = sum_deg_neighbors[mask] / deg[mask]
    
    # Score: We want to maximize this score.
    # Higher score = Higher priority.
    # We want to pick vertices with LOW degree and HIGH neighbor degree.
    # So Score = avg_deg_neighbors - deg
    scores = avg_deg_neighbors - deg
    
    return scores
