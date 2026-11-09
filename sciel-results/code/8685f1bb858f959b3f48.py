import numpy as np

def priority(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Sum of degrees of neighbors
    # adj @ degrees
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    
    # Sum of squared degrees of neighbors
    neighbor_sq_degree_sums = (adj @ (degrees ** 2)).astype(np.float64)
    
    # Average degree
    avg_deg = degrees.mean()
    
    # Penalty for blocking low-degree neighbors
    # For each vertex v, sum over neighbors u of max(0, avg_deg - d(u))
    # This is equivalent to:
    # P(v) = sum_{u in N(v)} (avg_deg - d(u)) where d(u) < avg_deg
    # We can compute this using matrix ops or a loop.
    # Since N=200, a loop is fast enough, but let's try vectorization.
    # Create a matrix of penalties for each edge (v, u)
    # penalty_matrix[v, u] = max(0, avg_deg - degrees[u]) if adj[v, u] else 0
    
    # To avoid O(N^2) memory if N was large, but N=200 is small.
    # deg_diff = avg_deg - degrees (shape N)
    # pos_diff = np.maximum(0, deg_diff) (shape N)
    # P(v) = sum_u adj[v, u] * pos_diff[u]
    pos_diff = np.maximum(0, avg_deg - degrees)
    penalties = (adj @ pos_diff).astype(np.float64)
    
    # Coefficients
    # C1: Weight for degree (penalty)
    # C2: Weight for neighbor degree sum (bonus)
    # C3: Weight for neighbor squared degree sum (bonus)
    # C4: Weight for low-neighbor penalty
    
    # We want degree to be the primary factor.
    # Max deg ~ 200. Max sum_deg ~ 40000.
    # If C1 is too small, high degree vertices with high neighbor sums might beat low degree vertices.
    # We want deg=1 to beat deg=2 even if deg=1 has bad neighbors and deg=2 has great neighbors?
    # Usually, yes. Low degree is king.
    
    # Let's set C1 = 1000
    # C2 = 1
    # C3 = 0.01
    # C4 = 10
    
    C1 = 1000.0
    C2 = 1.0
    C3 = 0.01
    C4 = 10.0
    
    scores = -degrees * C1 + neighbor_degree_sums * C2 + neighbor_sq_degree_sums * C3 - penalties * C4
    
    return scores
