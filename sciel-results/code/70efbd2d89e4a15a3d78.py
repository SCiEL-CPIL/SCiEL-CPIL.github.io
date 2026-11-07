import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # Initialize heuristic matrix
    eta = np.ones((n, n), dtype=np.float64)
    
    # Base: Inverse distance
    # Avoid division by zero
    dm_safe = dm + 1e-9
    eta = 1.0 / dm_safe
    
    # Depot Proximity Factor
    # d_i0 is distance from node i to depot (node 0)
    d_to_depot = dm[:, 0] # Shape (n,)
    
    # We want to penalize moving to a node j that is much further from the depot than i is.
    # Ratio R[i,j] = d_i0 / d_j0
    # If d_j0 > d_i0, R < 1 (penalty)
    # If d_j0 < d_i0, R > 1 (bonus)
    
    # Handle depot (i=0): d_00 = 0.
    # For i=0, we don't want the ratio to be 0. 
    # Standard practice: For the first step, just use inverse distance.
    # For subsequent steps, apply the ratio.
    
    # Create ratio matrix
    # d_i0 is row vector, d_j0 is column vector
    d_i = d_to_depot[:, np.newaxis] # (n, 1)
    d_j = d_to_depot[np.newaxis, :] # (1, n)
    
    # Add epsilon to avoid div by zero in ratio
    ratio = (d_i + 1e-9) / (d_j + 1e-9)
    
    # Exponent kappa
    kappa = 0.5
    geo_factor = ratio ** kappa
    
    # For i=0, d_i is 0, so ratio is 0. geo_factor is 0.
    # We want the depot row to be just inverse distance.
    # So, for row 0, set geo_factor to 1.
    geo_factor[0, :] = 1.0
    
    # Also, we don't want to move to self (eta[i,i] should be 0 or very small, but ACO handles unvisited)
    # The task says "picks an unvisited customer". The heuristic matrix is used for all i,j.
    # Usually diagonal is 0 or 1. Let's set diagonal to 0 to discourage self-loops if logic allows, 
    # but the prompt says "picks an unvisited customer", so the code likely masks visited nodes.
    # However, if the code doesn't mask, we should ensure eta[i,i] is low.
    np.fill_diagonal(eta, 0.0)
    np.fill_diagonal(geo_factor, 0.0) # Not strictly needed if eta is 0
    
    # Apply geometric factor
    eta = eta * geo_factor
    
    # Demand Adjustment?
    # The previous program used a demand penalty. 
    # Let's try a slight demand-awareness: 
    # High demand nodes are "heavier". Maybe we should visit them when we have more space?
    # Static heuristic can't know space.
    # However, clustering high demand nodes might be good.
    # Let's skip demand for now and rely on geometry, as demand penalty was removed in the better inspiration?
    # Actually, inspiration 1 (1.0349) was better than current (1.0164). 
    # The diff for inspiration 1 was truncated but it seemed to simplify.
    
    # Normalize
    # We want the heuristic to be of similar magnitude to pheromones (which start at 1).
    # Let's scale so the mean is 1.
    mean_eta = np.mean(eta)
    if mean_eta > 0:
        eta = eta / mean_eta
    
    # Ensure positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
