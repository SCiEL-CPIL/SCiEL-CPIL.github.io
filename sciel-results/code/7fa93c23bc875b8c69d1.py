import numpy as np

def heuristics(distance_matrix):
    # Add a small epsilon to avoid division by zero
    d = distance_matrix + 1e-9
    
    # Use a stronger inverse power to bias towards shorter edges
    # 1.5 is a moderate increase over the seed's 1.0 and the known good 1.25
    h = 1.0 / np.power(d, 1.5)
    
    # Normalize to mean 1 to keep the scale comparable to the seed
    # This ensures the pheromone term (starting at 1) remains relevant
    mean_h = np.mean(h)
    h = h / mean_h
    
    # Ensure all values are positive (they should be, but just in case)
    h = np.where(h > 0, h, 1e-9)
    
    return h
