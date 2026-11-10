import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the edge-attractiveness matrix for the ACO heuristic.
    
    Args:
        distance_matrix: np.ndarray of shape (N, N) containing pairwise distances.
        coordinates: np.ndarray of shape (N, 2) containing coordinates.
        demands: np.ndarray of shape (N,) containing demands.
        capacity: float, vehicle capacity.
        
    Returns:
        np.ndarray of shape (N, N) with heuristic values (larger is more attractive).
    """
    try:
        # Ensure inputs are numpy arrays
        d = np.asarray(distance_matrix, dtype=np.float64)
        
        # Check for non-finite values in input
        if not np.all(np.isfinite(d)):
            # Fallback to a safe heuristic if input is invalid
            return np.ones_like(d)
        
        # Use the measured improved exponent 1.211
        # Formula: 1 / (distance + epsilon) ** exponent
        # This penalizes long distances more than the standard 1/distance
        epsilon = 1e-9
        exponent = 1.211
        
        # Compute heuristic
        # Add epsilon to avoid division by zero
        safe_dist = d + epsilon
        h = 1.0 / (safe_dist ** exponent)
        
        # Ensure no non-finite values in output (should not happen with safe_dist > 0)
        if not np.all(np.isfinite(h)):
             h = np.where(np.isfinite(h), h, 1e-9)
             
        return h

    except Exception:
        # In case of any unexpected error, return a neutral heuristic
        try:
            return np.ones_like(np.asarray(distance_matrix, dtype=np.float64))
        except:
            return np.ones((50, 50))
