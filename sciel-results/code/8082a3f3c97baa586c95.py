import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
        # distance_matrix shape (N, N)
        # demands shape (N,)
        
        # Add small epsilon to avoid division by zero
        d = distance_matrix + 1e-9
        
        # Calculate base heuristic: inverse distance
        # We want to penalize high demand nodes to encourage packing small items
        # or visiting big items only when very close.
        # Formula: 1 / (d_ij * (1 + demand_j))
        
        # demands is a vector. We need to broadcast it to the columns (destination nodes)
        # demand_j corresponds to column j in the matrix
        # So we multiply each column j by (1 + demands[j])
        
        denom = d * (1.0 + demands)[:, np.newaxis].T # Wait, demands[j] is for column j
        # d is (N, N). demands is (N,).
        # We want to multiply column j by (1 + demands[j]).
        # So we can do: d * (1 + demands)[np.newaxis, :]
        
        denom = d * (1.0 + demands)[np.newaxis, :]
        
        eta = 1.0 / denom
        
        # Ensure positivity
        eta[eta <= 0] = 1e-9
        # Also handle any inf/nan just in case, though d>0 ensures finite
        eta[~np.isfinite(eta)] = 1e-9
        
        return eta
