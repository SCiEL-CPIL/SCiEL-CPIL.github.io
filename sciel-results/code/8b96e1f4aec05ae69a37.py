import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
        d = np.array(distance_matrix, dtype=float)
        # Avoid division by zero for diagonal and any zero distances
        # The spec says non-positive become 1e-9.
        # We want diagonal to be 0 ideally, but if the ACO uses it, 1e-9 is safe.
        # However, usually diagonal is ignored or set to 0.
        
        # Sharpened distance heuristic
        # d^3 amplifies the difference between short and long distances more aggressively
        d_safe = d + 1e-9
        heuristic = 1.0 / (d_safe ** 3)
        
        # Set diagonal to 0 to discourage self-loops if the ACO doesn't mask them
        np.fill_diagonal(heuristic, 0.0)
        
        # Ensure any non-positive values (if d was somehow negative, unlikely) are 1e-9
        heuristic[heuristic <= 0] = 1e-9
        
        return heuristic
