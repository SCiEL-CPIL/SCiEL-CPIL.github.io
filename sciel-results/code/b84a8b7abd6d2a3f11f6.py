import numpy as np

def heuristics(demands, capacity):
       demands = np.asarray(demands, dtype=float)
       n = len(demands)
       
       # Calculate pair sums
       sums = demands[:, None] + demands[None, :]
       
       # Identify valid pairs (sum <= capacity)
       valid = sums <= capacity
       
       # Calculate gap for valid pairs
       gaps = capacity - sums
       
       # Initialize heuristic with a very small value
       eta = np.full((n, n), 1e-9, dtype=float)
       
       # Calculate score for valid pairs
       # Combine gap penalty with size bias
       # Gap penalty: exp(-alpha * gap) - favors tight fits
       # Size bias: (1 + beta * d_j / capacity) - favors larger items when gaps are similar
       
       # Use alpha=0.05 (less aggressive than 0.1) to not over-penalize moderate gaps
       # Use beta=0.5 for moderate size bias
       alpha = 0.05
       beta = 0.5
       
       # For valid pairs, compute combined score
       gap_scores = np.exp(-alpha * gaps[valid])
       size_bias = 1.0 + beta * demands[None, :][valid] / capacity
       
       scores = gap_scores * size_bias
       
       # Assign
       eta[valid] = scores
       
       # Diagonal: set to 1e-9
       np.fill_diagonal(eta, 1e-9)
       
       return eta
