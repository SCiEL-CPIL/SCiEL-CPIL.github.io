import numpy as np

def heuristics(prizes, weights):
       # Calculate ratio for each constraint
       # Add small epsilon to avoid division by zero
       eps = 1e-9
       ratios = prizes / (weights + eps)
       
       # Harmonic mean of the ratios
       # H = n / sum(1/x_i)
       inv_ratios = 1.0 / ratios
       sum_inv = np.sum(inv_ratios, axis=0)
       eta = 5.0 / sum_inv
       
       # Ensure positive finite
       eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
       
       return eta
