import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    eps = 1e-9
    
    # Local prize density: for each node j, sum of prizes of nearby nodes
    # Use a Gaussian kernel with sigma scaled to the problem
    sigma = 0.3  # in unit square, typical neighbor distance
    # local_density[j] = sum_k prize[k] * exp(-d[j,k]^2 / (2*sigma^2))
    # Exclude self (k=j) since prize[j] is already in the numerator
    d_sq = distance_matrix ** 2
    kernel = np.exp(-d_sq / (2 * sigma ** 2))
    # Zero out diagonal to exclude self
    np.fill_diagonal(kernel, 0.0)
    local_density = kernel @ prizes  # shape (n,)
    
    # Normalize local_density to [0, 1] range for stable weighting
    max_ld = local_density.max() if local_density.max() > 0 else 1.0
    local_density_norm = local_density / max_ld
    
    # Prize amplification: square the prize to emphasize high-value nodes
    prize_amp = prizes ** 2
    
    # Combined node attractiveness: prize * (1 + alpha * local_density_norm)
    alpha = 1.0
    node_attr = prize_amp * (1.0 + alpha * local_density_norm)
    
    # Distance penalty: step cost + return cost
    # d[i,j] is the step, d[j,0] is the return
    d_to_origin = distance_matrix[:, 0]  # shape (n,), d[j,0]
    
    # eta[i,j] = node_attr[j] / (d[i,j] + beta * d[j,0] + eps)
    beta = 1.0
    denom = distance_matrix + beta * d_to_origin[None, :] + eps
    
    eta = node_attr[None, :] / denom
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
