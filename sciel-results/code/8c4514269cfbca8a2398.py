import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    # distance_matrix is n x n
    # coordinates is n x 2
    # demands is n,
    
    # Initialize heuristic with inverse distance
    # Add small epsilon to avoid division by zero
    d = distance_matrix + 1e-9
    h = 1.0 / d
    
    # Compute savings
    # s_ij = d_i0 + d_0j - d_ij
    d_i0 = distance_matrix[:, 0]
    d_0j = distance_matrix[0, :]
    
    # s is n x n
    s = d_i0[:, np.newaxis] + d_0j[np.newaxis, :] - distance_matrix
    
    # The savings term is more meaningful when normalized or used as a boost.
    # We want to boost edges that have high savings (good clusters).
    # Normalize savings by the potential max savings (which is approx 2 * min(d_i0, d_0j))
    # Or simply use the savings directly if it's positive.
    # Since h is 1/d, and s is roughly proportional to d_i0, 
    # multiplying h by (1 + s) might be too strong or weak.
    
    # Let's use a normalized savings factor.
    # max_s for a pair is approx 2 * min(d_i0, d_0j).
    # Let's just use the raw savings scaled to be comparable to 1.
    # Typical distances are in [0, 1.4] (unit square).
    # Savings are in [0, ~2].
    
    # Strategy: h_ij = (1/d_ij) * (1 + alpha * s_ij)
    # If s_ij is large, the edge is very attractive.
    # If s_ij is 0 (far apart), it falls back to 1/d_ij.
    
    # Let's pick alpha = 1.0
    # s_ij can be negative? Triangle inequality says d_ij >= |d_i0 - d_0j|, so s_ij >= 2*min(d_i0, d_0j) - d_ij? No.
    # s_ij = d_i0 + d_0j - d_ij.
    # Since d_ij >= d_i0 - d_0j and d_ij >= d_0j - d_i0,
    # d_ij >= |d_i0 - d_0j|.
    # So s_ij <= d_i0 + d_0j - |d_i0 - d_0j| = 2 * min(d_i0, d_0j).
    # And s_ij >= d_i0 + d_0j - (d_i0 + d_0j) = 0? No, d_ij can be larger than d_i0 + d_0j? No, triangle inequality d_ij <= d_i0 + d_0j.
    # So s_ij >= 0.
    
    # So s is non-negative.
    # We can use h = (1/d) * (1 + s)
    
    # However, for the depot (0), s_0j = d_00 + d_0j - d_0j = 0.
    # So edges from depot get h = 1/d.
    # Edges between customers get boosted by savings.
    
    # This seems reasonable.
    
    h = h * (1.0 + s)
    
    # Ensure non-finite and positive
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    return h
