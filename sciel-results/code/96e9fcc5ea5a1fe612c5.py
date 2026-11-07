import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
            n = distance_matrix.shape[0]
            # distance_matrix is n x n
            # coordinates is n x 2
            # demands is n
            # capacity is float
            
            # 1. Compute Savings
            # S_ij = d_i0 + d_0j - d_ij
            d_to_depot = distance_matrix[:, 0] # shape (n,)
            # S_ij = d_i0 + d_0j - d_ij
            # d_i0 is column 0? No, distance_matrix[i, 0] is d_i0.
            # d_0j is distance_matrix[0, j].
            
            # Vectorized computation of Savings Matrix S
            # S[i, j] = d[i, 0] + d[0, j] - d[i, j]
            d_i0 = distance_matrix[:, 0].reshape(-1, 1) # (n, 1)
            d_0j = distance_matrix[0, :].reshape(1, -1) # (1, n)
            S = d_i0 + d_0j - distance_matrix
            
            # Note: S[0, j] = d[0,0] + d[0,j] - d[0,j] = 0.
            # S[i, 0] = d[i,0] + d[0,0] - d[i,0] = 0.
            # S[i, i] = 2*d[i,0] - 0 = 2*d[i,0]. (But i->i is not a valid move).
            
            # We only care about j != i and j is a customer (j > 0).
            # The heuristic matrix is returned for all i, j.
            # The ACO loop will only use it for valid moves.
            
            S_max = np.max(S)
            if S_max == 0:
                S_norm = np.zeros_like(S)
            else:
                S_norm = S / S_max
            
            # Clip negative savings? Savings can be negative if d_ij > d_i0 + d_0j (triangle inequality violation? No, triangle inequality says d_ij <= d_i0 + d_0j, so S >= 0).
            # So S is non-negative.
            
            # 2. Construct Heuristic
            # eta_ij = (1 / d_ij^2) * (1 + lambda * S_norm_ij)
            
            alpha = 2.0
            lambda_s = 1.0
            
            # Avoid division by zero for i==j (though not used, good practice)
            d_safe = distance_matrix + 1e-9
            
            dist_term = 1.0 / (d_safe ** alpha)
            
            # For i=0, S_norm is 0, so multiplier is 1.
            # For i!=0, multiplier is 1 + lambda * S_norm.
            
            multiplier = 1.0 + lambda_s * S_norm
            
            eta = dist_term * multiplier
            
            # Ensure positive
            eta[eta <= 0] = 1e-9
            
            return eta
