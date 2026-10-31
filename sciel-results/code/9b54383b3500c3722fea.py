import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Outer Strassen coefficients (2x2 block level)
    # beta[s][I][J], gamma[s][J][K], delta[s][I][K]
    beta_outer = np.zeros((7, 2, 2), dtype=int)
    gamma_outer = np.zeros((7, 2, 2), dtype=int)
    delta_outer = np.zeros((7, 2, 2), dtype=int)
    
    # s=0
    beta_outer[0, 0, 0] = 1; beta_outer[0, 1, 1] = 1
    gamma_outer[0, 0, 0] = 1; gamma_outer[0, 1, 1] = 1
    delta_outer[0, 0, 0] = 1; delta_outer[0, 1, 1] = 1
    
    # s=1
    beta_outer[1, 1, 0] = 1; beta_outer[1, 1, 1] = 1
    gamma_outer[1, 0, 0] = 1
    delta_outer[1, 1, 0] = 1; delta_outer[1, 1, 1] = -1
    
    # s=2
    beta_outer[2, 0, 0] = 1
    gamma_outer[2, 0, 1] = 1; gamma_outer[2, 1, 1] = -1
    delta_outer[2, 0, 1] = 1; delta_outer[2, 1, 1] = 1
    
    # s=3
    beta_outer[3, 1, 1] = 1
    gamma_outer[3, 1, 0] = 1; gamma_outer[3, 0, 0] = -1
    delta_outer[3, 0, 0] = 1; delta_outer[3, 1, 0] = 1
    
    # s=4
    beta_outer[4, 0, 0] = 1; beta_outer[4, 0, 1] = 1
    gamma_outer[4, 1, 1] = 1
    delta_outer[4, 0, 0] = -1; delta_outer[4, 0, 1] = 1
    
    # s=5
    beta_outer[5, 1, 0] = 1; beta_outer[5, 0, 0] = -1
    gamma_outer[5, 0, 0] = 1; gamma_outer[5, 0, 1] = 1
    delta_outer[5, 1, 1] = 1
    
    # s=6
    beta_outer[6, 0, 1] = 1; beta_outer[6, 1, 1] = -1
    gamma_outer[6, 1, 0] = 1; gamma_outer[6, 1, 1] = 1
    delta_outer[6, 0, 0] = 1
    
    # Inner Strassen coefficients (2x2 scalar level)
    # U2[i][t], V2[j][t], W2[k][t]
    # Indices 0,1,2,3 correspond to 00, 01, 10, 11
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # t=0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # t=2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # t=3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # t=4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # t=5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # t=6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct U, V, W using vectorized operations
    # Global indices: i, j, k in 0..3
    # I = i//2, i' = i%2, etc.
    
    # Precompute block indices and local indices
    i_indices = np.arange(4)
    j_indices = np.arange(4)
    k_indices = np.arange(4)
    
    # Meshgrid for U
    I_grid, i_prime_grid = np.meshgrid(i_indices // 2, i_indices % 2, indexing='ij')
    J_grid, j_prime_grid = np.meshgrid(j_indices // 2, j_indices % 2, indexing='ij')
    
    # We need to map (i, j) to row_u = i*4 + j
    # And fill U[row_u, 7*s + t]
    
    # Reshape grids to (16, 2, 2) to match beta_outer[7, 2, 2]
    # I_grid is (4,4), we want (4,4)
    # Let's flatten U to (16, 49)
    
    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            
            # U
            # beta_outer[s] is (2,2)
            # I_grid is (4,4), i_prime_grid is (4,4)
            # We need to sum over I, i' ? No, each cell (i,j) has specific I, i'.
            # U[i, j, c] = beta_outer[s, I, J] * U2[local_u_idx, t]
            # local_u_idx = 2*i_prime + j_prime
            
            # Vectorized calculation for U[:, c]
            # I_grid shape (4,4), J_grid shape (4,4)
            # beta_outer[s] shape (2,2)
            # We need to index beta_outer[s] with I_grid and J_grid
            # numpy advanced indexing
            b_coeffs = beta_outer[s, I_grid.flatten(), J_grid.flatten()]
            # local_u_idx
            local_u = 2 * i_prime_grid.flatten() + j_prime_grid.flatten()
            u2_coeffs = U2[local_u, t]
            
            U[:, c] = b_coeffs * u2_coeffs
            
            # V
            # V[j, k, c] = gamma_outer[s, J, K] * V2[local_v_idx, t]
            # local_v_idx = 2*j_prime + k_prime
            # J_grid is for j, K_grid for k
            # j_indices, k_indices
            J_v, j_prime_v = np.meshgrid(j_indices // 2, j_indices % 2, indexing='ij')
            K_v, k_prime_v = np.meshgrid(k_indices // 2, k_indices % 2, indexing='ij')
            
            g_coeffs = gamma_outer[s, J_v.flatten(), K_v.flatten()]
            local_v = 2 * j_prime_v.flatten() + k_prime_v.flatten()
            v2_coeffs = V2[local_v, t]
            
            V[:, c] = g_coeffs * v2_coeffs
            
            # W
            # W[k, i, c] = delta_outer[s, I, K] * W2[local_w_idx, t]
            # Trace convention: W[k*n+i, r] -> row = k*4 + i
            # local_w_idx = 2*k_prime + i_prime
            K_w, k_prime_w = np.meshgrid(k_indices // 2, k_indices % 2, indexing='ij')
            I_w, i_prime_w = np.meshgrid(i_indices // 2, i_indices % 2, indexing='ij')
            
            # Note: W row is k*4 + i.
            # Meshgrid K_w (4,4), I_w (4,4)
            # We need to align them with the row index.
            # Row = k*4 + i.
            # The meshgrid K_w, I_w has shape (4,4) where axis 0 is k, axis 1 is i?
            # np.meshgrid(k_indices, i_indices, indexing='ij') gives K_w (k varies by row), I_w (i varies by col).
            # So K_w.flatten() and I_w.flatten() correspond to row order?
            # Row 0: k=0, i=0 -> K_w[0,0], I_w[0,0]
            # Row 1: k=0, i=1 -> K_w[0,1], I_w[0,1]
            # ...
            # Row 4: k=1, i=0 -> K_w[1,0], I_w[1,0]
            # This matches row = k*4 + i.
            
            d_coeffs = delta_outer[s, I_w.flatten(), K_w.flatten()]
            local_w = 2 * k_prime_w.flatten() + i_prime_w.flatten()
            w2_coeffs = W2[local_w, t]
            
            W[:, c] = d_coeffs * w2_coeffs
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
                
    return {"U": U, "V": V, "W": W}
