import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank-9 decomposition for 2x3x3
    n_small, m_small, p_small = 2, 3, 3
    R_small = 9
    
    U_small = np.zeros((n_small * m_small, R_small))
    V_small = np.zeros((m_small * p_small, R_small))
    W_small = np.zeros((p_small * n_small, R_small))
    
    for j in range(m_small):
        for k in range(p_small):
            r = j * p_small + k
            U_small[0 * m_small + j, r] = 1.0
            U_small[1 * m_small + j, r] = 1.0
            V_small[j * p_small + k, r] = 1.0
            W_small[k * n_small + 0, r] = 1.0
            W_small[k * n_small + 1, r] = 1.0
    
    # Verify
    U_3d = U_small.reshape(n_small, m_small, R_small)
    V_3d = V_small.reshape(m_small, p_small, R_small)
    W_3d = W_small.reshape(p_small, n_small, R_small)
    recon = np.einsum('ijr,jkr,ikr->ijk', U_3d, V_3d, W_3d)
    T_target = np.ones((n_small, m_small, p_small))
    
    if not np.allclose(recon, T_target, atol=1e-10):
        R_small = n_small * m_small * p_small
        U_small = np.zeros((n_small * m_small, R_small))
        V_small = np.zeros((m_small * p_small, R_small))
        W_small = np.zeros((p_small * n_small, R_small))
        col = 0
        for i in range(n_small):
            for j in range(m_small):
                for k in range(p_small):
                    U_small[i * m_small + j, col] = 1.0
                    V_small[j * p_small + k, col] = 1.0
                    W_small[k * n_small + i, col] = 1.0
                    col += 1
    
    # Expand to 2x3x9
    R_total = 3 * R_small
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    for t in range(3):
        r_offset = t * R_small
        for i in range(n_small):
            for j in range(m_small):
                idx_u = i * m + j
                U[idx_u, r_offset:r_offset + R_small] = U_small[i * m_small + j, :]
        for j in range(m_small):
            for k_prime in range(p_small):
                k = t * p_small + k_prime
                idx_v = j * p + k
                V[idx_v, r_offset:r_offset + R_small] = V_small[j * p_small + k_prime, :]
        for k_prime in range(p_small):
            for i in range(n_small):
                k = t * p_small + k_prime
                idx_w = k * n + i
                W[idx_w, r_offset:r_offset + R_small] = W_small[k_prime * n_small + i, :]
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
