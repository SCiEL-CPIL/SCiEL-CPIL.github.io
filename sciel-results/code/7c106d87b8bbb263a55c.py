import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook rank 54
    def get_rank_54():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Target: Rank 33 (3 * 11)
    # First, find a rank 11 decomposition for 2x3x3 multiplication.
    
    # Tensor for 2x3x3
    n_l, m_l, p_l = 2, 3, 3
    U_size = n_l * m_l  # 6
    V_size = m_l * p_l  # 9
    W_size = p_l * n_l  # 6
    
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n_l):
        for j in range(m_l):
            for k in range(p_l):
                u_idx = i * m_l + j
                v_idx = j * p_l + k
                w_idx = k * n_l + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Try to find rank 11 decomposition using ALS
    rank_l = 11
    np.random.seed(42)
    U_l = np.random.randn(U_size, rank_l)
    V_l = np.random.randn(V_size, rank_l)
    W_l = np.random.randn(W_size, rank_l)
    
    def als_step(U, V, W, T, max_iters=200, tol=1e-10, deadline=deadline):
        I, J, K = T.shape
        R = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Solve for U
            M_VW = np.zeros((J * K, R))
            for j in range(J):
                for k in range(K):
                    M_VW[j * K + k, :] = V[j, :] * W[k, :]
            
            U_new = np.zeros_like(U)
            for i in range(I):
                target = T[i, :, :].flatten()
                U_new[i, :] = np.linalg.lstsq(M_VW.T, target, rcond=None)[0]
            U = U_new
            
            # Solve for V
            M_UW = np.zeros((I * K, R))
            for i in range(I):
                for k in range(K):
                    M_UW[i * K + k, :] = U[i, :] * W[k, :]
            
            V_new = np.zeros_like(V)
            for j in range(J):
                target = T[:, j, :].flatten()
                V_new[j, :] = np.linalg.lstsq(M_UW.T, target, rcond=None)[0]
            V = V_new
            
            # Solve for W
            M_UV = np.zeros((I * J, R))
            for i in range(I):
                for j in range(J):
                    M_UV[i * J + j, :] = U[i, :] * V[j, :]
            
            W_new = np.zeros_like(W)
            for k in range(K):
                target = T[:, :, k].flatten()
                W_new[k, :] = np.linalg.lstsq(M_UV.T, target, rcond=None)[0]
            W = W_new
            
            # Check convergence
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(T - T_approx)
            if error < tol:
                break
                
        return U, V, W, error

    U_l, V_l, W_l, error = als_step(U_l, V_l, W_l, T, max_iters=200, tol=1e-8)
    
    # Round to nearest integer or half-integer
    def round_half(x):
        return np.round(x * 2) / 2.0

    U_l_r = round_half(U_l)
    V_l_r = round_half(V_l)
    W_l_r = round_half(W_l)
    
    # Verify
    T_approx = np.einsum('ir,jr,kr->ijk', U_l_r, V_l_r, W_l_r)
    if np.allclose(T_approx, T, atol=1e-6):
        # Success! Construct global rank 33 decomposition
        rank_g = 33
        U_g = np.zeros((n * m, rank_g))
        V_g = np.zeros((m * p, rank_g))
        W_g = np.zeros((p * n, rank_g))
        
        # Map local factors to global factors for each of the 3 blocks
        for b in range(3):
            # U is same for all blocks
            U_g[:, b * 11 : (b + 1) * 11] = U_l_r
            
            # V mapping:
            # Local V row: j * 3 + k_loc
            # Global V row: j * 9 + (3 * b + k_loc)
            for r in range(11):
                for j in range(3):
                    for k_loc in range(3):
                        v_idx_local = j * 3 + k_loc
                        v_idx_global = j * 9 + (3 * b + k_loc)
                        V_g[v_idx_global, b * 11 + r] = V_l_r[v_idx_local, r]
                        
            # W mapping:
            # Local W row: k_loc * 2 + i
            # Global W row: (3 * b + k_loc) * 2 + i
            for r in range(11):
                for k_loc in range(3):
                    for i in range(2):
                        w_idx_local = k_loc * 2 + i
                        w_idx_global = (3 * b + k_loc) * 2 + i
                        W_g[w_idx_global, b * 11 + r] = W_l_r[w_idx_local, r]
                        
        # Convert to lists of lists
        return {
            "U": U_g.tolist(),
            "V": V_g.tolist(),
            "W": W_g.tolist()
        }
    else:
        # Fallback
        return get_rank_54()

# EVOLVE-BLOCK-START
# The construct function above is the implementation.
# To satisfy the "EVOLVE-BLOCK" structure if required by the harness, 
# I will wrap the logic or ensure the function name is correct.
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has `construct`.
# I will just define `construct` in the EVOLVE-BLOCK.
# However, the code above defines `construct` outside. 
# I will move the entire logic into the EVOLVE-BLOCK.
# EVOLVE-BLOCK-END
