"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build the target tensor T of shape (dim_A, dim_B, dim_C)
    T = np.zeros((dim_A, dim_B, dim_C))
    for i_n in range(n):
        for i_m in range(m):
            for k_p in range(p):
                i = i_n * m + i_m
                j = i_m * p + k_p
                k = i_n * p + k_p
                T[i, j, k] = 1.0
                
    def verify_decomposition(U, V, W, rank):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        return err

    def _get_rank14_decomposition():
        """Return the known rank-14 Strassen decomposition."""
        rank = 14
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen for B1 (cols 0,1)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1

        # Strassen for B2 (cols 2,3)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1

        return {"U": U, "V": V, "W": W}

    def try_rank13_search(time_limit=60):
        """
        Try to find a rank-13 decomposition using diverse random restarts of ALS.
        """
        start_time = time.time()
        rank = 13
        rng = np.random.RandomState(123)
        
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            U = rng.randn(dim_A, rank)
            V = rng.randn(dim_B, rank)
            W = rng.randn(dim_C, rank)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            prev_err = verify_decomposition(U, V, W, rank)
            
            for step in range(100):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U
                M = np.einsum('jr,kr->jkr', V, W).reshape(dim_B * dim_C, rank)
                T_flat = T.reshape(dim_A, dim_B * dim_C)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                    
                # Update V
                N = np.einsum('ir,kr->ikr', U, W).reshape(dim_A * dim_C, rank)
                T_flat_V = T.transpose(1, 0, 2).reshape(dim_B, dim_A * dim_C)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(N, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                    
                # Update W
                M = np.einsum('ir,jr->ijr', U, V).reshape(dim_A * dim_B, rank)
                T_flat_W = T.transpose(2, 0, 1).reshape(dim_C, dim_A * dim_B)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                    
                err = verify_decomposition(U, V, W, rank)
                if err < 1e-8:
                    # Check if entries are close to integers/half-integers
                    is_valid = True
                    for mat in [U, V, W]:
                        for i in range(mat.shape[0]):
                            for j in range(mat.shape[1]):
                                val = mat[i, j]
                                k = round(val * 2)
                                if abs(val - k * 0.5) > 1e-4:
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if not is_valid:
                            break
                    
                    if is_valid:
                        U_list = [[round(val * 2) * 0.5 for val in row] for row in U.tolist()]
                        V_list = [[round(val * 2) * 0.5 for val in row] for row in V.tolist()]
                        W_list = [[round(val * 2) * 0.5 for val in row] for row in W.tolist()]
                        
                        U_np = np.array(U_list)
                        V_np = np.array(V_list)
                        W_np = np.array(W_list)
                        
                        if verify_decomposition(U_np, V_np, W_np, rank) < 1e-6:
                            return {"U": U_list, "V": V_list, "W": W_list}
                
                if err > prev_err * 1.01:
                    break
                prev_err = err
                
        return None

    overall_start = time.time()
    overall_deadline = 140

    # Try rank 13
    if time.time() - overall_start < overall_deadline - 20:
        res13 = try_rank13_search(time_limit=min(60, overall_deadline - (time.time() - overall_start) - 10))
        if res13 is not None:
            return res13

    # Fallback to rank 14
    return _get_rank14_decomposition()
# EVOLVE-BLOCK-END
