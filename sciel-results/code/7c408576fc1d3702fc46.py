"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Strassen coefficients for M_{2,2,2}
    # U: 4x7, V: 4x7, W: 4x7
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    # Schoolbook coefficients for M_{2,2,1}
    U_school = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    
    V_school = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    
    W_school = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)

    def build_solution(num_strassen, use_m223=False, m223_coeffs=None):
        """Build the full U, V, W matrices for the given decomposition strategy."""
        if use_m223 and m223_coeffs is not None:
            # 6 blocks of M_{2,2,2} (12 cols) + 1 block of M_{2,2,3} (3 cols)
            rank = 6 * 7 + 10
        else:
            # 7 blocks of M_{2,2,2} (14 cols) + 1 block of M_{2,2,1} (1 col)
            rank = 7 * 7 + 4
        
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col_offset = 0
        
        # Fill Strassen blocks
        for block in range(num_strassen):
            k0 = 2 * block
            k1 = 2 * block + 1
            
            for r in range(7):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, c] = U_strassen[i * 2 + j, r]
                for j in range(2):
                    V[j * p + k0, c] = V_strassen[j * 2 + 0, r]
                    V[j * p + k1, c] = V_strassen[j * 2 + 1, r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W[kk * n + i, c] = W_strassen[k * 2 + i, r]
            col_offset += 7
        
        if use_m223 and m223_coeffs is not None:
            # Fill M_{2,2,3} block at k=12,13,14
            k_start = 2 * num_strassen
            U_223, V_223, W_223 = m223_coeffs
            for r in range(10):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, c] = U_223[i * 2 + j, r]
                for j in range(2):
                    for k in range(3):
                        V[j * p + k_start + k, c] = V_223[j * 3 + k, r]
                for k in range(3):
                    for i in range(2):
                        W[(k_start + k) * n + i, c] = W_223[k * 2 + i, r]
        else:
            # Fill M_{2,2,1} block at k=14
            k_last = 14
            for r in range(4):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, c] = U_school[i * 2 + j, r]
                for j in range(2):
                    V[j * p + k_last, c] = V_school[j, r]
                for i in range(2):
                    W[k_last * n + i, c] = W_school[i, r]
        
        # Convert to lists of ints/half-ints
        U_list = [[round(U[i, r], 1) for r in range(rank)] for i in range(n * m)]
        V_list = [[round(V[i, r], 1) for r in range(rank)] for i in range(m * p)]
        W_list = [[round(W[i, r], 1) for r in range(rank)] for i in range(p * n)]
        
        return U_list, V_list, W_list

    def verify_solution(U_list, V_list, W_list):
        """Verify that the tensor decomposition is exact."""
        U = np.array(U_list, dtype=float)
        V = np.array(V_list, dtype=float)
        W = np.array(W_list, dtype=float)
        rank = U.shape[1]
        
        # Check tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def try_find_m223_rank10():
        """Try to find a rank-10 decomposition of M_{2,2,3}."""
        # Variables: U (4x10), V (2x3x10), W (3x2x10) -> 40 + 60 + 60 = 160 variables
        # Target: for all i in {0,1}, j in {0,1}, k in {0,1,2}:
        #   sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        
        from scipy.optimize import minimize
        
        def objective(params):
            U_flat = params[0:40]
            V_flat = params[40:100]
            W_flat = params[100:160]
            
            U = U_flat.reshape(2, 2, 10)  # i, j, r
            V = V_flat.reshape(2, 3, 10)  # j, k, r
            W = W_flat.reshape(3, 2, 10)  # k, i, r
            
            err = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                        err += (val - 1.0) ** 2
            return err
        
        # Try a few random initializations
        np.random.seed(42)
        best_err = float('inf')
        best_params = None
        
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            # Initialize with small random values
            params = np.random.randn(160) * 0.5
            result = minimize(objective, params, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_err:
                best_err = result.fun
                best_params = result.x
        
        if best_err < 1e-6 and best_params is not None:
            U = best_params[0:40].reshape(2, 2, 10)
            V = best_params[40:100].reshape(2, 3, 10)
            W = best_params[100:160].reshape(3, 2, 10)
            
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check if rounded solution is valid
            err = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = np.sum(U_rounded[i, j, :] * V_rounded[j, k, :] * W_rounded[k, i, :])
                        err += (val - 1.0) ** 2
            
            if err < 1e-6:
                return U_rounded, V_rounded, W_rounded
        
        return None

    # First, try to find a rank-10 M_{2,2,3} decomposition
    m223_coeffs = try_find_m223_rank10()
    
    if m223_coeffs is not None:
        U_list, V_list, W_list = build_solution(6, use_m223=True, m223_coeffs=m223_coeffs)
        if verify_solution(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: rank 53 solution
    if time.time() > deadline - 5:
        U_list, V_list, W_list = build_solution(7, use_m223=False)
        return {"U": U_list, "V": V_list, "W": W_list}
    
    U_list, V_list, W_list = build_solution(7, use_m223=False)
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

