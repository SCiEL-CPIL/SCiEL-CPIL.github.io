"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Uses vectorized NumPy operations for speed.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all tensor entries using vectorized computation
    for a_idx in range(n * m):
        i, j = divmod(a_idx, m)
        
        # Compute result[b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # M[b, r] = V[b, r] * U[a, r]
        M = V_arr * U_arr[a_idx, :][None, :]
        # result[b, c] = sum_r M[b, r] * W[c, r] = (M @ W.T)[b, c]
        result = M @ W_arr.T  # shape (m*p, p*n)
        
        # Build expected tensor slice
        expected = np.zeros((m * p, p * n))
        for b_idx in range(m * p):
            j2, k = divmod(b_idx, p)
            if j2 == j:
                for c_idx in range(p * n):
                    k2, i2 = divmod(c_idx, n)
                    if k2 == k and i2 == i:
                        expected[b_idx, c_idx] = 1.0
        
        if np.max(np.abs(result - expected)) > 1e-10:
            return False
    return True

def _build_strassen_35():
    """Constructs the baseline rank-35 decomposition."""
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    strassen_B = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    w_contrib[0][0] = 1; w_contrib[0][3] = 1
    w_contrib[1][2] = 1; w_contrib[1][3] = -1
    w_contrib[2][1] = 1; w_contrib[2][3] = 1
    w_contrib[3][0] = 1; w_contrib[3][2] = 1
    w_contrib[4][0] = -1; w_contrib[4][1] = 1
    w_contrib[5][3] = 1
    w_contrib[6][0] = 1

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            col = b * 7 + r
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
            
    return U, V, W

def _search_lower_rank(target_rank, start_time, deadline):
    """
    Attempts to find a decomposition of the given target_rank using numerical optimization.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = DIMENSIONS
    # Pre-compute target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n*m):
        i_A = a // m
        j_A = a % m
        for b in range(m*p):
            j_B = b // p
            k_B = b % p
            if j_B != j_A: continue
            for c in range(p*n):
                k_C = c // n
                i_C = c % n
                if k_C == k_B and i_C == i_A:
                    T[a, b, c] = 1.0

    def objective(x):
        U_flat = x[:n*m*target_rank]
        V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = x[n*m*target_rank + m*p*target_rank : ]
        
        U_mat = U_flat.reshape(n*m, target_rank)
        V_mat = V_flat.reshape(m*p, target_rank)
        W_mat = W_flat.reshape(p*n, target_rank)
        
        # Reconstruct tensor
        R = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        
        return np.sum((R - T)**2)

    best_error = float('inf')
    best_x = None
    
    # Try a few random restarts
    for i in range(10):
        if time.time() > deadline:
            break
            
        x0 = np.random.randn(n*m*target_rank + m*p*target_rank + p*n*target_rank)
        
        # Optimization
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.fun < best_error:
            best_error = res.fun
            best_x = res.x
            
            # Check if we found a solution
            if best_error < 1e-6:
                # Round to nearest half-integer
                U_flat = best_x[:n*m*target_rank]
                V_flat = best_x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                W_flat = best_x[n*m*target_rank + m*p*target_rank : ]
                
                U_mat = (np.round(U_flat * 2) / 2).reshape(n*m, target_rank)
                V_mat = (np.round(V_flat * 2) / 2).reshape(m*p, target_rank)
                W_mat = (np.round(W_flat * 2) / 2).reshape(p*n, target_rank)
                
                # Verify exactness
                U_list = [[float(val) for val in row] for row in U_mat]
                V_list = [[float(val) for val in row] for row in V_mat]
                W_list = [[float(val) for val in row] for row in W_mat]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list

    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find rank 32
    if time.time() - start_time < DEADLINE - 10:
        result = _search_lower_rank(32, start_time, DEADLINE - 5)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to rank 35
    U, V, W = _build_strassen_35()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
