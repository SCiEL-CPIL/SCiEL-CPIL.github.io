import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 16]

def _verify_decomp(U, V, W, n, m, p, num_tests=20):
    """Verify the decomposition is correct."""
    for seed in range(num_tests):
        np.random.seed(seed)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms
        C_test_flat = C_test.T.flatten()
        
        if not np.allclose(C_decomp, C_test_flat, rtol=1e-9, atol=1e-9):
            return False
    return True

def _try_rank_R(R, n, m, p, deadline, max_attempts=30):
    """Try to find a rank-R decomposition using least squares."""
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            return None
        np.random.seed(attempt * 7919 + R * 131)
        
        # Initialize
        U_init = np.random.randn(n*m, R) * 0.3
        V_init = np.random.randn(m*p, R) * 0.3
        W_init = np.random.randn(p*n, R) * 0.3
        
        # Build residual vector: for each (i,j,k), residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        def make_residuals(U, V, W):
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        res[idx] = val - 1.0
                        idx += 1
            return res
        
        # Use a smarter approach: fix W to be identity-like and solve for U,V
        # Or use alternating minimization
        
        # Alternating minimization approach:
        U_curr = U_init.copy()
        V_curr = V_init.copy()
        W_curr = W_init.copy()
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i,j,r), minimize sum_k (target - sum_{r'} V[j*p+k,r']*W[k*n+i,r'])^2 * U[i*m+j,r']^2
            # This is a linear system in U for fixed V,W
            for i in range(n):
                for j in range(m):
                    # For fixed V,W, we want: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                    # Let A_mat[k,r] = V[j*p+k,r] * W[k*n+i,r]
                    # Then A_mat @ U[i*m+j,:] = ones(p)
                    A_mat = np.zeros((p, R))
                    for k in range(p):
                        A_mat[k, :] = V_curr[j*p+k, :] * W_curr[k*n+i, :]
                    # Solve A_mat @ u = ones(p) in least squares
                    u_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                    U_curr[i*m+j, :] = u_sol
            
            # Update V: for each (j,k,r), similar
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, R))
                    for i in range(n):
                        A_mat[i, :] = U_curr[i*m+j, :] * W_curr[k*n+i, :]
                    v_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                    V_curr[j*p+k, :] = v_sol
            
            # Update W: for each (k,i,r), similar
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, R))
                    for j in range(m):
                        A_mat[j, :] = U_curr[i*m+j, :] * V_curr[j*p+k, :]
                    w_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                    W_curr[k*n+i, :] = w_sol
        
        # Check convergence
        res = make_residuals(U_curr, V_curr, W_curr)
        if np.max(np.abs(res)) < 1e-6:
            # Round to nearest integers/half-integers
            U_rounded = np.round(U_curr * 2) / 2
            V_rounded = np.round(V_curr * 2) / 2
            W_rounded = np.round(W_curr * 2) / 2
            
            if _verify_decomp(U_rounded, V_rounded, W_rounded, n, m, p):
                return (U_rounded, V_rounded, W_rounded)
            
            # Try integer rounding
            U_rounded = np.round(U_curr)
            V_rounded = np.round(V_curr)
            W_rounded = np.round(W_curr)
            
            if _verify_decomp(U_rounded, V_rounded, W_rounded, n, m, p):
                return (U_rounded, V_rounded, W_rounded)
    
    return None

def _strassen_decomposition(n, m, p):
    """Strassen-based rank-56 decomposition."""
    num_blocks = p // 2
    R = num_blocks * 7
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    W_contribs = [
        {0: 1, 3: 1},
        {2: 1},
        {1: 1, 3: -1},
        {0: -1, 2: 1},
        {1: 1, 3: 1},
        {0: 1, 3: -1},
        {0: 1}
    ]
    
    for q in range(num_blocks):
        b_00 = 0 * p + 2 * q
        b_01 = 0 * p + (2 * q + 1)
        b_10 = 1 * p + 2 * q
        b_11 = 1 * p + (2 * q + 1)
        
        c_00 = (2 * q) * n + 0
        c_01 = (2 * q + 1) * n + 0
        c_10 = (2 * q) * n + 1
        c_11 = (2 * q + 1) * n + 1
        
        b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}
        c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}
        
        for s in range(7):
            r = q * 7 + s
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
            for local_idx, coeff in V_coeffs[s].items():
                V[b_map[local_idx], r] = coeff
            for local_out_idx, coeff in W_contribs[s].items():
                W[c_map[local_out_idx], r] = coeff
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Schoolbook rank-64 decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Try lower ranks first
    for R in [48, 49, 50, 51, 52, 53, 54, 55]:
        if time.time() > deadline - 15:
            break
        result = _try_rank_R(R, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomp(U, V, W, n, m, p, num_tests=30):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to Strassen rank 56
    if time.time() < deadline - 5:
        U, V, W = _strassen_decomposition(n, m, p)
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
