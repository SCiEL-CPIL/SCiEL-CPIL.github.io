"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def _verify_decomposition(U, V, W, n=2, m=2, p=10):
    """Verify that U @ diag(1) @ V^T reconstructs the tensor correctly via W."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The decomposition says: T[i_A, j_B, k_C] = sum_r U[i_A, r] * V[j_B, r] * W[k_C, r]
    # where i_A = i*m+j (A index), j_B = j*p+k (B index), k_C = k*n+i (C index, trace convention)
    
    # Build the tensor T from U, V, W
    # T has shape (n*m, m*p, p*n) = (4, 20, 20)
    # T[i_A, j_B, k_C] = sum_r U[i_A, r] * V[j_B, r] * W[k_C, r]
    
    # Efficient computation: T = U @ (V * W^T) reshaped... 
    # Actually, T[i_A, j_B, k_C] = sum_r U[i_A, r] * V[j_B, r] * W[k_C, r]
    # This is the "mode-3" product. We can compute it as:
    # For each r, outer product U[:,r] @ V[:,r]^T @ W[:,r]^T
    # T = sum_r np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # Now build the expected tensor
    # C[i_C_row, k_C_col] = sum_j A[i_C_row, j] * B[j, k_C_col]
    # In tensor notation: T[i_A, j_B, k_C] should equal delta if the bilinear form matches
    # i_A = i*m + j_A (row i, col j_A of A)
    # j_B = j_B_row * p + k_B (row j_B_row, col k_B of B)  
    # k_C = k_C_col * n + i_C_row (col k_C_col, row i_C_row of C) -- trace convention
    
    # The multiplication: C[i, k] = sum_j A[i, j] * B[j, k]
    # So T[i_A, j_B, k_C] = 1 if (i_A, j_B, k_C) corresponds to a valid multiplication term
    # i_A = i * m + j_A  => i = i_A // m, j_A = i_A % m
    # j_B = j_B_row * p + k_B => j_B_row = j_B // p, k_B = j_B % p
    # k_C = k_C_col * n + i_C => k_C_col = k_C // n, i_C = k_C % n
    
    # T[i_A, j_B, k_C] = 1 if j_A == j_B_row and k_B == k_C_col and i == i_C
    # Otherwise 0
    
    expected = np.zeros((n*m, m*p, p*n))
    for i_A in range(n*m):
        i = i_A // m
        j_A = i_A % m
        for j_B in range(m*p):
            j_B_row = j_B // p
            k_B = j_B % p
            for k_C in range(p*n):
                k_C_col = k_C // n
                i_C = k_C % n
                if j_A == j_B_row and k_B == k_C_col and i == i_C:
                    expected[i_A, j_B, k_C] = 1.0
    
    diff = np.max(np.abs(T - expected))
    return diff

def _build_strassen_rank35():
    """Build the rank-35 Strassen block-diagonal decomposition."""
    n, m, p = DIMENSIONS
    rank = 35
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_A = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1 (e+h)
        [1, 0, 0, 0],  # P2 (e)
        [0, 1, 0, -1], # P3 (f-h)
        [-1, 0, 1, 0], # P4 (g-e)
        [0, 0, 0, 1],  # P5 (h)
        [1, 1, 0, 0],  # P6 (e+f)
        [0, 0, 1, 1]   # P7 (g+h)
    ]
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    w_contrib[0][0] = 1;  w_contrib[0][3] = 1   # P1: C00, C11
    w_contrib[1][2] = 1;  w_contrib[1][3] = -1  # P2: C10, C11
    w_contrib[2][1] = 1;  w_contrib[2][3] = 1   # P3: C01, C11
    w_contrib[3][0] = 1;  w_contrib[3][2] = 1   # P4: C00, C10
    w_contrib[4][0] = -1; w_contrib[4][1] = 1   # P5: C00, C01
    w_contrib[5][3] = 1                              # P6: C11
    w_contrib[6][0] = 1                              # P7: C00
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            U[0, col] = strassen_A[r][0]
            U[1, col] = strassen_A[r][1]
            U[2, col] = strassen_A[r][2]
            U[3, col] = strassen_A[r][3]
            
            k0 = 2 * b
            k1 = 2 * b + 1
            V[0 * p + k0, col] = strassen_B[r][0]
            V[0 * p + k1, col] = strassen_B[r][1]
            V[1 * p + k0, col] = strassen_B[r][2]
            V[1 * p + k1, col] = strassen_B[r][3]
            
            W[(2 * b) * 2 + 0, col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0, col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1, col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1, col] = w_contrib[r][3]
    
    return U, V, W

def _build_expected_tensor(n=2, m=2, p=10):
    """Build the expected multiplication tensor."""
    expected = np.zeros((n*m, m*p, p*n))
    for i_A in range(n*m):
        i = i_A // m
        j_A = i_A % m
        for j_B in range(m*p):
            j_B_row = j_B // p
            k_B = j_B % p
            for k_C in range(p*n):
                k_C_col = k_C // n
                i_C = k_C % n
                if j_A == j_B_row and k_B == k_C_col and i == i_C:
                    expected[i_A, j_B, k_C] = 1.0
    return expected

def _try_lower_rank(target_rank, expected, n=2, m=2, p=10, max_iters=300, seed=42, time_limit=30):
    """Try to find a rank-target_rank decomposition using L-BFGS optimization."""
    start_time = time.time()
    
    U35, V35, W35 = _build_strassen_rank35()
    rng = np.random.RandomState(seed)
    
    best_result = None
    best_err = float('inf')
    
    for attempt in range(2):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize: take target_rank columns from rank-35, add small noise
        idx = rng.choice(35, size=target_rank, replace=False)
        U0 = U35[:, idx].copy() + rng.randn(n*m, target_rank) * 0.05
        V0 = V35[:, idx].copy() + rng.randn(m*p, target_rank) * 0.05
        W0 = W35[:, idx].copy() + rng.randn(p*n, target_rank) * 0.05
        
        # Flatten parameters
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        dim_U = n*m * target_rank
        dim_V = m*p * target_rank
        
        def objective(x):
            U = x[:dim_U].reshape(n*m, target_rank)
            V = x[dim_U:dim_U+dim_V].reshape(m*p, target_rank)
            W = x[dim_U+dim_V:].reshape(p*n, target_rank)
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = T - expected
            return np.sum(residual**2)
        
        def gradient(x):
            U = x[:dim_U].reshape(n*m, target_rank)
            V = x[dim_U:dim_U+dim_V].reshape(m*p, target_rank)
            W = x[dim_U+dim_V:].reshape(p*n, target_rank)
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = T - expected
            
            # grad_U[i,r] = sum_{j,k} V[j,r] * W[k,r] * residual[i,j,k]
            grad_U = np.einsum('jr,kr,ijk->ir', V, W, residual)
            grad_V = np.einsum('ir,kr,ijk->jr', U, W, residual)
            grad_W = np.einsum('ir,jr,ijk->kr', U, V, residual)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        from scipy.optimize import minimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                          options={'maxiter': max_iters, 'gtol': 1e-12, 'maxls': 20})
            if res.success or res.fun < best_err:
                best_err = res.fun
                U = res.x[:dim_U].reshape(n*m, target_rank)
                V = res.x[dim_U:dim_U+dim_V].reshape(m*p, target_rank)
                W = res.x[dim_U+dim_V:].reshape(p*n, target_rank)
                best_result = (U, V, W)
        except Exception:
            pass
    
    if best_result is not None and best_err < 1e-8:
        U, V, W = best_result
        # Try rounding to half-integers
        for factor in [2, 1]:
            U_r = np.round(U * factor) / factor
            V_r = np.round(V * factor) / factor
            W_r = np.round(W * factor) / factor
            T_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            diff_r = np.max(np.abs(T_r - expected))
            if diff_r < 1e-6:
                return U_r, V_r, W_r
    return None

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 105  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build expected tensor
    expected = _build_expected_tensor(n, m, p)
    
    # First, build the known rank-35 solution
    U35, V35, W35 = _build_strassen_rank35()
    
    # Verify it works
    T35 = np.einsum('ir,jr,kr->ijk', U35, V35, W35)
    err35 = np.max(np.abs(T35 - expected))
    if err35 > 1e-10:
        # Should not happen, but fallback
        return {"U": U35.tolist(), "V": V35.tolist(), "W": W35.tolist()}
    
    # Try to find lower rank solutions
    for target_rank in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 15:
            break
        
        result = _try_lower_rank(target_rank, expected, n, m, p, max_iters=200, seed=target_rank, time_limit=15)
        if result is not None:
            U, V, W = result
            # Final verification
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(T - expected))
            if err < 1e-6:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the rank-35 solution
    return {"U": U35.tolist(), "V": V35.tolist(), "W": W35.tolist()}
# EVOLVE-BLOCK-END
