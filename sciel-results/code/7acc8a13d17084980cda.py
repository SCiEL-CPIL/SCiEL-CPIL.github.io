"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, n_tests=10):
    """Verify that the decomposition correctly computes C = A*B."""
    np.random.seed(42)
    for _ in range(n_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_reconstructed = np.zeros((n, p))
        for r in range(U.shape[1]):
            a_vals = np.zeros(n)
            b_vals = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    a_vals[i] += U[i * m + j, r] * A[i, j]
            for k in range(p):
                for j in range(m):
                    b_vals[k] += V[j * p + k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_reconstructed[i, k] += W[k * n + i, r] * a_vals[i] * b_vals[k]
        if not np.allclose(C_reconstructed, C_expected, atol=1e-8):
            return False
    return True

def _als_decompose(n, m, p, rank, max_iter=100, tol=1e-8, time_budget=20, seed=0):
    """
    Use Alternating Least Squares to find a rank-r decomposition.
    Returns (U, V, W) or None if failed.
    """
    start = time.time()
    np.random.seed(seed)
    
    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T[a, b, c]
    # a = i'*m + j'
    # b = j''*p + k'
    # c = k''*n + i
    # T[a, b, c] = 1 if i(a)==i(c) and k(b)==k(c) else 0
    # Note: j(a) and j(b) are free.
    
    # We will work with unfolded matrices.
    # T1: dim_u x (dim_v * dim_w)
    # T2: dim_v x (dim_w * dim_u)
    # T3: dim_w x (dim_u * dim_v)
    
    # Construct T1 sparsely or densely?
    # dim_u=9, dim_v=21, dim_w=21.
    # T1 size 9 x 441. Dense is fine.
    
    T1 = np.zeros((dim_u, dim_v * dim_w))
    T2 = np.zeros((dim_v, dim_w * dim_u))
    T3 = np.zeros((dim_w, dim_u * dim_v))
    
    for i in range(n):
        for k in range(p):
            # c index for this i, k
            c = k * n + i
            # a indices for this i
            a_indices = [i * m + j for j in range(m)]
            # b indices for this k
            b_indices = [j * p + k for j in range(m)]
            
            for a in a_indices:
                for b in b_indices:
                    T1[a, b * dim_w + c] = 1.0
                    T2[b, c * dim_u + a] = 1.0
                    T3[c, a * dim_v + b] = 1.0
    
    # Initialize
    U = np.random.randn(dim_u, rank)
    V = np.random.randn(dim_v, rank)
    W = np.random.randn(dim_w, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
            
        # Update U
        # Minimize || T1 - U * M1 ||_F^2
        # M1[r, (b,c)] = V[b,r] * W[c,r]
        M1 = np.outer(V.ravel(), W.ravel()) # Wrong shape logic, need to be careful
        # V is dim_v x rank. W is dim_w x rank.
        # M1 should be rank x (dim_v * dim_w)
        # M1[r, b*dim_w + c] = V[b, r] * W[c, r]
        M1 = np.zeros((rank, dim_v * dim_w))
        for r in range(rank):
            M1[r, :] = np.outer(V[:, r], W[:, r]).ravel()
            
        # Solve U * M1 = T1
        # U = T1 * M1^T * (M1 * M1^T)^-1
        # Add regularization for stability
        reg = 1e-6
        M1T_M1 = M1.T @ M1 + reg * np.eye(rank)
        U = T1 @ M1.T @ np.linalg.inv(M1T_M1)
        
        # Update V
        # Minimize || T2 - V * M2 ||_F^2
        # M2[r, (c,a)] = W[c,r] * U[a,r]
        M2 = np.zeros((rank, dim_w * dim_u))
        for r in range(rank):
            M2[r, :] = np.outer(W[:, r], U[:, r]).ravel()
            
        M2T_M2 = M2.T @ M2 + reg * np.eye(rank)
        V = T2 @ M2.T @ np.linalg.inv(M2T_M2)
        
        # Update W
        # Minimize || T3 - W * M3 ||_F^2
        # M3[r, (a,b)] = U[a,r] * V[b,r]
        M3 = np.zeros((rank, dim_u * dim_v))
        for r in range(rank):
            M3[r, :] = np.outer(U[:, r], V[:, r]).ravel()
            
        M3T_M3 = M3.T @ M3 + reg * np.eye(rank)
        W = T3 @ M3.T @ np.linalg.inv(M3T_M3)
        
        # Calculate Error
        # Err = || T1 - U * M1 ||_F^2
        # Recalculate M1 with new U? No, M1 depends on V,W which are fixed in this step?
        # Actually, we updated U, then V, then W.
        # For error, use current U,V,W.
        M1_new = np.zeros((rank, dim_v * dim_w))
        for r in range(rank):
            M1_new[r, :] = np.outer(V[:, r], W[:, r]).ravel()
        err = np.sum((T1 - U @ M1_new.T)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < tol:
            break
            
    if best_err < 1e-4: # Loose threshold for ALS, verify later
        return best_U, best_V, best_W
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 * 3x7.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook solution: rank = n * m * p = 3 * 3 * 7 = 63
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank decompositions
    # Hint suggests rank 24 is possible. Try from 24 upwards.
    # If 24 fails, try 25, etc.
    for try_rank in range(24, 63):
        if time.time() > deadline:
            break
        
        # Try a few seeds for robustness
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            result = _als_decompose(n, m, p, try_rank, max_iter=50, time_budget=5, seed=seed)
            if result is not None:
                U_try, V_try, W_try = result
                if _verify_decomposition(U_try, V_try, W_try, n, m, p, n_tests=10):
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = try_rank
                    found = True
                    break
        
        if found:
            break
    
    # Round to half-integers if possible
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    if best_rank < 63:
        U_r = round_to_half(best_U)
        V_r = round_to_half(best_V)
        W_r = round_to_half(best_W)
        if _verify_decomposition(U_r, V_r, W_r, n, m, p, n_tests=10):
            best_U, best_V, best_W = U_r, V_r, W_r
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
