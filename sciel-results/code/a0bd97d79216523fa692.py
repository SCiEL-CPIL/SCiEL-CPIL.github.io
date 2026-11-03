"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}


def _verify(U, V, W, n=3, m=3, p=6, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == tensor for matrix mult."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Contract: tensor[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Build target tensor
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0  # In matrix multiplication, the coefficient is 1 for each (i,j,k) triple
    # Compute reconstructed
    recon = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        # The contribution to tensor[i,j,k] is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    return np.allclose(recon, target, atol=tol)


def _try_rank_search(n, m, p, target_rank, time_budget=100.0):
    """
    Try to find a decomposition of rank = target_rank for n x m x p matrix multiplication.
    Uses ALS (Alternating Least Squares) with integer rounding.
    """
    deadline = time.time() + time_budget
    np.random.seed(42)
    
    # Target tensor
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline - 2:
                break
            
            # Update U: for each (i,j), minimize ||target[i,j,:] - V[j*p+k,:] * W[k*n+i,:]||_2
            # Actually, we need to be careful. Let's use the standard ALS approach.
            # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Fix V and W, solve for U.
            # For fixed (i,j), U[i*m+j, :] should minimize sum_k (target[i,j,k] - sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r])^2
            # This is a least squares problem.
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build A: A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = target[i, j, :]
                    # Solve A^T A U_vec = A^T b
                    try:
                        sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i*m+j, :] = sol
                    except:
                        U_new[i*m+j, :] = np.random.randn(target_rank)
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = target[:, j, k]
                    try:
                        sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j*p+k, :] = sol
                    except:
                        V_new[j*p+k, :] = np.random.randn(target_rank)
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = target[i, j, k]
                    try:
                        sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k*n+i, :] = sol
                    except:
                        W_new[k*n+i, :] = np.random.randn(target_rank)
            W = W_new
            
            # Compute error
            recon = np.zeros((n, m, p))
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            err = np.sum((recon - target) ** 2)
            
            if err < 1e-12:
                break
        
        # After ALS, round to integers and verify
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)
        
        if _verify(U_int, V_int, W_int, n, m, p):
            return U_int, V_int, W_int
        
        # Also try half-integers
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        if _verify(U_half, V_half, W_half, n, m, p):
            return U_half, V_half, W_half
        
        # Track best
        recon = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        err = np.sum((recon - target) ** 2)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Check if best is close enough and can be rounded
    if best_U is not None and best_err < 1e-6:
        U_int = np.round(best_U).astype(int)
        V_int = np.round(best_V).astype(int)
        W_int = np.round(best_W).astype(int)
        if _verify(U_int, V_int, W_int, n, m, p):
            return U_int, V_int, W_int
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 54
    
    # First, try to find a decomposition for 3x3x3 (rank 23 known)
    # If we find it, we can construct 3x3x6 with rank 46
    n2, m2, p2 = 3, 3, 3
    target_rank_333 = 23
    
    result = None
    if time.time() < deadline - 30:
        result = _try_rank_search(n2, m2, p2, target_rank_333, time_budget=min(100.0, deadline - time.time() - 5))
    
    if result is not None:
        U3, V3, W3 = result
        # Construct 3x3x6 from 3x3x3
        # U3: (9, 23), V3: (9, 23), W3: (9, 23)
        # For 3x3x6: U: (9, 46), V: (18, 46), W: (18, 46)
        rank_336 = 46
        U = np.zeros((n * m, rank_336))
        V = np.zeros((m * p, rank_336))
        W = np.zeros((p * n, rank_336))
        
        # First 23 columns: use 3x3x3 decomposition for first half of p
        U[:, :23] = U3
        V[:9, :23] = V3  # V3 is (9, 23) for m*p2=9, pad to m*p=18
        W[:9, :23] = W3   # W3 is (9, 23) for p2*n=9, pad to p*n=18
        
        # Last 23 columns: use 3x3x3 decomposition for second half of p
        U[:, 23:] = U3
        V[9:, 23:] = V3   # Pad second half
        W[9:, 23:] = W3    # Pad second half
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fallback: try direct search for 3x3x6 with various ranks
    for target_rank in [46, 48, 50, 52]:
        if time.time() > deadline - 20:
            break
        res = _try_rank_search(n, m, p, target_rank, time_budget=min(20.0, deadline - time.time() - 5))
        if res is not None:
            U, V, W = res
            if _verify(U, V, W, n, m, p):
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fall back to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
