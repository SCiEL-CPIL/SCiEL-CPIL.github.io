"""Matrix multiplication 2,4,6 – rank-minimal decomposition via block decomposition."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p).
    Returns U (n*m x nmp), V (m*p x nmp), W (p*n x nmp)."""
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _als_2x4x6(n, m, p, rank, max_iters=100, tol=1e-10, seed=42):
    """
    Attempt to find a rank-`rank` decomposition using Alternating Least Squares.
    """
    np.random.seed(seed)
    
    # Initialize U, V, W with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    for it in range(max_iters):
        # Update U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Equations: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
                # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then A @ U[i,j,:] = ones(p)
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    # Solve min ||A u - 1||^2
                    # Using lstsq
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=1e-12)
                    U_new[i * m + j, :] = sol
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        U = U_new
        
        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # Equations: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i
                # Let B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Then B @ V[j,k,:] = ones(n)
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    sol, residuals, rank_B, sv = np.linalg.lstsq(B, np.ones(n), rcond=1e-12)
                    V_new[j * p + k, :] = sol
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        V = V_new
        
        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # Equations: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all j
                # Let C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then C @ W[k,i,:] = ones(m)
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    sol, residuals, rank_C, sv = np.linalg.lstsq(C, np.ones(m), rcond=1e-12)
                    W_new[k * n + i, :] = sol
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        W = W_new
        
        # Check convergence
        if it % 5 == 0 or it == max_iters - 1:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        err += (val - 1.0)**2
            if err < tol:
                break
                
    return U, V, W

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Try to find rank 24 decomposition
    # Theoretical lower bound is 24? 
    # Let's try 24 first.
    target_rank = 24
    
    if time.time() < deadline:
        U, V, W = _als_2x4x6(n, m, p, target_rank, max_iters=50, seed=1)
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # Try different seeds if time permits
        for seed in range(2, 10):
            if time.time() > deadline - 5.0:
                break
            U, V, W = _als_2x4x6(n, m, p, target_rank, max_iters=50, seed=seed)
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to higher rank if 24 fails or times out
    # Try rank 28, 32, etc? Or just schoolbook?
    # Schoolbook is 48.
    # Let's try to get a better fallback than 48 if 24 fails.
    # We can try rank 30.
    if time.time() < deadline - 10.0:
        for r in [30, 32, 36]:
            U, V, W = _als_2x4x6(n, m, p, r, max_iters=50, seed=1)
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: Schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
