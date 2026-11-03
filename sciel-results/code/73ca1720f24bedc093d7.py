import numpy as np
import time

DIMENSIONS = [3, 3, 4]
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition computes matrix multiplication exactly."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed[i, k] = s
    
    return np.allclose(C_computed, C_expected, atol=1e-9)

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank_alts(n, m, p, target_rank, max_iters=2000, seed_offset=0):
    """
    Alternating minimization: fix U,V -> solve for W (linear), then fix U,W -> solve for V, etc.
    The tensor equation: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    """
    np.random.seed(seed_offset)
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.zeros((p * n, target_rank))
    
    for it in range(max_iters):
        # Update W: for each (i,k), need sum_r S[i,k,r] * W[k*n+i,r] = 1
        # where S[i,k,r] = sum_j U[i*m+j,r] * V[j*p+k,r]
        S = np.zeros((n, p, target_rank))
        for i in range(n):
            for k in range(p):
                for r in range(target_rank):
                    s = 0.0
                    for j in range(m):
                        s += U[i*m+j, r] * V[j*p+k, r]
                    S[i, k, r] = s
        
        for i in range(n):
            for k in range(p):
                sv = S[i, k, :]
                nrm = np.dot(sv, sv)
                if nrm > 1e-15:
                    W[k*n+i, :] = sv / nrm
                else:
                    W[k*n+i, :] = 0
        
        # Update U: for each (i,j), need sum_k sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = ...
        # This is: for fixed (i,j), sum_r U[i*m+j,r] * (sum_k V[j*p+k,r] * W[k*n+i,r]) = ?
        # The target is not simply 1; it's part of the overall equation.
        # Actually, the equation is: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # Fixing V and W, for each (i,j): this is a system where U[i*m+j,:] appears in multiple equations (one per k).
        # For fixed (i,j): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for each k.
        # Let M[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
        # Then: sum_r U[i*m+j,r] * M[i,j,k,r] = 1 for each k.
        # This is: U[i*m+j,:] @ M[i,j,:,:].T = ones(p)
        
        for i in range(n):
            for j in range(m):
                # Build M: shape (p, target_rank) where M[k,r] = V[j*p+k,r] * W[k*n+i,r]
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve U[i*m+j,:] @ M.T = ones(p)
                # U[i*m+j,:] = ones(p) @ pinv(M.T)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), need sum_i sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = ...
        # For fixed (j,k): sum_r V[j*p+k,r] * (sum_i U[i*m+j,r] * W[k*n+i,r]) = 1
        # Let N[j,k,r] = sum_i U[i*m+j,r] * W[k*n+i,r]
        # Then: sum_r V[j*p+k,r] * N[j,k,r] = 1
        
        for j in range(m):
            for k in range(p):
                N = np.zeros(target_rank)
                for i in range(n):
                    N += U[i*m+j, :] * W[k*n+i, :]
                nrm = np.dot(N, N)
                if nrm > 1e-15:
                    V[j*p+k, :] = N / nrm
                else:
                    V[j*p+k, :] = 0
        
        # Check convergence
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_gauss_newton(n, m, p, target_rank, max_iters=500, seed_offset=0):
    """
    Gauss-Newton style: treat the tensor equation as a nonlinear least squares problem.
    Variables: all entries of U, V, W.
    Residuals: for each (i,j,k), R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
    """
    np.random.seed(seed_offset)
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.zeros((p * n, target_rank))
    
    # Initialize W from U,V
    for i in range(n):
        for k in range(p):
            sv = np.zeros(target_rank)
            for j in range(m):
                sv += U[i*m+j, :] * V[j*p+k, :]
            nrm = np.dot(sv, sv)
            if nrm > 1e-15:
                W[k*n+i, :] = sv / nrm
    
    for it in range(max_iters):
        # Compute residuals
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residuals[i, j, k] = s - 1.0
        
        res_norm = np.linalg.norm(residuals)
        if res_norm < 1e-8:
            return U.tolist(), V.tolist(), W.tolist()
        
        # Update W (linear solve)
        for i in range(n):
            for k in range(p):
                sv = np.zeros(target_rank)
                for j in range(m):
                    sv += U[i*m+j, :] * V[j*p+k, :]
                nrm = np.dot(sv, sv)
                if nrm > 1e-15:
                    W[k*n+i, :] = sv / nrm
        
        # Update U (linear solve)
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V (linear solve)
        for j in range(m):
            for k in range(p):
                N = np.zeros(target_rank)
                for i in range(n):
                    N += U[i*m+j, :] * W[k*n+i, :]
                nrm = np.dot(N, N)
                if nrm > 1e-15:
                    V[j*p+k, :] = N / nrm
                else:
                    V[j*p+k, :] = 0
        
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    
    best_result = None
    best_rank = n * m * p  # 36
    
    # Try ranks from 34 down to 30
    for target_rank in range(34, 29, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try multiple seeds
        for seed in range(50):
            if time.time() - start_time > DEADLINE:
                break
            
            result = _try_rank_gauss_newton(n, m, p, target_rank, max_iters=300, seed_offset=seed)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
