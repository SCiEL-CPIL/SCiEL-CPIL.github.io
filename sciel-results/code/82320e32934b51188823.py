import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Reconstruct tensor
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = U[:, c]  # (n*m,)
        vc = V[:, c]  # (m*p,)
        wc = W[:, c]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += uc[i*m + j] * vc[j*p + k] * wc[k*n + i]
    
    # Expected tensor: T[i,j,k] = 1 for all i,j,k (since each entry is A[i,j]*B[j,k])
    T_expected = np.ones((n, m, p), dtype=np.float64)
    
    return np.allclose(T_recon, T_expected, atol=1e-10)

def _try_rank(r, n, m, p, timeout=140.0):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    start_time = time.time()
    
    # The tensor is T[i,j,k] = 1 for all i,j,k in the "product" sense
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 if we think of it as
    # the bilinear form, but more precisely it's the identity on the middle index.
    # 
    # Let me reconsider. The matrix multiplication tensor for A(n×m) * B(m×p) -> C(n×p)
    # is a trilinear form. The tensor T has T[i,j,k] = 1 for all valid i,j,k when we
    # consider it as a 3-index tensor where the multiplication is C[i,k] = sum_j A[i,j]*B[j,k].
    #
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # Use a simple iterative approach: start with random initialization and try to minimize
    # the error, or use a structured approach.
    
    # For small ranks, let's try a structured approach.
    # Key insight: if we can find rank r such that the system is solvable.
    
    # Let's set up the linear system. For each (i,j,k), we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is nonlinear in the variables. Let's use a different approach.
    #
    # Alternative: fix two of the matrices and solve for the third, iterate.
    
    np.random.seed(42)
    rank = r
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Iterative refinement
    for iteration in range(500):
        if time.time() - start_time > timeout - 5:
            break
        
        # Update U: for each i,j, solve for U[i*m+j, :] given V and W
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # This is: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
        # Let M[i,j,k,c] = V[j*p+k, c] * W[k*n+i, c]
        # Then for fixed i,j: sum_c U[i*m+j, c] * M[i,j,k,c] = 1 for all k
        # This is a linear system in U[i*m+j, :] for each k... but it's coupled.
        #
        # Actually, for fixed i,j: we have p equations (one for each k) and r unknowns.
        # M_kc = V[j*p+k, c] * W[k*n+i, c] for k=0..p-1, c=0..r-1
        # System: sum_c U_c * M_kc = 1 for all k
        # In matrix form: M @ U_ij = 1 where M is p x r
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        M[k, c] = V[j*p + k, c] * W[k*n + i, c]
                rhs = np.ones(p)
                # Solve M @ x = rhs in least squares
                x, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                U[i*m + j, :] = x
        
        # Update V: for each j,k, solve for V[j*p+k, :] given U and W
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        M[i, c] = U[i*m + j, c] * W[k*n + i, c]
                rhs = np.ones(n)
                x, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                V[j*p + k, :] = x
        
        # Update W: for each k,i, solve for W[k*n+i, :] given U and V
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        M[j, c] = U[i*m + j, c] * V[j*p + k, c]
                rhs = np.ones(m)
                x, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                W[k*n + i, :] = x
        
        # Check convergence
        if iteration % 50 == 0:
            T_recon = np.zeros((n, m, p), dtype=np.float64)
            for c in range(rank):
                uc = U[:, c]
                vc = V[:, c]
                wc = W[:, c]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i, j, k] += uc[i*m + j] * vc[j*p + k] * wc[k*n + i]
            err = np.max(np.abs(T_recon - 1.0))
            if err < 1e-10:
                return U, V, W
    
    # Final check
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += uc[i*m + j] * vc[j*p + k] * wc[k*n + i]
    err = np.max(np.abs(T_recon - 1.0))
    if err < 1e-10:
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start from schoolbook rank and try to reduce
    schoolbook_rank = n * m * p  # 45
    
    # Try ranks from lower to higher
    # The minimum possible rank is at least max(n*p, m*p, n*m) = max(15, 15, 9) = 15
    # But realistically it's higher. Let's search from 15 upward.
    
    best_result = None
    
    for r in range(15, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(r, n, m, p, timeout=deadline - time.time())
        if result is not None:
            U, V, W = result
            best_result = (r, U, V, W)
            break
    
    if best_result is None:
        # Fall back to schoolbook
        r = schoolbook_rank
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
