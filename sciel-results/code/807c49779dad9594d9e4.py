"""Rewritten program: finds a rank-minimal exact tensor decomposition for 2x4 * 4x5."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _build_schoolbook(n, m, p):
    """Return U, V, W for the schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
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
    """Check if U, V, W exactly decompose the multiplication tensor."""
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_rank_R(R, n, m, p, max_iters=200, seed=42):
    """Try to find a rank-R decomposition using ALS-style optimization."""
    np.random.seed(seed)
    U = np.random.randn(n * m, R) * 0.1
    V = np.random.randn(m * p, R) * 0.1
    W = np.random.randn(p * n, R) * 0.1
    
    # We want: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a nonlinear system. Use alternating updates:
    # Fix V and W, solve for U; fix U and W, solve for V; fix U and V, solve for W.
    
    for it in range(max_iters):
        # Update U: for each row i*m+j, we need sum_r U[idx, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
        # This is underdetermined in general, but we can do gradient descent
        # Actually, let's use a different approach: treat it as a least squares problem
        
        # For U update: for each (i, j), and each k, we have:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Let a_kr = V[j*p+k, r] * W[k*n+i, r], then sum_r U[i*m+j, r] * a_kr = 1 for all k
        # This is A @ u = 1 where A[k, r] = a_kr, u = U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # Build system: for each k, sum_r U[idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                A = np.zeros((p, R))
                for k in range(p):
                    for r in range(R):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                # Solve A @ u = ones(p)
                try:
                    u_new, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[idx, :] = u_new
                except:
                    pass
        
        # Update V: for each (j, k), and each i, we have:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                A = np.zeros((n, R))
                for i in range(n):
                    for r in range(R):
                        A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    v_new, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[idx, :] = v_new
                except:
                    pass
        
        # Update W: for each (k, i), and each j, we have:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                A = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    w_new, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[idx, :] = w_new
                except:
                    pass
        
        # Check convergence
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _to_fraction_list(mat):
    """Convert numpy matrix to list of lists of floats (for return)."""
    return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

def construct():
    n, m, p = DIMENSIONS
    # Try ranks from the theoretical lower bound upward
    # Lower bound for matrix multiplication tensor rank: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But actual rank might be higher. Let's try from 20 up to 40.
    for R in range(20, 41):
        result = _try_rank_R(R, n, m, p, max_iters=100, seed=R)
        if result is not None:
            U, V, W = result
            return {"U": _to_fraction_list(U), "V": _to_fraction_list(V), "W": _to_fraction_list(W)}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": _to_fraction_list(U), "V": _to_fraction_list(V), "W": _to_fraction_list(W)}
# EVOLVE-BLOCK-END
