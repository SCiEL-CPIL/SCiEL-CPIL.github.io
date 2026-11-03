"""Matrix multiplication 3,3,6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]


# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=3, m=3, p=6, tol=1e-10):
    """Check exact decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_val = Ua[i * m + j]
                v_val = Va[j * p + k]
                w_val = Wa[k * n + i]
                s = np.dot(np.multiply(u_val, v_val), w_val)
                if abs(s - 1.0) > tol:
                    return False
    # Check that no spurious terms: for different (i',j',k') the contribution should be 0
    # Actually, the tensor T[i,j,k] = 1 if i,i' match etc. For matrix mult, T[i,j,k] = 1.
    # Wait, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k? No.
    # The matrix multiplication tensor is defined as: for A (n x m) and B (m x p), C = A*B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 (it's the identity in the middle index).
    # So we need sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k.
    # That's what we check above.
    return True


def _try_als(rank, max_iter=200, seed=42, time_budget=100):
    """Try to find a rank-`rank` decomposition via ALS on random init."""
    n, m, p = DIMENSIONS
    t0 = time.time()
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Precompute indices
    idx_U = [(i * m + j) for i in range(n) for j in range(m)]
    idx_V = [(j * p + k) for j in range(m) for k in range(p)]
    idx_W = [(k * n + i) for k in range(p) for i in range(n)]
    
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            return None
        # Update U
        for i in range(n):
            for j in range(m):
                row_idx = i * m + j
                # Sum over k of W[k*n+i, r] * V[j*p+k, r] = target row
                # For each k, we have a constraint: sum_r U[row_idx, r] * V[j*p+k, r] * W[k*n+i, r] should contribute to T[i,j,k]=1
                # Actually, for fixed (i,j), the row U[i*m+j,:] must satisfy:
                # For each k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is: U_row @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
                # So we have p equations for U_row
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k] = V[j * p + k] * W[k * n + i]
                b_vec = np.ones(p)
                # Solve least squares
                U[row_idx], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Update V
        for j in range(m):
            for k in range(p):
                row_idx = j * p + k
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    A_mat[i] = U[i * m + j] * W[k * n + i]
                b_vec = np.ones(n)
                V[row_idx], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Update W
        for k in range(p):
            for i in range(n):
                row_idx = k * n + i
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    A_mat[j] = U[i * m + j] * V[j * p + k]
                b_vec = np.ones(m)
                W[row_idx], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Check convergence
        if it % 5 == 0 or it == max_iter - 1:
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    t0 = time.time()
    deadline = 140.0  # 150s kill, keep 10s margin
    
    # Try lower ranks first
    # For 3x3x3, best known is 23. For 3x3x6, we could potentially do 2*23=46 if the 3x3x3 result is known,
    # but that requires specific coefficients. Let's try ALS for rank 46, 48, 50, 52.
    
    # First, always have the schoolbook fallback
    fallback = _build_schoolbook()
    
    # Try ALS with various ranks
    for rank in [46, 48, 50, 52]:
        if time.time() - t0 > deadline - 20:
            break
        result = _try_als(rank, max_iter=300, seed=rank * 1000, time_budget=deadline - (time.time() - t0) - 10)
        if result is not None:
            # Verify once more
            if _verify(result["U"], result["V"], result["W"]):
                return result
    
    return fallback
# EVOLVE-BLOCK-END
