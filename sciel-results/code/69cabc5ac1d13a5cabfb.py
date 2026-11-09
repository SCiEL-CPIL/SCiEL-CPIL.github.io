import time
import numpy as np

DIMENSIONS = [2, 3, 5]
DEADLINE = 140.0

def _build_correct_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[I, J, K] = 1 if I=i*m+j, J=j*p+k, K=k*n+i for some i,j,k.
    Note: The problem statement says W is (p*n, rank) and index is k*n+i.
    So the third dimension is p*n.
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    return T

def _verify_correct(U, V, W, n, m, p):
    """
    Verify decomposition of the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    T_target = _build_correct_tensor(n, m, p)
    # Check if they match
    # Due to floating point, use allclose
    return np.allclose(T_recon, T_target, atol=1e-9)

def _als_decompose_correct(n, m, p, rank, max_iter=100, tol=1e-8, seed=0):
    """
    ALS on the correct tensor.
    """
    T = _build_correct_tensor(n, m, p)
    # Flatten T to (n*m, m*p*p*n)? No, keep 3-way.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    best_err = np.inf
    best_factors = (U.copy(), V.copy(), W.copy())
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
            
        # Update U
        # For each row i of U (index I), solve (V^T ⊗ W^T) u_I = T[I, :, :].ravel()
        # V is (m*p, rank), W is (p*n, rank)
        # V^T ⊗ W^T is (m*p * p*n, rank)
        # This is large. (15*10=150, rank~15). 150x15 matrix.
        # T[I, :, :] is (m*p, p*n) -> ravel to (m*p*p*n)
        
        # Precompute Kronecker products?
        # V.T is (rank, m*p). W.T is (rank, p*n).
        # kron(V.T, W.T) is (rank*rank, m*p*p*n)? No.
        # We need matrix M such that M @ u = T[I,:,:].ravel()
        # sum_c U[I,c] V[J,c] W[K,c] = T[I,J,K]
        # For fixed I, vector u = U[I,:].
        # sum_c u_c (V[:,c] * W[:,c]) = T[I,:,:].ravel()
        # Let Z = V * W, shape (m*p, p*n, rank)? No.
        # Z_c = V[:,c] * W[:,c] is not defined directly.
        # V[:,c] is (m*p,). W[:,c] is (p*n,).
        # We need outer product?
        # T[I, J, K] = sum_c U[I,c] V[J,c] W[K,c]
        # Fix I. T[I, :, :] is (m*p, p*n).
        # T[I, J, K] = sum_c u_c V[J,c] W[K,c]
        # This is sum_c u_c (V[:,c] * W[:,c]^T)? No.
        # V[J,c] * W[K,c] is the (J,K) entry of the rank-1 matrix V_c W_c^T.
        # So T[I,:,:] = sum_c u_c (V_c W_c^T)
        # Where V_c is column c of V, W_c is column c of W.
        # Vectorize: vec(T[I,:,:]) = sum_c u_c vec(V_c W_c^T)
        # vec(V_c W_c^T) = W_c ⊗ V_c
        # So we build matrix A of shape (m*p*p*n, rank)
        # Column c is W_c ⊗ V_c
        
        A_U = np.zeros((m * p * p * n, rank))
        for c in range(rank):
            A_U[:, c] = np.kron(W[:, c], V[:, c])
            
        for I in range(n * m):
            rhs = T[I, :, :].ravel()
            try:
                u_I, _, _, _ = np.linalg.lstsq(A_U, rhs, rcond=tol)
            except:
                u_I = np.zeros(rank)
            U[I, :] = u_I
            
        # Update V
        # T[:, J, K] = sum_c U[I,c] V[J,c] W[K,c]
        # Fix J. T[:, J, :] is (n*m, p*n).
        # T[I, J, K] = sum_c v_c U[I,c] W[K,c]
        # vec(T[:,J,:]) = sum_c v_c (W_c ⊗ U_c)
        
        A_V = np.zeros((n * m * p * n, rank))
        for c in range(rank):
            A_V[:, c] = np.kron(W[:, c], U[:, c])
            
        for J in range(m * p):
            rhs = T[:, J, :].ravel()
            try:
                v_J, _, _, _ = np.linalg.lstsq(A_V, rhs, rcond=tol)
            except:
                v_J = np.zeros(rank)
            V[J, :] = v_J
            
        # Update W
        # T[:, :, K] = sum_c U[I,c] V[J,c] W[K,c]
        # Fix K. T[:, :, K] is (n*m, m*p).
        # T[I, J, K] = sum_c w_c U[I,c] V[J,c]
        # vec(T[:,:,K]) = sum_c w_c (V_c ⊗ U_c)
        
        A_W = np.zeros((n * m * m * p, rank))
        for c in range(rank):
            A_W[:, c] = np.kron(V[:, c], U[:, c])
            
        for K in range(p * n):
            rhs = T[:, :, K].ravel()
            try:
                w_K, _, _, _ = np.linalg.lstsq(A_W, rhs, rcond=tol)
            except:
                w_K = np.zeros(rank)
            W[K, :] = w_K
            
        # Compute error
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
        
        if err < best_err:
            best_err = err
            best_factors = (U.copy(), V.copy(), W.copy())
            
        if err < tol:
            break
            
    if best_err < 1e-6:
        return best_factors
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                U[I, col] = 1.0
                V[J, col] = 1.0
                W[K, col] = 1.0
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try ranks from low to high
    # The rank of 2x3x5 is likely between 10 and 20.
    for rank in [10, 12, 14, 15, 16, 18, 20, 22, 24, 26, 28]:
        if time.time() > DEADLINE:
            break
        
        found = False
        for seed in range(20):
            if time.time() > DEADLINE:
                break
            
            result = _als_decompose_correct(n, m, p, rank, max_iter=30, tol=1e-8, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_correct(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if time.time() > DEADLINE:
            break
            
    # Fall back to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
