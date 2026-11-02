import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _construct_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
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
                
    return {"U": U, "V": V, "W": W}

def _als_decompose(n, m, p, target_rank, max_iter=50, tol=1e-10, seed=42):
    """Alternating least squares for 3-tensor decomposition."""
    np.random.seed(seed)
    dim1, dim2, dim3 = n * m, m * p, p * n
    
    U = np.random.randn(dim1, target_rank) * 0.1
    V = np.random.randn(dim2, target_rank) * 0.1
    W = np.random.randn(dim3, target_rank) * 0.1
    
    prev_err = np.inf
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                A = np.zeros((target_rank, target_rank))
                b = np.zeros(target_rank)
                for k in range(p):
                    vw = V[j * p + k, :] * W[k * n + i, :]
                    A += np.outer(vw, vw)
                    b += vw
                U[idx, :] = np.linalg.lstsq(A + 1e-12 * np.eye(target_rank), b, rcond=None)[0]
        
        # Update V
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                A = np.zeros((target_rank, target_rank))
                b = np.zeros(target_rank)
                for i in range(n):
                    uw = U[i * m + j, :] * W[k * n + i, :]
                    A += np.outer(uw, uw)
                    b += uw
                V[idx, :] = np.linalg.lstsq(A + 1e-12 * np.eye(target_rank), b, rcond=None)[0]
        
        # Update W
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                A = np.zeros((target_rank, target_rank))
                b = np.zeros(target_rank)
                for j in range(m):
                    uv = U[i * m + j, :] * V[j * p + k, :]
                    A += np.outer(uv, uv)
                    b += uv
                W[idx, :] = np.linalg.lstsq(A + 1e-12 * np.eye(target_rank), b, rcond=None)[0]
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (pred - 1.0) ** 2
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 125
    
    # Start with trivial decomposition
    best_result = _construct_trivial(n, m, p)
    best_rank = n * m * p
    
    # Try ALS with decreasing rank
    for rank in [85, 80, 75, 70, 65, 60, 55, 50, 45, 40]:
        if time.time() > deadline - 15:
            break
        
        U, V, W, err = _als_decompose(n, m, p, rank, max_iter=30, seed=42)
        
        if err < 1e-4:
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
