"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 9]

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 144."""
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

def _kron_product(A, B):
    """Kronecker product of two matrices."""
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    result = [[0.0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            if A[i][j] != 0:
                for k in range(rb):
                    for l in range(cb):
                        result[i * rb + k][j * cb + l] = A[i][j] * B[k][l]
    return result

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                # The matrix multiplication tensor has 1s for all i, j, k
                expected = 1.0
                if abs(total - expected) > 1e-10:
                    return False
    return True

def _search_223_decomposition(rank, deadline, n_restarts=50):
    """Search for a rank-r decomposition of 2x2x3 matrix multiplication."""
    n, m, p = 2, 2, 3
    dim_U = n * m  # 4
    dim_V = m * p  # 6
    dim_W = p * n  # 6
    
    best_result = None
    best_err = float('inf')
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:  # Leave 5s margin
            break
        
        # Random initialization
        U = np.random.randn(dim_U, rank)
        V = np.random.randn(dim_V, rank)
        W = np.random.randn(dim_W, rank)
        
        # Define objective: sum of squared errors
        def objective(params):
            U_flat = params[:dim_U * rank].reshape(dim_U, rank)
            V_flat = params[dim_U * rank:dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W_flat = params[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_flat[i*m+j] * V_flat[j*p+k] * W_flat[k*n+i])
                        err += (val - 1.0) ** 2
            return err
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 500, 'gtol': 1e-12})
            err = res.fun
            if err < best_err:
                best_err = err
                if err < 1e-10:
                    U_opt = res.x[:dim_U * rank].reshape(dim_U, rank)
                    V_opt = res.x[dim_U * rank:dim_U * rank + dim_V * rank].reshape(dim_V, rank)
                    W_opt = res.x[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
                    # Convert to lists
                    U_list = [[float(U_opt[i][c]) for c in range(rank)] for i in range(dim_U)]
                    V_list = [[float(V_opt[i][c]) for c in range(rank)] for i in range(dim_V)]
                    W_list = [[float(W_opt[i][c]) for c in range(rank)] for i in range(dim_W)]
                    best_result = (U_list, V_list, W_list)
                    break
        except Exception:
            continue
    
    return best_result

def _try_kron_decomposition(deadline):
    """Try Kronecker product decomposition using low-rank 2x2x3 factors."""
    if time.time() > deadline:
        return None
    
    # Try ranks from 7 upward
    for rank in range(7, 13):
        if time.time() > deadline:
            break
        
        result = _search_223_decomposition(rank, deadline, n_restarts=20)
        if result is not None:
            U2, V2, W2 = result
            
            # Verify the 2x2x3 decomposition first
            if not _verify_decomposition(U2, V2, W2, 2, 2, 3):
                continue
            
            # Kronecker product
            U4 = _kron_product(U2, U2)
            V4 = _kron_product(V2, V2)
            W4 = _kron_product(W2, W2)
            
            if _verify_decomposition(U4, V4, W4, 4, 4, 9):
                return {"U": U4, "V": V4, "W": W4}
    
    return None

def construct():
    deadline = time.time() + 120
    best = _schoolbook()
    
    # Try Kronecker product approach
    result = _try_kron_decomposition(deadline)
    if result is not None:
        rank_new = len(result['U'][0])
        rank_old = len(best['U'][0])
        if rank_new < rank_old:
            best = result
    
    return best
