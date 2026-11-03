"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+l, :], V_arr[l*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_54(n, m, p):
    """Build the known rank-54 Strassen decomposition."""
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    return U, V, W, col

def _try_reduce_rank(U, V, W, n, m, p, rank, deadline):
    """Try to reduce the rank by finding linear dependencies or using optimization."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor as a matrix and check rank
    # The tensor T[i,l,k] = sum_r U[i*m+l,r] * V[l*p+k,r] * W[k*n+i,r]
    # We can form a matrix where each column r corresponds to a rank-1 term
    
    # Try to find if we can merge or eliminate columns
    # Approach: use SVD-like decomposition on the "flattened" tensor
    
    # For each column r, the contribution is a rank-1 tensor U_r ⊗ V_r ⊗ W_r
    # where U_r is indexed by (i,l), V_r by (l,k), W_r by (k,i)
    
    # Try numerical optimization to find a lower-rank decomposition
    from scipy.optimize import least_squares
    
    triples = [(i, l, k) for i in range(n) for l in range(m) for k in range(p)]
    ntriples = len(triples)
    
    def residuals(x, R):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R:n*m*R + m*p*R + p*n*R].reshape(p*n, R)
        
        res = np.zeros(ntriples)
        for idx, (i, l, k) in enumerate(triples):
            res[idx] = np.dot(U[i*m+l, :], V[l*p+k, :] * W[k*n+i, :]) - 1.0
        return res
    
    # Try ranks from rank-1 down
    for R in range(rank - 1, max(rank - 5, 20), -1):
        if time.time() > deadline:
            break
        
        # Initialize with first R columns
        U_init = U_arr[:, :R]
        V_init = V_arr[:, :R]
        W_init = W_arr[:, :R]
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(residuals, x0, args=(R,), method='lm', 
                                   max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            
            if np.max(np.abs(result.fun)) < 1e-9:
                U_sol = result.x[:n*m*R].reshape(n*m, R)
                V_sol = result.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = result.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                if _verify_decomp(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, R):
                    return U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), R
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140
    
    # Build the rank-54 decomposition
    U, V, W, rank = _build_strassen_54(n, m, p)
    
    if not _verify_decomp(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n*m)]
        V = [[0] * rank for _ in range(m*p)]
        W = [[0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U[i*m+l][col] = 1
                    V[l*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    reduced = _try_reduce_rank(U, V, W, n, m, p, rank, _deadline)
    if reduced is not None:
        U, V, W, rank = reduced
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
