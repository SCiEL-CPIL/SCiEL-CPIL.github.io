import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _strassen_rank55():
    """Known-good rank-55 decomposition: 5 Strassen blocks + 20 schoolbook terms."""
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    W_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j][r] = val
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k)][r] = val
            for local_idx, val in W_coeffs[s].items():
                i = local_idx // 2
                k = local_idx % 2
                W[(k_offset + k) * n + i][r] = val
            col += 1
            
    for i in range(2):
        for k in range(10):
            r = col
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}


def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] = 1 if valid."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _verify(U, V, W, T):
    """Check if U, V, W exactly decompose T."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float).reshape(n * m, rank)
    Va = np.array(V, dtype=float).reshape(m * p, rank)
    Wa = np.array(W, dtype=float).reshape(p * n, rank)
    
    # Reconstruct: C_ik = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Tensor form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    recon = np.zeros((n, m, p))
    for r in range(rank):
        ua = Ua[:, r].reshape(n, m)
        va = Va[:, r].reshape(m, p)
        wa = Wa[:, r].reshape(p, n)
        # outer product structure
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += ua[i, j] * va[j, k] * wa[k, i]
    
    return np.allclose(recon, T, atol=1e-9)


def _als_reduce(T, rank_target, max_iters=200, n_restarts=10, deadline=None):
    """Try to find a rank-rank_target decomposition via ALS."""
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    
    best_result = None
    best_err = float('inf')
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(Nm, rank_target)
        V = np.random.randn(Mp, rank_target)
        W = np.random.randn(Pn, rank_target)
        
        # Scale for numerical stability
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_err = float('inf')
        converged = False
        
        for it in range(max_iters):
            if deadline and time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), U[i*m+j, r] such that
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = T[i,j,k] for all k
            # This is: U[i*m+j, :] * (V_j ⊙ W_i)^T = T[i,j,:]
            # where V_j is row j of V reshaped to (p, rank) and W_i is column i of W reshaped to (p, rank)
            for i in range(n):
                for j in range(m):
                    # Build the matrix A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, rank_target))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve: U[i*m+j, :] @ A.T = T[i,j,:]
                    # i.e., A @ U[i*m+j, :].T = T[i,j,:].T
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, T[i, j, :], rcond=None)
                        U[i * m + j, :] = sol
                    except:
                        pass
            
            # Update V: for each (j,k), V[j*p+k, r]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n * m, rank_target))
                    for i in range(n):
                        for jj in range(m):
                            A[i * m + jj, :] = U[i * m + jj, :] * W[k * n + i, :]
                    # We need: sum_r U[i*m+jj, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,jj,k]
                    # For fixed j,k: sum_r (U[i*m+jj,r] * W[k*n+i,r]) * V[j*p+k,r] = T[i,jj,k]
                    # A[i*m+jj, r] = U[i*m+jj,r] * W[k*n+i,r]
                    # A @ V[j*p+k,:].T = T[:,j,k].T
                    rhs = T[:, j, k].reshape(-1)
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=None)
                        V[j * p + k, :] = sol
                    except:
                        pass
            
            # Update W: for each (k,i), W[k*n+i, r]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m * p, rank_target))
                    for j in range(m):
                        for kk in range(p):
                            A[j * p + kk, :] = U[i * m + j, :] * V[j * p + kk, :]
                    # sum_r U[i*m+j,r] * V[j*p+kk,r] * W[k*n+i,r] = T[i,j,kk]
                    # For fixed k,i: sum_r (U[i*m+j,r] * V[j*p+kk,r]) * W[k*n+i,r] = T[i,j,kk]
                    rhs = T[i, :, :].reshape(-1)
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=None)
                        W[k * n + i, :] = sol
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank_target):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err += (val - T[i, j, k]) ** 2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                converged = True
                break
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        if converged:
            break
    
    if best_result is not None and best_err < 1e-9:
        U, V, W = best_result
        # Convert to lists with integer/half-integer rounding
        U_list = [[round(U[i, r] * 2) / 2 for r in range(rank_target)] for i in range(Nm)]
        V_list = [[round(V[i, r] * 2) / 2 for r in range(rank_target)] for i in range(Mp)]
        W_list = [[round(W[i, r] * 2) / 2 for r in range(rank_target)] for i in range(Pn)]
        
        # Verify the rounded version
        if _verify(U_list, V_list, W_list, T):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try integer rounding
        U_list = [[round(U[i, r]) for r in range(rank_target)] for i in range(Nm)]
        V_list = [[round(V[i, r]) for r in range(rank_target)] for i in range(Mp)]
        W_list = [[round(W[i, r]) for r in range(rank_target)] for i in range(Pn)]
        if _verify(U_list, V_list, W_list, T):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Start with known rank-55 solution
    best = _strassen_rank55()
    best_rank = 55
    
    # Try to find lower rank via ALS
    for target_rank in range(54, 30, -1):
        if time.time() > deadline - 10:
            break
        
        result = _als_reduce(T, target_rank, max_iters=100, n_restarts=5, deadline=deadline)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
