"""Matrix multiplication 3,3,8 - rank-minimal decomposition search."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, rank):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_target_tensor(n, m, p):
    """Build the target tensor values for all valid (i,j,k) triples."""
    # T[i,j,k] = 1 for all valid i,j,k
    # We store as a dict keyed by (i,j,k)
    targets = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets[(i, j, k)] = 1.0
    return targets

def _compute_residual(U, V, W, n, m, p, rank):
    """Compute the residual for all (i,j,k) triples."""
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    residual = {}
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r] for r in range(rank))
                res = val - 1.0
                residual[(i, j, k)] = res
                max_res = max(max_res, abs(res))
    return residual, max_res

def _als_optimize(U, V, W, n, m, p, rank, max_iters=50, tol=1e-12, deadline=None):
    """Alternating Least Squares to minimize residual."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] appears in terms where first index is i*m+j
                # Constraint: for all k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # This is: U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for each k
                # Build system: for each k, row k: (V[j*p+k,:] * W[k*n+i,:]) @ U[i*m+j,:] = 1
                A_mat = np.zeros((p, rank))
                b_vec = np.ones(p)
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A_mat @ U[i*m+j,:] = b_vec
                if rank > p:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                else:
                    try:
                        sol = np.linalg.solve(A_mat, b_vec)
                    except:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[i*m+j, :] = sol
        
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, rank))
                b_vec = np.ones(n)
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                if rank > n:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                else:
                    try:
                        sol = np.linalg.solve(A_mat, b_vec)
                    except:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[j*p+k, :] = sol
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, rank))
                b_vec = np.ones(m)
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                if rank > m:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                else:
                    try:
                        sol = np.linalg.solve(A_mat, b_vec)
                    except:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[k*n+i, :] = sol
        
        # Check convergence
        _, max_res = _compute_residual(U, V, W, n, m, p, rank)
        if max_res < tol:
            break
    
    return U, V, W

def _try_rank(n, m, p, rank, deadline, num_restarts=3):
    """Try to find a valid decomposition of given rank."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    for restart in range(num_restarts):
        if deadline and time.time() > deadline:
            break
        
        # Structured initialization
        U = np.zeros((n_m, rank))
        V = np.zeros((m_p, rank))
        W = np.zeros((p_n, rank))
        
        # Initialize with small random values
        np.random.seed(42 + restart)
        U += np.random.randn(n_m, rank) * 0.1
        V += np.random.randn(m_p, rank) * 0.1
        W += np.random.randn(p_n, rank) * 0.1
        
        # Run ALS
        U, V, W = _als_optimize(U, V, W, n, m, p, rank, max_iters=100, deadline=deadline)
        
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound is at least max(n*p, n*m, m*p) = max(24, 9, 24) = 24
    # Try ranks from 24 up to schoolbook-1
    for rank in range(24, best_rank):
        if deadline and time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, rank, deadline, num_restarts=2)
        if result is not None:
            best = result
            best_rank = rank
            break  # Found the minimum, stop
    
    return best
# EVOLVE-BLOCK-END
