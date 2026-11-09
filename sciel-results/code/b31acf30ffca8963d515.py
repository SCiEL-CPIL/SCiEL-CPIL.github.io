"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]
import time
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s

# EVOLVE-BLOCK-START
import random
import math
import numpy as np
from itertools import product

def _build_schoolbook():
    """Standard schoolbook matrix multiplication tensor decomposition of rank n*m*p."""
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

def _validate_decomposition(result, ntest=3, seed=42):
    """Validate that the decomposition computes exact matrix multiplication."""
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    if not U or not V or not W:
        return False
    if len(U[0]) != len(V[0]) or len(U[0]) != len(W[0]):
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    rng = random.Random(seed)
    for t in range(ntest):
        A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        Ctrue = [[sum(A[i][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)]
                 for i in range(n)]
        Ccomp = [[0.0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(len(U[0])):
                    uvprod = 0.0
                    for jj in range(m):
                        uvprod += U[i*m+jj][r] * V[jj*p+k][r]
                    acc += uvprod * W[k*n+i][r]
                Ccomp[i][k] = acc
        for i in range(n):
            for k in range(p):
                if abs(Ccomp[i][k] - Ctrue[i][k]) > 1e-9 * max(1, abs(Ctrue[i][k])):
                    return False
    return True

def _compute_residual(U, V, W, rank, n, m, p):
    """Compute max residual of the tensor decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _refine_W(U, V, rank, n, m, p):
    """Given U and V, find W that minimizes residual using Least Squares.
    
    The tensor equation is:
    1 = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    For each (i, j, k), we have a linear constraint on the vector W[:, k*n+i] 
    weighted by U[i,j,r]*V[j,k,r].
    
    Actually, it's easier to view this as:
    For each r, the term is U[:, :, r] * V[:, :, r] * W[:, :, r].
    
    Let's flatten the indices.
    We want to find W such that:
    sum_r (U[i,j,r] * V[j,k,r]) * W[k,i,r] = 1 for all i,j,k.
    
    This is a linear system for each (i,k) pair? No, W[k,i,r] is a variable.
    The variables are W[k,i,r].
    The constraints are for each (i,j,k):
    sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r]) = 1.
    
    Let x be the vector of all W elements. x is of size p*n*rank.
    The system is A x = b, where b is 1s of size n*m*p.
    A has size (n*m*p) x (p*n*rank).
    
    We can solve this least squares problem.
    """
    # Construct A and b
    num_constraints = n * m * p
    num_vars = p * n * rank
    
    # A is sparse-ish, but let's build it densely for small sizes
    # n=3, m=4, p=8, rank<=96.
    # num_constraints = 96
    # num_vars = 8*3*96 = 2304.
    # This is small enough for dense LSQ.
    
    A = np.zeros((num_constraints, num_vars))
    b = np.ones(num_constraints)
    
    # Map constraint index (i,j,k) to row
    # Map variable index (k,i,r) to col
    def cidx(i, j, k):
        return i * m * p + j * p + k
        
    def vidx(k, i, r):
        return k * n * rank + i * rank + r
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = cidx(i, j, k)
                for r in range(rank):
                    coeff = U[i*m+j][r] * V[j*p+k][r]
                    if coeff != 0:
                        col = vidx(k, i, r)
                        A[row, col] = coeff
                        
    # Solve LSQ
    try:
        W_flat, residuals, rank_lsq, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None, 1e10
        
    # Reshape W_flat to W matrix
    W = [[0.0] * rank for _ in range(p * n)]
    for r in range(rank):
        for k in range(p):
            for i in range(n):
                val = W_flat[vidx(k, i, r)]
                W[k*n+i][r] = val
                
    # Compute residual
    err = _compute_residual(U, V, W, rank, n, m, p)
    return W, err

def _refine_U(U, V, W, rank, n, m, p):
    """Given V and W, find U that minimizes residual."""
    num_constraints = n * m * p
    num_vars = n * m * rank
    
    A = np.zeros((num_constraints, num_vars))
    b = np.ones(num_constraints)
    
    def cidx(i, j, k):
        return i * m * p + j * p + k
        
    def uidx(i, j, r):
        return i * m * rank + j * rank + r
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = cidx(i, j, k)
                for r in range(rank):
                    coeff = V[j*p+k][r] * W[k*n+i][r]
                    if coeff != 0:
                        col = uidx(i, j, r)
                        A[row, col] = coeff
                        
    try:
        U_flat, residuals, rank_lsq, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None, 1e10
        
    U = [[0.0] * rank for _ in range(n * m)]
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                val = U_flat[uidx(i, j, r)]
                U[i*m+j][r] = val
                
    err = _compute_residual(U, V, W, rank, n, m, p)
    return U, err

def _refine_V(U, V, W, rank, n, m, p):
    """Given U and W, find V that minimizes residual."""
    num_constraints = n * m * p
    num_vars = m * p * rank
    
    A = np.zeros((num_constraints, num_vars))
    b = np.ones(num_constraints)
    
    def cidx(i, j, k):
        return i * m * p + j * p + k
        
    def vidx(j, k, r):
        return j * p * rank + k * rank + r
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = cidx(i, j, k)
                for r in range(rank):
                    coeff = U[i*m+j][r] * W[k*n+i][r]
                    if coeff != 0:
                        col = vidx(j, k, r)
                        A[row, col] = coeff
                        
    try:
        V_flat, residuals, rank_lsq, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None, 1e10
        
    V = [[0.0] * rank for _ in range(m * p)]
    for r in range(rank):
        for j in range(m):
            for k in range(p):
                val = V_flat[vidx(j, k, r)]
                V[j*p+k][r] = val
                
    err = _compute_residual(U, V, W, rank, n, m, p)
    return V, err

def _try_rank_reduction_lsq(rank, n, m, p, seed):
    """Try to find a valid decomposition of given rank using LSQ refinement."""
    rng = random.Random(seed)
    
    # Initialize U, V, W randomly or sparsely
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Initialize with random sparse patterns
    for r in range(rank):
        i = rng.randint(0, n-1)
        j = rng.randint(0, m-1)
        k = rng.randint(0, p-1)
        U[i*m+j][r] = 1.0
        V[j*p+k][r] = 1.0
        W[k*n+i][r] = 1.0
        
    current_err = _compute_residual(U, V, W, rank, n, m, p)
    
    for it in range(100):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
            
        # Refine W
        W_new, err_w = _refine_W(U, V, rank, n, m, p)
        if W_new is not None and err_w < current_err:
            W = W_new
            current_err = err_w
            
        # Refine U
        U_new, err_u = _refine_U(U, V, W, rank, n, m, p)
        if U_new is not None and err_u < current_err:
            U = U_new
            current_err = err_u
            
        # Refine V
        V_new, err_v = _refine_V(U, V, W, rank, n, m, p)
        if V_new is not None and err_v < current_err:
            V = V_new
            current_err = err_v
            
        if current_err < 1e-9:
            # Snap to nearest 0.5
            U_snapped = [[round(x*2)/2 for x in row] for row in U]
            V_snapped = [[round(x*2)/2 for x in row] for row in V]
            W_snapped = [[round(x*2)/2 for x in row] for row in W]
            
            # Verify snapped
            err_snapped = _compute_residual(U_snapped, V_snapped, W_snapped, rank, n, m, p)
            if err_snapped < 1e-9:
                result = {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                if _validate_decomposition(result, ntest=5, seed=seed+rank):
                    return result, current_err
            else:
                # If snapping failed, maybe we are close but not exact.
                # Continue refining? Or accept if floating point error is small?
                # The task requires "exact" decomposition with integer/half-integer entries.
                # So we must snap.
                pass
        
        # If stuck, perturb
        if it % 10 == 9:
             # Perturb a random entry in U
             r = rng.randint(0, rank-1)
             i = rng.randint(0, n-1)
             j = rng.randint(0, m-1)
             U[i*m+j][r] += rng.choice([-1, 1, 0.5, -0.5])
             
    return None, current_err

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME
    budget = _TIME_LIMIT - elapsed
    if budget <= 10:
        return _build_schoolbook()
    
    best = _build_schoolbook()
    best_rank = len(best['U'][0])
    
    # Try to reduce rank
    # Lower bound is max(n*m, m*p, p*n) = 32.
    # We try ranks from 95 down to 80 first, then maybe lower.
    # Given time constraints, we focus on close-to-schoolbook ranks.
    
    for rank in range(best_rank - 1, max(60, best_rank - 10), -1):
        if time.time() - _START_TIME > _TIME_LIMIT - 15:
            break
            
        # Try a few seeds
        found = False
        for seed in range(5):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            result, err = _try_rank_reduction_lsq(rank, n, m, p, seed)
            if result:
                best = result
                best_rank = rank
                found = True
                break
                
        if found:
            # If we found a better rank, we can try to go even lower, 
            # but we must stop if time is low.
            if best_rank <= 70:
                break
                
    return {
        "U": [[float(x) for x in row] for row in best['U']],
        "V": [[float(x) for x in row] for row in best['V']],
        "W": [[float(x) for x in row] for row in best['W']]
    }
# EVOLVE-BLOCK-END
