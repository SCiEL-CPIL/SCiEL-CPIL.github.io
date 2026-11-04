"""Matrix multiplication 2x3 by 3x9 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, max_iters=200, seed=42):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    rng = np.random.RandomState(seed)
    for it in range(max_iters):
        # Random initialization
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Iteratively refine: for each (i,j,k), we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        # Use a simple gradient-free approach: adjust W to satisfy constraints
        # This is a bilinear/trilinear system - use alternating least squares style
        
        # Actually, let's use a different strategy: fix U and V, solve for W
        # For each k,i: W[k*n+i, :] needs to satisfy for all j:
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a linear system in W[k*n+i, :] for each (k,i)
        
        # Build the system: for fixed k, we have n rows (for each i) and m equations per row
        # Actually for each (k,i), the constraint is: for all j, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is m linear equations in r unknowns W[k*n+i, 0..r-1]
        
        # Let's set up: for each k, for each i, we have m equations:
        # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1 for j=0..m-1
        # So for each (k,i), we have a system A_{ki} * W[k*n+i,:]^T = 1
        # where A_{ki}[j,c] = U[i*m+j,c] * V[j*p+k,c]
        
        # If r >= m, we might be able to solve this. Let's try least squares.
        W_new = np.zeros((p * n, r))
        feasible = True
        for k in range(p):
            for i in range(n):
                # Build A: m x r
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                b = np.ones(m)
                # Solve A * x = b
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if solution is exact
                    if residuals.size > 0 and residuals[0] > 1e-7:
                        feasible = False
                        break
                    W_new[k * n + i, :] = x
                except:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            # Verify
            if _verify(U, V, W_new, n, m, p):
                return [U.tolist(), V.tolist(), W_new.tolist()]
        
        # Try another random init with perturbation
        U += 0.1 * rng.randn(n * m, r)
        V += 0.1 * rng.randn(m * p, r)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook as fallback
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Also, rank >= max(n*m, m*p, p*n) is a known lower bound for matrix mult tensor
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound up to best_rank-1
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, max_iters=50, seed=r * 1000 + 42)
        if result is not None:
            best_rank = r
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            # Continue trying lower ranks
    
    if time.time() > deadline:
        return best_result
    
    # If we found a rank, try to go even lower
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, max_iters=100, seed=r * 2000 + 123)
        if result is not None:
            best_rank = r
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
    
    return best_result
