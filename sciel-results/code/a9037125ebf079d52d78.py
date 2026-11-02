"""Matrix multiplication 3x4x7 decomposition - search for lower rank."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook_solution():
    """Return the standard rank-84 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    r = n * m * p
    
    U = [[0] * r for _ in range(n*m)]
    V = [[0] * r for _ in range(m*p)]
    W = [[0] * r for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=3, m=4, p=7, tol=1e-10):
    """
    Verify that U, V, W form a valid tensor decomposition for n*m*p.
    The condition is: for all i,j,k:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is the (i,j,k) entry
    Actually, the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    and the decomposition requires:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n=3, m=4, p=7, deadline=None):
    """
    Try to find a rank-r decomposition.
    Uses a random initialization + iterative refinement approach.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    # The tensor has n*m*p = 84 non-zero entries (all 1s)
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a system of 84 equations in 3*r variables (but nonlinear).
    # We'll use a simple iterative method: start with random init, then
    # try to satisfy the equations one at a time.
    
    np.random.seed(42)
    
    # Initialize with small random values
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    # List of all (i,j,k) triples
    triples = [(i,j,k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Iterative refinement: for each triple, adjust the matrices to satisfy the equation
    for iteration in range(50):
        if deadline is not None and time.time() > deadline:
            return None
        
        for idx, (i,j,k) in enumerate(triples):
            # Current value
            val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
            # Residual
            res = 1.0 - val
            
            if abs(res) < 1e-12:
                continue
            
            # Try to adjust: we can scale one of the vectors
            # Simple approach: scale U[i*m+j] by (1 + res * 0.01)
            # This is very crude and may not converge.
            
            # Better: use gradient-like step
            # d/dU[i*m+j,r] of (sum U*V*W)^2 = 2 * val * V[j*p+k,r] * W[k*n+i,r]
            # So update: U[i*m+j] += lr * res * V[j*p+k] * W[k*n+i]
            
            lr = 0.01
            U[i*m+j] += lr * res * V[j*p+k] * W[k*n+i]
            V[j*p+k] += lr * res * U[i*m+j] * W[k*n+i]
            W[k*n+i] += lr * res * U[i*m+j] * V[j*p+k]
    
    # Check if we got a valid solution
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err < 1e-8:
        # Convert to list of lists with rounded values
        U_list = [[round(U[i][j], 10) for j in range(r)] for i in range(n*m)]
        V_list = [[round(V[i][j], 10) for j in range(r)] for i in range(m*p)]
        W_list = [[round(W[i][j], 10) for j in range(r)] for i in range(p*n)]
        
        # Final verification
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """
    Returns an exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Searches for lower rank, falls back to schoolbook rank 84.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 84
    
    # Set a deadline: 120s total, leave 20s safety margin
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Try ranks from just below schoolbook down to some minimum
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Also rank >= n*m*p / max(n,m,p) roughly... let's just try a few
    
    best_solution = _schoolbook_solution()
    
    # Try to find lower rank solutions
    for r in range(schoolbook_rank - 1, 27, -1):
        if time.time() > deadline:
            break
        
        sol = _try_rank(r, n, m, p, deadline)
        if sol is not None:
            best_solution = sol
            break  # Found a lower rank, use it
    
    return best_solution
# EVOLVE-BLOCK-END
