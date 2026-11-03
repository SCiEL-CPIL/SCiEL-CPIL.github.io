"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
  The trivial schoolbook decomposition has rank 63 = 3*3*7.
  We attempt to find a lower-rank exact decomposition.
  
  For this specific case, we use a structural insight:
  The multiplication C = A * B where A is 3x3 and B is 3x7 can be
  decomposed by viewing B as three 3x3 blocks plus a 3x1 block, but
  a more direct approach is to use the fact that we're computing
  7 columns of the result, each being a matrix-vector product.
  
  However, the most promising approach for exact decomposition with
  integer/half-integer entries is to use known low-rank algorithms
  or to find a decomposition through careful construction.
  
  For 3x3 * 3x3, the best known rank is 21 (Strassen-like).
  For 3x3 * 3x7, we can potentially do better than 63.
  
  Let me try a constructive approach:
  - Use the trivial decomposition as baseline (rank 63)
  - Try to find a decomposition with rank < 63 using:
    1. Randomized search with local optimization
    2. Structural exploitation of the tensor's symmetry
  
  Given the time constraint (150s), I'll implement a search that:
  - Tries to find a valid decomposition for various ranks
  - Uses coordinate descent / alternating least squares
  - Verifies exactness by checking all tensor entries
  
  The key equations: for all (i,j,k) in [0,3) x [0,3) x [0,7):
  sum_r U[i*3+j, r] * V[j*7+k, r] * W[k*3+i, r] = 1
  And for all other (a,b,c):
  sum_r U[a, r] * V[b, r] * W[c, r] = 0
"""

import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def _compute_error(U, V, W, n, m, p):
    """Compute the total error of the decomposition."""
    err = 0.0
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Check diagonal terms (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0) ** 2
    
    # Check a subset of off-diagonal terms (should be 0)
    # For efficiency, check all terms
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if not (i == i2 and j == j2 and k == k2):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += val ** 2
    
    return err

def _is_exact(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact within tolerance."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Check diagonal terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal terms
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if not (i == i2 and j == j2 and k == k2):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Trivial decomposition (rank = n*m*p = 63)
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    # Try to find lower-rank decompositions
    # Start from rank 62 down to some lower bound
    lower_bound = 27  # heuristic lower bound
    
    for R in range(trivial_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline - 25:
            break
        
        # Try multiple random initializations for this rank
        max_trials = 3
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with small values
            np.random.seed(42 + trial)
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # Alternating least squares
            converged = False
            for iteration in range(30):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each i,j
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Update V: for each j,k
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, R))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass
                
                # Update W: for each k,i
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, R))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass
                
                # Check convergence
                err = _compute_error(U, V, W, n, m, p)
                if err < 1e-12:
                    converged = True
                    break
            
            if converged:
                if _is_exact(U, V, W, n, m, p):
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = R
                    break  # Found valid decomposition, try lower rank
        
        if time.time() > deadline - 10:
            break
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
