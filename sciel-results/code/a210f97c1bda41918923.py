import numpy as np
import itertools
from fractions import Fraction
from math import gcd

DIMENSIONS = [3, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    # Check all (i, j, k) tuples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > tol:
                    return False
    # Check orthogonality: for (i', k') != (i, k), the sum should be 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        if ip == i and kp == k:
                            continue
                        s = 0
                        for c in range(r):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[kp * n + ip][c]
                        if abs(s) > tol:
                            return False
    return True

def _try_rank_with_random_init(r, n, m, p, num_trials=200, seed_offset=0):
    """Try to find a decomposition of rank r using random initialization."""
    for trial in range(num_trials):
        seed = seed_offset + trial
        np.random.seed(seed)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        
        # Solve for W
        W = np.zeros((p * n, r))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build system: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                b = np.ones(m)
                
                try:
                    w, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                    if residuals and residuals[0] > 1e-10:
                        valid = False
                        break
                    W[k * n + i, :] = w
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Try to find a rational approximation
            # First, check if the float solution is close to rational numbers
            # Try rounding to nearest integer, half-integer, quarter-integer, etc.
            for denom in [1, 2, 4, 8, 16]:
                scale = denom
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                
                # Convert to lists for verification
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 84
    
    # Try ranks from schoolbook-1 down to some minimum
    # The known upper bounds for 3x4x7 matrix multiplication are not well-studied,
    # but we can try to find improvements.
    
    # First, try a few ranks below schoolbook
    for r in range(rank_school - 1, max(rank_school - 20, 1), -1):
        result = _try_rank_with_random_init(r, n, m, p, num_trials=100, seed_offset=0)
        if result is not None:
            return result
    
    # If we couldn't find a better rank quickly, return schoolbook
    rank = rank_school
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
