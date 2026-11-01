"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    # T[i,j,k] = 1 for all i,j,k
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_rank(r, n, m, p, timeout=30):
    """Try to find a rank-r decomposition using random initialization and Newton's method."""
    start = time.time()
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() - start > timeout:
            break
        
        # Random initialization with integer entries
        U = np.random.randint(-2, 3, (n*m, r))
        V = np.random.randint(-2, 3, (m*p, r))
        W = np.random.randint(-2, 3, (p*n, r))
        
        # Build the tensor equation: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a system of n*m*p equations in the variables U, V, W
        
        # Use a greedy/coordinate descent approach
        # Actually, let's try a different approach: fix U and V, solve for W, etc.
        
        # For each (i,j,k): W[k*n+i, :] must satisfy sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is: dot(U[i*m+j,:]*V[j*p+k,:], W[k*n+i,:]) = 1
        
        # Let's try to construct W from U and V
        W_sol = np.zeros((p*n, r))
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec = U[i*m+j, :] * V[j*p+k, :]
                    if np.all(vec == 0):
                        valid = False
                        break
                    # We need dot(vec, W[k*n+i, :]) = 1
                    # This is one linear equation in W[k*n+i, :]
                    # If vec has at least one nonzero, we can set W[k*n+i, :] = vec / dot(vec,vec)
                    dot_val = np.dot(vec, vec)
                    if dot_val == 0:
                        valid = False
                        break
                    W_sol[k*n+i, :] = vec / dot_val
        
        if not valid:
            continue
        
        # Now check if this W is consistent (each W[k*n+i,:] is determined by multiple (i,j) pairs)
        # For a given (k,i), W[k*n+i,:] must satisfy: for all j, dot(U[i*m+j,:]*V[j*p+k,:], W[k*n+i,:]) = 1
        # So we need to check consistency
        
        consistent = True
        for k in range(p):
            for i in range(n):
                # Collect all constraints on W[k*n+i,:]
                constraints = []
                for j in range(m):
                    vec = U[i*m+j, :] * V[j*p+k, :]
                    if np.any(vec != 0):
                        constraints.append(vec)
                if not constraints:
                    consistent = False
                    break
                # Check if all constraints are consistent with W_sol[k*n+i,:]
                for vec in constraints:
                    if abs(np.dot(vec, W_sol[k*n+i, :]) - 1) > 1e-6:
                        consistent = False
                        break
                if not consistent:
                    break
            if not consistent:
                break
        
        if consistent:
            # Convert to integer/half-integer if possible
            U_int = U.tolist()
            V_int = V.tolist()
            W_int = W_sol.tolist()
            
            # Check if all values are integer or half-integer
            all_valid = True
            for matrix in [U_int, V_int, W_int]:
                for row in matrix:
                    for val in row:
                        if abs(val * 2 - round(val * 2)) > 1e-6:
                            all_valid = False
                            break
                    if not all_valid:
                        break
                if not all_valid:
                    break
            
            if all_valid:
                # Round to nearest half-integer
                U_rounded = [[round(v*2)/2 for v in row] for row in U_int]
                V_rounded = [[round(v*2)/2 for v in row] for row in V_int]
                W_rounded = [[round(v*2)/2 for v in row] for row in W_int]
                
                if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                    return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Start with schoolbook as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best = (schoolbook_rank, U_fb, V_fb, W_fb)
    
    # Try to find lower rank decompositions
    for r in range(21, schoolbook_rank):  # Try ranks 21, 22, 23
        if time.time() > _DEADLINE - 10:
            break
        result = _try_rank(r, n, m, p, timeout=20)
        if result is not None:
            U, V, W = result
            best = (r, U, V, W)
            break
    
    rank, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
