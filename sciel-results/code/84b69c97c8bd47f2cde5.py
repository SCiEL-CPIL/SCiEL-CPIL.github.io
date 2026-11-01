"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_nice(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    # Check integer
    nearest_int = round(x)
    if abs(x - nearest_int) < tol:
        return True
    # Check half-integer
    nearest_half = round(2*x) / 2.0
    if abs(x - nearest_half) < tol:
        return True
    return False

def _all_nice(matrix, tol=1e-9):
    """Check if all entries in matrix are nice (integer or half-integer)."""
    for row in matrix:
        for val in row:
            if not _is_nice(val, tol):
                return False
    return True

def _try_rank_search(n, m, p, target_rank, deadline, max_trials=500):
    """Try to find a decomposition of given rank by random search."""
    U_dim = n * m  # 10
    V_dim = m * p  # 30
    W_dim = p * n  # 12
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        # Random initialization with small values
        np.random.seed(trial)
        U = np.random.randn(U_dim, target_rank) * 0.5
        V = np.random.randn(V_dim, target_rank) * 0.5
        
        # Fix U and V, solve for W
        # For each (i,k): sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a linear system in W
        
        # Build the linear system: for each (i,k), we have one equation
        # The unknowns are W[k*n+i, r] for r in 0..target_rank-1
        # Total unknowns: W_dim * target_rank = 12 * target_rank
        # Total equations: n * p = 12
        
        # Actually, we have 12 equations (one per (i,k) pair) and 12*target_rank unknowns
        # This is underdetermined for target_rank > 1
        
        # Let's set up: for each (i,k), equation is:
        # sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
        
        # So for each (i,k), we have: sum_r W[k*n+i, r] * coeff[i,k,r] = 1
        # where coeff[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]
        
        # This gives us 12 equations, but they are coupled through W.
        # Actually no - for each (i,k), the equation only involves W[k*n+i, :]
        # So we have 12 independent equations, each involving target_rank unknowns.
        
        # For each (i,k): sum_r W[k*n+i, r] * coeff[i,k,r] = 1
        # This is a system of 12 equations in 12*target_rank unknowns, but decoupled!
        # Each (i,k) gives one equation in target_rank unknowns.
        # So for each (i,k), we need to find W[k*n+i, :] such that dot(W[k*n+i,:], coeff[i,k,:]) = 1
        
        # This is possible if coeff[i,k,:] is not all zeros.
        # We can find a particular solution and add null space.
        
        # Let's find a particular solution for each (i,k)
        W = np.zeros((W_dim, target_rank))
        valid = True
        for i in range(n):
            for k in range(p):
                coeff = np.zeros(target_rank)
                for j in range(m):
                    coeff += U[i*m+j, :] * V[j*p+k, :]
                
                if np.allclose(coeff, 0):
                    valid = False
                    break
                
                # Find w such that w @ coeff = 1
                # Particular solution: w = coeff / (coeff @ coeff)
                norm_sq = np.dot(coeff, coeff)
                if norm_sq < 1e-12:
                    valid = False
                    break
                W[k*n+i, :] = coeff / norm_sq
        
        if not valid:
            continue
        
        # Now verify the full decomposition
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-8):
            # Check if entries are nice
            if _all_nice(U.tolist()) and _all_nice(V.tolist()) and _all_nice(W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        # Even if not nice, keep track of best (closest to nice)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = 2, 5, 6
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But actually for matrix multiplication, the rank is at least max(n, m, p) * something
    # Known: rank >= n*m + m*p + p*n - n - m - p + 1? Not sure.
    # Let's try ranks from 55 down to 35
    
    for r in range(best_rank - 1, 34, -1):
        if time.time() > deadline - 20:
            break
        result = _try_rank_search(n, m, p, r, deadline - 10, max_trials=200)
        if result is not None:
            U, V, W = result
            best_rank = r
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
