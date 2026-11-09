"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(r):
                    u_val = U[i*m+j, c]
                    v_val = V[j*p+k, c]
                    w_val = W[k*n+i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += Fraction(int(u_val * 2), 2) * Fraction(int(v_val * 2), 2) * Fraction(int(w_val * 2), 2)
                if val != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Quick float verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _trivial_solution(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_fixed_uv_solve_w(U, V, n, m, p, r):
    """Given fixed U and V, try to solve for W using least squares."""
    W = np.zeros((p * n, r))
    
    for i in range(n):
        for k in range(p):
            A_mat = np.zeros((m, r))
            for j in range(m):
                A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            # Check if system is solvable
            b = np.ones(m)
            try:
                # Use np.linalg.lstsq
                w_vec, residuals, rank, s = np.linalg.lstsq(A_mat, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-8:
                return None
            
            # Check if solution is close to half-integers
            w_scaled = w_vec * 2
            w_rounded = np.round(w_scaled)
            if np.linalg.norm(A_mat @ (w_rounded / 2) - b) > 1e-8:
                return None
            
            W[k * n + i, :] = w_rounded / 2
    
    # Final verification
    if _verify_float(U, V, W, n, m, p):
        return W
    return None

def _random_search(n, m, p, r, max_attempts, seed_base, deadline):
    """Random search for rank-r decomposition."""
    np.random.seed(seed_base)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random U and V with small integer/half-integer entries
        # Use {-1, 0, 1} with some probability, or {-0.5, 0, 0.5, 1}
        scale = 1.0
        U = np.random.randint(-1, 2, size=(n * m, r)) * scale
        V = np.random.randint(-1, 2, size=(m * p, r)) * scale
        
        # Ensure no row is all zeros (would make system unsolvable)
        U_nonzero = np.any(U != 0, axis=1)
        V_nonzero = np.any(V != 0, axis=1)
        if not np.all(U_nonzero) or not np.all(V_nonzero):
            continue
        
        W = _try_fixed_uv_solve_w(U, V, n, m, p, r)
        if W is not None:
            # Verify with exact arithmetic
            if _verify_exact(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _structured_search(n, m, p, r, deadline):
    """Try structured patterns for U and V."""
    # Try block-diagonal structures
    # For now, just do random search with different distributions
    
    results = []
    
    # Try with entries from {-1, 0, 1}
    res = _random_search(n, m, p, r, 500, 0, deadline)
    if res:
        return res
    
    # Try with entries from {-2, -1, 0, 1, 2}
    np.random.seed(1000)
    for attempt in range(200):
        if time.time() > deadline:
            break
        U = np.random.randint(-2, 3, size=(n * m, r))
        V = np.random.randint(-2, 3, size=(m * p, r))
        U_nonzero = np.any(U != 0, axis=1)
        V_nonzero = np.any(V != 0, axis=1)
        if not np.all(U_nonzero) or not np.all(V_nonzero):
            continue
        W = _try_fixed_uv_solve_w(U, V, n, m, p, r)
        if W is not None:
            if _verify_exact(U, V, W, n, m, p):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin (evaluator kills at 150s)
    
    # Trivial solution fallback
    U_triv, V_triv, W_triv = _trivial_solution(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Lower bound for rank
    lb = max(n * m, m * p, p * n)  # = 35 for 3x5x7
    
    # Also, rank must be at least the dimension of the matrix multiplication tensor
    # The tensor has rank at least max(n*m, m*p, p*n) by matrix multiplication properties
    
    # Try ranks from lower bound upwards
    # We'll try a few ranks and see what we can find
    ranks_to_try = []
    
    # Start from lower bound, go up
    # But we know trivial is 105, so we can try to beat that
    # Let's try a range of ranks
    
    # First, try to find any solution better than trivial
    # Try ranks from lb to lb + 20, then jump
    for r in range(lb, min(lb + 30, best_rank)):
        if time.time() > deadline:
            break
        
        # Time budget per rank
        rank_deadline = min(deadline, time.time() + 10)
        
        res = _structured_search(n, m, p, r, rank_deadline)
        if res is not None:
            U, V, W = res
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break  # Found a solution at this rank, try to go lower
    
    # If we found a solution, we can't easily go lower since we started from lb
    # But lb might not be tight, so let's also try a few specific ranks
    
    # Try some known good ranks for small matrix multiplication
    # For 3x5x7, let's try ranks around 40-60
    if best_rank > 60:
        for r in [40, 45, 50, 55, 60]:
            if time.time() > deadline:
                break
            if r >= best_rank:
                continue
            
            rank_deadline = min(deadline, time.time() + 8)
            res = _random_search(n, m, p, r, 300, r * 100, rank_deadline)
            if res is not None:
                U, V, W = res
                best_rank = r
                best_U, best_V, best_W = U, V, W
    
    # Final attempt: try to improve with more random search at the best rank found
    if best_rank > lb and time.time() < deadline - 5:
        # Try to find a solution at best_rank - 1
        r_try = best_rank - 1
        if r_try >= lb:
            rank_deadline = min(deadline, time.time() + 5)
            res = _random_search(n, m, p, r_try, 100, best_rank * 1000, rank_deadline)
            if res is not None:
                U, V, W = res
                best_rank = r_try
                best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
