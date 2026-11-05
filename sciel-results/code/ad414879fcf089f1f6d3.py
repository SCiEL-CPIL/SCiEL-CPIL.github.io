import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = np.dot(U_arr[idx_u, :], V_arr[idx_v, :] * W_arr[idx_w, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_rank28(n, m, p):
    """Known rank-28 decomposition for 2x3x5 matrix multiplication."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, all k (10 terms) - schoolbook for 2x5
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms) - schoolbook for 2x2x1
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    # M1: (A00+A11)(B00+B11)
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    
    # M2: (A10+A11)B00
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1
    r += 1
    
    # M3: A00(B11-B00)
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    
    # M4: A11(B00-B10)
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1
    r += 1
    
    # M5: (A00+A01)B11
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1
    r += 1
    
    # M6: (A10-A00)(B00+B01)
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7: (A01-A11)(B10+B11)
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    # M1: (A00+A11)(B02+B13)
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1
    r += 1
    
    # M2: (A10+A11)B02
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1
    r += 1
    
    # M3: A00(B13-B02)
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1
    r += 1
    
    # M4: A11(B02-B12)
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1
    r += 1
    
    # M5: (A00+A01)B13
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1
    r += 1
    
    # M6: (A10-A00)(B02+B03)
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7: (A01-A11)(B12+B13)
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_27(n, m, p, time_limit):
    """
    Try to find a rank-27 decomposition.
    
    Strategy: The key insight is that we can potentially save one term by
    finding a more efficient decomposition of the 2x2x5 subproblem.
    
    For 2x2x5, we need rank 17 (instead of 18 = 7+7+4).
    
    Approach: Use a global optimization over all 27 terms simultaneously,
    starting from a perturbed rank-28 solution and trying to drive one
    term to zero.
    """
    start_time = time.time()
    
    # Start from the rank-28 solution and try to optimize
    base = _strassen_rank28(n, m, p)
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Try different strategies to achieve rank 27
    
    # Strategy 1: Merge some terms from the Strassen blocks
    # The two Strassen blocks (k=0,1 and k=2,3) each use 7 terms.
    # Maybe we can share some structure between them.
    
    # Actually, let's try a direct numerical optimization approach.
    # We'll parameterize 27 terms and try to find a solution.
    
    # For efficiency, let's try a specific structured approach:
    # Use 10 terms for j=2 (schoolbook)
    # Use 17 terms for j=0,1 (all k=0..4)
    
    # For the 17 terms, try a combination of:
    # - 7 Strassen-like terms for k=0,1
    # - 7 Strassen-like terms for k=2,3  
    # - 3 terms for k=4 (instead of 4)
    
    # To save one term in the k=4 part, we need to express the 2x2x1
    # multiplication with 3 terms instead of 4. But 2x2x1 has rank 4
    # (it's just 4 independent products), so we can't do it with 3.
    
    # Alternative: try to save a term by making the Strassen blocks
    # more efficient when combined with the k=4 part.
    
    # Let's try a different split:
    # j=2: 10 terms
    # j=0,1,k=0..4: 17 terms using a clever decomposition
    
    # For 2x2x5, known results suggest rank might be 17 or 18.
    # Let's try to find rank 17 for 2x2x5.
    
    # Use optimization: start from a 18-term solution and try to
    # drive one coefficient to zero.
    
    if time.time() - start_time > time_limit:
        return None
    
    # Actually, let's try a more direct approach:
    # Try to find a rank-27 solution by solving the system directly.
    # We have 30 equations (2*3*5) and 27*(6+15+10) = 27*31 = 837 unknowns.
    # This is very underdetermined, so solutions should exist if rank 27 is possible.
    
    # Let's try a randomized approach with local optimization.
    
    from scipy.optimize import minimize
    
    def _residual(params):
        """Compute the residual for a given parameterization of rank-27 solution."""
        nm, mp, pn = n*m, m*p, p*n
        total = rank * (nm + mp + pn)
        
        if len(params) != total:
            return 1e10
        
        U_flat = params[:rank*nm].reshape(rank, nm)
        V_flat = params[rank*nm:rank*(nm+mp)].reshape(rank, mp)
        W_flat = params[rank*(nm+mp):].reshape(rank, pn)
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_flat[r, idx_u] * V_flat[r, idx_v] * W_flat[r, idx_w]
                    residual += (s - 1.0) ** 2
        return residual
    
    # Try multiple random initializations
    best_result = None
    best_residual = np.inf
    
    num_trials = min(5, max(1, int((time_limit - (time.time() - start_time)) / 10)))
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit - 5:
            break
        
        np.random.seed(trial * 12345 + 67890)
        
        # Initialize with a structured guess based on the rank-28 solution
        # but with one term removed and adjusted
        params = np.random.randn(rank * (nm + mp + pn)) * 0.1
        
        try:
            result = minimize(_residual, params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_residual:
                best_residual = result.fun
                best_result = result.x
        except Exception:
            continue
    
    if best_result is None or best_residual > 1e-8:
        return None
    
    # Extract the solution
    nm, mp, pn = n*m, m*p, p*n
    U_flat = best_result[:rank*nm].reshape(rank, nm)
    V_flat = best_result[rank*nm:rank*(nm+mp)].reshape(rank, mp)
    W_flat = best_result[rank*(nm+mp):].reshape(rank, pn)
    
    # Check if entries are close to half-integers
    all_vals = np.concatenate([U_flat.flatten(), V_flat.flatten(), W_flat.flatten()])
    for val in all_vals:
        nearest_half = np.round(val * 2) / 2
        if abs(val - nearest_half) > 1e-6:
            return None
    
    # Convert to fractions
    U = [[Fraction(int(np.round(U_flat[r, i*m+j] * 2)), 2) for j in range(m) for i in range(n)] for r in range(rank)]
    # Wait, we need U[i*m+j, r], not U[r, i*m+j]
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for i in range(nm):
        for r in range(rank):
            U[i][r] = Fraction(int(np.round(U_flat[r, i] * 2)), 2)
    for j in range(mp):
        for r in range(rank):
            V[j][r] = Fraction(int(np.round(V_flat[r, j] * 2)), 2)
    for k in range(pn):
        for r in range(rank):
            W[k][r] = Fraction(int(np.round(W_flat[r, k] * 2)), 2)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s margin for the 150s limit
    
    # First, get the known rank-28 solution as fallback
    fallback = _strassen_rank28(n, m, p)
    
    # Try to find rank-27 solution
    time_for_27 = min(60, deadline - (time.time() - start_time) - 10)
    if time_for_27 > 10:
        result_27 = _try_rank_27(n, m, p, time_for_27)
        if result_27 is not None:
            return result_27
    
    # Return the rank-28 solution
    return fallback

