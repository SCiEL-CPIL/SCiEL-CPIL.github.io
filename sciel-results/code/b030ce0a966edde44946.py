"""Matrix multiplication 3x5x7 - rank minimization via structured algebraic search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if val != 1:
                    return False
    return True

def _to_half_int(x):
    """Convert to nearest half-integer if close enough."""
    h = round(x * 2) / 2.0
    return h if abs(x - h) < 1e-8 else None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p = 105."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_solve_for_U(V, W, r):
        """Given V and W of rank r, try to solve for U with half-integer entries."""
        U = [[0.0]*r for _ in range(nm)]
        for i in range(n):
            # Build system: for each (j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            A = np.zeros((m*p, m*r))
            b = np.ones(m*p)
            for j in range(m):
                for k in range(p):
                    eq = j * p + k
                    for c in range(r):
                        A[eq, j*r + c] = V[j*p+k][c] * W[k*n+i][c]
            
            try:
                sol, res, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-12)
            except np.linalg.LinAlgError:
                return None
            
            # Check if solution is close to half-integers
            sol_h = np.array([_to_half_int(x) for x in sol])
            if np.any(sol_h == None):
                return None
            sol_h = np.array(sol_h, dtype=np.float64)
            
            if np.max(np.abs(sol - sol_h)) > 1e-7:
                return None
            
            # Verify this i-block
            for j in range(m):
                for k in range(p):
                    val = sum(sol_h[j*r+c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                    if abs(val - 1.0) > 1e-7:
                        return None
            
            for j in range(m):
                for c in range(r):
                    U[i*m+j][c] = sol_h[j*r+c]
        
        return U
    
    def try_rank(r, num_attempts=200):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        strategies = [
            lambda: [-1, 0, 1],
            lambda: [-2, -1, 0, 1, 2],
            lambda: [0, 1],
            lambda: [-1, 1],
            lambda: [-3, -2, -1, 0, 1, 2, 3],
        ]
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            random.seed(42 + r * 1000 + attempt)
            strat_idx = attempt % len(strategies)
            vals = strategies[strat_idx]()
            
            V = [[random.choice(vals) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(pn)]
            
            U = try_solve_for_U(V, W, r)
            if U is None:
                continue
            
            # Full verification
            if _verify_np(U, V, W, n, m, p):
                # Convert to exact fractions
                U_f = [[Fraction(U[i][j]).limit_denominator(100) for j in range(r)] for i in range(nm)]
                V_f = [[Fraction(V[i][j]).limit_denominator(100) for j in range(r)] for i in range(mp)]
                W_f = [[Fraction(W[i][j]).limit_denominator(100) for j in range(r)] for i in range(pn)]
                
                if _verify_exact(U_f, V_f, W_f, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_f],
                        "V": [[float(x) for x in row] for row in V_f],
                        "W": [[float(x) for x in row] for row in W_f]
                    }
        
        return None
    
    # Try ranks from just below schoolbook downward
    # Theoretical lower bounds: rank >= max(n*m, m*p, p*n) = 35
    # Known results suggest rank around 60-80 for these dimensions
    for r in range(best_rank - 1, 34, -1):
        if time.time() > deadline:
            break
        
        # Use fewer attempts for larger ranks (easier to find), more for smaller
        if r > 90:
            attempts = 50
        elif r > 70:
            attempts = 100
        else:
            attempts = 200
        
        result = try_rank(r, attempts)
        if result is not None:
            best_result = result
            best_rank = r
            # Found a valid solution; try to go lower
            continue
    
    # Final safety check
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
