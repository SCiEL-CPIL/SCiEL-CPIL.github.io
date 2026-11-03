"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Schoolbook solution as fallback
    best = _schoolbook_solution(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank solutions
    # Lower bound: each rank-1 term can cover at most min(n*m, m*p, p*n) entries
    # But realistically for 4x4*4x9, try ranks from 50 to 143
    for target_rank in range(50, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            sol = _try_rank(n, m, p, target_rank, restart)
            if sol is not None:
                if target_rank < best_rank:
                    best = sol
                    best_rank = target_rank
                    break  # found a solution at this rank, move to lower
    
    return best


def _schoolbook_solution(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                if total != 1:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast numerical verification."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
                if abs(total - 1.0) > tol:
                    return False
    return True


def _try_rank(n, m, p, target_rank, seed):
    """Try to find a valid decomposition at the given rank."""
    np.random.seed(seed * 1000 + target_rank)
    
    # Initialize randomly
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Alternating least squares iterations
    for it in range(30):
        # Update U
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                A = np.zeros((p, target_rank))
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    A[k, :] = V[v_idx, :] * W[w_idx, :]
                b = np.ones(p)
                try:
                    U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                A = np.zeros((n, target_rank))
                for i in range(n):
                    u_idx = i * m + j
                    w_idx = k * n + i
                    A[i, :] = U[u_idx, :] * W[w_idx, :]
                b = np.ones(n)
                try:
                    V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                A = np.zeros((m, target_rank))
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    A[j, :] = U[u_idx, :] * V[v_idx, :]
                b = np.ones(m)
                try:
                    W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        if _verify_fast(U, V, W, n, m, p, tol=1e-8):
            # Convert to fractions for exact verification
            U_frac = [[_to_fraction(x) for x in U[r]] for r in range(n * m)]
            V_frac = [[_to_fraction(x) for x in V[r]] for r in range(m * p)]
            W_frac = [[_to_fraction(x) for x in W[r]] for r in range(p * n)]
            
            if _verify(U_frac, V_frac, W_frac, n, m, p):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None


def _to_fraction(x):
    """Convert a float to a Fraction with reasonable precision."""
    if abs(x) < 1e-12:
        return Fraction(0)
    # Try to find a simple fraction
    for denom in range(1, 100):
        num = round(x * denom)
        if abs(x - num / denom) < 1e-10:
            return Fraction(num, denom)
    # Fallback: use the float value as a fraction
    return Fraction(str(x))
# EVOLVE-BLOCK-END
