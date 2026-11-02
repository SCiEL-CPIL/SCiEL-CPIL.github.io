"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find minimum rank decomposition
    # Start with schoolbook rank and try to reduce
    
    best_result = None
    deadline = time.time() + 120
    
    # Try ranks from low to high
    for rank in range(35, 43):
        if time.time() > deadline - 10:
            break
        result = _try_rank(n, m, p, rank)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

def _try_rank(n, m, p, rank, max_iters=500):
    """Try to find a rank decomposition using iterative method."""
    np.random.seed(42)
    
    # Initialize with random small values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    for iteration in range(max_iters):
        # Solve for W given U, V
        # For each (i,k): sum_s (sum_j U[i*m+j,s] * V[j*p+k,s]) * W[k*n+i,s] = 1
        W_new = np.zeros((p * n, rank))
        for i in range(n):
            for k in range(p):
                # Coefficients for W[k*n+i, s]
                coeffs = np.zeros(rank)
                for j in range(m):
                    coeffs += U[i * m + j, :] * V[j * p + k, :]
                # coeffs @ W[k*n+i, :] = 1
                # This is a single equation: coeffs dot W_row = 1
                # Underdetermined if rank > 1, so we use least squares
                if np.linalg.norm(coeffs) > 1e-10:
                    W_new[k * n + i, :] = coeffs / np.dot(coeffs, coeffs)
                else:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        # Solve for V given U, W
        V_new = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                coeffs = np.zeros(rank)
                for i in range(n):
                    coeffs += U[i * m + j, :] * W[k * n + i, :]
                if np.linalg.norm(coeffs) > 1e-10:
                    V_new[j * p + k, :] = coeffs / np.dot(coeffs, coeffs)
                else:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Solve for U given V, W
        U_new = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                coeffs = np.zeros(rank)
                for k in range(p):
                    coeffs += V[j * p + k, :] * W[k * n + i, :]
                if np.linalg.norm(coeffs) > 1e-10:
                    U_new[i * m + j, :] = coeffs / np.dot(coeffs, coeffs)
                else:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Check convergence
        U, V, W = U_new, V_new, W_new
    
    # Verify the solution
    if _verify_exact(n, m, p, U, V, W, rank):
        # Convert to exact fractions if possible
        return _to_exact(n, m, p, U, V, W, rank)
    return None

def _verify_exact(n, m, p, U, V, W, rank, tol=1e-6):
    """Check if the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1) > tol:
                    return False
    return True

def _to_exact(n, m, p, U, V, W, rank):
    """Convert numerical solution to exact fractions."""
    U_exact = [[0] * rank for _ in range(n * m)]
    V_exact = [[0] * rank for _ in range(m * p)]
    W_exact = [[0] * rank for _ in range(p * n)]
    
    for i in range(n * m):
        for s in range(rank):
            U_exact[i][s] = _float_to_frac(U[i, s])
    for i in range(m * p):
        for s in range(rank):
            V_exact[i][s] = _float_to_frac(V[i, s])
    for i in range(p * n):
        for s in range(rank):
            W_exact[i][s] = _float_to_frac(W[i, s])
    
    # Verify with exact arithmetic
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for s in range(rank):
                    total += Fraction(U_exact[i * m + j][s]) * Fraction(V_exact[j * p + k][s]) * Fraction(W_exact[k * n + i][s])
                if total != 1:
                    return None
    
    return {"U": U_exact, "V": V_exact, "W": W_exact}

def _float_to_frac(x):
    """Convert float to nearest simple fraction."""
    if abs(x) < 1e-10:
        return 0
    # Try simple fractions
    for den in [1, 2, 3, 4, 6, 8, 12, 16]:
        num = round(x * den)
        if abs(x - num / den) < 1e-9:
            return Fraction(num, den)
    return Fraction(str(x))
# EVOLVE-BLOCK-END
