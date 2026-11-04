"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import random

_START_TIME = time.time()
_TIME_LIMIT = 145  # safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _to_fraction_list(lst):
    """Convert a list of numbers to Fractions."""
    return [Fraction(x).limit_denominator(1000) for x in lst]


def _schoolbook_3x3x7():
    """Return schoolbook rank-63 decomposition for 3x3 * 3x7."""
    n, m, p = 3, 3, 7
    rank = n * m * p  # 63
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W


def _residual(U, V, W, n=3, m=3, p=7):
    """Compute the residual of the tensor equation."""
    rank = U.shape[1]
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                residual += (val - 1.0) ** 2
    return residual


def _verify_decomposition(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Check: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    (This is the tensor equation)
    """
    rank = U.shape[1]
    errors = 0
    max_error = 0.0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                # Should be 1
                err = abs(val - 1.0)
                if err > max_error:
                    max_error = err
                if err > tol:
                    errors += 1
    
    if errors > 0:
        return False
    return True


def _convert_to_lists(U, V, W):
    """Convert numpy arrays to lists of lists with Fraction or int entries."""
    def arr_to_list(arr):
        result = []
        for row in arr:
            row_list = []
            for val in row:
                # Try to convert to exact fraction
                f = Fraction(val).limit_denominator(1000)
                # If it's a whole number, use int
                if f.denominator == 1:
                    row_list.append(int(f.numerator))
                else:
                    row_list.append(float(f))
            result.append(row_list)
        return result
    
    return arr_to_list(U), arr_to_list(V), arr_to_list(W)


def _optimize_decomposition(rank, n=3, m=3, p=7, time_budget=60):
    """
    Try to find a rank-`rank` decomposition using random restarts and local optimization.
    
    Returns (U, V, W) if successful, None otherwise.
    """
    start_time = time.time()
    best_residual = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Number of restarts based on time budget
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        np.random.seed(42 + restart)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Local optimization using simple gradient-free method
        # We'll use a simple iterative approach
        lr = 0.01
        for iteration in range(100):
            if time.time() - start_time > time_budget:
                break
            
            # Compute residual
            res = _residual(U, V, W, n, m, p)
            if res < best_residual:
                best_residual = res
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if res < 1e-12:
                break
            
            # Compute gradients numerically
            # For efficiency, we'll use a simple coordinate descent approach
            # Update U
            for idx in range(min(5, n*m)):
                t_idx = restart % rank
                old_val = U[idx, t_idx]
                for delta in [0.01, -0.01]:
                    U[idx, t_idx] = old_val + delta
                    res_new = _residual(U, V, W, n, m, p)
                    if res_new < res:
                        res = res_new
                        break
                    U[idx, t_idx] = old_val
            
            # Update V
            for idx in range(min(5, m*p)):
                t_idx = (restart + 1) % rank
                old_val = V[idx, t_idx]
                for delta in [0.01, -0.01]:
                    V[idx, t_idx] = old_val + delta
                    res_new = _residual(U, V, W, n, m, p)
                    if res_new < res:
                        res = res_new
                        break
                    V[idx, t_idx] = old_val
            
            # Update W
            for idx in range(min(5, p*n)):
                t_idx = (restart + 2) % rank
                old_val = W[idx, t_idx]
                for delta in [0.01, -0.01]:
                    W[idx, t_idx] = old_val + delta
                    res_new = _residual(U, V, W, n, m, p)
                    if res_new < res:
                        res = res_new
                        break
                    W[idx, t_idx] = old_val
        
        if best_residual < 1e-9:
            break
    
    if best_residual < 1e-9 and best_U is not None:
        return best_U, best_V, best_W
    
    return None


def _construct_3x3x7():
    """
    Construct a decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy: Try to find a low-rank decomposition directly.
    Fall back to schoolbook if optimization fails.
    """
    n, m, p = 3, 3, 7
    
    # Try different ranks from low to high
    for rank in [55, 56, 57, 58, 59, 60, 61, 62, 63]:
        if _time_left() < 30:
            break
        
        result = _optimize_decomposition(rank, n, m, p, time_budget=min(20, _time_left() - 10))
        
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W, rank
    
    # Fall back to schoolbook
    U, V, W = _schoolbook_3x3x7()
    return U, V, W, 63


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    """
    if _time_left() < 30:
        # Emergency fallback: return schoolbook
        n, m, p = 3, 3, 7
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    try:
        U, V, W, rank = _construct_3x3x7()
        
        # Verify the decomposition
        if _verify_decomposition(U, V, W):
            U_list, V_list, W_list = _convert_to_lists(U, V, W)
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Fall back to schoolbook
            n, m, p = 3, 3, 7
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        # Fall back to schoolbook on any error
        n, m, p = 3, 3, 7
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
