import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    num_eqs = n * m * p  # 84
    
    def get_schoolbook():
        rank = num_eqs
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def check_half_integer(W_sol, tol=1e-8):
        """Check if W_sol entries are close to half-integers"""
        W_scaled = W_sol * 2
        W_rounded = np.round(W_scaled)
        max_err = np.max(np.abs(W_scaled - W_rounded))
        return max_err < tol, W_rounded / 2.0
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly satisfies the tensor equation"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a better rank using structured search
    # For 3x4x7, known lower bounds suggest rank >= 28 (max of dimensions)
    # Let's try a few ranks with different strategies
    
    best_result = None
    
    # Strategy: Try ranks from 28 upward, but limit total time
    # Use random initialization with values from {-1, 0, 1} and also try small integers
    
    ranks_to_try = [28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 83]
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Number of tries per rank - more for lower ranks
        num_tries = 5 if r < 40 else 3
        
        for attempt in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with integers
            np.random.seed(42 + r * 100 + attempt)  # Reproducible
            U = np.random.choice([-1, 0, 1], size=(dim_U, r))
            V = np.random.choice([-1, 0, 1], size=(dim_V, r))
            
            # Build linear system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # Variables: W[k*n+i, c] for all k,i,c
            # A[eq, var] = U[i*m+j, c] * V[j*p+k, c] where var = (k*n+i)*r + c
            
            A = np.zeros((num_eqs, dim_W * r))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_W = k * n + i
                        for c in range(r):
                            A[eq_idx, row_W * r + c] = U[i * m + j, c] * V[j * p + k, c]
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=1e-12)
            except Exception:
                continue
            
            # Check if solution is close to half-integers
            is_half_int, W_int = check_half_integer(W_flat.reshape(dim_W, r))
            
            if is_half_int:
                # Verify exactness
                if verify_decomposition(U, V, W_int):
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_int.tolist()
                    }
                    return best_result
    
    # Fallback to schoolbook
    return get_schoolbook()
