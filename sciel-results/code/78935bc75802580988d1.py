"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _tensor_error(U, V, W, n, m, p):
    """Compute the tensor reconstruction error."""
    rank = len(U[0])
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                err += abs(val - 1.0)
    return err

def _try_reduce_rank(max_rank, time_budget):
    """Try to find a decomposition with rank < max_rank."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    best_rank = max_rank
    best_result = _schoolbook()
    
    for target_rank in range(max_rank - 1, max(1, max_rank - 20)):
        if time.time() - start_time > time_budget:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() - start_time > time_budget:
                break
                
            # Random initialization with small values
            U = [[random.gauss(0, 0.1) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.gauss(0, 0.1) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.gauss(0, 0.1) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Gradient descent on the tensor decomposition
            lr = 0.01
            for step in range(500):
                if time.time() - start_time > time_budget:
                    break
                    
                # Compute gradients
                dU = [[0.0]*target_rank for _ in range(n*m)]
                dV = [[0.0]*target_rank for _ in range(m*p)]
                dW = [[0.0]*target_rank for _ in range(p*n)]
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(target_rank):
                                u_val = U[i*m+j][c]
                                v_val = V[j*p+k][c]
                                w_val = W[k*n+i][c]
                                val = u_val * v_val * w_val
                                err = val - 1.0
                                dU[i*m+j][c] += err * v_val * w_val
                                dV[j*p+k][c] += err * u_val * w_val
                                dW[k*n+i][c] += err * u_val * v_val
                
                # Update
                for idx in range(n*m):
                    for c in range(target_rank):
                        U[idx][c] -= lr * dU[idx][c]
                for idx in range(m*p):
                    for c in range(target_rank):
                        V[idx][c] -= lr * dV[idx][c]
                for idx in range(p*n):
                    for c in range(target_rank):
                        W[idx][c] -= lr * dW[idx][c]
            
            err = _tensor_error(U, V, W, n, m, p)
            if err < 1e-6:
                # Found a valid decomposition
                # Convert to fractions
                U_frac = [[Fraction(round(U[i][c], 6)).limit_denominator(1000) for c in range(target_rank)] for i in range(n*m)]
                V_frac = [[Fraction(round(V[i][c], 6)).limit_denominator(1000) for c in range(target_rank)] for i in range(m*p)]
                W_frac = [[Fraction(round(W[i][c], 6)).limit_denominator(1000) for c in range(target_rank)] for i in range(p*n)]
                
                # Verify
                if _tensor_error(U_frac, V_frac, W_frac, n, m, p) < 1e-9:
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
                else:
                    # Use float version
                    return {"U": U, "V": V, "W": W}
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    time_budget = 120  # seconds
    
    start_time = time.time()
    
    # Try to find a better decomposition
    result = _try_reduce_rank(schoolbook_rank, time_budget)
    
    return result
# EVOLVE-BLOCK-END
