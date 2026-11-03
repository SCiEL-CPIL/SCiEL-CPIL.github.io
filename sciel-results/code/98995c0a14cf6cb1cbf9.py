"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction
import random
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 145  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_school
    
    # Try to find lower rank solutions
    # Lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    for rank in range(20, min(40, rank_school)):  # Try ranks 20-39
        if time.time() > deadline:
            break
        
        success = False
        # Multiple random restarts
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Use adaptive learning rate gradient descent
            lr = 0.01
            best_err = float('inf')
            
            for step in range(300):
                if time.time() > deadline:
                    break
                
                # Compute error and gradients
                err = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            e = val - 1.0
                            err += e * e
                            grad_U[i*m+j, :] += 2 * e * V[j*p+k, :] * W[k*n+i, :]
                            grad_V[j*p+k, :] += 2 * e * U[i*m+j, :] * W[k*n+i, :]
                            grad_W[k*n+i, :] += 2 * e * U[i*m+j, :] * V[j*p+k, :]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                max_err = max(
                    abs(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) - 1.0)
                    for i in range(n) for j in range(m) for k in range(p)
                )
                
                if max_err < 1e-9:
                    success = True
                    break
                
                # Reduce learning rate if stuck
                if err > best_err:
                    lr *= 0.5
                else:
                    best_err = err
            
            if success:
                best_U = U
                best_V = V
                best_W = W
                best_rank = rank
                break
        
        if success:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
