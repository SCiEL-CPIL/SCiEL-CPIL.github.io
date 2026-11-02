"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    r = len(U[0])
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _loss_and_grad(U, V, W, n, m, p):
    """Compute loss and gradients for the decomposition."""
    r = len(U[0])
    loss = 0.0
    dU = [[0.0]*r for _ in range(n*m)]
    dV = [[0.0]*r for _ in range(m*p)]
    dW = [[0.0]*r for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Target is 1
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                err = s - 1.0
                loss += err * err
                for c in range(r):
                    dU[i*m+j][c] += 2.0 * err * V[j*p+k][c] * W[k*n+i][c]
                    dV[j*p+k][c] += 2.0 * err * U[i*m+j][c] * W[k*n+i][c]
                    dW[k*n+i][c] += 2.0 * err * U[i*m+j][c] * V[j*p+k][c]
    
    return loss, dU, dV, dW

def _optimize_decomposition(U, V, W, n, m, p, max_iter=1000, lr=0.01, tol=1e-10):
    """Optimize the decomposition using gradient descent."""
    r = len(U[0])
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    
    best_loss = float('inf')
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    for it in range(max_iter):
        loss, dU, dV, dW = _loss_and_grad(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        
        if loss < tol:
            break
        
        # Gradient descent step
        for i in range(n*m):
            for c in range(r):
                U[i][c] -= lr * dU[i][c]
        for i in range(m*p):
            for c in range(r):
                V[i][c] -= lr * dV[i][c]
        for i in range(p*n):
            for c in range(r):
                W[i][c] -= lr * dW[i][c]
    
    return best_U, best_V, best_W, best_loss

def _try_find_decomposition(n, m, p, rank, deadline, n_attempts=5):
    """Try to find a decomposition of given rank."""
    best_U = None
    best_V = None
    best_W = None
    best_loss = float('inf')
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)]
        
        # Optimize
        opt_U, opt_V, opt_W, loss = _optimize_decomposition(
            U, V, W, n, m, p, 
            max_iter=500, 
            lr=0.05, 
            tol=1e-8
        )
        
        if loss < best_loss:
            best_loss = loss
            best_U = opt_U
            best_V = opt_V
            best_W = opt_W
        
        # Check if we found an exact solution
        if best_loss < 1e-6:
            # Try to round to integers/half-integers
            U_round = [[round(x*2)/2 for x in row] for row in best_U]
            V_round = [[round(x*2)/2 for x in row] for row in best_V]
            W_round = [[round(x*2)/2 for x in row] for row in best_W]
            
            if _verify_decomposition(U_round, V_round, W_round, n, m, p):
                return U_round, V_round, W_round
    
    if best_U is not None and best_loss < 1e-6:
        return best_U, best_V, best_W
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate the schoolbook solution as a fallback
    U_school = [[0]*rank_schoolbook for _ in range(n*m)]
    V_school = [[0]*rank_schoolbook for _ in range(m*p)]
    W_school = [[0]*rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition
    # Start from rank 35 down to 25
    for r in range(rank_schoolbook - 1, 24, -1):
        if time.time() > deadline:
            break
        
        U, V, W = _try_find_decomposition(n, m, p, r, deadline, n_attempts=3)
        
        if U is not None:
            best_U = U
            best_V = V
            best_W = W
            best_rank = r
            # Don't break - try to find even lower rank
            # But we found a valid one, so we have a good result
    
    # Final verification
    if best_rank < rank_schoolbook:
        if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
            # Fall back to schoolbook if verification fails
            best_U = U_school
            best_V = V_school
            best_W = W_school
            best_rank = rank_schoolbook
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
