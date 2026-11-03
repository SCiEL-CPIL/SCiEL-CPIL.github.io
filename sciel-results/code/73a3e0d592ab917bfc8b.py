"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(V[0]) if V else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _try_strassen_extension(n, m, p, deadline):
    """
    Try to construct a low-rank decomposition by extending Strassen's idea.
    For 3x3x3, best known is 23. For 3x3x5, we can try to combine.
    
    Strategy: Use the fact that 5 = 3 + 2. We can think of the 3x5 matrix as
    [A_3x3 | B_3x2]. Then C = A_3x3 * [A_3x3 | B_3x2] = [A*A_3x3 | A*B_3x2].
    
    We can use a rank-23 decomposition for the 3x3x3 part and extend it.
    """
    # This is a placeholder - we'll implement a more systematic search
    pass

def _optimize_decomposition(U_init, V_init, W_init, n, m, p, max_iter=100, lr=0.01, deadline=None):
    """
    Optimize a tensor decomposition using gradient descent on the loss:
    L = sum_{i,j,k} (sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1)^2
    """
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    rank = U.shape[1]
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # s = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    err = s - 1.0
                    for r in range(rank):
                        dU[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                        dV[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                        dW[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    loss += (s - 1.0) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adaptive learning rate
        if it % 10 == 9 and lr > 1e-6:
            lr *= 0.9
        
        # Check convergence
        if best_loss < 1e-12:
            break
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_loss

def _try_rank_search(n, m, p, deadline):
    """
    Search for a low-rank decomposition by trying various ranks.
    """
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound upward
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            
            np_rng = np.random.RandomState(42 + trial)
            U_init = np_rng.randn(n * m, rank) * 0.1
            V_init = np_rng.randn(m * p, rank) * 0.1
            W_init = np_rng.randn(p * n, rank) * 0.1
            
            U_opt, V_opt, W_opt, loss = _optimize_decomposition(
                U_init, V_init, W_init, n, m, p, 
                max_iter=50, lr=0.01, deadline=deadline
            )
            
            if loss < 1e-9:
                if _validate(U_opt, V_opt, W_opt, n, m, p):
                    return {"U": U_opt, "V": V_opt, "W": W_opt}
    
    return best

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Try to find a better decomposition than schoolbook
    result = _try_rank_search(n, m, p, deadline)
    
    return result

# EVOLVE-BLOCK-END
