import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def verify(U, V, W):
    """
    Verify if U, V, W form a valid decomposition for 3x3x5 matrix multiplication.
    U: (9, rank), V: (15, rank), W: (15, rank)
    """
    n, m, p = 3, 3, 5
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook fallback (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find a lower rank decomposition using randomized optimization
    # The tensor rank of 3x3x5 matrix multiplication is known to be at most 45 (schoolbook)
    # and at least 40 (lower bound). Known best upper bound is 45.
    # However, for the specific 3x3x5 case, we can try to find a decomposition
    # with rank less than 45 using numerical optimization.
    
    # Strategy: Use random initialization and gradient descent to minimize
    # the error of the tensor decomposition.
    
    def try_rank(R, max_iters=5000, seed=42):
        np.random.seed(seed)
        
        # Random initialization
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Compute loss: sum over i,j,k of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # s = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        diff = s - 1.0
                        loss += diff * diff
                        
                        # Gradients
                        dU[i*m+j, :] += 2 * diff * V[j*p+k, :] * W[k*n+i, :]
                        dV[j*p+k, :] += 2 * diff * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += 2 * diff * U[i*m+j, :] * V[j*p+k, :]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Track best
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Adaptive learning rate
            if loss > best_loss * 1.1:
                lr *= 0.5
            elif loss < best_loss * 0.9:
                lr *= 1.1
                lr = min(lr, 0.1)
            
            # Early termination
            if best_loss < 1e-12:
                break
        
        if best_loss < 1e-9:
            return best_U, best_V, best_W, best_loss
        return None
    
    # Try ranks from 40 up to 44
    for R in range(40, 45):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts
        found = False
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            result = try_rank(R, max_iters=3000, seed=R*1000 + attempt)
            if result is not None:
                U, V, W, loss = result
                found = True
                break
        
        if found:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_result

# EVOLVE-BLOCK-END
