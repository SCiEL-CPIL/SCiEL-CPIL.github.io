import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

def _verify(M, U, V, W):
    """Verify exact decomposition of matrix multiplication tensor.
    For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = 3, 3, 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # safety margin
    
    # Schoolbook decomposition
    U_school = np.zeros((9, 27))
    V_school = np.zeros((9, 27))
    W_school = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_school[i*3+j, col] = 1
                V_school[j*3+k, col] = 1
                W_school[k*3+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = 27
    
    # Try to reduce rank from 27 down to 23
    for target_rank in range(26, 22, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations for this rank
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with schoolbook, then randomly project to lower rank
            # Or start with random small values
            np.random.seed(attempt + 1000)
            U = np.random.randn(9, target_rank) * 0.5
            V = np.random.randn(9, target_rank) * 0.5
            W = np.random.randn(9, target_rank) * 0.5
            
            # Gradient descent
            lr = 0.01
            prev_err = np.inf
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Compute error
                err = 0.0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                            err += (s - 1.0) ** 2
                
                if err < 1e-12:
                    found = True
                    break
                
                # Compute gradients
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                            diff = s - 1.0
                            for r in range(target_rank):
                                U_grad[i*3+j, r] += 2 * diff * V[j*3+k, r] * W[k*3+i, r]
                                V_grad[j*3+k, r] += 2 * diff * U[i*3+j, r] * W[k*3+i, r]
                                W_grad[k*3+i, r] += 2 * diff * U[i*3+j, r] * V[j*3+k, r]
                
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                if abs(prev_err - err) < 1e-15:
                    # Stagnant, try a random kick
                    U += np.random.randn(9, target_rank) * 0.01
                    V += np.random.randn(9, target_rank) * 0.01
                    W += np.random.randn(9, target_rank) * 0.01
                    lr *= 0.5
                prev_err = err
            
            if found:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if found:
            # Try to reduce further in next iteration
            continue
        else:
            # If we failed at this rank, we might still try lower ranks? 
            # Usually if 26 fails, 25 might fail too. But let's try one more time for 23 specifically.
            pass
    
    # Final check
    if not _verify(None, best_U, best_V, best_W):
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = 27
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
