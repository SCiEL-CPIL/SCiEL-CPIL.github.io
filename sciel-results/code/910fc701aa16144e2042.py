import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Checks if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    return True

def tensor_residual(U, V, W, n, m, p):
    """Compute the residual tensor T - 1."""
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return T - np.ones((n, m, p))

def objective(U, V, W, n, m, p):
    """Compute the sum of squares of the residual."""
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    residual = T - np.ones((n, m, p))
    return np.sum(residual ** 2)

def gradient_step(U, V, W, n, m, p, lr=0.01):
    """
    Perform one gradient descent step on the rank-R decomposition.
    Uses the structure of the matrix multiplication tensor.
    """
    global R
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    residual = T - np.ones((n, m, p))
    
    # Gradients
    # dT/dU[i,j,r] = sum_k V[j,k,r] * W[k,i,r]
    # dT/dV[j,k,r] = sum_i U[i,j,r] * W[k,i,r]  
    # dT/dW[k,i,r] = sum_j U[i,j,r] * V[j,k,r]
    
    dU = np.einsum('jkr,kir,ijk->ijr', V_3d, W_3d, residual)
    dV = np.einsum('ijr,kir,ijk->jkr', U_3d, W_3d, residual)
    dW = np.einsum('ijr,jkr,ijk->kir', U_3d, V_3d, residual)
    
    U -= lr * dU.reshape(n * m, R)
    V -= lr * dV.reshape(m * p, R)
    W -= lr * dW.reshape(p * n, R)
    
    return U, V, W

def try_rank(R_target, n, m, p, max_iters=5000, tol=1e-8, time_limit=10):
    """
    Try to find a rank-R_target decomposition using gradient descent.
    Returns (success, U, V, W) if successful.
    """
    global R
    R = R_target
    
    start_time = time.time()
    
    for restart in range(5):
        # Initialize with random values
        np.random.seed(restart * 1000 + R_target)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Add schoolbook structure as a starting point
        # This helps with convergence
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i * m + j < n * m:
                        # Map to a random rank component
                        r = (i * m + j + k) % R
                        U[i * m + j, r] += 0.5
                        V[j * p + k, r] += 0.5
                        W[k * n + i, r] += 0.5
        
        best_obj = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.005
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            U, V, W = gradient_step(U, V, W, n, m, p, lr=lr)
            
            obj = objective(U, V, W, n, m, p)
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            # Adaptive learning rate
            if obj > best_obj * 1.1:
                lr *= 0.5
            elif obj < best_obj * 0.99:
                lr *= 1.1
                lr = min(lr, 0.01)
            
            if obj < tol:
                # Check if it's actually valid
                if check_decomposition(U, V, W, n, m, p, tol=1e-7):
                    return True, U, V, W
                # Try polishing
                for polish_it in range(100):
                    U, V, W = gradient_step(U, V, W, n, m, p, lr=lr * 0.1)
                    if objective(U, V, W, n, m, p) < 1e-10:
                        if check_decomposition(U, V, W, n, m, p, tol=1e-9):
                            return True, U, V, W
                        break
                break
        
        if best_obj < 1e-6:
            # Try more polishing
            U, V, W = best_U, best_V, best_W
            for polish_it in range(200):
                U, V, W = gradient_step(U, V, W, n, m, p, lr=0.001)
                if objective(U, V, W, n, m, p) < 1e-10:
                    if check_decomposition(U, V, W, n, m, p, tol=1e-9):
                        return True, U, V, W
                    break
    
    return False, None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known results suggest rank around 35-45 for 3x3x6
    for R_target in [40, 38, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 20:
            break
        
        success, U, V, W = try_rank(R_target, n, m, p, max_iters=3000, tol=1e-8, time_limit=15)
        
        if success:
            # Verify with tighter tolerance
            if check_decomposition(U, V, W, n, m, p, tol=1e-9):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = R_target
                print(f"Found rank {R_target} decomposition!")
                # Continue trying to go lower
            else:
                print(f"Rank {R_target} passed but failed strict check")
    
    return best_result
