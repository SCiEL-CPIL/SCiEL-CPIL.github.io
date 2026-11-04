import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    r = U.shape[1]
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Condition: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1 for all i, j, k
    # Note: The indices in the problem statement are:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The sum is over c.
    # The result should be 1 for all i, j, k.
    
    # We can vectorize this check.
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    # Compute the sum for all i, j, k
    # result[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # This is a bit tricky to vectorize directly without loops or einsum.
    # Let's use einsum.
    # U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want to sum over c.
    # np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False, 1e10
        
    # The target is 1 for all i, j, k
    target = np.ones((n, m, p))
    error = np.linalg.norm(result - target)
    return error < 1e-9, error

def find_decomposition(r, n, m, p, max_time, num_restarts=100):
    """Try to find a rank-r decomposition using random restarts and gradient descent."""
    start_time = time.time()
    best_error = 1e10
    best_solution = None
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Random initialization
        # Scale: entries around 1
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Simple gradient descent
        lr = 0.01
        for step in range(1000):
            if time.time() - start_time > max_time:
                break
                
            # Compute result and error
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            
            result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            target = np.ones((n, m, p))
            error = np.linalg.norm(result - target)
            
            if error < 1e-9:
                # Found a solution
                # Try rounding
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                is_exact, _ = check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p)
                if is_exact:
                    return U_rounded, V_rounded, W_rounded
                
                # Also try integer rounding
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                is_exact, _ = check_decomposition(U_int, V_int, W_int, n, m, p)
                if is_exact:
                    return U_int, V_int, W_int
                
                # If not exact after rounding, we can still return the float solution 
                # if the problem allows floats? 
                # "integer/half-integer entries" is specified.
                # So we must return rounded values.
                # If rounding fails, we continue searching.
                
            # Compute gradients
            # dE/dU = 2 * (result - target) * V * W
            # dE/dV = 2 * (result - target) * U * W
            # dE/dW = 2 * (result - target) * U * V
            
            diff = result - target
            
            # Gradient for U
            # dU[i,j,c] = 2 * sum_{k} diff[i,j,k] V[j,k,c] W[k,i,c]
            # This is another einsum
            grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', diff, V_reshaped, W_reshaped)
            grad_U = grad_U.reshape(n*m, r)
            
            # Gradient for V
            # dV[j,k,c] = 2 * sum_{i} diff[i,j,k] U[i,j,c] W[k,i,c]
            grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', diff, U_reshaped, W_reshaped)
            grad_V = grad_V.reshape(m*p, r)
            
            # Gradient for W
            # dW[k,i,c] = 2 * sum_{j} diff[i,j,k] U[i,j,c] V[j,k,c]
            grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', diff, U_reshaped, V_reshaped)
            grad_W = grad_W.reshape(p*n, r)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if error > best_error:
                lr *= 0.5
            else:
                lr = min(lr * 1.05, 0.1)
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
                
    # If we didn't find an exact rounded solution, check best_solution
    if best_solution is not None:
        U, V, W = best_solution
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        is_exact, _ = check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p)
        if is_exact:
            return U_rounded, V_rounded, W_rounded
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 100  # seconds
    
    # Try rank 16
    if time.time() - start_time < deadline:
        sol = find_decomposition(16, n, m, p, deadline - (time.time() - start_time))
        if sol is not None:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try rank 17
    if time.time() - start_time < deadline:
        sol = find_decomposition(17, n, m, p, deadline - (time.time() - start_time))
        if sol is not None:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
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
    return {"U": U, "V": V, "W": W}
