"""Rank-minimal tensor decomposition for 2x3 * 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def _build_target_tensor():
    """Build the matmul tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T, dim_u, dim_v, dim_w

def _validate_decomposition(U, V, W, T, dim_u, dim_v, dim_w):
    """Check if decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('at,bt,ct->abc', U, V, W)
    return np.allclose(T_rec, T, atol=1e-8)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    T, dim_u, dim_v, dim_w = _build_target_tensor()
    
    # Schoolbook rank 18 as fallback
    rank_std = n * m * p
    U_std = np.zeros((dim_u, rank_std))
    V_std = np.zeros((dim_v, rank_std))
    W_std = np.zeros((dim_w, rank_std))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_std[i * m + j, col] = 1
                V_std[j * p + k, col] = 1
                W_std[k * n + i, col] = 1
                col += 1
    
    best_result = None
    best_rank = rank_std
    
    # Try ranks from 12 up to 17
    for target_rank in range(12, rank_std):
        if time.time() > deadline - 20:
            break
        
        found = False
        n_restarts = 30
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(restart * 7 + target_rank)
            
            # Initialize with random values
            U = np.random.randn(dim_u, target_rank) * 0.5
            V = np.random.randn(dim_v, target_rank) * 0.5
            W = np.random.randn(dim_w, target_rank) * 0.5
            
            # Alternating least squares optimization
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each column t, minimize ||T - sum U[:,t] outer V[:,t] outer W[:,t]||
                # Fix V, W and solve for U
                # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                # For fixed t: U[a,t] = argmin sum_{b,c} (T[a,b,c] - sum_t' U[a,t'] V[b,t'] W[c,t'])^2
                # This is a linear system in U[:,t]
                
                # More efficient: use the fact that for each t, 
                # U[:,t] should minimize ||T - (U[:,t] outer V[:,t] outer W[:,t]) - (sum of other terms)||
                
                # Let's do gradient descent instead
                T_rec = np.einsum('at,bt,ct->abc', U, V, W)
                E = T - T_rec
                
                # Gradients
                grad_U = np.einsum('abc,bt,ct->at', E, V, W)
                grad_V = np.einsum('abc,at,ct->bt', E, U, W)
                grad_W = np.einsum('abc,at,bt->ct', E, U, V)
                
                # Adaptive learning rate
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(E)) < 1e-8:
                    # Check if entries are close to half-integers
                    is_half_int = True
                    for mat in [U, V, W]:
                        doubled = 2.0 * mat
                        rounded = np.round(doubled)
                        if np.max(np.abs(doubled - rounded)) > 1e-4:
                            is_half_int = False
                            break
                    
                    if is_half_int:
                        U_res = np.round(2.0 * U) / 2.0
                        V_res = np.round(2.0 * V) / 2.0
                        W_res = np.round(2.0 * W) / 2.0
                        
                        if _validate_decomposition(U_res, V_res, W_res, T, dim_u, dim_v, dim_w):
                            best_result = (U_res, V_res, W_res, target_rank)
                            best_rank = target_rank
                            found = True
                            break
                    else:
                        # Try to polish
                        for polish_iter in range(50):
                            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
                            E = T - T_rec
                            if np.max(np.abs(E)) < 1e-10:
                                break
                            grad_U = np.einsum('abc,bt,ct->at', E, V, W)
                            grad_V = np.einsum('abc,at,ct->bt', E, U, W)
                            grad_W = np.einsum('abc,at,bt->ct', E, U, V)
                            lr = 0.001
                            U -= lr * grad_U
                            V -= lr * grad_V
                            W -= lr * grad_W
                        
                        if np.max(np.abs(T - np.einsum('at,bt,ct->abc', U, V, W))) < 1e-8:
                            # Round to half-integers
                            U_res = np.round(2.0 * U) / 2.0
                            V_res = np.round(2.0 * V) / 2.0
                            W_res = np.round(2.0 * W) / 2.0
                            if _validate_decomposition(U_res, V_res, W_res, T, dim_u, dim_v, dim_w):
                                best_result = (U_res, V_res, W_res, target_rank)
                                best_rank = target_rank
                                found = True
                                break
            
            if found:
                break
        
        if found:
            break
    
    if best_result is not None:
        U_res, V_res, W_res, _ = best_result
        return {
            "U": U_res.tolist(),
            "V": V_res.tolist(),
            "W": W_res.tolist()
        }
    
    # Return schoolbook
    return {
        "U": U_std.tolist(),
        "V": V_std.tolist(),
        "W": W_std.tolist()
    }
