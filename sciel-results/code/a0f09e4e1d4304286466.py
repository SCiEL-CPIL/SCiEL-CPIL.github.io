import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if the multiplication pattern matches
    # a = i*m + j (index into A, i in [0,n), j in [0,m))
    # b = j'*p + k (index into B, j' in [0,m), k in [0,p))
    # c = k'*n + i' (index into C, k' in [0,p), i' in [0,n))
    # T[a, b, c] = 1 if j == j' and i == i' and k == k'
    
    # Actually, for matrix multiplication C = A * B:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # With the trace convention W[k*n+i, r], the output index is k*n + i
    # So C is stored as a vector of length p*n where C_vec[k*n+i] = C[i,k]... 
    # Wait, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # This means W is of shape (p*n, rank) and W[k*n+i, r] gives the coefficient
    # for the output element at position (k, i) in the flattened output.
    # So the output is C[k, i] = sum_j A[i, j] B[j, k]
    
    # The tensor T has dimensions (n*m) x (m*p) x (p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k'
    # Otherwise 0
    
    # Let's build this tensor explicitly
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose T."""
        # Compute sum_r U[:,r] outer V[:,r] outer W[:,r]
        # and compare with T
        recon = np.zeros_like(T)
        for r in range(U.shape[1]):
            recon += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(recon, T, atol=tol, rtol=0)
    
    def try_rank_gradient(target_rank, time_budget, rng):
        """Try to find a rank-1 decomposition of T with given rank using gradient descent."""
        if time_budget <= 0:
            return None
        
        # Initialize U, V, W randomly
        U = rng.standard_normal((n*m, target_rank))
        V = rng.standard_normal((m*p, target_rank))
        W = rng.standard_normal((p*n, target_rank))
        
        # Scale to reasonable magnitude
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        lr = 0.01
        best_err = np.inf
        best_U_copy, best_V_copy, best_W_copy = U.copy(), V.copy(), W.copy()
        
        t_start = time.time()
        step = 0
        while time.time() - t_start < time_budget * 0.7:
            # Compute gradient of ||sum_r u_r v_r w_r - T||^2
            # d/dU[ar] = 2 * sum_bc (recon[abc] - T[abc]) * V[br] * W[cr]
            
            recon = np.zeros_like(T)
            for r in range(target_rank):
                recon += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
            
            err = np.sum((recon - T) ** 2)
            if err < best_err:
                best_err = err
                best_U_copy, best_V_copy, best_W_copy = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            # Compute gradients
            diff = recon - T
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(target_rank):
                dU[:, r] = 2 * np.einsum('abc,br,cr->a', diff, V[:, r], W[:, r])
                dV[:, r] = 2 * np.einsum('abc,ar,cr->b', diff, U[:, r], W[:, r])
                dW[:, r] = 2 * np.einsum('abc,ar,br->c', diff, U[:, r], V[:, r])
            
            U -= lr * dU
            V -= lr * dW  # Note: dW is gradient w.r.t. W, but we use dW for W update
            # Actually let me fix this:
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            step += 1
            if step % 100 == 0:
                # Adaptive learning rate
                if err < best_err * 0.9:
                    lr *= 1.1
                else:
                    lr *= 0.9
                lr = max(1e-4, min(0.1, lr))
        
        # Check if we found a valid decomposition
        if best_err < 1e-6:
            # Try to snap to integers/half-integers
            U_snap = np.round(best_U_copy * 2) / 2
            V_snap = np.round(best_V_copy * 2) / 2
            W_snap = np.round(best_W_copy * 2) / 2
            
            if check_decomposition(U_snap, V_snap, W_snap, tol=1e-6):
                return U_snap, V_snap, W_snap
            else:
                return best_U_copy, best_V_copy, best_W_copy
        
        return None
    
    def try_rank_random_restart(target_rank, time_budget, rng):
        """Try multiple random restarts for a given rank."""
        n_restarts = min(20, max(1, int(time_budget / 2)))
        for _ in range(n_restarts):
            if time.time() > deadline - 20:
                break
            result = try_rank_gradient(target_rank, time_budget / n_restarts, rng)
            if result is not None:
                U, V, W = result
                if check_decomposition(U, V, W, tol=1e-5):
                    return U, V, W
        return None
    
    # Try ranks from low to high
    # For 2x5x5, the rank is at most 50 (schoolbook)
    # Known lower bounds suggest rank >= 25 or so
    # Try from 25 up to 50
    
    rng = np.random.default_rng(42)
    
    for target_rank in range(25, rank_school + 1):
        if time.time() > deadline - 25:
            break
        
        time_budget = min(30, deadline - time.time() - 20)
        if time_budget <= 0:
            break
        
        result = try_rank_random_restart(target_rank, time_budget, rng)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W, tol=1e-5):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # If we found a decomposition, try to polish it
    if best_rank < rank_school:
        if time.time() < deadline - 15:
            # Try to improve with more precise optimization
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                U_polish = best_U.copy()
                V_polish = best_V.copy()
                W_polish = best_W.copy()
                
                lr = 1e-3
                for it in range(1000):
                    if time.time() > deadline - 10:
                        break
                    recon = np.zeros_like(T)
                    for r in range(best_rank):
                        recon += np.einsum('ar,br,cr->abc', U_polish[:, r], V_polish[:, r], W_polish[:, r])
                    
                    err = np.sum((recon - T) ** 2)
                    if err < 1e-10:
                        break
                    
                    diff = recon - T
                    dU = np.zeros_like(U_polish)
                    dV = np.zeros_like(V_polish)
                    dW = np.zeros_like(W_polish)
                    for r in range(best_rank):
                        dU[:, r] = 2 * np.einsum('abc,br,cr->a', diff, V_polish[:, r], W_polish[:, r])
                        dV[:, r] = 2 * np.einsum('abc,ar,cr->b', diff, U_polish[:, r], W_polish[:, r])
                        dW[:, r] = 2 * np.einsum('abc,ar,br->c', diff, U_polish[:, r], V_polish[:, r])
                    
                    U_polish -= lr * dU
                    V_polish -= lr * dV
                    W_polish -= lr * dW
                    lr *= 0.999
                
                # Snap to half-integers
                U_snap = np.round(U_polish * 2) / 2
                V_snap = np.round(V_polish * 2) / 2
                W_snap = np.round(W_polish * 2) / 2
                
                if check_decomposition(U_snap, V_snap, W_snap, tol=1e-6):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    break
    
    # Convert to lists for the return format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
