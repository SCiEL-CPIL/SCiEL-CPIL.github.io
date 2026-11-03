"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    # Build the target tensor T of shape (dim_u, dim_v, dim_w)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback (rank = n*m*p = 48)
    rank_fb = n * m * p
    U_fb = np.zeros((dim_u, rank_fb))
    V_fb = np.zeros((dim_v, rank_fb))
    W_fb = np.zeros((dim_w, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fb
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_rec, T, atol=tol)
    
    def compute_loss(U, V, W):
        """Compute the sum of squared errors."""
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.sum((T_rec - T)**2)
    
    def gradient_step(U, V, W, lr=0.01):
        """One step of gradient descent to minimize loss."""
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        err = T_rec - T  # shape (dim_u, dim_v, dim_w)
        
        dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
        dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
        dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        return U, V, W, np.sum(err**2)
    
    def random_decomposition(R, scale=1.0):
        U = np.random.randn(dim_u, R) * scale
        V = np.random.randn(dim_v, R) * scale
        W = np.random.randn(dim_w, R) * scale
        return U, V, W
    
    # Try ranks from higher to lower, starting just below schoolbook
    # Use a structured approach: try ranks that might be achievable
    # Known: 4x4x4 has rank at most 49 (or less with better decompositions)
    # For 3x4x4, we can try various ranks
    
    # Strategy: Try ranks from 47 down to 30
    # For each rank, do multiple random restarts with gradient descent
    
    for R in range(47, 29, -1):
        if time.time() > DEADLINE - 20:
            break
        
        # Multiple random restarts
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() > DEADLINE - 15:
                break
            
            # Random initialization
            U, V, W = random_decomposition(R, scale=0.5)
            
            # Run gradient descent with adaptive learning rate
            lr = 0.005
            prev_loss = compute_loss(U, V, W)
            
            for step in range(8000):
                if time.time() > DEADLINE - 12:
                    break
                
                U, V, W, loss = gradient_step(U, V, W, lr)
                
                # Adaptive learning rate
                if loss < prev_loss * 0.99:
                    lr = min(lr * 1.05, 0.05)
                elif loss > prev_loss * 1.01:
                    lr *= 0.95
                
                prev_loss = loss
                
                if loss < 1e-14:
                    break
                if loss > 1e6:  # diverged
                    break
            
            # Check if we found a valid decomposition
            if check_decomposition(U, V, W, tol=1e-8):
                best_rank = R
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # Found a valid decomposition at rank R, try to go lower
                break
        
        # If we found a valid decomposition at rank R, continue to try lower
        if best_rank == R:
            # Continue searching for lower ranks
            continue
    
    # If we have time, try even lower ranks with more aggressive search
    if time.time() < DEADLINE - 10:
        for R in range(min(best_rank - 1, 47), 24, -1):
            if time.time() > DEADLINE - 8:
                break
            
            n_restarts = 5
            for restart in range(n_restarts):
                if time.time() > DEADLINE - 6:
                    break
                
                U, V, W = random_decomposition(R, scale=0.5)
                
                lr = 0.003
                prev_loss = compute_loss(U, V, W)
                
                for step in range(5000):
                    if time.time() > DEADLINE - 5:
                        break
                    
                    U, V, W, loss = gradient_step(U, V, W, lr)
                    
                    if loss < prev_loss * 0.99:
                        lr = min(lr * 1.05, 0.03)
                    elif loss > prev_loss * 1.01:
                        lr *= 0.95
                    
                    prev_loss = loss
                    
                    if loss < 1e-14:
                        break
                    if loss > 1e6:
                        break
                
                if check_decomposition(U, V, W, tol=1e-8):
                    best_rank = R
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if best_rank == R:
                continue
    
    # Try to convert to integer/half-integer entries if possible
    def to_half_int_array(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    U_h = to_half_int_array(best_U)
    V_h = to_half_int_array(best_V)
    W_h = to_half_int_array(best_W)
    
    if check_decomposition(U_h, V_h, W_h, tol=1e-8):
        U_out, V_out, W_out = U_h, V_h, W_h
    else:
        U_out, V_out, W_out = best_U, best_V, best_W
    
    # Return as lists of lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
