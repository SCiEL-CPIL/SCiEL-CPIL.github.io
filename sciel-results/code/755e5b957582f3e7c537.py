"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time
import random

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify decomposition
    def verify(U, V, W, tol=1e-10):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(S - T))
        return err < tol
    
    # Try to find lower rank using randomized local search with gradient descent
    # Use normalization to handle scale invariance
    
    # We'll try ranks from best_rank-1 down, but limit the search to save time
    # Start with a few attempts at each rank, decreasing as we go lower
    
    for target_rank in range(best_rank - 1, max(best_rank - 60, 40), -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations - more for higher ranks
        n_attempts = 5 if target_rank > best_rank - 20 else 3
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            # Normalize columns to unit norm
            for c in range(target_rank):
                norm_u = np.linalg.norm(U_r[:, c])
                norm_v = np.linalg.norm(V_r[:, c])
                norm_w = np.linalg.norm(W_r[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    # Scale so that product of norms is 1
                    scale = (norm_u * norm_v * norm_w) ** (-1/3)
                    U_r[:, c] *= scale
                    V_r[:, c] *= scale
                    W_r[:, c] *= scale
            
            # Gradient descent with adaptive learning rate
            lr = 0.1
            best_err = np.inf
            for step in range(800):
                if time.time() > deadline - 20:
                    break
                
                # Compute S and error
                S = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                diff = S - T
                err = np.sum(diff**2)
                
                if err < 1e-10:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break
                
                if err < best_err:
                    best_err = err
                    best_U_r, best_V_r, best_W_r = U_r.copy(), V_r.copy(), W_r.copy()
                
                # Compute gradients
                grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V_r, W_r)
                grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U_r, W_r)
                grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U_r, V_r)
                
                # Normalize gradients
                norm_gU = np.linalg.norm(grad_U)
                norm_gV = np.linalg.norm(grad_V)
                norm_gW = np.linalg.norm(grad_W)
                if norm_gU > 1e-10:
                    grad_U /= norm_gU
                if norm_gV > 1e-10:
                    grad_V /= norm_gV
                if norm_gW > 1e-10:
                    grad_W /= norm_gW
                
                # Update
                U_r -= lr * grad_U
                V_r -= lr * grad_V
                W_r -= lr * grad_W
                
                # Renormalize columns
                for c in range(target_rank):
                    norm_u = np.linalg.norm(U_r[:, c])
                    norm_v = np.linalg.norm(V_r[:, c])
                    norm_w = np.linalg.norm(W_r[:, c])
                    if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                        scale = (norm_u * norm_v * norm_w) ** (-1/3)
                        U_r[:, c] *= scale
                        V_r[:, c] *= scale
                        W_r[:, c] *= scale
                
                # Reduce learning rate
                lr *= 0.995
                
                if lr < 1e-5:
                    break
            
            if best_rank == target_rank:
                break
    
    # If we found a lower rank, try to polish with more precise steps
    if best_rank < n * m * p:
        # Final high-precision polish pass
        for polish_iter in range(100):
            if time.time() > deadline - 10:
                break
            
            S = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
            diff = S - T
            err = np.sum(diff**2)
            
            if err < 1e-15:
                break
            
            grad_U = 2 * np.einsum('abc,br,cr->ar', diff, best_V, best_W)
            grad_V = 2 * np.einsum('abc,ar,cr->br', diff, best_U, best_W)
            grad_W = 2 * np.einsum('abc,ar,br->cr', diff, best_U, best_V)
            
            norm_gU = np.linalg.norm(grad_U)
            norm_gV = np.linalg.norm(grad_V)
            norm_gW = np.linalg.norm(grad_W)
            if norm_gU > 1e-10:
                grad_U /= norm_gU
            if norm_gV > 1e-10:
                grad_V /= norm_gV
            if norm_gW > 1e-10:
                grad_W /= norm_gW
            
            lr = 0.01 * (1 - polish_iter / 200)
            best_U -= lr * grad_U
            best_V -= lr * grad_V
            best_W -= lr * grad_W
            
            # Renormalize
            for c in range(best_rank):
                norm_u = np.linalg.norm(best_U[:, c])
                norm_v = np.linalg.norm(best_V[:, c])
                norm_w = np.linalg.norm(best_W[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    scale = (norm_u * norm_v * norm_w) ** (-1/3)
                    best_U[:, c] *= scale
                    best_V[:, c] *= scale
                    best_W[:, c] *= scale
    
    # Convert to lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
