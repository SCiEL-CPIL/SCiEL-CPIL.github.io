import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 140.0
    
    def elapsed():
        return time.time() - start_time
    
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook solution
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower rank decomposition
    # Use a randomized approach with rank minimization
    
    def verify_decomp(U, V, W, rank):
        """Verify that the decomposition is exact"""
        recon = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(rank):
            recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(T - recon)) < 1e-8
    
    def try_rank(target_rank, max_restarts=10):
        """Try to find a decomposition of given target_rank"""
        if elapsed() > DEADLINE - 5:
            return None
        
        for restart in range(max_restarts):
            if elapsed() > DEADLINE - 5:
                return None
            
            # Initialize with random vectors
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize to prevent divergence
            for r in range(target_rank):
                norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
                if norm > 1e-10:
                    scale = norm ** (1/3)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
            
            # Gradient descent
            lr = 0.01
            for it in range(1000):
                if elapsed() > DEADLINE - 5:
                    return None
                
                # Compute residual
                recon = np.zeros((n*m, m*p, p*n), dtype=float)
                for r in range(target_rank):
                    recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                err = np.max(np.abs(T - recon))
                if err < 1e-8:
                    return U, V, W
                
                # Compute gradients
                res = T - recon
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(target_rank):
                    u, v, w = U[:, r], V[:, r], W[:, r]
                    grad_U[:, r] = np.einsum('ijk,j,k->i', res, v, w)
                    grad_V[:, r] = np.einsum('ijk,i,k->j', res, u, w)
                    grad_W[:, r] = np.einsum('ijk,i,j->k', res, u, v)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                lr *= 0.995
                
                # Renormalize
                for r in range(target_rank):
                    norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
                    if norm > 1e-10:
                        scale = norm ** (1/3)
                        U[:, r] /= scale
                        V[:, r] /= scale
                        W[:, r] /= scale
        
        return None
    
    # Try ranks from lower to higher
    # Lower bound is 19, so start from there
    for target_rank in range(19, schoolbook_rank):
        if elapsed() > DEADLINE - 10:
            break
        
        result = try_rank(target_rank, max_restarts=5)
        if result is not None:
            U, V, W = result
            if verify_decomp(U, V, W, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Round to nearest 0.5 if close
    def round_half(arr):
        rounded = np.round(arr * 2) / 2
        if np.max(np.abs(arr - rounded)) < 1e-6:
            return rounded.tolist()
        return arr.tolist()
    
    return {
        "U": round_half(best_U),
        "V": round_half(best_V),
        "W": round_half(best_W)
    }
