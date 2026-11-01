"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses a structured search for low-rank decompositions, starting from the
    schoolbook rank of 50 and attempting to reduce it via optimization.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Build the full tensor T[a, b, c] where:
    #   a = i*m + j  (row of A, col of A)
    #   b = j*p + k  (row of B, col of B)
    #   c = k*n + i  (output, trace convention: k*n + i)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p  # 50
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook
    Rec_sb = np.einsum('ik,jk,lk->ijl', U_sb, V_sb, W_sb)
    err_sb = np.max(np.abs(T - Rec_sb))
    
    # Try to find a lower-rank decomposition using optimization
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Attempt optimization for ranks below schoolbook
    # We'll try a range of ranks and use multiple random initializations
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 20, 10), -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values scaled appropriately
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use a more sophisticated optimization approach
            # We'll use alternating least squares (ALS) for better convergence
            lr = 0.05
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
                err = np.max(np.abs(T - Rec))
                
                if err < 1e-8:
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break
                
                # Compute gradients
                diff = T - Rec
                grad_U = np.einsum('ijl,jk,lk->ik', diff, V, W)
                grad_V = np.einsum('ijl,ik,lk->jk', diff, U, W)
                grad_W = np.einsum('ijl,ik,jk->lk', diff, U, V)
                
                # Adaptive learning rate based on gradient magnitude
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 0:
                    adaptive_lr = lr * (1.0 / (1.0 + grad_norm))
                else:
                    adaptive_lr = lr
                
                # Update
                U -= adaptive_lr * grad_U
                V -= adaptive_lr * grad_V
                W -= adaptive_lr * grad_W
            
            # Check final error
            Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
            err = np.max(np.abs(T - Rec))
            if err < 1e-8:
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
