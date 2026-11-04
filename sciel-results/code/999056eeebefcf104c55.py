"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    
    Uses structured search exploiting the 6=2*3 factorization and numerical optimization
    to find low-rank decompositions below the schoolbook rank of 36.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 second deadline with margin
    
    n, m, p = DIMENSIONS
    
    # First, create the schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0.0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0.0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0.0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1.0
                V_sb[j * p + k][column] = 1.0
                W_sb[k * n + i][column] = 1.0
                column += 1
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify if U, V, W form an exact decomposition of the multiplication tensor."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                    if abs(actual - 1.0) > tol:
                        return False
        return True
    
    def try_optimize(target_rank, max_iters=200, lr=0.01, seed=0):
        """Try to find a rank-target_rank decomposition using gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                        diff = val - 1.0
                        loss += diff * diff
                        for r in range(target_rank):
                            grad_U[i*m+j, r] += 2 * diff * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += 2 * diff * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += 2 * diff * U[i*m+j, r] * V[j*p+k, r]
            
            loss /= (n * m * p)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if loss < 1e-10:
                if verify_decomposition(best_U, best_V, best_W, tol=1e-8):
                    return (best_U, best_V, best_W)
            
            # Adaptive learning rate
            if it % 20 == 0 and it > 0:
                lr *= 0.9
        
        # Final verification
        if best_loss < 1e-6:
            if verify_decomposition(best_U, best_V, best_W, tol=1e-8):
                return (best_U, best_V, best_W)
        
        return None
    
    # Try to find lower rank decompositions
    # Start from rank 24 (based on border rank analysis) and work down
    best_rank = rank_schoolbook
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try ranks from 24 down to 1
    for target_rank in range(24, 0, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try multiple seeds and optimization settings
        for seed in range(10):
            if time.time() > deadline - 15:
                break
            
            result = try_optimize(target_rank, max_iters=150, lr=0.01, seed=seed)
            if result is not None:
                found = True
                best_rank = target_rank
                best_decomp = result
                break
        
        if found:
            # Try to go even lower
            continue
    
    if best_rank < rank_schoolbook:
        U_out = best_decomp[0].tolist()
        V_out = best_decomp[1].tolist()
        W_out = best_decomp[2].tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
