"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank = n*m*p = 42
    # Try to find a lower rank decomposition
    
    deadline = time.time() + 140  # 140 seconds, leave 10s margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Try to construct a lower rank decomposition
    # For 2x3x7, try rank 34 using 2x3x3 (rank 14) * 2 + 2x3x1 (rank 6)
    # But we need the actual decomposition matrices
    
    # Let's try a direct approach: use bilinear optimization
    # The tensor T has shape (n*m, m*p, p*n) = (6, 21, 14)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try to find a rank-r decomposition for decreasing r
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # For each candidate rank, try to solve
    for rank in range(30, 42):
        if time.time() > deadline:
            break
        
        # Try random initialization and gradient descent
        # This is a non-convex problem, so we'll try multiple restarts
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W with small random values
            np.random.seed(42 + restart)
            U_init = np.random.randn(n * m, rank) * 0.1
            V_init = np.random.randn(m * p, rank) * 0.1
            W_init = np.random.randn(p * n, rank) * 0.1
            
            # Use a simple gradient descent on the loss:
            # L = ||T - sum_r U[:,r] V[:,r] W[:,r]||^2
            # This is expensive, so let's skip for now
            
            # Instead, try a constructive approach for specific ranks
            found = False
            break
        
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
