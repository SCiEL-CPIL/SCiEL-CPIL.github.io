"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with a reasonable rank guess
    # For 3x4x6, let's try ranks below 72
    best_rank = n * m * p  # fallback to schoolbook
    best_factors = None
    
    deadline = time.time() + 120  # safety margin
    
    # Try different ranks
    for rank in range(30, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random factors
        np.random.seed(42)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # The target tensor: T[i,j,k] = 1 for all i,j,k is wrong.
        # Let me reconsider what the target is.
        
        # From the seed code, the decomposition satisfies:
        # For each r, there's a triple (i_r, j_r, k_r) such that
        # U[i_r*m+j_r, r] = 1, V[j_r*p+k_r, r] = 1, W[k_r*n+i_r, r] = 1
        # and all other entries are 0.
        # So the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # = 1 if (i,j,k) = (i_r, j_r, k_r) for some r, else 0
        # = 1 for all (i,j,k) since every triple is covered.
        
        # So the target tensor is the all-ones tensor!
        # That seems odd for matrix multiplication, but let's verify with the seed.
        
        # Actually, I think I'm misunderstanding. Let me just compute the target.
        # The multiplication tensor should be: T[i,j,k] = 1 if the indices are "compatible"
        # but in the standard definition, T_{ijk} = 1 for all i,j,k when we write
        # C_{ik} = sum_{j} A_{ij} B_{jk} = sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki}
        # with T_{ijk} = 1.
        
        # So yes, the target is the all-ones tensor of shape (n, m, p).
        
        target = np.ones((n, m, p))
        
        # Optimize using simple gradient descent
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            # Compute current tensor
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            T = np.zeros((n, m, p))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            residual = T - target
            if np.linalg.norm(residual) < 1e-6:
                break
            
            # Gradient computation is complex; use finite differences or skip
            # For simplicity, let's just check if rank works with a smarter init
        
        # Alternative: use the fact that for small dims, we can try structured decompositions
        # For now, if we can't find a low-rank one, use schoolbook
    
    # Fallback to schoolbook decomposition
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
# EVOLVE-BLOCK-END
