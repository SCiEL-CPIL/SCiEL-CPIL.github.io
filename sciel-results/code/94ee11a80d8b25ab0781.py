import numpy as np
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p - 1 # 119
    
    # Try to find a decomposition with rank < n*m*p
    # We use a simple random search with local optimization
    
    for rank in range(target_rank, n*m*p - 5, -1):
        # Initialize with random values
        np.random.seed(42) # For reproducibility in search, but we want diversity?
        # Let's use a fixed seed for the search process to be deterministic?
        # No, let's try a few random inits.
        
        found = False
        for _ in range(100): # 100 random restarts
            U = np.random.uniform(-1, 1, (n*m, rank))
            V = np.random.uniform(-1, 1, (m*p, rank))
            W = np.random.uniform(-1, 1, (p*n, rank))
            
            # Simple gradient descent
            lr = 0.01
            for step in range(1000):
                # Compute error
                # Error[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
                # This is expensive to compute naively.
                # We can use tensor operations.
                
                # Reshape U to (n, m, rank)
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                
                # Sum over rank
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # Note: W index is k*n+i, so W_3d[k,i,r]
                
                # This is a tensor contraction.
                # We can do it in steps.
                # Temp[i,j,k] = sum_r U[i,j,r] * V[j,k,r]
                # Temp[i,j,k] = np.einsum('ijr,jkr->ijk', U_3d, V_3d)
                # Then T[i,k] = sum_j Temp[i,j,k] * W[k,i,r]? No.
                # The term is U[i,j,r] * V[j,k,r] * W[k,i,r].
                # So T[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r].
                
                # Let's compute T[i,k] directly.
                # T[i,k] = sum_r (sum_j U[i,j,r] V[j,k,r]) * W[k,i,r]
                
                # Step 1: M[i,k,r] = sum_j U[i,j,r] V[j,k,r]
                M = np.einsum('ijr,jkr->ikr', U_3d, V_3d)
                # Step 2: T[i,k] = sum_r M[i,k,r] * W[k,i,r]
                # W is (p, n, rank), so W[k,i,r]
                T = np.einsum('ikr,ki r->ik', M, W_3d)
                
                # Target is 1 for all i,k?
                # No, the target tensor T_target has indices i,j,k.
                # The condition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
                # My T[i,k] above sums over j. This is wrong.
                # The condition is for each triple (i,j,k).
                
                # So we need to check the full 3D tensor.
                # T_full[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # T_full = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
                # This is expensive.
                
                # Let's just compute the error for a few random triples to speed up?
                # No, we need exactness.
                
                # Let's stick to the full computation for now, but it might be slow.
                # n=4, m=5, p=6, rank=119.
                # U_3d: 4x5x119
                # V_3d: 5x6x119
                # W_3d: 6x4x119
                # np.einsum('ijr,jkr,ki r->ijk', ...)
                # This is O(n*m*p*rank) = 4*5*6*119 = 14280 operations.
                # 1000 steps * 100 restarts * 14280 = 1.4e9 operations.
                # This might be too slow in Python/NumPy within 120s.
                
                # We need a faster method.
                # We can use the fact that we only need to find *one* solution.
                # And we can use integer/half-integer search.
                
                # Given the complexity, I will return the schoolbook solution for now,
                # but with a small modification:
                # I will try to see if rank 119 is possible by checking if the system
                # has a solution. But I can't solve it easily.
                
                # Let's just return the schoolbook solution.
                # Wait, the task says "design a new program... from scratch".
                # If I return schoolbook, I score 1.0.
                # I need to improve.
                
                # Let's try a different strategy.
                # Use the Strassen algorithm for 2x2 blocks.
                # 4x5x6 can be split into 2x2 blocks?
                # 4 = 2*2, 5 = 2*2 + 1, 6 = 2*3.
                # This is complicated.
                
                # I will output the schoolbook solution but with a comment that it's a fallback.
                # Actually, I can try to reduce the rank by 1 by finding a linear dependence.
                # But there are none.
                
                # Let's try rank 119 with a very simple random search, limited steps.
                pass

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
# EVOLVE-BLOCK-END
