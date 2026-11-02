"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: 120s limit, use 110s for safety
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook decomposition: rank = n*m*p = 50
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using random search
    # Start from rank 49 and go down
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # For small dimensions, we can try random initialization and Newton's method
    # to find a rank-r decomposition
    # The tensor T[i,j,k] = delta_{i,k} (for matrix multiplication)
    # We need U[i,j], V[j,k], W[k,i] such that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,k}
    
    # Try rank 49 first
    for target_rank in range(rank_sb - 1, 40, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        max_tries = 5
        found = False
        
        for attempt in range(max_tries):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Convert to numpy for computation
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Compute residual: T - sum_r U_r * V_r * W_r
            # T[i,j,k] = 1 if i == k, else 0
            residual = np.zeros((n, m, p))
            for i in range(n):
                for k in range(p):
                    if i == k:
                        residual[i, :, k] = 1
            
            # Compute the tensor from U, V, W
            T_computed = np.zeros((n, m, p))
            for r in range(target_rank):
                U_r = U_np[:, r]  # shape (n*m,)
                V_r = V_np[:, r]  # shape (m*p,)
                W_r = W_np[:, r]  # shape (p*n,)
                
                # Reshape
                U_r_reshaped = U_r.reshape(n, m)
                V_r_reshaped = V_r.reshape(m, p)
                W_r_reshaped = W_r.reshape(p, n)
                
                # T[i,j,k] += U_r[i,j] * V_r[j,k] * W_r[k,i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_computed[i, j, k] += U_r_reshaped[i, j] * V_r_reshaped[j, k] * W_r_reshaped[k, i]
            
            diff = T_computed - residual
            if np.allclose(diff, 0, atol=1e-6):
                found = True
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
