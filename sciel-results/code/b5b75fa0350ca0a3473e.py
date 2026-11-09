"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin
    
    # Try to find a lower-rank decomposition
    # Start with schoolbook and try to reduce rank
    best_rank = n * m * p  # 90
    best_U, best_V, best_W = None, None, None
    
    # Generate schoolbook decomposition
    def schoolbook():
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = schoolbook()
    best_U, best_V, best_W = U0.tolist(), V0.tolist(), W0.tolist()
    best_rank = n * m * p
    
    # Try random rank reduction: pick random subsets of columns and see if they span
    # This is a heuristic - we try to find a smaller set of rank-1 tensors that still works
    
    # For each target rank, try random linear combinations
    for target_rank in range(50, 90):
        if time.time() > deadline - 30:
            break
        
        # Try to find a decomposition with this rank
        # Use random initialization and check if it's valid
        for attempt in range(100):
            if time.time() > deadline - 30:
                break
            
            # Random U, V, W with integer/half-integer entries
            scale = 1.0
            U_rand = np.random.choice([-1, 0, 0.5, 1, 1.5, 2], size=(n*m, target_rank)) * scale
            V_rand = np.random.choice([-1, 0, 0.5, 1, 1.5, 2], size=(m*p, target_rank)) * scale
            W_rand = np.random.choice([-1, 0, 0.5, 1, 1.5, 2], size=(p*n, target_rank)) * scale
            
            # Check if this decomposition is valid
            # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # should equal 1 if i,j,k form a valid triple, 0 otherwise
            # Actually, the tensor for matrix multiplication is:
            # T[i,j,k] = 1 if the computation uses this (i,j,k) triple
            # More precisely, C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor is T[i,j,k] = 1 for all valid (i,j,k)
            
            # Let's verify: for each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # This should equal 1 for all i,j,k (since each product term is used)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
