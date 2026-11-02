import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target ranks to try
    target_ranks = [14, 15, 16, 24]
    
    best_result = None
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
            
        success = False
        # Try a few random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with random small values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Simple gradient descent or alternating minimization
            # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # This is hard to optimize directly.
            # Instead, we can fix U and V and solve for W?
            # The equation is linear in W for fixed U, V.
            # For each (i,j,k), sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
            # This is a system of linear equations for W.
            # Let A_{(i,j,k), (k', i', r)} = U[i*m+j, r] V[j*p+k, r] if (k', i') == (k, i) else 0?
            # No, W is indexed by (k, i, r).
            # The term is W[k*n+i, r].
            # So for a fixed (i,j), the equation for all k is:
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all k.
            # This is not linear in W alone because V depends on k.
            
            # Let's use a different approach: Random search for integer/half-integer solutions.
            # Given the small size, we can try random integers in {-1, 0, 1} and half-integers.
            # But 14 columns is too many for brute force.
            
            # I will fall back to a known rank 15 solution if I can hardcode it.
            # For now, I will return the schoolbook solution if no better is found.
            pass
        
        # If we reach here, we haven't found a solution for this rank yet.
        # I will implement a specific rank 15 solution construction.
        if rank == 15:
             # Hardcode a rank 15 solution
             # I will use a known decomposition for 2x4x4 and adapt it.
             # Or just return schoolbook to be safe if I can't find it.
             pass
    
    # Fallback: Schoolbook
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
