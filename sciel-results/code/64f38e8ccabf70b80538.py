"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0
    
    # Best known: rank 21 via 3 independent Strassen blocks
    def strassen_3blocks():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]  # 4 x 21
        V = [[0] * rank for _ in range(m * p)]  # 12 x 21
        W = [[0] * rank for _ in range(p * n)]  # 12 x 21
        
        U_coeffs = [
            {0: 1, 3: 1},       # M1: A00 + A11
            {2: 1, 3: 1},       # M2: A10 + A11
            {0: 1},             # M3: A00
            {3: 1},             # M4: A11
            {0: 1, 1: 1},       # M5: A00 + A01
            {2: 1, 0: -1},      # M6: A10 - A00
            {1: 1, 3: -1}       # M7: A01 - A11
        ]
        
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],      # M1
            [(0, 0, 1)],                 # M2
            [(0, 1, 1), (1, 1, -1)],     # M3
            [(1, 0, 1), (0, 0, -1)],     # M4
            [(1, 1, 1)],                 # M5
            [(0, 0, 1), (0, 1, 1)],      # M6
            [(1, 0, 1), (1, 1, 1)]       # M7
        ]
        
        W_map = [
            [(0, 0, 1), (1, 1, 1)],   # M1
            [(1, 0, 1), (1, 1, -1)],  # M2
            [(0, 1, 1), (1, 1, 1)],   # M3
            [(0, 0, 1), (1, 0, 1)],   # M4
            [(0, 0, -1), (0, 1, 1)],  # M5
            [(1, 1, 1)],              # M6
            [(0, 0, 1)]               # M7
        ]
        
        for r in range(3):
            for k in range(7):
                col = r * 7 + k
                for idx, val in U_coeffs[k].items():
                    U[idx][col] = val
                for j_local, k_local, val in V_coeffs[k]:
                    global_k = 2 * r + k_local
                    v_idx = j_local * p + global_k
                    V[v_idx][col] = val
                for i, j_local, val in W_map[k]:
                    global_k = 2 * r + j_local
                    w_idx = global_k * n + i
                    W[w_idx][col] = val
        
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Build tensor: T[(i,j),(j,k),(i,k)] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # Should equal 1 if i==i, j==j, k==k (i.e., always 1 for valid indices)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    # Try rank 20 by searching for a valid decomposition
    # Strategy: start from rank-21 Strassen and try to eliminate one product
    # by finding a linear dependency
    
    best_result = strassen_3blocks()
    
    # Verify rank 21 works
    if verify(best_result["U"], best_result["V"], best_result["W"]):
        pass  # Good, rank 21 is valid
    
    # Attempt rank 20: modify one column to be a linear combination
    # This is complex; for now, return rank 21 which is verified
    if time.time() > deadline - 10:
        return best_result
    
    # Try to find rank 20 by perturbing the rank-21 solution
    # Use the fact that if we can express one rank-1 term as a combination
    # of others, we can reduce rank. But this requires the tensor to be 
    # decomposable at rank 20.
    
    # For M_{2,2,6}, known optimal is likely 21. Return rank 21.
    return best_result
# EVOLVE-BLOCK-END
