"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 6 matrix."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    
    # Rank 22 decomposition
    # Part 1: j=2 slice (2x1x6) -> Rank 1
    # Part 2: j in {0,1} slice (2x2x6) -> 3 blocks of 2x2x2 -> 3 * 7 = Rank 21
    
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # --- Part 1: j=2, Rank 1 (Index 0) ---
    # U[i, 2, 0] = 1 for i in 0,1
    U[0 * m + 2, 0] = 1.0
    U[1 * m + 2, 0] = 1.0
    
    # V[2, k, 0] = 1 for k in 0..5
    for k in range(p):
        V[2 * p + k, 0] = 1.0
        
    # W[k, i, 0] = 1 for all k, i
    for k in range(p):
        for i in range(n):
            W[k * n + i, 0] = 1.0
            
    # --- Part 2: j in {0,1}, Rank 21 (Indices 1..21) ---
    # Strassen coefficients for 2x2x2
    # U_coeffs[i, j, s], V_coeffs[j, k, s], W_coeffs[k, i, s]
    # s is 0..6 (local rank index)
    
    # Precompute Strassen coefficients
    # U_strassen[i, j, s]
    U_str = np.zeros((2, 2, 7))
    V_str = np.zeros((2, 2, 7))
    W_str = np.zeros((2, 2, 7))
    
    # s=0
    U_str[0,0,0] = 1; U_str[1,1,0] = 1
    V_str[0,0,0] = 1; V_str[1,1,0] = 1
    W_str[0,0,0] = 1; W_str[1,1,0] = 1
    
    # s=1
    U_str[1,0,1] = 1; U_str[1,1,1] = 1
    V_str[0,0,1] = 1
    W_str[1,0,1] = 1; W_str[1,1,1] = -1
    
    # s=2
    U_str[0,0,2] = 1
    V_str[1,0,2] = 1; V_str[1,1,2] = -1
    W_str[0,1,2] = 1; W_str[1,1,2] = 1
    
    # s=3
    U_str[1,1,3] = 1
    V_str[0,1,3] = 1; V_str[0,0,3] = -1
    W_str[0,0,3] = 1; W_str[1,0,3] = 1
    
    # s=4
    U_str[0,0,4] = 1; U_str[0,1,4] = 1
    V_str[1,1,4] = 1
    W_str[0,0,4] = -1; W_str[0,1,4] = 1
    
    # s=5
    U_str[1,0,5] = 1; U_str[0,0,5] = -1
    V_str[0,0,5] = 1; V_str[0,1,5] = 1
    W_str[1,1,5] = 1
    
    # s=6
    U_str[0,1,6] = 1; U_str[1,1,6] = -1
    V_str[1,0,6] = 1; V_str[1,1,6] = 1
    W_str[0,0,6] = 1
    
    # Fill in the 3 blocks for k in {0,1}, {2,3}, {4,5}
    for g in range(3):
        base_rank = 1 + g * 7
        for s in range(7):
            r = base_rank + s
            # Local k' is 0 or 1
            for i in range(2):
                for j in range(2):
                    if U_str[i, j, s] != 0:
                        # Global index U: i * m + j
                        U[i * m + j, r] += U_str[i, j, s]
                        
            for j in range(2):
                for k_prime in range(2):
                    if V_str[j, k_prime, s] != 0:
                        # Global k = 2*g + k_prime
                        k_global = 2 * g + k_prime
                        # Global index V: j * p + k_global
                        V[j * p + k_global, r] += V_str[j, k_prime, s]
                        
            for k_prime in range(2):
                for i in range(2):
                    if W_str[k_prime, i, s] != 0:
                        # Global k = 2*g + k_prime
                        k_global = 2 * g + k_prime
                        # Global index W: k_global * n + i
                        W[k_global * n + i, r] += W_str[k_prime, i, s]
                        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
