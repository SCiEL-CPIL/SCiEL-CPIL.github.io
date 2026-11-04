"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build M_{2,2,2} rank 7 decomposition (Strassen)
    U_222 = [[0]*7 for _ in range(4)]
    V_222 = [[0]*7 for _ in range(4)]
    W_222 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U_222[0][0] = 1; U_222[3][0] = 1
    V_222[0][0] = 1; V_222[3][0] = 1
    W_222[0][0] = 1; W_222[3][0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U_222[2][1] = 1; U_222[3][1] = 1
    V_222[0][1] = 1
    W_222[1][1] = 1 # C10 -> k=0, i=1 -> idx 1
    
    # r=2: M3 = A00(B01-B11)
    U_222[0][2] = 1
    V_222[1][2] = 1; V_222[3][2] = -1
    W_222[2][2] = 1 # C01 -> k=1, i=0 -> idx 2
    
    # r=3: M4 = A11(B10-B00)
    U_222[3][3] = 1
    V_222[2][3] = 1; V_222[0][3] = -1
    W_222[1][3] = 1 # C10 -> k=0, i=1 -> idx 1
    # Wait, M4 contributes to C00 and C10?
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # So W for M4 must contribute to C00 (k=0, i=0 -> idx 0) and C10 (k=0, i=1 -> idx 1).
    # But W is a single vector for this rank-1 component.
    # This means M4 cannot be a single rank-1 component in this simple mapping if it contributes to multiple C entries with different signs/coefficients?
    # No, the tensor T_{ijk} must be 1.
    # The decomposition sums to 1.
    # If W_222[1][3]=1, it contributes to T_{100} (i=1, j=0, k=0).
    # Does it contribute to T_{000}? W_222[0][3] is 0.
    # But C00 needs M4.
    # This implies the simple "one rank-1 per Strassen product" mapping is NOT sufficient to build the tensor directly?
    # Yes, because Strassen products are intermediate. The final C is a linear combination.
    # The tensor decomposition must represent the FINAL multiplication.
    # So we cannot just concatenate the rank-1 components of the Strassen products.
    # We must expand them into the final U, V, W.
    
    # The expansion logic:
    # U_444[i, 7s+t] = Sum_{j'} CoeffA[s, i'j'] * U_222[i''*2+j'', t]
    # This handles the linear combinations automatically.
    
    # Let's verify the W mapping in the expansion.
    # W_444[k, 7s+t] = Sum_{i'} CoeffC[s, k'i'] * W_222[k''*2+i'', t]
    # This should work.
    
    # Define CoeffA, CoeffB, CoeffC for Strassen
    # s=0..6
    # CoeffA[s][block] where block is 00, 01, 10, 11 (index 0,1,2,3)
    CoeffA = [
        [1, 0, 0, 1], # s=0: A00 + A11
        [0, 0, 1, 1], # s=1: A10 + A11
        [1, 0, 0, 0], # s=2: A00
        [0, 0, 0, 1], # s=3: A11
        [1, 1, 0, 0], # s=4: A00 + A01
        [-1, 0, 1, 0],# s=5: A10 - A00
        [0, 1, 0, -1] # s=6: A01 - A11
    ]
    
    CoeffB = [
        [1, 0, 0, 1], # s=0: B00 + B11
        [1, 0, 0, 0], # s=1: B00
        [0, 1, 0, -1],# s=2: B01 - B11
        [0, 0, 1, -1],# s=3: B10 - B00
        [0, 0, 0, 1], # s=4: B11
        [1, 1, 0, 0], # s=5: B00 + B01
        [0, 0, 1, 1]  # s=6: B10 + B11
    ]
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # CoeffC[s][block]
    CoeffC = [
        [1, 0, 0, 1], # s=0: M1 -> C00, C11
        [0, 0, 1, -1],# s=1: M2 -> C10, -C11
        [0, 1, 0, 0], # s=2: M3 -> C01
        [1, 0, 1, 0], # s=3: M4 -> C00, C10
        [-1, 1, 0, 0],# s=4: M5 -> -C00, C01
        [0, 0, 0, 1], # s=5: M6 -> C11
        [1, 0, 0, 1]  # s=6: M7 -> C00, C11
    ]
    
    # Build 4x4x4 part (Rank 49)
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    for s in range(7):
        for t in range(7):
            r = s*7 + t
            for i_prime in range(2):
                for i_double in range(2):
                    i = i_prime*2 + i_double
                    for j_prime in range(2):
                        for j_double in range(2):
                            j = j_prime*2 + j_double
                            u_idx = i*4 + j
                            u_222_idx = i_double*2 + j_double
                            val = CoeffA[s][i_prime*2 + j_prime] * U_222[u_222_idx][t]
                            U_444[u_idx][r] += val
                            
                            v_222_idx = j_double*2 + (j_prime*2 + i_prime) # Wait, V index is j*k? No, V is m*p.
                            # For 2x2x2, V row is j*2 + k.
                            # Here j is global j. j = 2j' + j''.
                            # k is global k? No, V_444 row is j*4 + k? No, V shape is (m*p, rank).
                            # m=4, p=4. V row j*4 + k.
                            # In 2x2x2, V row j*2 + k.
                            # CoeffB[s][block] where block is j'k'?
                            # B_s[j'', k''] = Sum CoeffB[s, j'k'] B_{j'k''}[j'', k'']
                            # So V_444[j*4+k, r] += CoeffB[s, j'k'] * V_222[j''*2+k''][t]
                            pass # Implemented below correctly

    # Correct Expansion Loop
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    for s in range(7):
        for t in range(7):
            r = s*7 + t
            for i_p in range(2):
                for i_d in range(2):
                    i = i_p*2 + i_d
                    for j_p in range(2):
                        for j_d in range(2):
                            j = j_p*2 + j_d
                            u_idx = i*4 + j
                            u_222_idx = i_d*2 + j_d
                            U_444[u_idx][r] = CoeffA[s][i_p*2 + j_p] * U_222[u_222_idx][t]
                            
            for j_p in range(2):
                for j_d in range(2):
                    for k_p in range(2):
                        for k_d in range(2):
                            j = j_p*2 + j_d
                            k = k_p*2 + k_d
                            v_idx = j*4 + k
                            v_222_idx = j_d*2 + k_d
                            V_444[v_idx][r] = CoeffB[s][j_p*2 + k_p] * V_222[v_222_idx][t]
                            
            for k_p in range(2):
                for k_d in range(2):
                    for i_p in range(2):
                        for i_d in range(2):
                            k = k_p*2 + k_d
                            i = i_p*2 + i_d
                            w_idx = k*4 + i
                            w_222_idx = k_d*2 + i_d
                            W_444[w_idx][r] = CoeffC[s][k_p*2 + i_p] * W_222[w_222_idx][t]

    # Build 4x4x5 (Add rank 4 for 5th column)
    rank = 53
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    # Copy 4x4x4 part
    for i in range(16):
        for r in range(49):
            U[i][r] = U_444[i][r]
    for j in range(16):
        for r in range(49):
            V[j][r] = V_444[j][r]
    for k in range(16):
        for r in range(49):
            W[k][r] = W_444[k][r]
            
    # Add 5th column (k=4)
    # T_{i,j,4} = 1
    # U_{ij, 49+r} = delta_{jr}
    # V_{j4, 49+r} = delta_{jr}
    # W_{4i, 49+r} = 1
    for r in range(4):
        for i in range(4):
            for j in range(4):
                if j == r:
                    U[i*4 + j][49+r] = 1
                    V[j*5 + 4][49+r] = 1
                W[4*4 + i][49+r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
