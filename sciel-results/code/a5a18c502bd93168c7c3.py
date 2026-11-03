import time
import sys
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    t_limit = 115.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # The 2x4x8 matrix multiplication tensor is a Kronecker product of two 2x2x2 tensors.
    # We use a rank-7 Strassen decomposition for 2x2x2 and take the Kronecker product
    # to get a rank-49 decomposition for 2x4x8.
    
    # Strassen 2x2x2 decomposition (rank 7)
    # Local indices for 2x2:
    # A: flat i*2+j (i,j in {0,1})
    # B: flat j*2+k (j,k in {0,1})
    # C: flat k*2+i (W convention: k*n+i with n=2)
    
    # Strassen components
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1)
    U_local[0][0] = 1; U_local[3][0] = 1 # a00, a11
    V_local[0][0] = 1; V_local[3][0] = 1 # b00, b11
    W_local[0][0] = 1 # C00 (k=0,i=0)
    W_local[3][0] = 1 # C11 (k=1,i=1)
    
    # r=1 (M2)
    U_local[2][1] = 1; U_local[3][1] = 1 # a10, a11
    V_local[0][1] = 1 # b00
    W_local[2][1] = 1 # C10 (k=0,i=1)
    W_local[3][1] = -1 # C11 (k=1,i=1)
    
    # r=2 (M3)
    U_local[0][2] = 1 # a00
    V_local[1][2] = 1; V_local[3][2] = -1 # b01, b11
    W_local[1][2] = 1 # C01 (k=1,i=0)
    W_local[3][2] = 1 # C11 (k=1,i=1)
    
    # r=3 (M4)
    U_local[3][3] = 1 # a11
    V_local[2][3] = 1; V_local[0][3] = -1 # b10, b00
    W_local[0][3] = 1 # C00 (k=0,i=0)
    W_local[2][3] = 1 # C10 (k=0,i=1)
    
    # r=4 (M5)
    U_local[0][4] = 1; U_local[1][4] = 1 # a00, a01
    V_local[3][4] = 1 # b11
    W_local[0][4] = -1 # C00 (k=0,i=0)
    W_local[1][4] = 1 # C01 (k=1,i=0)
    
    # r=5 (M6)
    U_local[2][5] = 1; U_local[0][5] = -1 # a10, a00
    V_local[0][5] = 1; V_local[1][5] = 1 # b00, b01
    W_local[3][5] = 1 # C11 (k=1,i=1)
    
    # r=6 (M7)
    U_local[1][6] = 1; U_local[3][6] = -1 # a01, a11
    V_local[2][6] = 1; V_local[3][6] = 1 # b10, b11
    W_local[0][6] = 1 # C00 (k=0,i=0)

    # Kronecker product to get rank-49 decomposition for 2x4x8
    # The 2x4x8 tensor = T_2x2 ⊗ T_2x2
    # So U_global = U_local ⊗ U_local, etc.
    
    total_rank = 49
    
    U = [[0] * total_rank for _ in range(n * m)] # 8 x 49
    V = [[0] * total_rank for _ in range(m * p)] # 32 x 49
    W = [[0] * total_rank for _ in range(p * n)] # 16 x 49
    
    # For Kronecker product, we need to map local indices to global indices
    # Global A: 2x4, flat i*4+j where i in {0,1}, j in {0,1,2,3}
    # Global B: 4x8, flat j*8+k where j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    # Global C: 2x8, W convention k*2+i where i in {0,1}, k in {0,1,2,3,4,5,6,7}
    
    # The Kronecker product structure:
    # A_global[i,j] where j = j1*2 + j0 (j1,j0 in {0,1})
    # B_global[j,k] where j = j1*2 + j0, k = k1*2 + k0
    # C_global[i,k] where k = k1*2 + k0
    # W_global[k*2+i] = sum over r1,r2 of U_local[i,j0,r1] * U_local[i,j1,r2] * V_local[j0,k0,r1] * V_local[j1,k1,r2] * W_local[k0*2+i,r1] * W_local[k1*2+i,r2]
    
    # Actually, let me think about this more carefully.
    # The 2x4x8 multiplication can be seen as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # where i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    
    # We can write j = j1*2 + j0 and k = k1*2 + k0 where j1,j0,k1,k0 in {0,1}
    # Then C[i,k1*2+k0] = sum_{j1,j0} A[i,j1*2+j0] * B[j1*2+j0,k1*2+k0]
    
    # This is a Kronecker product structure if we group appropriately.
    # Let's define:
    # A1[i,j0] = A[i, j0] (columns 0,1)
    # A2[i,j0] = A[i, 2+j0] (columns 2,3)
    # B1[j0,k0] = B[j0, k0] (rows 0,1; cols 0,1)
    # B2[j0,k0] = B[2+j0, k0] (rows 2,3; cols 0,1)
    # B3[j0,k0] = B[j0, 2+k0] (rows 0,1; cols 2,3)
    # B4[j0,k0] = B[2+j0, 2+k0] (rows 2,3; cols 2,3)
    
    # Then:
    # C[i, k0] = sum_{j0} A1[i,j0]*B1[j0,k0] + A2[i,j0]*B3[j0,k0]
    # C[i, 2+k0] = sum_{j0} A1[i,j0]*B2[j0,k0] + A2[i,j0]*B4[j0,k0]
    
    # Hmm, this isn't a clean Kronecker product. Let me reconsider.
    
    # Actually, the correct way to see the Kronecker structure:
    # The 2x4x8 tensor is T_{2,4,8} = T_{2,2,2} ⊗ T_{1,2,4} or something similar.
    # But that doesn't help directly.
    
    # Let me just use the 8 independent Strassen blocks approach but try to find
    # a better rank-7 decomposition or use a different strategy.
    
    # For now, let's stick with the rank-56 solution and try to improve it.
    
    # Actually, let me try a different approach: use the fact that we can share
    # some computations between blocks.
    
    # The 8 blocks are indexed by (j_block, s) where j_block in {0,1,2,3} and s in {0,1}
    # Block (j, s) uses:
    # A rows 0,1; cols 2j, 2j+1 (if s=0) or cols 2j, 2j+1 (if s=1) -- wait, this is wrong
    
    # Let me re-examine the mapping.
    # For block (j, s):
    # A: uses rows 0,1 and cols 2j, 2j+1 (for s=0) or cols 2j, 2j+1 (for s=1)
    # Wait, that's the same. Let me re-read the original code.
    
    # In the original code:
    # s=0: A1 cols 0,1; B top half rows 0,1
    # s=1: A2 cols 2,3; B bottom half rows 2,3
    
    # But j varies from 0 to 3, so:
    # (j=0, s=0): A cols 0,1; B rows 0,1 cols 0,1
    # (j=1, s=0): A cols 0,1; B rows 0,1 cols 2,3
    # (j=2, s=0): A cols 0,1; B rows 0,1 cols 4,5
    # (j=3, s=0): A cols 0,1; B rows 0,1 cols 6,7
    # (j=0, s=1): A cols 2,3; B rows 2,3 cols 0,1
    # (j=1, s=1): A cols 2,3; B rows 2,3 cols 2,3
    # (j=2, s=1): A cols 2,3; B rows 2,3 cols 4,5
    # (j=3, s=1): A cols 2,3; B rows 2,3 cols 6,7
    
    # So there are 8 blocks, each using a 2x2 submatrix of A and a 2x2 submatrix of B.
    # The A submatrix depends only on s, and the B submatrix depends on both j and s.
    
    # For s=0: A uses cols 0,1 (same for all j)
    # For s=1: A uses cols 2,3 (same for all j)
    
    # So we can factor out the A part! The U matrix for all 4 blocks with s=0 is the same,
    # and the U matrix for all 4 blocks with s=1 is the same.
    
    # This means we can potentially reduce the rank by sharing U factors.
    
    # Let's construct the rank-56 solution first, then try to optimize.
    
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)] # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)] # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)] # 16 x 56
    
    # Mapping logic
    # j: 0..3 (col block of C)
    # s: 0..1 (selector for A row-block/B row-block)
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # Define mappings for local indices 0..3 to global indices
            # Local A: i_loc (0,1), j_loc (0,1) -> flat i_loc*2+j_loc
            # Global A: i_g (0,1), j_g (0,1,2,3) -> flat i_g*4+j_g
            a_map = [0]*4
            if s == 0:
                # A1: cols 0,1
                a_map[0] = 0*4 + 0 # i=0, j=0
                a_map[1] = 0*4 + 1 # i=0, j=1
                a_map[2] = 1*4 + 0 # i=1, j=0
                a_map[3] = 1*4 + 1 # i=1, j=1
            else:
                # A2: cols 2,3
                a_map[0] = 0*4 + 2
                a_map[1] = 0*4 + 3
                a_map[2] = 1*4 + 2
                a_map[3] = 1*4 + 3
                
            # Local B: j_loc (0,1), k_loc (0,1) -> flat j_loc*2+k_loc
            # Global B: j_g (0..3), k_g (0..7) -> flat j_g*8+k_g
            b_map = [0]*4
            if s == 0:
                # Top half (rows 0,1), cols 2j, 2j+1
                b_map[0] = 0*8 + (2*j)   # j_g=0, k_g=2j
                b_map[1] = 0*8 + (2*j+1) # j_g=0, k_g=2j+1
                b_map[2] = 1*8 + (2*j)   # j_g=1, k_g=2j
                b_map[3] = 1*8 + (2*j+1) # j_g=1, k_g=2j+1
            else:
                # Bottom half (rows 2,3), cols 2j, 2j+1
                b_map[0] = 2*8 + (2*j)
                b_map[1] = 2*8 + (2*j+1)
                b_map[2] = 3*8 + (2*j)
                b_map[3] = 3*8 + (2*j+1)
                
            # Local C: i_loc (0,1), k_loc (0,1) -> flat i_loc*2+k_loc
            # Global C: i_g (0,1), k_g (0..7)
            # W Index Convention: k_g * n + i_g = k_g * 2 + i_g
            c_map = [0]*4
            # k_g is 2j or 2j+1
            # i_g is 0 or 1
            
            # Local C index 0: i=0, k_loc=0 -> k_g=2j
            c_map[0] = (2*j) * 2 + 0
            # Local C index 1: i=0, k_loc=1 -> k_g=2j+1
            c_map[1] = (2*j+1) * 2 + 0
            # Local C index 2: i=1, k_loc=0 -> k_g=2j
            c_map[2] = (2*j) * 2 + 1
            # Local C index 3: i=1, k_loc=1 -> k_g=2j+1
            c_map[3] = (2*j+1) * 2 + 1

            # Fill U, V, W
            for l in range(7):
                r = base_r + l
                
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        ga = a_map[la]
                        U[ga][r] += val
                        
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        gb = b_map[lb]
                        V[gb][r] += val
                        
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        gc = c_map[lc]
                        W[gc][r] += val
                        
    if time.time() - start_time > t_limit:
        # Safety check, though this is deterministic and fast
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
