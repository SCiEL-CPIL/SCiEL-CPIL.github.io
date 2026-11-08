import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: 6x6 as 3x3 blocks of 2x2 matrices.
    # Rank = Rank(3x3) * Rank(2x2)
    
    # 1. Define 2x2 Strassen Decomposition (Rank 7)
    # U, V, W are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices: 0:00, 1:01, 2:10, 3:11
    # U2[row, col]
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00 gets M1, C11 gets M1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C10 gets M2
    
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C01 gets M3
    
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00 gets M4, C10 gets M4
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00 gets -M5, C01 gets M5
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11 gets M6
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00 gets M7
    
    # 2. Define 3x3 Schoolbook Decomposition (Rank 27)
    # We use schoolbook for safety, Rank 27.
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j, col] = 1
                V3[j * 3 + k, col] = 1
                W3[k * 3 + i, col] = 1
                col += 1
                
    R2 = 7
    R3 = 27
    R_total = R2 * R3
    
    # 3. Construct 6x6 Decomposition
    U6 = np.zeros((36, R_total))
    V6 = np.zeros((36, R_total))
    W6 = np.zeros((36, R_total))
    
    for b in range(R3):
        for c in range(R2):
            col_idx = b * R2 + c
            
            # U6
            for ib in range(3):
                if U3[ib * 3 + 0, b] != 0 or U3[ib * 3 + 1, b] != 0 or U3[ib * 3 + 2, b] != 0:
                    # U3 index is i_block * 3 + j_block? No, U3 is for A.
                    # A is 3x3 blocks. U3 row index is i_block * 3 + j_block?
                    # Wait, U3 corresponds to A. A has indices i, j.
                    # In the 3x3 problem, U3[i, j, r].
                    # Here U3 is flattened.
                    # My U3 construction above: U3[i*3+j, col].
                    # So for a fixed block multiplication index b (which is col in U3),
                    # we need to iterate over the non-zero entries in U3[:, b].
                    
                    pass
            
            # Simpler: Iterate over all entries
            for i3 in range(9):
                val_u3 = U3[i3, b]
                if val_u3 == 0: continue
                # i3 = i_block * 3 + j_block? No.
                # In my U3 construction: i is row, j is col of A.
                # i3 = i * 3 + j.
                i_block = i3 // 3
                j_block = i3 % 3
                
                # U2 index is i_local * 2 + j_local? No.
                # U2 is for 2x2 A. U2[row, col].
                # row = i_local * 2 + j_local? No, A is 2x2.
                # U2 row index is i * 2 + j.
                
                # We need to combine U3[i_block, j_block, b] with U2[i_local, j_local, c].
                # But U3 is flattened.
                # The block (i_block, j_block) contains 4 elements.
                # For each element (i_local, j_local) in the 2x2 block:
                for il in range(2):
                    for jl in range(2):
                        # U2 row index: il * 2 + jl
                        val_u2 = U2[il * 2 + jl, c]
                        if val_u2 == 0: continue
                        
                        # Global row index in U6:
                        # i_global = i_block * 2 + il
                        # j_global = j_block * 2 + jl
                        # Row in U6 = i_global * 6 + j_global
                        row_u6 = (i_block * 2 + il) * 6 + (j_block * 2 + jl)
                        
                        U6[row_u6, col_idx] = val_u3 * val_u2
            
            # V6
            for j3 in range(9):
                val_v3 = V3[j3, b]
                if val_v3 == 0: continue
                j_block = j3 // 3
                k_block = j3 % 3
                
                for jl in range(2):
                    for kl in range(2):
                        val_v2 = V2[jl * 2 + kl, c]
                        if val_v2 == 0: continue
                        
                        row_v6 = (j_block * 2 + jl) * 6 + (k_block * 2 + kl)
                        V6[row_v6, col_idx] = val_v3 * val_v2
            
            # W6
            for k3 in range(9):
                val_w3 = W3[k3, b]
                if val_w3 == 0: continue
                k_block = k3 // 3
                i_block = k3 % 3
                
                for kl in range(2):
                    for il in range(2):
                        val_w2 = W2[kl * 2 + il, c]
                        if val_w2 == 0: continue
                        
                        row_w6 = (k_block * 2 + kl) * 6 + (i_block * 2 + il)
                        W6[row_w6, col_idx] = val_w3 * val_w2
                        
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

# EVOLVE-BLOCK-END
