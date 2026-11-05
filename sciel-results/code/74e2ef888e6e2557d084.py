"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 # seconds
    
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen M2 factors (Rank 7)
    # Indices: U2[i2*2+j2, c2], V2[j2*2+k2, c2], W2[k2*2+i2, c2]
    # i2, j2, k2 in {0, 1}
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # c=0: M1 = (A00+A11)(B00+B11) -> C00+C11
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # c=1: M2 = (A10+A11)B00 -> C10-C11 (Wait, C10+M2, C11-M2)
    # C10 index (1,0) -> W2[2]? No, W2 index is k2*2+i2.
    # C10: k=1, i=0 -> idx 2. C11: k=1, i=1 -> idx 3.
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # c=2: M3 = A00(B01-B11) -> C01+C11
    # C01: k=0, i=1 -> idx 1. C11: k=1, i=1 -> idx 3.
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # c=3: M4 = A11(B10-B00) -> C10+C00
    # C10: idx 2. C00: idx 0.
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1; W2[0][3] = 1
    
    # c=4: M5 = (A00+A01)B11 -> -C00+C01
    # C00: idx 0. C01: idx 1.
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # c=5: M6 = (A10-A00)(B00+B01) -> C11
    # C11: idx 3.
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # c=6: M7 = (A01-A11)(B10+B11) -> C00
    # C00: idx 0.
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # 2. Define Schoolbook M3 factors (Rank 27)
    # Indices: U3[i3*3+j3, c3], V3[j3*3+k3, c3], W3[k3*3+i3, c3]
    # i3, j3, k3 in {0, 1, 2}
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    c3 = 0
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                U3[i3*3+j3][c3] = 1
                V3[j3*3+k3][c3] = 1
                W3[k3*3+i3][c3] = 1
                c3 += 1
                
    # 3. Construct M6 factors via Tensor Product
    # Rank = 7 * 27 = 189
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Loop over all combinations of M2 and M3 indices
    # i = 3*i2 + i3
    # j = 3*j2 + j3
    # k = 3*k2 + k3
    
    for i2 in range(2):
        for j2 in range(2):
            for k2 in range(2):
                for i3 in range(3):
                    for j3 in range(3):
                        for k3 in range(3):
                            # This loops over all elements of the 6x6 matrices
                            # But we need to fill the columns based on c2, c3
                            pass # This is inefficient. Better to loop over columns.
                            
    # Efficient construction:
    # Column c = c2 * 27 + c3
    for c2 in range(7):
        for c3 in range(27):
            c = c2 * 27 + c3
            # Iterate over non-zero entries in U2[:, c2]
            for i2 in range(2):
                for j2 in range(2):
                    val_u2 = U2[i2*2+j2][c2]
                    if val_u2 == 0: continue
                    # Iterate over non-zero entries in U3[:, c3]
                    for i3 in range(3):
                        for j3 in range(3):
                            val_u3 = U3[i3*3+j3][c3]
                            if val_u3 == 0: continue
                            
                            i = 3*i2 + i3
                            j = 3*j2 + j3
                            row_u = i*6 + j
                            U[row_u][c] += val_u2 * val_u3
                            
                            # V part
                            # V2 index is j2, k2. V3 index is j3, k3.
                            # We need to sum over k2, k3 as well?
                            # No, the tensor product structure means:
                            # U6[i,j, c2,c3] = U2[i2,j2,c2] * U3[i3,j3,c3]
                            # V6[j,k, c2,c3] = V2[j2,k2,c2] * V3[j3,k3,c3]
                            # W6[k,i, c2,c3] = W2[k2,i2,c2] * W3[k3,i3,c3]
                            
                            # So for a fixed c, we need to fill U, V, W rows.
                            # The above loop only filled U.
                            # We need to fill V and W in the same pass or separate passes.
                            # Since the indices are coupled (j2, j3 for U and V), 
                            # it's best to iterate over all indices for each c.
                            
    # Let's restart the construction loop properly.
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for c2 in range(7):
        for c3 in range(27):
            c = c2 * 27 + c3
            
            # Fill U
            for i2 in range(2):
                for j2 in range(2):
                    u2_val = U2[i2*2+j2][c2]
                    if u2_val == 0: continue
                    for i3 in range(3):
                        for j3 in range(3):
                            u3_val = U3[i3*3+j3][c3]
                            if u3_val == 0: continue
                            
                            i = 3*i2 + i3
                            j = 3*j2 + j3
                            row_u = i*6 + j
                            U[row_u][c] = u2_val * u3_val
                            
            # Fill V
            for j2 in range(2):
                for k2 in range(2):
                    v2_val = V2[j2*2+k2][c2]
                    if v2_val == 0: continue
                    for j3 in range(3):
                        for k3 in range(3):
                            v3_val = V3[j3*3+k3][c3]
                            if v3_val == 0: continue
                            
                            j = 3*j2 + j3
                            k = 3*k2 + k3
                            row_v = j*6 + k
                            V[row_v][c] = v2_val * v3_val
                            
            # Fill W
            for k2 in range(2):
                for i2 in range(2):
                    w2_val = W2[k2*2+i2][c2]
                    if w2_val == 0: continue
                    for k3 in range(3):
                        for i3 in range(3):
                            w3_val = W3[k3*3+i3][c3]
                            if w3_val == 0: continue
                            
                            k = 3*k2 + k3
                            i = 3*i2 + i3
                            row_w = k*6 + i
                            W[row_w][c] = w2_val * w3_val
                            
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
