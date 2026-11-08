"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # 1. Construct M_{2,2,2} Strassen decomposition (Rank 7)
    # U: 4x7, V: 4x7, W: 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1
    
    # 2. Construct M_{4,4,4} via Kronecker Product
    # M_{4,4,4} = M_{2,2,2} \otimes M_{2,2,2}
    # Rank 49
    U4 = np.kron(U2, U2) # 16x49
    V4 = np.kron(V2, V2) # 16x49
    W4 = np.kron(W2, W2) # 16x49
    
    # 3. Construct M_{1,2,2} decomposition (Rank 1)
    # A: 1x2, B: 2x2, C: 1x2
    U1 = np.array([[1], [0]]) # 2x1
    V1 = np.array([[1], [1], [1], [1]]) # 4x1
    W1 = np.array([[1], [1]]) # 2x1
    
    # 4. Construct M_{4,8,8} = M_{4,4,4} \otimes M_{1,2,2}
    # Rank 49
    U488 = np.kron(U4, U1) # 32x49 (4*2 rows)
    V488 = np.kron(V4, V1) # 64x49 (4*4*2*2? No. V4 is 16x49. V1 is 4x1. 16*4=64)
    W488 = np.kron(W4, W1) # 32x49
    
    # 5. Extract M_{3,5,7} from M_{4,8,8}
    # A is 4x8. Keep rows i < 3.
    # B is 8x8. Keep rows j < 5, cols k < 7.
    # C is 4x8. Keep rows i < 3, cols k < 7.
    
    # Row indices for U (A): i = i1*2 + i2. Keep if i1 < 3 and i < 5?
    # Wait, A is 4x8 in M_{4,8,8}.
    # i1 in {0,1,2,3}, i2 in {0,1}.
    # We want A to be 3x5.
    # Rows of A: 0,1,2.
    # Cols of A: 0,1,2,3,4.
    
    u_rows = []
    for i1 in range(4):
        for i2 in range(2):
            i = i1 * 2 + i2
            if i1 < 3 and i < 5:
                u_rows.append(i)
                
    # Row indices for V (B): j = j1*2 + j2, k = k1*2 + k2.
    # V is indexed by j (rows of B) and k (cols of B)?
    # In M_{n,m,p}, V is (m x p) flattened.
    # M_{4,8,8}: m=8, p=8.
    # j in {0..7}, k in {0..7}.
    # We want B to be 5x7.
    # Rows j < 5, Cols k < 7.
    
    v_rows = []
    for j1 in range(4):
        for j2 in range(2):
            j = j1 * 2 + j2
            if j < 5:
                for k1 in range(4):
                    for k2 in range(2):
                        k = k1 * 2 + k2
                        if k < 7:
                            # Index in V488:
                            # V4 is 16x49. V4 index = j1*4 + j2?
                            # V4 is kron(V2, V2).
                            # V2 is 4x7.
                            # V4 index = (j1*2+j2)*4 + (k1*2+k2)?
                            # No, V2 indices are 0..3.
                            # V4 index = j1*4 + j2? No.
                            # V2 is for 2x2 matrix. Indices 0,1,2,3.
                            # V4 is for 4x4 matrix.
                            # Row index in V4: j4 = j1*2 + j2?
                            # No, V2 has 4 rows.
                            # V4 has 16 rows.
                            # Row index = j1*4 + j2?
                            # Let's check kron.
                            # V4[r, c] = V2[r1, c1] * V2[r2, c2].
                            # r = r1*4 + r2.
                            # r1 in 0..3, r2 in 0..3.
                            # r1 corresponds to j in 2x2.
                            # r2 corresponds to k in 2x2?
                            # No, V2 is for B (2x2).
                            # So r1 is row of B (2x2), r2 is col of B (2x2).
                            # So j4 = r1*2 + r2?
                            # Wait, V2 has 4 rows.
                            # If B is 2x2, rows are 0,1. Cols are 0,1.
                            # Flattened index = row*2 + col.
                            # So r2 is col?
                            # In kron(V2, V2), the index is r1*4 + r2.
                            # r1 is the "row" index of the first 2x2 block?
                            # Actually, V2[i,j] where i is row, j is col.
                            # Flattened V2 has 4 rows.
                            # V4 has 16 rows.
                            # Index = i1*4 + i2.
                            # i1 is row of 4x4 B. i2 is col of 4x4 B.
                            # i1 = j1*2 + j2?
                            # j1 is row of first 2x2, j2 is row of second 2x2?
                            # No, 4x4 B is formed by 2x2 blocks.
                            # B[i,j] where i,j in 0..3.
                            # i = i1*2 + i2.
                            # j = j1*2 + j2.
                            # So flattened index = i*4 + j = (i1*2+i2)*4 + (j1*2+j2).
                            
                            v_idx = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                            v_rows.append(v_idx)
                            
    # Row indices for W (C): i = i1*2 + i2, k = k1*2 + k2.
    # W is (p x n) flattened?
    # Task says W[k*n+i, r].
    # So W index = k*n + i.
    # In M_{4,8,8}, n=4, p=8.
    # W index = k*4 + i.
    # k in 0..7, i in 0..3.
    # We want C to be 3x7.
    # i < 3, k < 7.
    
    w_rows = []
    for k1 in range(4):
        for k2 in range(2):
            k = k1 * 2 + k2
            if k < 7:
                for i1 in range(4):
                    for i2 in range(2):
                        i = i1 * 2 + i2
                        if i < 3:
                            # W488 index:
                            # W4 is 16x49.
                            # W4 index = k4*4 + i4.
                            # k4 = k1*2 + k2.
                            # i4 = i1*2 + i2.
                            w_idx = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                            w_rows.append(w_idx)
                            
    U_final = U488[u_rows, :]
    V_final = V488[v_rows, :]
    W_final = W488[w_rows, :]
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
