import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strassen decomposition for 2x2 * 2x2 (rank 7)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # A flat: [a, b, c, d] (indices 0,1,2,3)
    # B flat: [e, f, g, h] (indices 0,1,2,3)
    # C flat: [C11, C12, C21, C22] (indices 0,1,2,3)
    
    strassen_U = [
        [1, 0, 0, 0],   # M1: a
        [1, 0, 1, 0],   # M2: a+c
        [0, 0, 1, 1],   # M3: c+d
        [0, 1, 0, -1],  # M4: b-d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, -1, 0],  # M7: b-c
    ]
    
    strassen_V = [
        [1, 0, 0, 1],   # M1: e+h
        [0, 0, 0, 1],   # M2: h
        [1, 0, 0, 0],   # M3: e
        [0, 0, 1, 1],   # M4: g+h
        [1, 0, 0, 0],   # M5: e
        [0, 1, 0, 1],   # M6: f+h
        [1, 1, 0, 0],   # M7: e+f
    ]
    
    strassen_W = [
        [1, 0, 0, 1],   # M1: C11 += 1, C22 += 1
        [0, 0, 1, -1],  # M2: C21 += 1, C22 -= 1
        [0, 0, 1, 1],   # M3: C21 += 1, C22 += 1
        [1, 0, 1, 0],   # M4: C11 += 1, C21 += 1
        [-1, 1, 0, 0],  # M5: C11 -= 1, C12 += 1
        [0, 0, 0, 1],   # M6: C22 += 1
        [1, 0, 0, 0],   # M7: C11 += 1
    ]
    
    # Schoolbook for 2x2 * 2x1 (rank 4)
    # A = [[a, b], [c, d]], B = [[x], [y]], C = A*B (2x1)
    # A flat: [a, b, c, d] (indices 0,1,2,3)
    # B flat: [x, y] (indices 0,1)
    # C flat: [C0, C1] (indices 0,1)
    # C0 = a*x + b*y, C1 = c*x + d*y
    
    mv_U = [
        [1, 0, 0, 0],  # M1: a
        [0, 1, 0, 0],  # M2: b
        [0, 0, 1, 0],  # M3: c
        [0, 0, 0, 1],  # M4: d
    ]
    
    mv_V = [
        [1, 0],  # M1: x
        [0, 1],  # M2: y
        [1, 0],  # M3: x
        [0, 1],  # M4: y
    ]
    
    mv_W = [
        [1, 0],  # M1: C0 += 1
        [1, 0],  # M2: C0 += 1
        [0, 1],  # M3: C1 += 1
        [0, 1],  # M4: C1 += 1
    ]
    
    # Build the full decomposition for 2x4 * 4x7
    # A(2x4) flat: [A00,A01,A02,A03,A10,A11,A12,A13] (indices 0-7)
    # B(4x7) flat: [B00..B06,B10..B16,B20..B26,B30..B36] (indices 0-27)
    # C(2x7) flat: [C00..C06,C10..C16] (indices 0-13)
    
    # Partition:
    # A1 = [[A00,A01],[A10,A11]] -> A flat indices [0,1,4,5]
    # A2 = [[A02,A03],[A12,A13]] -> A flat indices [2,3,6,7]
    
    # B1 = rows 0-1 of B (2x7): [B00..B06, B10..B16] -> B flat indices [0-6, 7-13]
    # B2 = rows 2-3 of B (2x7): [B20..B26, B30..B36] -> B flat indices [14-20, 21-27]
    
    # C(2x7) flat: [C00..C06, C10..C16]
    # C00..C06 are indices 0-6, C10..C16 are indices 7-13
    
    U_list = []
    V_list = []
    W_list = []
    
    # A index mappings for Strassen (local [a,b,c,d] to global A flat)
    # A1: a=A00(idx0), b=A01(idx1), c=A10(idx4), d=A11(idx5)
    a1_map = [0, 1, 4, 5]
    # A2: a=A02(idx2), b=A03(idx3), c=A12(idx6), d=A13(idx7)
    a2_map = [2, 3, 6, 7]
    
    # B block index mappings
    # For B1 (rows 0-1):
    #   B1_1 = B[0:2,0:2] -> B flat [B00,B01,B10,B11] = indices [0,1,7,8]
    #   B1_2 = B[0:2,2:4] -> B flat [B02,B03,B12,B13] = indices [2,3,9,10]
    #   B1_3 = B[0:2,4:6] -> B flat [B04,B05,B14,B15] = indices [4,5,11,12]
    #   B1_4 = B[0:2,6:7] -> B flat [B06,B16] = indices [6,13]
    # For B2 (rows 2-3):
    #   B2_1 = B[2:4,0:2] -> B flat [B20,B21,B30,B31] = indices [14,15,21,22]
    #   B2_2 = B[2:4,2:4] -> B flat [B22,B23,B32,B33] = indices [16,17,23,24]
    #   B2_3 = B[2:4,4:6] -> B flat [B24,B25,B34,B35] = indices [18,19,25,26]
    #   B2_4 = B[2:4,6:7] -> B flat [B26,B36] = indices [20,27]
    
    b1_maps = [
        [0, 1, 7, 8],   # B1_1
        [2, 3, 9, 10],  # B1_2
        [4, 5, 11, 12], # B1_3
        [6, 13],         # B1_4 (2x1, only 2 elements)
    ]
    b2_maps = [
        [14, 15, 21, 22],  # B2_1
        [16, 17, 23, 24],  # B2_2
        [18, 19, 25, 26],  # B2_3
        [20, 27],           # B2_4 (2x1, only 2 elements)
    ]
    
    # C index mappings
    # For C(2x7) flat: [C00..C06, C10..C16]
    # C[0:2, 0:2] -> [C00,C01,C10,C11] = indices [0,1,7,8]
    # C[0:2, 2:4] -> [C02,C03,C12,C13] = indices [2,3,9,10]
    # C[0:2, 4:6] -> [C04,C05,C14,C15] = indices [4,5,11,12]
    # C[0:2, 6:7] -> [C06,C16] = indices [6,13]
    
    c_maps = [
        [0, 1, 7, 8],   # cols 0-1
        [2, 3, 9, 10],  # cols 2-3
        [4, 5, 11, 12], # cols 4-5
        [6, 13],         # col 6 (2x1, only 2 elements)
    ]
    
    # For each row-block i (0 for A1/B1, 1 for A2/B2):
    for i in range(2):
        a_map = a1_map if i == 0 else a2_map
        b_maps = b1_maps if i == 0 else b2_maps
        
        for j in range(4):
            c_map = c_maps[j]
            
            if j < 3:
                # 2x2 * 2x2 block: use Strassen (rank 7)
                b_map = b_maps[j]  # 4 elements
                for r in range(7):
                    # U: 8 elements (A flat)
                    u = [0.0] * 8
                    for k in range(4):
                        u[a_map[k]] = strassen_U[r][k]
                    
                    # V: 28 elements (B flat)
                    v = [0.0] * 28
                    for k in range(4):
                        v[b_map[k]] = strassen_V[r][k]
                    
                    # W: 14 elements (C flat)
                    w = [0.0] * 14
                    for k in range(4):
                        w[c_map[k]] = strassen_W[r][k]
                    
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
            else:
                # 2x2 * 2x1 block: use schoolbook (rank 4)
                b_map = b_maps[j]  # 2 elements
                for r in range(4):
                    # U: 8 elements (A flat)
                    u = [0.0] * 8
                    for k in range(4):
                        u[a_map[k]] = mv_U[r][k]
                    
                    # V: 28 elements (B flat)
                    v = [0.0] * 28
                    for k in range(2):
                        v[b_map[k]] = mv_V[r][k]
                    
                    # W: 14 elements (C flat)
                    w = [0.0] * 14
                    for k in range(2):
                        w[c_map[k]] = mv_W[r][k]
                    
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
    
    rank = len(U_list)
    
    # Verify correctness
    U = np.array(U_list, dtype=float).T  # (8, rank)
    V = np.array(V_list, dtype=float).T  # (28, rank)
    W = np.array(W_list, dtype=float).T  # (14, rank)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(2, 4)
    B_test = np.random.randn(4, 7)
    C_expected = A_test @ B_test
    
    # Reconstruct using tensor decomposition:
    # C_flat[ci] = sum_k W[ci,k] * (sum_ai A[ai]*U[ai,k]) * (sum_bi B[bi]*V[bi,k])
    A_flat = A_test.flatten()  # row-major
    B_flat = B_test.flatten()  # row-major
    C_flat = np.zeros(14)
    
    for k in range(rank):
        a_proj = np.dot(A_flat, U[:, k])
        b_proj = np.dot(B_flat, V[:, k])
        C_flat += W[:, k] * a_proj * b_proj
    
    C_reconstructed = C_flat.reshape(2, 7)
    
    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        U_sb = np.zeros((8, 56))
        V_sb = np.zeros((28, 56))
        W_sb = np.zeros((14, 56))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(7):
                    U_sb[i * 4 + j, col] = 1
                    V_sb[j * 7 + k, col] = 1
                    W_sb[k * 2 + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
