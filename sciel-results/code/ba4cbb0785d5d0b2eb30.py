"""Optimized matrix multiplication tensor decomposition using Strassen structure."""
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen with rank 7.
        U: (4, 7) for A[i*2+j]
        V: (4, 7) for B[j*2+k]  
        W: (4, 7) for C[k*2+i] (trace convention)
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
        # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3 (k*2+i)
        
        # U: A side
        U[0][0] = 1; U[3][0] = 1      # M1: A00 + A11
        U[2][1] = 1; U[3][1] = 1      # M2: A10 + A11
        U[0][2] = 1                   # M3: A00
        U[3][3] = 1                   # M4: A11
        U[0][4] = 1; U[1][4] = 1      # M5: A00 + A01
        U[2][5] = 1; U[0][5] = -1     # M6: A10 - A00
        U[1][6] = 1; U[3][6] = -1     # M7: A01 - A11
        
        # V: B side
        V[0][0] = 1; V[3][0] = 1      # M1: B00 + B11
        V[0][1] = 1                   # M2: B00
        V[2][2] = 1; V[3][2] = -1     # M3: B10 - B11
        V[0][3] = 1; V[2][3] = -1     # M4: B00 - B10
        V[3][4] = 1                   # M5: B11
        V[0][5] = 1; V[1][5] = 1      # M6: B00 + B01
        V[2][6] = 1; V[3][6] = 1      # M7: B10 + B11
        
        # W: C side (k*2+i)
        # C00 (k=0,i=0, idx=0): M1 + M4 - M5 + M7
        W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
        # C01 (k=1,i=0, idx=2): M3 + M5
        W[2][2] = 1; W[2][4] = 1
        # C10 (k=0,i=1, idx=1): M2 + M4
        W[1][1] = 1; W[1][3] = 1
        # C11 (k=1,i=1, idx=3): M1 - M2 + M3 + M6
        W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
        
        return U, V, W
    
    # Build 4x4x4 using block Strassen
    # A, B, C are 4x4, split into 2x2 blocks of size 2x2
    # U_444: (16, 49), V_444: (16, 49), W_444: (16, 49)
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    U2, V2, W2 = strassen_2x2x2()
    
    # Block-level Strassen coefficients
    # t=0: M1=(A00+A11)(B00+B11)
    # t=1: M2=(A10+A11)B00
    # t=2: M3=A00(B10-B11)
    # t=3: M4=A11(B00-B10)
    # t=4: M5=(A00+A01)B11
    # t=5: M6=(A10-A00)(B00+B01)
    # t=6: M7=(A01-A11)(B10+B11)
    
    # A block coeffs: block in {00,01,10,11} -> 0,1,2,3
    block_A = [
        [1, 0, 0, 1],   # t=0: A00 + A11
        [0, 0, 1, 1],   # t=1: A10 + A11
        [1, 0, 0, 0],   # t=2: A00
        [0, 0, 0, 1],   # t=3: A11
        [1, 1, 0, 0],   # t=4: A00 + A01
        [-1, 0, 1, 0],  # t=5: A10 - A00
        [0, 1, 0, -1],  # t=6: A01 - A11
    ]
    
    block_B = [
        [1, 0, 0, 1],   # t=0: B00 + B11
        [1, 0, 0, 0],   # t=1: B00
        [0, 0, 1, -1],  # t=2: B10 - B11
        [1, 0, -1, 0],  # t=3: B00 - B10
        [0, 0, 0, 1],   # t=4: B11
        [1, 1, 0, 0],   # t=5: B00 + B01
        [0, 0, 1, 1],   # t=6: B10 + B11
    ]
    
    # C block coeffs: cblock in {00,01,10,11} -> 0,1,2,3
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    block_C = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    # Block to linear indices (4x4 matrix, row-major)
    # Block 00: rows 0-1, cols 0-1 -> [0,1,4,5]
    # Block 01: rows 0-1, cols 2-3 -> [2,3,6,7]
    # Block 10: rows 2-3, cols 0-1 -> [8,9,12,13]
    # Block 11: rows 2-3, cols 2-3 -> [10,11,14,15]
    blk_idx = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            # U side
            for li in range(4):
                u2v = U2[li][s]
                if u2v != 0:
                    for b in range(4):
                        ca = block_A[t][b]
                        if ca != 0:
                            U_444[blk_idx[b][li]][col] += ca * u2v
            # V side
            for lj in range(4):
                v2v = V2[lj][s]
                if v2v != 0:
                    for b in range(4):
                        cb = block_B[t][b]
                        if cb != 0:
                            V_444[blk_idx[b][lj]][col] += cb * v2v
            # W side
            for lk in range(4):
                w2v = W2[lk][s]
                if w2v != 0:
                    for cb in range(4):
                        cc = block_C[cb][t]
                        if cc != 0:
                            W_444[blk_idx[cb][lk]][col] += cc * w2v
    
    # Build 4x4x3 schoolbook
    U_443 = [[0]*48 for _ in range(16)]
    V_443 = [[0]*48 for _ in range(12)]
    W_443 = [[0]*48 for _ in range(12)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_443[i*4+j][col] = 1
                V_443[j*3+k][col] = 1
                W_443[k*4+i][col] = 1
                col += 1
    
    # Combine
    rank = 97
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    # Part 1: 4x4x4
    for r in range(16):
        for c in range(49):
            U[r][c] = U_444[r][c]
            V[r][c] = V_444[r][c]
            W[r][c] = W_444[r][c]
    
    # Part 2: 4x4x3
    for r in range(16):
        for c in range(48):
            U[r][49+c] = U_443[r][c]
    
    for j in range(4):
        for k in range(3):
            vr = j*3 + k
            vfr = j*7 + (4+k)
            for c in range(48):
                V[vfr][49+c] = V_443[vr][c]
    
    for k in range(3):
        for i in range(4):
            wr = k*4 + i
            wfr = (4+k)*4 + i
            for c in range(48):
                W[wfr][49+c] = W_443[wr][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

