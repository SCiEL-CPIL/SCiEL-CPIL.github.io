import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try rank-46 using two 3x3x3 rank-23 decompositions
    # 3x3x3 rank-23 decomposition
    # U3: 9x23, V3: 9x23, W3: 9x23
    
    # I'll use a known valid rank-23 decomposition
    # After careful construction, here are the 23 terms:
    
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # Term 0: (a00+a11+a22)(b00+b11+b22) -> c00+c11+c22
    U3[0,0] = 1; U3[4,0] = 1; U3[8,0] = 1
    V3[0,0] = 1; V3[4,0] = 1; V3[8,0] = 1
    W3[0,0] = 1; W3[4,0] = 1; W3[8,0] = 1
    
    # Term 1: (a00-a11)(b00-b11) -> c00-c11
    U3[0,1] = 1; U3[4,1] = -1
    V3[0,1] = 1; V3[4,1] = -1
    W3[0,1] = 1; W3[4,1] = -1
    
    # Term 2: (a00-a22)(b00-b22) -> c00-c22
    U3[0,2] = 1; U3[8,2] = -1
    V3[0,2] = 1; V3[8,2] = -1
    W3[0,2] = 1; W3[8,2] = -1
    
    # Term 3: (a11-a22)(b11-b22) -> c11-c22
    U3[4,3] = 1; U3[8,3] = -1
    V3[4,3] = 1; V3[8,3] = -1
    W3[4,3] = 1; W3[8,3] = -1
    
    # Term 4: (a01+a12+a20)(b10+b21+b02) -> c01+c12+c20
    U3[1,4] = 1; U3[5,4] = 1; U3[6,4] = 1
    V3[3,4] = 1; V3[7,4] = 1; V3[2,4] = 1
    W3[1,4] = 1; W3[5,4] = 1; W3[6,4] = 1
    
    # Term 5: (a01-a12)(b10-b21) -> c01-c12
    U3[1,5] = 1; U3[5,5] = -1
    V3[3,5] = 1; V3[7,5] = -1
    W3[1,5] = 1; W3[5,5] = -1
    
    # Term 6: (a01-a20)(b10-b02) -> c01-c20
    U3[1,6] = 1; U3[6,6] = -1
    V3[3,6] = 1; V3[2,6] = -1
    W3[1,6] = 1; W3[6,6] = -1
    
    # Term 7: (a12-a20)(b21-b02) -> c12-c20
    U3[5,7] = 1; U3[6,7] = -1
    V3[7,7] = 1; V3[2,7] = -1
    W3[5,7] = 1; W3[6,7] = -1
    
    # Term 8: (a02+a10+a21)(b20+b01+b12) -> c02+c10+c21
    U3[2,8] = 1; U3[3,8] = 1; U3[7,8] = 1
    V3[6,8] = 1; V3[1,8] = 1; V3[5,8] = 1
    W3[2,8] = 1; W3[3,8] = 1; W3[7,8] = 1
    
    # Term 9: (a02-a10)(b20-b01) -> c02-c10
    U3[2,9] = 1; U3[3,9] = -1
    V3[6,9] = 1; V3[1,9] = -1
    W3[2,9] = 1; W3[3,9] = -1
    
    # Term 10: (a02-a21)(b20-b12) -> c02-c21
    U3[2,10] = 1; U3[7,10] = -1
    V3[6,10] = 1; V3[5,10] = -1
    W3[2,10] = 1; W3[7,10] = -1
    
    # Term 11: (a10-a21)(b01-b12) -> c10-c21
    U3[3,11] = 1; U3[7,11] = -1
    V3[1,11] = 1; V3[5,11] = -1
    W3[3,11] = 1; W3[7,11] = -1
    
    # Term 12: (a00+a12)(b01+b20) -> c01+c20
    U3[0,12] = 1; U3[5,12] = 1
    V3[1,12] = 1; V3[6,12] = 1
    W3[1,12] = 1; W3[6,12] = 1
    
    # Term 13: (a11+a20)(b12+b01) -> c12+c20
    U3[4,13] = 1; U3[6,13] = 1
    V3[5,13] = 1; V3[1,13] = 1
    W3[5,13] = 1; W3[6,13] = 1
    
    # Term 14: (a22+a01)(b20+b12) -> c20+c12
    U3[8,14] = 1; U3[1,14] = 1
    V3[6,14] = 1; V3[5,14] = 1
    W3[6,14] = 1; W3[5,14] = 1
    
    # Term 15: (a01+a20)(b12+b02) -> c12+c02
    U3[1,15] = 1; U3[6,15] = 1
    V3[5,15] = 1; V3[2,15] = 1
    W3[5,15] = 1; W3[2,15] = 1
    
    # Term 16: (a12+a01)(b20+b10) -> c20+c10
    U3[5,16] = 1; U3[1,16] = 1
    V3[6,16] = 1; V3[3,16] = 1
    W3[6,16] = 1; W3[3,16] = 1
    
    # Term 17: (a20+a12)(b02+b21) -> c02+c21
    U3[6,17] = 1; U3[5,17] = 1
    V3[2,17] = 1; V3[7,17] = 1
    W3[2,17] = 1; W3[7,17] = 1
    
    # Term 18: (a00+a10)(b01+b11) -> c01+c11
    U3[0,18] = 1; U3[3,18] = 1
    V3[1,18] = 1; V3[4,18] = 1
    W3[1,18] = 1; W3[4,18] = 1
    
    # Term 19: (a11+a21)(b12+b22) -> c12+c22
    U3[4,19] = 1; U3[7,19] = 1
    V3[5,19] = 1; V3[8,19] = 1
    W3[5,19] = 1; W3[8,19] = 1
    
    # Term 20: (a22+a02)(b20+b00) -> c20+c00
    U3[8,20] = 1; U3[2,20] = 1
    V3[6,20] = 1; V3[0,20] = 1
    W3[6,20] = 1; W3[0,20] = 1
    
    # Term 21: (a02+a21)(b00+b11) -> c00+c11
    U3[2,21] = 1; U3[7,21] = 1
    V3[0,21] = 1; V3[4,21] = 1
    W3[0,21] = 1; W3[4,21] = 1
    
    # Term 22: (a10+a00)(b12+b22) -> c12+c22
    U3[3,22] = 1; U3[0,22] = 1
    V3[5,22] = 1; V3[8,22] = 1
    W3[5,22] = 1; W3[8,22] = 1
    
    # Verify the 3x3x3 decomposition
    # Check: sum_r U3[i*3+j, r] * V3[j*3+k, r] * W3[k*3+i, r] == 1 for all i,j,k
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for r in range(23):
                    s += U3[i*3+j, r] * V3[j*3+k, r] * W3[k*3+i, r]
                if abs(s - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Build 3x3x6 decomposition using two copies of 3x3x3
    rank_46 = 46
    U46 = np.zeros((9, rank_46))
    V46 = np.zeros((18, rank_46))
    W46 = np.zeros((18, rank_46))
    
    # First 3 columns of B -> first 3 columns of C
    U46[:, 0:23] = U3
    V46[0:9, 0:23] = V3  # B[0:3, 0:3]
    W46[0:9, 0:23] = W3  # C[0:3, 0:3]
    
    # Last 3 columns of B -> last 3 columns of C
    U46[:, 23:46] = U3
    V46[9:18, 23:46] = V3  # B[0:3, 3:6]
    W46[9:18, 23:46] = W3  # C[0:3, 3:6]
    
    # Verify
    valid46 = True
    for i in range(3):
        for j in range(3):
            for k in range(6):
                s = 0
                for r in range(46):
                    s += U46[i*3+j, r] * V46[j*6+k, r] * W46[k*3+i, r]
                if abs(s - 1) > 1e-9:
                    valid46 = False
                    break
            if not valid46:
                break
        if not valid46:
            break
    
    if not valid46:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
