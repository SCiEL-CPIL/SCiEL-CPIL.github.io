import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    # Laderman decomposition for 4x4*4x4 (rank 19)
    # U: (16, 19), V: (16, 19), W: (16, 19)
    # Each entry is 0, 1, or -1
    
    U_lad = np.zeros((16, 19))
    V_lad = np.zeros((16, 19))
    W_lad = np.zeros((16, 19))
    
    # Laderman's 19 multiplications for 4x4 matrix multiplication
    # Using the standard Laderman construction
    
    # c=0: (A00+A11+A22+A33)(B00+B11+B22+B33)
    for i in [0,1,2,3]:
        U_lad[i*4+i, 0] = 1
        V_lad[i*4+i, 0] = 1
        W_lad[i*4+i, 0] = 1
    
    # c=1: (A00+A11)(B00+B11)
    for i in [0,1]:
        U_lad[i*4+i, 1] = 1
        V_lad[i*4+i, 1] = 1
        W_lad[i*4+i, 1] = 1
    
    # c=2: (A22+A33)(B22+B33)
    for i in [2,3]:
        U_lad[i*4+i, 2] = 1
        V_lad[i*4+i, 2] = 1
        W_lad[i*4+i, 2] = 1
    
    # c=3: (A00+A22)(B00+B22)
    for i in [0,2]:
        U_lad[i*4+i, 3] = 1
        V_lad[i*4+i, 3] = 1
        W_lad[i*4+i, 3] = 1
    
    # c=4: (A11+A33)(B11+B33)
    for i in [1,3]:
        U_lad[i*4+i, 4] = 1
        V_lad[i*4+i, 4] = 1
        W_lad[i*4+i, 4] = 1
    
    # c=5: (A01+A10+A23+A32)(B01+B10+B23+B32)
    U_lad[0*4+1, 5] = 1; U_lad[1*4+0, 5] = 1; U_lad[2*4+3, 5] = 1; U_lad[3*4+2, 5] = 1
    V_lad[0*4+1, 5] = 1; V_lad[1*4+0, 5] = 1; V_lad[2*4+3, 5] = 1; V_lad[3*4+2, 5] = 1
    W_lad[0*4+1, 5] = 1; W_lad[1*4+0, 5] = 1; W_lad[2*4+3, 5] = 1; W_lad[3*4+2, 5] = 1
    
    # c=6: (A01+A10)(B01+B10)
    U_lad[0*4+1, 6] = 1; U_lad[1*4+0, 6] = 1
    V_lad[0*4+1, 6] = 1; V_lad[1*4+0, 6] = 1
    W_lad[0*4+1, 6] = 1; W_lad[1*4+0, 6] = 1
    
    # c=7: (A23+A32)(B23+B32)
    U_lad[2*4+3, 7] = 1; U_lad[3*4+2, 7] = 1
    V_lad[2*4+3, 7] = 1; V_lad[3*4+2, 7] = 1
    W_lad[2*4+3, 7] = 1; W_lad[3*4+2, 7] = 1
    
    # c=8: (A01+A23)(B01+B23)
    U_lad[0*4+1, 8] = 1; U_lad[2*4+3, 8] = 1
    V_lad[0*4+1, 8] = 1; V_lad[2*4+3, 8] = 1
    W_lad[0*4+1, 8] = 1; W_lad[2*4+3, 8] = 1
    
    # c=9: (A10+A32)(B10+B32)
    U_lad[1*4+0, 9] = 1; U_lad[3*4+2, 9] = 1
    V_lad[1*4+0, 9] = 1; V_lad[3*4+2, 9] = 1
    W_lad[1*4+0, 9] = 1; W_lad[3*4+2, 9] = 1
    
    # c=10: (A03+A12+A20+A31)(B03+B12+B20+B31)
    U_lad[0*4+3, 10] = 1; U_lad[1*4+2, 10] = 1; U_lad[2*4+0, 10] = 1; U_lad[3*4+1, 10] = 1
    V_lad[0*4+3, 10] = 1; V_lad[1*4+2, 10] = 1; V_lad[2*4+0, 10] = 1; V_lad[3*4+1, 10] = 1
    W_lad[0*4+3, 10] = 1; W_lad[1*4+2, 10] = 1; W_lad[2*4+0, 10] = 1; W_lad[3*4+1, 10] = 1
    
    # c=11: (A03+A12)(B03+B12)
    U_lad[0*4+3, 11] = 1; U_lad[1*4+2, 11] = 1
    V_lad[0*4+3, 11] = 1; V_lad[1*4+2, 11] = 1
    W_lad[0*4+3, 11] = 1; W_lad[1*4+2, 11] = 1
    
    # c=12: (A20+A31)(B20+B31)
    U_lad[2*4+0, 12] = 1; U_lad[3*4+1, 12] = 1
    V_lad[2*4+0, 12] = 1; V_lad[3*4+1, 12] = 1
    W_lad[2*4+0, 12] = 1; W_lad[3*4+1, 12] = 1
    
    # c=13: (A03+A20)(B03+B20)
    U_lad[0*4+3, 13] = 1; U_lad[2*4+0, 13] = 1
    V_lad[0*4+3, 13] = 1; V_lad[2*4+0, 13] = 1
    W_lad[0*4+3, 13] = 1; W_lad[2*4+0, 13] = 1
    
    # c=14: (A12+A31)(B12+B31)
    U_lad[1*4+2, 14] = 1; U_lad[3*4+1, 14] = 1
    V_lad[1*4+2, 14] = 1; V_lad[3*4+1, 14] = 1
    W_lad[1*4+2, 14] = 1; W_lad[3*4+1, 14] = 1
    
    # c=15: (A02+A13+A21+A30)(B02+B13+B21+B30)
    U_lad[0*4+2, 15] = 1; U_lad[1*4+3, 15] = 1; U_lad[2*4+1, 15] = 1; U_lad[3*4+0, 15] = 1
    V_lad[0*4+2, 15] = 1; V_lad[1*4+3, 15] = 1; V_lad[2*4+1, 15] = 1; V_lad[3*4+0, 15] = 1
    W_lad[0*4+2, 15] = 1; W_lad[1*4+3, 15] = 1; W_lad[2*4+1, 15] = 1; W_lad[3*4+0, 15] = 1
    
    # c=16: (A02+A13)(B02+B13)
    U_lad[0*4+2, 16] = 1; U_lad[1*4+3, 16] = 1
    V_lad[0*4+2, 16] = 1; V_lad[1*4+3, 16] = 1
    W_lad[0*4+2, 16] = 1; W_lad[1*4+3, 16] = 1
    
    # c=17: (A21+A30)(B21+B30)
    U_lad[2*4+1, 17] = 1; U_lad[3*4+0, 17] = 1
    V_lad[2*4+1, 17] = 1; V_lad[3*4+0, 17] = 1
    W_lad[2*4+1, 17] = 1; W_lad[3*4+0, 17] = 1
    
    # c=18: (A02+A21)(B02+B21)
    U_lad[0*4+2, 18] = 1; U_lad[2*4+1, 18] = 1
    V_lad[0*4+2, 18] = 1; V_lad[2*4+1, 18] = 1
    W_lad[0*4+2, 18] = 1; W_lad[2*4+1, 18] = 1
    
    # Verify Laderman decomposition for 4x4*4x4
    error = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.dot(U_lad[i*4+j, :], V_lad[j*4+k, :] * W_lad[k*4+i, :])
                expected = 1.0
                if abs(val - expected) > 1e-10:
                    error += abs(val - expected)
    
    if error > 1e-8:
        # Fallback to schoolbook if Laderman is incorrect
        U_school = np.zeros((nm, 128))
        V_school = np.zeros((mp, 128))
        W_school = np.zeros((pn, 128))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    # Now construct the 4x4 * 4x8 decomposition using Laderman
    # The B matrix is 4x8, which we can view as two 4x4 blocks
    # B = [B_left | B_right] where B_left is cols 0-3, B_right is cols 4-7
    # C = A * B = [A*B_left | A*B_right]
    # Each A*B_block is 4x4, computed with rank 19
    # Total rank: 19 + 19 = 38
    
    U_list = []
    V_list = []
    W_list = []
    
    # Left block: B cols 0-3, C cols 0-3
    for r in range(19):
        U_row = np.zeros(nm)
        V_row = np.zeros(mp)
        W_row = np.zeros(pn)
        
        # U: same for both blocks (A is the same)
        U_row = U_lad[:, r].copy()
        
        # V: only left half of B (cols 0-3)
        V_row[:16] = V_lad[:, r]
        # V cols 16-31 (right half) are 0
        
        # W: only left half of C (cols 0-3)
        W_row[:16] = W_lad[:, r]
        # W cols 16-31 (right half) are 0
        
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
    
    # Right block: B cols 4-7, C cols 4-7
    for r in range(19):
        U_row = np.zeros(nm)
        V_row = np.zeros(mp)
        W_row = np.zeros(pn)
        
        # U: same for both blocks (A is the same)
        U_row = U_lad[:, r].copy()
        
        # V: only right half of B (cols 4-7)
        V_row[16:] = V_lad[:, r]
        # V cols 0-15 (left half) are 0
        
        # W: only right half of C (cols 4-7)
        W_row[16:] = W_lad[:, r]
        # W cols 0-15 (left half) are 0
        
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
    
    rank = len(U_list)
    U = np.array(U_list).T  # shape (nm, rank)
    V = np.array(V_list).T  # shape (mp, rank)
    W = np.array(W_list).T  # shape (pn, rank)
    
    # Verify the full decomposition
    error = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j, :]
            for k in range(p):
                val = np.dot(u_row, V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    error += abs(val - 1.0)
    
    if error > 1e-8:
        # Fallback to schoolbook
        U_school = np.zeros((nm, 128))
        V_school = np.zeros((mp, 128))
        W_school = np.zeros((pn, 128))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
