import numpy as np

def construct():
    """
    Construct a rank-44 exact tensor decomposition for 3x4 times 4x4 matrix multiplication.
    Part 1: Strassen-style 2x2 blocks for A[0:2,:]*B (rank 28).
    Part 2: Direct 1x2 blocks for A[2,:]*B (rank 16).
    """
    n, m, p = 3, 4, 4  # A is n x m, B is m x p, C is n x p
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)  # shape (12, 44)
    V = np.zeros((m * p, rank), dtype=float)  # shape (16, 44)
    W = np.zeros((p * n, rank), dtype=float)  # shape (12, 44)

    # Strassen 2x2x2 decomposition (rank 7)
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    # S1
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1
    # S2
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1; W_s[3,1]=-1
    # S3
    U_s[0,2]=1
    V_s[1,2]=1; V_s[3,2]=-1
    W_s[1,2]=1; W_s[3,2]=1
    # S4
    U_s[3,3]=1
    V_s[2,3]=1; V_s[0,3]=-1
    W_s[0,3]=1; W_s[2,3]=1
    # S5
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[0,4]=-1; W_s[1,4]=1
    # S6
    U_s[2,5]=1; U_s[0,5]=-1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[3,5]=1
    # S7
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[0,6]=1

    # 1x2x2 direct decomposition (rank 4)
    U_d = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    V_d = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    # Let's use zeros and manually set like the seed for consistency
    U_d = np.zeros((2,4), dtype=float); V_d = np.zeros((4,4), dtype=float); W_d = np.zeros((2,4), dtype=float)
    U_d[0,0]=1; V_d[0,0]=1; W_d[0,0]=1
    U_d[1,1]=1; V_d[2,1]=1; W_d[0,1]=1
    U_d[0,2]=1; V_d[1,2]=1; W_d[1,2]=1
    U_d[1,3]=1; V_d[3,3]=1; W_d[1,3]=1

    col_offset = 0

    # Block Part 1: 2x4x4
    strassen_blocks = [
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (0, 0, 0, 2),
        (0, 2, 2, 2)
    ]
    for ar_off, ac_off, br_off, bc_off in strassen_blocks:
        for r in range(2):
            for c in range(2):
                U[(ar_off+r)*4 + ac_off+c, col_offset:col_offset+7] += U_s[r*2+c, :]
        for r in range(2):
            for c in range(2):
                V[(br_off+r)*4 + bc_off+c, col_offset:col_offset+7] += V_s[r*2+c, :]
        for r in range(2):
            for c in range(2):
                W[(bc_off+c)*3 + ar_off+r, col_offset:col_offset+7] += W_s[r*2+c, :]
        col_offset += 7

    # Block Part 2: 1x4x4
    direct_blocks = [
        (0, 0, 0),
        (2, 2, 0),
        (0, 0, 2),
        (2, 2, 2)
    ]
    for ac_off, br_off, bc_off in direct_blocks:
        for c in range(2):
            U[8 + ac_off+c, col_offset:col_offset+4] += U_d[c, :]
        for r in range(2):
            for c in range(2):
                V[(br_off+r)*4 + bc_off+c, col_offset:col_offset+4] += V_d[r*2+c, :]
        for c in range(2):
            W[(bc_off+c)*3 + 2, col_offset:col_offset+4] += W_d[c, :]
        col_offset += 4

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
