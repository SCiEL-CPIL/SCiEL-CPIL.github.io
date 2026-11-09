"""
Rank-minimal tensor decomposition for 4x4 * 4x8 matrix multiplication.

Uses a Strassen-based 7-term decomposition for 2x2x2 blocks,
then composes to get 4x4x4 (rank 49), and extends to 4x4x8 (rank 98).
"""
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen, each of shape (4, 7).
    U: A indices (i,j) flattened as i*2+j
    V: B indices (j,k) flattened as j*2+k
    W: C indices (k,i) flattened as k*2+i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    
    # U vectors (A coefficients, flattened i*2+j):
    # A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    U[0][0] = 1  # A00 in M1
    U[3][0] = 1  # A11 in M1
    U[1][1] = 1  # A10 in M2
    U[3][1] = 1  # A11 in M2
    U[0][2] = 1  # A00 in M3
    U[3][3] = 1  # A11 in M4
    U[0][4] = 1  # A00 in M5
    U[1][4] = 1  # A01 in M5
    U[1][5] = 1  # A10 in M6
    U[0][5] = -1 # A00 in M6
    U[1][6] = 1  # A01 in M7
    U[3][6] = -1 # A11 in M7
    
    # V vectors (B coefficients, flattened j*2+k):
    # B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    V[0][0] = 1  # B00 in M1
    V[2][0] = 1  # B11 in M1
    V[0][1] = 1  # B00 in M2
    V[2][2] = 1  # B10 in M3
    V[3][2] = -1 # B11 in M3
    V[1][3] = 1  # B01 in M4
    V[0][3] = -1 # B00 in M4
    V[3][4] = 1  # B11 in M5
    V[0][5] = 1  # B00 in M6
    V[1][5] = 1  # B01 in M6
    V[2][6] = 1  # B10 in M7
    V[3][6] = 1  # B11 in M7
    
    # W vectors (C coefficients, flattened k*2+i):
    # C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    W[0][0] = 1   # C00 in M1
    W[0][4] = -1  # C00 in M5
    W[0][6] = 1   # C00 in M7
    W[2][2] = 1   # C01 in M3
    W[2][4] = 1   # C01 in M5
    W[1][1] = 1   # C10 in M2
    W[1][3] = 1   # C10 in M4
    W[3][0] = 1   # C11 in M1
    W[3][2] = -1  # C11 in M3
    W[3][5] = 1   # C11 in M6
    
    return U, V, W

def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2x2()
    
    # Build 4x4x4 decomposition via Kronecker product of two 2x2x2 Strassen
    # 4x4 matrix = 2x2 blocks of 2x2 matrices
    # A[i,j] where i = 2*i1+i0, j = 2*j1+j0
    # B[j,k] where j = 2*j1+j0, k = 2*k1+k0
    # C[i,k] where i = 2*i1+i0, k = 2*k1+k0
    #
    # The 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors.
    # Rank = 7 * 7 = 49
    
    rank_4x4x4 = 49
    U4 = [[0] * rank_4x4x4 for _ in range(16)]
    V4 = [[0] * rank_4x4x4 for _ in range(16)]
    W4 = [[0] * rank_4x4x4 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            # A index: i = 2*i1+i0, j = 2*j1+j0 -> flat: i*4+j = (2*i1+i0)*4 + (2*j1+j0)
                            ai = 2*i1 + i0
                            aj = 2*j1 + j0
                            U4[ai*4 + aj][c] = U2[i0*2+j0][c1] * U2[i1*2+j1][c2]
                        for k1 in range(2):
                            for k0 in range(2):
                                # B index: j = 2*j1+j0, k = 2*k1+k0 -> flat: j*4+k
                                bj = 2*j1 + j0
                                bk = 2*k1 + k0
                                V4[bj*4 + bk][c] = V2[j0*2+k0][c1] * V2[j1*2+k1][c2]
                                # C index: k = 2*k1+k0, i = 2*i1+i0 -> flat: k*4+i
                                ck = 2*k1 + k0
                                ci = 2*i1 + i0
                                W4[ck*4 + ci][c] = W2[k0*2+i0][c1] * W2[k1*2+i1][c2]
    
    # Now extend to 4x4x8: B is 4x8, so p=8
    # B[j,k] where j in [0,4), k in [0,8)
    # We can view B as two 4x4 blocks: B_left = B[:, 0:4], B_right = B[:, 4:8]
    # C = A * B = A * [B_left, B_right] = [A*B_left, A*B_right]
    # Each A*B_block is 4x4, computed with rank 49
    # Total rank = 49 + 49 = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # First block: B[:, 0:4], k in [0,4)
    for c in range(49):
        for row in range(16):
            U[row][c] = U4[row][c]
        for row in range(16):  # j*4+k where k in [0,4) -> rows 0..15
            V[row][c] = V4[row][c]
        for row in range(16):  # k*4+i where k in [0,4) -> rows 0..15
            W[row][c] = W4[row][c]
    
    # Second block: B[:, 4:8], k in [4,8)
    for c in range(49):
        c2 = c + 49
        for row in range(16):
            U[row][c2] = U4[row][c]
        for row in range(16):  # j*4+(k+4) where k in [0,4) -> rows 16..31
            V[row + 16][c2] = V4[row][c]
        for row in range(16):  # (k+4)*4+i where k in [0,4) -> rows 16..31
            W[row + 16][c2] = W4[row][c]
    
    return {"U": U, "V": V, "W": W}
