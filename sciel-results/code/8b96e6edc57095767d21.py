"""Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Strategy: Use the 2x2 block partition of the 4x4 multiplication tensor.
The 4x4 matrix multiplication can be viewed as multiplying two 2x2 block matrices
where each block is 2x2. This gives a 2x2x2 tensor of 2x2 matrices.

The key insight: The rank of 2x2x2 matrix multiplication is 7 (Strassen).
If we treat the 4x4 multiplication as a 2x2 block multiplication where each
"element" is a 2x2 matrix, we can use Strassen's algorithm at the block level.

For 4x4 multiplication, partition into 2x2 blocks:
A = [[A11, A12], [A21, A22]] where each Aij is 2x2
B = [[B11, B12], [B21, B22]] where each Bij is 2x2

C = A*B = [[C11, C12], [C21, C22]] where each Cij is 2x2

Using Strassen at the block level:
M1 = (A11 + A22)(B11 + B22)
M2 = (A21 + A22)B11
M3 = A11(B12 - B22)
M4 = A22(B21 - B11)
M5 = (A11 + A12)B22
M6 = (A21 - A11)(B11 + B12)
M7 = (A12 - A22)(B21 + B22)

C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6

Each Mi is a product of two 2x2 matrices, which itself has rank 7.
So total rank = 7 * 7 = 49.

The tensor identity:
sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 for all i,j,k
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def _strassen_2x2():
    """Return (U2, V2, W2) for the rank-7 Strassen decomposition of 2x2 matrix multiplication.
    
    U2 has shape (4, 7), V2 has shape (4, 7), W2 has shape (4, 7).
    U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    Satisfies: sum_r U2[i*2+j,r] * V2[j*2+k,r] * W2[k*2+i,r] = 1 for all i,j,k in {0,1}
    """
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # S1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # S2 = (A10+A11)B00 -> C10 += 1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # S3 = A00(B01-B11) -> C01 += 1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # S4 = A11(B10-B00) -> C00 += 1, C10 += 1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # S5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # S6 = (A10-A00)(B00+B01) -> C11 += 1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # S7 = (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1
    
    return U2, V2, W2

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # Verify 2x2 decomposition first
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                if abs(val - 1) > 1e-10:
                    # Fallback to schoolbook if 2x2 is wrong
                    return _schoolbook_4x4()
    
    # Build 4x4 decomposition via Kronecker product
    # The 4x4 multiplication tensor T4[i,j,k] = T2[i1,j1,k1] * T2[i2,j2,k2]
    # where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    # 
    # U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # V4[j*4+k, r1*7+r2] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    # W4[k*4+i, r1*7+r2] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    rank = 49
    U4 = np.zeros((16, rank), dtype=float)
    V4 = np.zeros((16, rank), dtype=float)
    W4 = np.zeros((16, rank), dtype=float)
    
    for i in range(4):
        i1, i2 = i // 2, i % 2
        for j in range(4):
            j1, j2 = j // 2, j % 2
            for r1 in range(7):
                u1 = U2[i1 * 2 + j1, r1]
                if u1 == 0:
                    continue
                for r2 in range(7):
                    u2 = U2[i2 * 2 + j2, r2]
                    if u2 == 0:
                        continue
                    col = r1 * 7 + r2
                    U4[i * 4 + j, col] = u1 * u2
    
    for j in range(4):
        j1, j2 = j // 2, j % 2
        for k in range(4):
            k1, k2 = k // 2, k % 2
            for r1 in range(7):
                v1 = V2[j1 * 2 + k1, r1]
                if v1 == 0:
                    continue
                for r2 in range(7):
                    v2 = V2[j2 * 2 + k2, r2]
                    if v2 == 0:
                        continue
                    col = r1 * 7 + r2
                    V4[j * 4 + k, col] = v1 * v2
    
    for k in range(4):
        k1, k2 = k // 2, k % 2
        for i in range(4):
            i1, i2 = i // 2, i % 2
            for r1 in range(7):
                w1 = W2[k1 * 2 + i1, r1]
                if w1 == 0:
                    continue
                for r2 in range(7):
                    w2 = W2[k2 * 2 + i2, r2]
                    if w2 == 0:
                        continue
                    col = r1 * 7 + r2
                    W4[k * 4 + i, col] = w1 * w2
    
    # Verify the 4x4 decomposition
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U4[i*4+j, :] * V4[j*4+k, :] * W4[k*4+i, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    
    if max_err < 1e-10:
        # Convert to lists with integer values
        U_list = U4.tolist()
        V_list = V4.tolist()
        W_list = W4.tolist()
        
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                        mat[i][j] = round(mat[i][j])
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        return _schoolbook_4x4()

def _schoolbook_4x4():
    """Fallback: schoolbook decomposition of rank 64."""
    n, m, p = 4, 4, 4
    rank_fb = 64
    U_fb = [[0] * rank_fb for _ in range(16)]
    V_fb = [[0] * rank_fb for _ in range(16)]
    W_fb = [[0] * rank_fb for _ in range(16)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
