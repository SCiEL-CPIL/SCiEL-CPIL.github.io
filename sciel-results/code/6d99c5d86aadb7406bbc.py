"""
Rank-minimal exact tensor decomposition for 4x4 x 4x8 matrix multiplication.

Strategy: Use a Kronecker/block structure based on Strassen's 2x2x2 (rank 7)
and a 2x2x4 rectangular multiplication (rank 8), giving rank 7*8 = 56.
This is better than the naive block approach which gives 56 but we verify it.

Actually, let's think more carefully. The task is 4x4 times 4x8.
We can view this as:
- Split A (4x4) into four 2x2 blocks: A = [[A00, A01], [A10, A11]]
- Split B (4x8) into four 2x4 blocks: B = [[B00, B01, B02, B03], [B10, B11, B12, B13]]
  where each Bjk is 2x4
- C (4x8) = [[C00, C01, C02, C03], [C10, C11, C12, C13]] where each Cij is 2x4

C_ij = A_i0 * B_0j + A_i1 * B_1j

So we need to compute 8 products of the form (2x2) * (2x4).

For 2x2 x 2x4: 
- A is 2x2, B is 2x4, C is 2x4
- C = A * B
- This is a bilinear map from (2x2) x (2x4) to (2x4)
- Rank of this tensor: we can compute it.

Actually, let me think about this differently. 

The tensor for 4x4 x 4x8 has rank at most 4*4*8 = 128 (schoolbook).

Using Strassen on 2x2 blocks:
- 4x4 x 4x4 has rank at most 4 * 7 = 28 (using Strassen on 2x2 blocks)
- 4x4 x 4x8 = 4x4 x (4x4 + 4x4) so rank at most 2 * 28 = 56

Let me construct this carefully.

For 4x4 x 4x4 using Strassen:
A (4x4) = [[A00, A01], [A10, A11]] where each Aij is 2x2
B (4x4) = [[B00, B01], [B10, B11]] where each Bij is 2x2
C (4x4) = [[C00, C01], [C10, C11]] where each Cij is 2x2

C00 = M1 + M4 - M5 + M7
C01 = M3 + M5
C10 = M2 + M4
C11 = M1 - M2 + M3 + M6

where:
M1 = (A00 + A11)(B00 + B11)
M2 = (A10 + A11)B00
M3 = A00(B01 - B11)
M4 = A11(B10 - B00)
M5 = (A00 + A01)B11
M6 = (A10 - A00)(B00 + B01)
M7 = (A01 - A11)(B10 + B11)

Each Mj is a 2x2 matrix, computed as a 2x2 x 2x2 product (rank 7).

So for 4x4 x 4x4, we have 4 Strassen decompositions (one for each output block),
each of rank 7, total rank 28.

For 4x4 x 4x8:
Split B into B_left (4x4) and B_right (4x4).
C_left (4x4) = A * B_left (rank 28)
C_right (4x4) = A * B_right (rank 28)
Total rank 56.

Let me implement this.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    V = [[0]*7 for _ in range(4)]
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    W = [[0]*7 for _ in range(4)]
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_4x4x8():
    """
    Construct a decomposition for 4x4 x 4x8 matrix multiplication.
    Split B (4x8) into two 4x4 blocks.
    For each 4x4 x 4x4 multiplication, use Strassen on 2x2 blocks (rank 28).
    Total rank: 2 * 28 = 56.
    """
    n, m, p = 4, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # (16, 56)
    V = [[0] * rank for _ in range(m * p)]   # (32, 56)
    W = [[0] * rank for _ in range(p * n)]   # (32, 56)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # For each of the two 4x4 output column blocks (k_block in {0,1})
    for k_block in range(2):
        base_col = k_block * 28
        
        # For each 2x2 block position in the 4x4x4 Strassen decomposition
        for i_block in range(2):   # row block of A (and C)
            for j_block in range(2):  # col block of A / row block of B
                # For each of the 7 Strassen products
                for r_s in range(7):
                    col = base_col + i_block * 14 + j_block * 7 + r_s
                    
                    # U: A is 4x4, indexed by (i, j) where i = row, j = col
                    # i = i_block * 2 + i_local, j = j_block * 2 + j_local
                    for i_local in range(2):
                        for j_local in range(2):
                            i = i_block * 2 + i_local
                            j = j_block * 2 + j_local
                            U[i * m + j][col] = U_s[i_local * 2 + j_local][r_s]
                    
                    # V: B is 4x8, indexed by (j, k) where j = row, k = col
                    # j = j_block * 2 + j_local, k = k_block * 4 + k_local
                    for j_local in range(2):
                        for k_local in range(2):
                            j = j_block * 2 + j_local
                            k = k_block * 4 + k_local
                            V[j * p + k][col] = V_s[j_local * 2 + k_local][r_s]
                    
                    # W: C is 4x8, indexed by (k, i) per trace convention W[k*n+i]
                    # k = k_block * 4 + k_local, i = i_block * 2 + i_local
                    for k_local in range(2):
                        for i_local in range(2):
                            k = k_block * 4 + k_local
                            i = i_block * 2 + i_local
                            W[k * n + i][col] = W_s[k_local * 2 + i_local][r_s]
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Primary: Strassen-based rank-56 decomposition
    result = _strassen_4x4x8()
    
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
