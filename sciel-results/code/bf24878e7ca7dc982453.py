"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: a = i*2+j (i,j in 0..1), b = j*2+k, c = k*2+i
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W


def _schoolbook_2x2x1():
    """
    Return schoolbook decomposition for 2x2 * 2x1 -> 2x1.
    U: shape (4, 4), V: shape (2, 4), W: shape (2, 4)
    Indexing: a = i*2+j, b = j*1+k = j, c = k*2+i = i
    """
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((2, 4), dtype=int)
    W = np.zeros((2, 4), dtype=int)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U[i*2+j, col] = 1
                V[j, col] = 1
                W[k*2+i, col] = 1
                col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    
    # Build the multiplication tensor
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Compute the decomposition
    # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_dec = np.zeros((nm, mp, pn), dtype=int)
    for r in range(rank):
        T_dec += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(nm, mp, pn)
    
    return np.array_equal(T, T_dec)


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Strategy: Use block Strassen decomposition
    # Split B (4x5) into B_top (2x5) and B_bottom (2x5)
    # C_top = A * B_top (2x5), C_bottom = A * B_bottom (2x5)
    # 
    # For C_top = A * B_top:
    #   Split A into A_left (2x2) and A_right (2x2)
    #   Split B_top into B_top_left (2x2), B_top_mid (2x1), B_top_right (2x2)
    #   C_top_left = A_left * B_top_left (2x2) - use Strassen, rank 7
    #   C_top_mid = A_left * B_top_mid (2x2*2x1) - schoolbook, rank 4
    #   C_top_right = A_right * B_top_right (2x2) - use Strassen, rank 7
    #   But wait, we also need cross terms...
    #
    # Actually, let me reconsider the block structure.
    # A is 2x4, B is 4x5.
    # 
    # Better approach: Split the 4 columns of A / 4 rows of B into two 2x2 blocks.
    # A = [A_L | A_R] where A_L, A_R are 2x2
    # B = [B_T; B_B] where B_T, B_B are 2x5
    # C = A_L * B_T + A_R * B_B
    #
    # For A_L * B_T (2x2 * 2x5):
    #   Split B_T into [B_TL | B_TM | B_TR] where B_TL, B_TR are 2x2, B_TM is 2x1
    #   A_L * B_TL (2x2*2x2): Strassen, rank 7
    #   A_L * B_TM (2x2*2x1): schoolbook, rank 4
    #   A_L * B_TR (2x2*2x2): Strassen, rank 7
    #   Total for A_L * B_T: 7 + 4 + 7 = 18
    #
    # For A_R * B_B (2x2 * 2x5):
    #   Similarly: 7 + 4 + 7 = 18
    #
    # Total rank: 18 + 18 = 36
    #
    # But wait, the inputs A_L and A_R are disjoint (different columns of A),
    # and B_T and B_B are disjoint (different rows of B).
    # The outputs C from A_L*B_T and A_R*B_B are added together,
    # but since we're summing rank-1 terms, we can just concatenate.
    
    # Let's build the full decomposition
    # A index: a = i*4 + j, i in 0..1, j in 0..3
    # B index: b = j*5 + k, j in 0..3, k in 0..4
    # C index: c = k*2 + i, k in 0..4, i in 0..1
    
    rank = 36
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    col = 0
    
    # Block 1: A_L * B_T (i in 0..1, j in 0..1 for A; j in 0..1, k in 0..4 for B)
    # A_L indices: a = i*4 + j for i in 0..1, j in 0..1
    # B_T indices: b = j*5 + k for j in 0..1, k in 0..4
    
    # Sub-block 1a: A_L * B_TL (j in 0..1, k in 0..1)
    # Use Strassen for 2x2 * 2x2
    sU, sV, sW = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                a = i * 4 + j  # global A index
                U[a, col] += sU[i*2+j, r]
        for j in range(2):
            for k in range(2):
                b = j * 5 + k  # global B index
                V[b, col] += sV[j*2+k, r]
        for k in range(2):
            for i in range(2):
                c = k * 2 + i  # global C index
                W[c, col] += sW[k*2+i, r]
        col += 1
    
    # Sub-block 1b: A_L * B_TM (j in 0..1, k in 2..2)
    # Use schoolbook for 2x2 * 2x1
    sU2, sV2, sW2 = _schoolbook_2x2x1()
    for r in range(4):
        for i in range(2):
            for j in range(2):
                a = i * 4 + j
                U[a, col] += sU2[i*2+j, r]
        for j in range(2):
            b = j * 5 + 2  # k=2
            V[b, col] += sV2[j, r]
        for i in range(2):
            c = 2 * 2 + i  # k=2
            W[c, col] += sW2[i, r]
        col += 1
    
    # Sub-block 1c: A_L * B_TR (j in 0..1, k in 3..4)
    # Use Strassen for 2x2 * 2x2
    sU, sV, sW = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                a = i * 4 + j
                U[a, col] += sU[i*2+j, r]
        for j in range(2):
            for k in range(2):
                b = j * 5 + (3 + k)  # k in 3..4
                V[b, col] += sV[j*2+k, r]
        for k in range(2):
            for i in range(2):
                c = (3 + k) * 2 + i  # k in 3..4
                W[c, col] += sW[k*2+i, r]
        col += 1
    
    # Block 2: A_R * B_B (i in 0..1, j in 2..3 for A; j in 2..3, k in 0..4 for B)
    # A_R indices: a = i*4 + j for i in 0..1, j in 2..3
    # B_B indices: b = j*5 + k for j in 2..3, k in 0..4
    
    # Sub-block 2a: A_R * B_BL (j in 2..3, k in 0..1)
    sU, sV, sW = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                a = i * 4 + (2 + j)  # j in 2..3
                U[a, col] += sU[i*2+j, r]
        for j in range(2):
            for k in range(2):
                b = (2 + j) * 5 + k  # j in 2..3
                V[b, col] += sV[j*2+k, r]
        for k in range(2):
            for i in range(2):
                c = k * 2 + i
                W[c, col] += sW[k*2+i, r]
        col += 1
    
    # Sub-block 2b: A_R * B_BM (j in 2..3, k in 2..2)
    sU2, sV2, sW2 = _schoolbook_2x2x1()
    for r in range(4):
        for i in range(2):
            for j in range(2):
                a = i * 4 + (2 + j)
                U[a, col] += sU2[i*2+j, r]
        for j in range(2):
            b = (2 + j) * 5 + 2
            V[b, col] += sV2[j, r]
        for i in range(2):
            c = 2 * 2 + i
            W[c, col] += sW2[i, r]
        col += 1
    
    # Sub-block 2c: A_R * B_BR (j in 2..3, k in 3..4)
    sU, sV, sW = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                a = i * 4 + (2 + j)
                U[a, col] += sU[i*2+j, r]
        for j in range(2):
            for k in range(2):
                b = (2 + j) * 5 + (3 + k)
                V[b, col] += sV[j*2+k, r]
        for k in range(2):
            for i in range(2):
                c = (3 + k) * 2 + i
                W[c, col] += sW[k*2+i, r]
        col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
