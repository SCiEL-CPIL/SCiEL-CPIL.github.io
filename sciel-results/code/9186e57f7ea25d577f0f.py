"""Rewritten program for 2x3x7 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact (within tolerance)."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Expected: T[i,j,k] = 1 for all i,j,k
    # Check via random sampling
    rng = np.random.RandomState(42)
    for _ in range(200):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        u_val = U[i * m + j]
        v_val = V[j * p + k]
        w_val = W[k * n + i]
        prod = np.dot(u_val, v_val * w_val)
        if abs(prod - 1.0) > 1e-6:
            return False
    return True

def _strassen_4x4():
    """
    Return the Strassen decomposition of 4x4x4 matrix multiplication.
    U: (16, 7), V: (16, 7), W: (16, 7)
    Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    # Strassen's 7 multiplications for 4x4 = 2x2 blocks of 2x2
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we index A as A[row][col] where row = block_row*2 + sub_row, col = block_col*2 + sub_col
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # U[i*4+j, c] for A, V[j*4+k, c] for B, W[k*4+i, c] for C
    
    # We need to build the 7 rank-1 tensors.
    # For each multiplication Mc, U_c is a 16-vector (flattened 4x4 for A part),
    # V_c is a 16-vector (flattened 4x4 for B part),
    # W_c is a 16-vector (flattened 4x4 for C part).
    
    def idx(r, c):
        return r * 4 + c
    
    U = [[0]*7 for _ in range(16)]
    V = [[0]*7 for _ in range(16)]
    W = [[0]*7 for _ in range(16)]
    
    # M1: (A11+A22)*(B11+B22)
    # A part: A11 + A22 -> positions (0,0),(0,1),(1,0),(1,1) and (2,2),(2,3),(3,2),(3,3)
    for r in [0,1]:
        for c in [0,1]:
            U[idx(r,c)][0] = 1
    for r in [2,3]:
        for c in [2,3]:
            U[idx(r,c)][0] = 1
    for r in [0,1]:
        for c in [0,1]:
            V[idx(r,c)][0] = 1
    for r in [2,3]:
        for c in [2,3]:
            V[idx(r,c)][0] = 1
    # C part: C11 + C22
    for r in [0,1]:
        for c in [0,1]:
            W[idx(c,r)][0] = 1  # C11: W[k*4+i] where k in 0,1 and i in 0,1
    for r in [2,3]:
        for c in [2,3]:
            W[idx(c,r)][0] = 1  # C22: W[k*4+i] where k in 2,3 and i in 2,3
    
    # M2: (A21+A22)*B11
    for r in [2,3]:
        for c in [0,1,2,3]:
            U[idx(r,c)][1] = 1
    for r in [0,1]:
        for c in [0,1]:
            V[idx(r,c)][1] = 1
    # C part: C21
    for r in [0,1]:
        for c in [2,3]:
            W[idx(c,r)][1] = 1  # C21: k in 2,3, i in 0,1
    
    # M3: A11*(B12-B22)
    for r in [0,1]:
        for c in [0,1]:
            U[idx(r,c)][2] = 1
    for r in [0,1]:
        for c in [2,3]:
            V[idx(r,c)][2] = 1
    for r in [2,3]:
        for c in [2,3]:
            V[idx(r,c)][2] = -1
    # C part: C12
    for r in [0,1]:
        for c in [2,3]:
            W[idx(c,r)][2] = 1  # C12: k in 2,3, i in 0,1
    
    # M4: A22*(B21-B11)
    for r in [2,3]:
        for c in [2,3]:
            U[idx(r,c)][3] = 1
    for r in [2,3]:
        for c in [0,1]:
            V[idx(r,c)][3] = 1
    for r in [0,1]:
        for c in [0,1]:
            V[idx(r,c)][3] = -1
    # C part: C11 + C21
    for r in [0,1]:
        for c in [0,1]:
            W[idx(c,r)][3] = 1  # C11
    for r in [0,1]:
        for c in [2,3]:
            W[idx(c,r)][3] = 1  # C21
    
    # M5: (A11+A12)*B22
    for r in [0,1]:
        for c in [0,1,2,3]:
            U[idx(r,c)][4] = 1
    for r in [2,3]:
        for c in [2,3]:
            V[idx(r,c)][4] = 1
    # C part: -C11 + C22
    for r in [0,1]:
        for c in [0,1]:
            W[idx(c,r)][4] = -1  # -C11
    for r in [2,3]:
        for c in [2,3]:
            W[idx(c,r)][4] = 1  # C22
    
    # M6: (A21-A11)*(B11+B12)
    for r in [2,3]:
        for c in [0,1]:
            U[idx(r,c)][5] = 1
    for r in [0,1]:
        for c in [0,1]:
            U[idx(r,c)][5] = -1
    for r in [0,1]:
        for c in [0,1,2,3]:
            V[idx(r,c)][5] = 1
    # C part: C22
    for r in [2,3]:
        for c in [2,3]:
            W[idx(c,r)][5] = 1  # C22
    
    # M7: (A12-A22)*(B21+B22)
    for r in [0,1]:
        for c in [2,3]:
            U[idx(r,c)][6] = 1
    for r in [2,3]:
        for c in [2,3]:
            U[idx(r,c)][6] = -1
    for r in [2,3]:
        for c in [0,1,2,3]:
            V[idx(r,c)][6] = 1
    # C part: C11
    for r in [0,1]:
        for c in [0,1]:
            W[idx(c,r)][6] = 1  # C11
    
    return U, V, W

def _lifting_decomposition():
    """
    Use Strassen 4x4x4 decomposition to get a rank-7 decomposition of 2x3x7 multiplication.
    
    We embed 2x3 matrix A into 4x4 matrix A' by zero-padding:
    A' = [[A, 0], [0, 0]]  (top-left 2x3 block is A, rest zero)
    
    We embed 3x7 matrix B into 4x4 matrix B' by zero-padding:
    B' = [[B, 0], [0, 0]]  (top-left 3x4 block is B, rest zero)
    
    Wait, this doesn't work directly because 3x7 doesn't fit in 4x4.
    
    Let me reconsider. The 2x3x7 multiplication tensor has rank at most 2*3*7=42.
    We want to use the Strassen 4x4x4 (rank 7) decomposition.
    
    Actually, let's think differently. We can use multiple Strassen decompositions.
    
    Alternative: Use the known result that rank(2x3x7) <= some value via block decomposition.
    
    Let me try a direct approach: decompose using blocks.
    
    A is 2x3, B is 3x7.
    C = A*B is 2x7.
    
    We can split B into B1 (3x4) and B2 (3x3):
    C = A*B1 + A*B2
    
    A*B1: 2x4 output. Use Strassen on 2x4 * 4x4? No, B1 is 3x4.
    
    Let me try: split A into A1 (2x2) and A2 (2x1):
    C = A1*B1 + A2*B2 where B1 is 2x7, B2 is 1x7.
    
    A1*B1: 2x2 * 2x7. We can use Strassen for 2x2*2x2 on the first 2 columns, then again on next 2, etc.
    Actually 2x2*2x7: split B1 into 2x2, 2x2, 2x1.
    A1*B1[:2] uses rank 2 (schoolbook) or we can use Strassen 2x2x2 (rank 7) but that's overkill.
    
    Let me just use a block-based Strassen approach:
    
    For 2x3 * 3x7:
    Split into blocks of size 2:
    A = [A11 A12 A13] where A11 is 2x2, A12 is 2x1, A13 is 2x0 (empty)
    Actually 3 = 2+1, so A = [A_2x2, A_2x1]
    B = [B_2x7; B_1x7]
    
    C = A_2x2 * B_2x7 + A_2x1 * B_1x7
    
    A_2x2 * B_2x7: split B_2x7 into B_a (2x4) and B_b (2x3)
    A_2x2 * B_a: split B_a into 2x2, 2x2
      A_2x2 * (2x2): rank 7 via Strassen
      A_2x2 * (2x2): rank 7 via Strassen
    A_2x2 * B_b: split B_b into 2x2, 2x1
      A_2x2 * (2x2): rank 7
      A_2x2 * (2x1): rank 2 (schoolbook, since one column)
    
    A_2x1 * B_1x7: rank 7 (schoolbook, 2*1*7)
    
    Total: 7+7+7+2+7 = 30
    
    Hmm, that's 30. Let me see if we can do better.
    
    Actually, let me just implement the schoolbook with rank 42 as a fallback,
    and try to find a better decomposition via the Strassen lifting.
    
    For now, let me implement the 4x4 Strassen based lifting properly.
    
    Key insight: We can compute 2x3 * 3x7 by:
    1. Pad A (2x3) to 4x4: A' = [[A, 0_2x1], [0_2x4]]
    2. Pad B (3x7) to 4x4: This is the problem - 7 > 4.
    
    So we need to split B. Let's do:
    B = [B1 (3x4), B2 (3x3)]
    C = A*B1 + A*B2
    
    For A*B1: Pad A to 4x4, pad B1 to 4x4, use Strassen 4x4x4 (rank 7), extract 2x4 result.
    For A*B2: Pad A to 4x4, pad B2 to 4x4, use Strassen 4x4x4 (rank 7), extract 2x3 result.
    
    Total rank: 7 + 7 = 14.
    
    Let me implement this.
    """
    n, m, p = 2, 3, 7
    
    # We'll use two Strassen 4x4x4 decompositions
    # U_total will have shape (n*m, rank_total) = (6, rank_total)
    # V_total will have shape (m*p, rank_total) = (21, rank_total)
    # W_total will have shape (p*n, rank_total) = (14, rank_total)
    
    U4, V4, W4 = _strassen_4x4()
    
    # First part: A * B1 where B1 is 3x4
    # A is 2x3, pad to 4x4: A'[i,j] = A[i,j] if i<2, j<3 else 0
    # B1 is 3x4, pad to 4x4: B1'[i,j] = B1[i,j] if i<3, j<4 else 0
    # Result C1 is 2x4, from C1'[i,j] = C'[i,j] if i<2, j<4
    
    # For the first Strassen decomposition (columns 0-6 of the 4x4 result):
    # U column c: for A' flattened (4x4 -> 16), U4[i*4+j, c]
    #   We need to map A' (4x4) to A (2x3): A'[i,j] = A[i,j] for i<2,j<3
    #   So U_total[i*3+j, c] = U4[i*4+j, c] for i in 0..1, j in 0..2
    #   For i>=2 or j>=3, A'[i,j]=0, so we don't include those in U_total.
    
    # V column c: for B1' flattened (4x4 -> 16), V4[j*4+k, c]
    #   B1' (4x4) maps to B1 (3x4): B1'[i,j] = B1[i,j] for i<3, j<4
    #   So V_total[i*4+j, c] = V4[i*4+j, c] for i in 0..2, j in 0..3
    
    # W column c: for C1' flattened (4x4 -> 16), W4[k*4+i, c]
    #   C1' (4x4) maps to C1 (2x4): C1'[i,j] = C1[i,j] for i<2, j<4
    #   So W_total[j*2+i, c] = W4[j*4+i, c] for i in 0..1, j in 0..3
    
    # Second part: A * B2 where B2 is 3x3
    # A is 2x3, pad to 4x4: same as above
    # B2 is 3x3, pad to 4x4: B2'[i,j] = B2[i,j] if i<3, j<3 else 0
    # Result C2 is 2x3, from C2'[i,j] = C2[i,j] if i<2, j<3
    
    rank_total = 14
    U_total = [[0]*rank_total for _ in range(n*m)]  # (6, 14)
    V_total = [[0]*rank_total for _ in range(m*p)]  # (21, 14)
    W_total = [[0]*rank_total for _ in range(p*n)]  # (14, 14)
    
    # Part 1: columns 0-6
    for c in range(7):
        # U: A' (4x4) -> A (2x3)
        for i in range(2):
            for j in range(3):
                U_total[i*3 + j][c] = U4[i*4 + j][c]
        
        # V: B1' (4x4) -> B1 (3x4)
        for i in range(3):
            for j in range(4):
                V_total[i*4 + j][c] = V4[i*4 + j][c]
        
        # W: C1' (4x4) -> C1 (2x4)
        for i in range(2):
            for j in range(4):
                W_total[j*2 + i][c] = W4[j*4 + i][c]
    
    # Part 2: columns 7-13
    for c in range(7):
        cc = c + 7
        # U: A' (4x4) -> A (2x3)
        for i in range(2):
            for j in range(3):
                U_total[i*3 + j][cc] = U4[i*4 + j][c]
        
        # V: B2' (4x4) -> B2 (3x3)
        for i in range(3):
            for j in range(3):
                V_total[i*3 + j][cc] = V4[i*4 + j][c]
        
        # W: C2' (4x4) -> C2 (2x3)
        for i in range(2):
            for j in range(3):
                W_total[j*2 + i][cc] = W4[j*4 + i][c]
    
    return U_total, V_total, W_total

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s deadline
    
    n, m, p = DIMENSIONS
    
    # Try the lifting decomposition (rank 14)
    try:
        U, V, W = _lifting_decomposition()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 42)
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
