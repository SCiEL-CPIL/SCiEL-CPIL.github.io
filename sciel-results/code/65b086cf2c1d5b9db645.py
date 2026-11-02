"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                if val != 1:
                    return False
    return True

def _strassen_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W as lists of lists.
    U: 4 x 7, V: 4 x 7, W: 4 x 7
    """
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U[i,j] where i in [0,4) is row index of A (0,1 for row0; 2,3 for row1)
    # Actually, let's use: U[row_of_A, col] where row_of_A is 0..3
    # V[col_of_B, row_of_B, col] 
    # Let me be more careful with indices.
    
    # A is 2x2: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B is 2x2: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C = A*B: C[0,0]=ae+bg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh
    
    # For tensor decomposition, we need:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1}
    # where (i,j) indexes A, (j,k) indexes B, (k,i) indexes C
    
    # U has 4 rows: (i,j) for i,j in {0,1}
    # V has 4 rows: (j,k) for j,k in {0,1}  
    # W has 4 rows: (k,i) for k,i in {0,1}
    
    # Let's map:
    # A index: (i,j) -> i*2+j
    # B index: (j,k) -> j*2+k
    # C index: (k,i) -> k*2+i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)*(e+h): U contribution to a and d, V to e and h
    # a is at A[0,0] -> index 0, d is at A[1,1] -> index 3
    # e is at B[0,0] -> index 0, h is at B[1,1] -> index 3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 += M1, C22 += M1
    # C11 is C[0,0] -> (k=0,i=0) -> index 0
    # C22 is C[1,1] -> (k=1,i=1) -> index 3
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)*e: c is A[1,0]->idx2, d is A[1,1]->idx3, e is B[0,0]->idx0
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 += M2, C22 -= M2
    # C21 is C[1,0] -> (k=0,i=1) -> index 1
    # C22 is C[1,1] -> (k=1,i=1) -> index 3
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a*(f-h): a is A[0,0]->idx0, f is B[0,1]->idx1, h is B[1,1]->idx3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C12 += M3, C22 += M3
    # C12 is C[0,1] -> (k=1,i=0) -> index 2
    # C22 is C[1,1] -> (k=1,i=1) -> index 3
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d*(g-e): d is A[1,1]->idx3, g is B[1,0]->idx2, e is B[0,0]->idx0
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C11 += M4, C21 += M4
    # C11 is C[0,0] -> (k=0,i=0) -> index 0
    # C21 is C[1,0] -> (k=0,i=1) -> index 1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)*h: a is A[0,0]->idx0, b is A[0,1]->idx1, h is B[1,1]->idx3
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 -= M5, C12 += M5
    # C11 is C[0,0] -> (k=0,i=0) -> index 0
    # C12 is C[0,1] -> (k=1,i=0) -> index 2
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c-a)*(e+f): c is A[1,0]->idx2, a is A[0,0]->idx0, e is B[0,0]->idx0, f is B[0,1]->idx1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 += M6
    # C22 is C[1,1] -> (k=1,i=1) -> index 3
    W[3][5] = 1
    
    # M7 = (b-d)*(g+h): b is A[0,1]->idx1, d is A[1,1]->idx3, g is B[1,0]->idx2, h is B[1,1]->idx3
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 += M7
    # C11 is C[0,0] -> (k=0,i=0) -> index 0
    W[0][6] = 1
    
    return U, V, W

def _block_decomposition(n, m, p):
    """
    Construct a block-structured decomposition for n x m * m x p.
    Uses Strassen for 2x2 blocks and handles remaining dimensions.
    """
    # Strategy: Split into 2x2 blocks where possible, handle remainder
    # For 3x3 * 3x4:
    # - Split A (3x3) into 2x2 block + 1x1
    # - Split B (3x4) into 2x2, 2x2 blocks + 1x4
    # - Use Strassen for the 2x2*2x2 parts
    
    # Let's use a different approach: direct construction using known rank-34 result
    # The key idea: use Strassen on overlapping 2x2 subproblems
    
    # For 3x3 * 3x4, we can achieve rank 34 using:
    # - 2x2 * 2x2 Strassen (rank 7) for the top-left block
    # - Handle the remaining rows/columns with additional terms
    
    # Let me construct this more carefully.
    # A is 3x3, B is 3x4
    # C = A * B is 3x4
    
    # Split: A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x2)
    # C21 = A21*B11 + A22*B21 (1x2)
    # C22 = A21*B12 + A22*B22 (1x2)
    
    # For C11: Use Strassen for A11*B11 (rank 7) + handle A12*B21 (2x1 * 1x2 = rank 2)
    # For C12: Use Strassen for A11*B12 (rank 7) + handle A12*B22 (2x1 * 1x2 = rank 2)
    # For C21: A21*B11 (1x2 * 2x2) + A22*B21 (1x1 * 1x2) = rank 2 + rank 2 = rank 4
    # For C22: A21*B12 (1x2 * 2x2) + A22*B22 (1x1 * 1x2) = rank 2 + rank 2 = rank 4
    
    # Total: 7+2 + 7+2 + 4 + 4 = 26? That seems too low. Let me reconsider.
    
    # Actually, the issue is that the Strassen terms for A11*B11 and A11*B12 share the A11 factors.
    # We need to be more careful about how we combine them.
    
    # Let me use a different strategy: build the decomposition column by column.
    
    # Actually, let's use a known result. The rank of 3x3 * 3x4 is at most 34.
    # Let me construct it using a systematic approach.
    
    # I'll use the following construction:
    # 1. Use Strassen for the 2x2 * 2x2 subproblem (top-left 2x2 of A times top-left 2x2 of B)
    # 2. Add terms for the remaining interactions
    
    # Let me just implement a proper search that finds rank 34 or better.
    
    n, m, p = 3, 3, 4
    r = 34
    
    # Initialize
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    # Use Strassen for the 2x2*2x2 part (rows 0,1 of A, cols 0,1 of B)
    # This gives us 7 terms for the top-left 2x2 block of C
    
    sU, sV, sW = _strassen_2x2()
    
    # Map Strassen indices to full matrix indices
    # A index (i,j) -> i*m+j where i,j in {0,1}
    # B index (j,k) -> j*p+k where j in {0,1}, k in {0,1}
    # C index (k,i) -> k*n+i where k in {0,1}, i in {0,1}
    
    for c in range(7):
        for a_idx in range(4):
            i, j = divmod(a_idx, 2)
            U[i*m+j][c] = sU[a_idx][c]
        for b_idx in range(4):
            j, k = divmod(b_idx, 2)
            V[j*p+k][c] = sV[b_idx][c]
        for c_idx in range(4):
            k, i = divmod(c_idx, 2)
            W[k*n+i][c] = sW[c_idx][c]
    
    # Now we need to handle:
    # - A[0,2]*B[2,0:4] -> C[0,0:4] (row 0 of A, col 2)
    # - A[1,2]*B[2,0:4] -> C[1,0:4] (row 1 of A, col 2)
    # - A[2,0:3]*B[0:2,0:4] -> C[2,0:4] (row 2 of A)
    # - A[0:2,0:2]*B[2,0:4] already partially handled by Strassen? No, Strassen only handles B[0:2,0:2]
    
    # Let me reconsider. The Strassen decomposition handles:
    # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    
    # Remaining terms needed:
    # 1. A[0:2,2] * B[2,0:2] -> C[0:2,0:2] (2x1 * 1x2)
    # 2. A[0:2,2] * B[2,2:4] -> C[0:2,2:4] (2x1 * 1x2)
    # 3. A[2,0:2] * B[0:2,0:2] -> C[2,0:2] (1x2 * 2x2)
    # 4. A[2,2] * B[2,0:2] -> C[2,0:2] (1x1 * 1x2)
    # 5. A[2,0:2] * B[0:2,2:4] -> C[2,2:4] (1x2 * 2x2)
    # 6. A[2,2] * B[2,2:4] -> C[2,2:4] (1x1 * 1x2)
    
    # Term 1: A[0,2]*B[2,0] -> C[0,0], A[0,2]*B[2,1] -> C[0,1], A[1,2]*B[2,0] -> C[1,0], A[1,2]*B[2,1] -> C[1,1]
    # These are 4 rank-1 terms: rank 4
    # But we can potentially combine some with existing terms.
    
    # Let me just add them as separate terms for now.
    
    col = 7
    
    # Term 1: A[i,2]*B[2,k] -> C[i,k] for i in {0,1}, k in {0,1}
    for i in range(2):
        for k in range(2):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    
    # Term 2: A[i,2]*B[2,k] -> C[i,k] for i in {0,1}, k in {2,3}
    for i in range(2):
        for k in range(2, 4):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    
    # Term 3: A[2,j]*B[j,k] -> C[2,k] for j in {0,1}, k in {0,1}
    for j in range(2):
        for k in range(2):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
    
    # Term 4: A[2,2]*B[2,k] -> C[2,k] for k in {0,1}
    for k in range(2):
        U[2*m+2][col] = 1
        V[2*p+k][col] = 1
        W[k*n+2][col] = 1
        col += 1
    
    # Term 5: A[2,j]*B[j,k] -> C[2,k] for j in {0,1}, k in {2,3}
    for j in range(2):
        for k in range(2, 4):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
    
    # Term 6: A[2,2]*B[2,k] -> C[2,k] for k in {2,3}
    for k in range(2, 4):
        U[2*m+2][col] = 1
        V[2*p+k][col] = 1
        W[k*n+2][col] = 1
        col += 1
    
    # Total: 7 + 4 + 4 + 4 + 2 + 4 + 2 = 27? That's less than 34.
    # But wait, I'm double-counting some terms. Let me verify.
    
    # Actually, the issue is that terms 1-6 might overlap with the Strassen terms in ways that cancel.
    # Let me just verify the result.
    
    if _verify(U, V, W, n, m, p):
        return U, V, W
    
    # If verification fails, fall back to a simpler construction
    # Let me try a different approach: use the fact that we can achieve rank 34
    # by a more careful combination.
    
    # For now, return schoolbook if this doesn't work
    rank_schoolbook = n * m * p
    U_school = [[0]*rank_schoolbook for _ in range(n*m)]
    V_school = [[0]*rank_schoolbook for _ in range(m*p)]
    W_school = [[0]*rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    return U_school, V_school, W_school

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds to be safe
    
    # Try block decomposition
    try:
        U, V, W = _block_decomposition(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0]*rank_schoolbook for _ in range(n*m)]
    V_school = [[0]*rank_schoolbook for _ in range(m*p)]
    W_school = [[0]*rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
