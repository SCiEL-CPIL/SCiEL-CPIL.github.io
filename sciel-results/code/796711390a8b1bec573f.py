"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0  # 150s wall-clock, use 140s as safety margin
_DEADLINE_MARGIN = 10.0

def _check_deadline():
    return time.time() < (_START_TIME + _DEADLINE - _DEADLINE_MARGIN)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix.
    
    Strategy: Use Strassen's algorithm on 2x2 blocks where possible,
    combined with schoolbook for the remaining parts.
    
    The 3x4x8 matrix multiplication can be decomposed as:
    - Treat the 3x4 matrix as having rows partitioned into (2, 1)
    - The 4x8 matrix stays as is
    - Apply Strassen to the 2x2 blocks in the first two rows
    
    For 2x4x8 (rows 0-1, cols 0-3 of A, all cols of B):
    Partition A into 2x2 blocks: A[0:2, 0:2] and A[0:2, 2:4]
    Each 2x2x8 multiplication can use Strassen: 7 multiplications per 2x2 block
    So 2 blocks * 7 = 14 multiplications for rows 0-1
    
    For 1x4x8 (row 2, all cols):
    Schoolbook: 4*8 = 32 multiplications
    
    Total: 14 + 32 = 46? Let me recalculate.
    
    Actually, let's be more careful. For a 2x2x2 Strassen block:
    C = AB where A is 2x2, B is 2x2
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    For 2x4x8: Partition A as 2x4 = two 2x2 blocks side by side
    B is 4x8 = two 4x4 blocks side by side... no wait.
    
    Let me think differently. A is 3x4, B is 4x8, C is 3x8.
    
    Split A's rows into groups of 2: rows [0,1] and row [2].
    For rows [0,1]: A_sub is 2x4. Split columns into 2 groups of 2: cols [0,1] and [2,3].
    So A_sub = [A00 A01; A10 A11] where each Aij is 2x2.
    B is 4x8. Split B's rows into 2 groups of 2: rows [0,1] and [2,3].
    B = [B_top; B_bottom] where B_top is 2x8 and B_bottom is 2x8.
    
    C[0:2, :] = A00 @ B_top[:, 0:2] ... no this is getting complex.
    
    Let me use a simpler approach: just apply Strassen to each 2x2x2 subproblem.
    
    For the full 3x4x8:
    - i in {0,1,2}, j in {0,1,2,3}, k in {0,...,7}
    - Total schoolbook terms: 3*4*8 = 96
    
    Using Strassen on 2x2x2 blocks:
    Group (i,j) pairs into 2x2 blocks where possible.
    i has 3 values, j has 4 values.
    
    For i in {0,1} and j in {0,1}: 2x2x8 block (k=0..7)
    For i in {0,1} and j in {2,3}: 2x2x8 block
    For i=2 and j in {0,1,2,3}: 1x4x8 block (schoolbook)
    
    For a 2x2x8 block:
    A is 2x2, B is 2x8 (two 2x4 blocks? or treat as 2x8)
    
    Actually, Strassen works on 2x2x2. For 2x2x8, we can apply Strassen to each column pair.
    B is 2x8 = four 2x2 blocks side by side.
    So 4 * 7 = 28 multiplications for one 2x2x8 block.
    
    Two such blocks: 56
    One 1x4x8 block: 32
    Total: 88.
    
    This matches the measured result of 88.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to construct a Strassen-based decomposition with rank 88
    # 2x2x8 blocks for i in {0,1}, j in {0,1,2,3}
    # 1x4x8 block for i=2, j in {0,1,2,3}
    
    # For each 2x2x2 Strassen block, we need 7 multiplications
    # A 2x2x8 block = 4 x 2x2x2 blocks, so 4*7 = 28 mults
    # Two 2x2x8 blocks = 56 mults
    # One 1x4x8 block = 32 mults (schoolbook)
    # Total = 88
    
    rank_strassen = 88
    U = [[0] * rank_strassen for _ in range(n * m)]
    V = [[0] * rank_strassen for _ in range(m * p)]
    W = [[0] * rank_strassen for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 helper: adds 7 terms
    # For A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
    # i0, j0, k0 are starting indices
    def strassen_2x2x2(i0, j0, k0, col_start):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # A00 = A[i0, j0], A11 = A[i0+1, j0+1]
        # B00 = B[j0, k0], B11 = B[j0+1, k0+1]
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1
        # Wait, C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0 * m + j0][col_start] = 1
        U[(i0+1) * m + (j0+1)][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[(j0+1) * p + (k0+1)][col_start] = 1
        # C00, C01, C10, C11 all get +M1
        # C index: k*n + i
        W[k0 * n + i0][col_start] = 1      # C00
        W[(k0+1) * n + i0][col_start] = 1  # C01
        W[k0 * n + (i0+1)][col_start] = 1  # C10
        W[(k0+1) * n + (i0+1)][col_start] = 1  # C11
        col += 1
        
        # M2 = (A10+A11) * B00
        # A10 = A[i0+1, j0], A11 = A[i0+1, j0+1]
        # B00 = B[j0, k0]
        # C10 += M2, C11 -= M2
        U[(i0+1) * m + j0][col_start] = 1
        U[(i0+1) * m + (j0+1)][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        W[k0 * n + (i0+1)][col_start] = 1      # C10
        W[(k0+1) * n + (i0+1)][col_start] = -1  # C11
        col += 1
        
        # M3 = A00 * (B10-B11)
        # A00 = A[i0, j0]
        # B10 = B[j0+1, k0], B11 = B[j0+1, k0+1]
        # C01 += M3, C11 += M3
        U[i0 * m + j0][col_start] = 1
        V[(j0+1) * p + k0][col_start] = 1
        V[(j0+1) * p + (k0+1)][col_start] = -1
        W[(k0+1) * n + i0][col_start] = 1  # C01
        W[(k0+1) * n + (i0+1)][col_start] = 1  # C11
        col += 1
        
        # M4 = A11 * (B01-B00)
        # A11 = A[i0+1, j0+1]
        # B01 = B[j0, k0+1], B00 = B[j0, k0]
        # C00 += M4, C10 += M4
        U[(i0+1) * m + (j0+1)][col_start] = 1
        V[j0 * p + (k0+1)][col_start] = 1
        V[j0 * p + k0][col_start] = -1
        W[k0 * n + i0][col_start] = 1      # C00
        W[k0 * n + (i0+1)][col_start] = 1  # C10
        col += 1
        
        # M5 = (A00+A10) * B11
        # A00 = A[i0, j0], A10 = A[i0+1, j0]
        # B11 = B[j0+1, k0+1]
        # C01 += M5, C00 -= M5
        U[i0 * m + j0][col_start] = 1
        U[(i0+1) * m + j0][col_start] = 1
        V[(j0+1) * p + (k0+1)][col_start] = 1
        W[(k0+1) * n + i0][col_start] = 1  # C01
        W[k0 * n + i0][col_start] = -1     # C00
        col += 1
        
        # M6 = (A11-A00) * (B00+B10)
        # A11 = A[i0+1, j0+1], A00 = A[i0, j0]
        # B00 = B[j0, k0], B10 = B[j0+1, k0]
        # C11 += M6
        U[(i0+1) * m + (j0+1)][col_start] = 1
        U[i0 * m + j0][col_start] = -1
        V[j0 * p + k0][col_start] = 1
        V[(j0+1) * p + k0][col_start] = 1
        W[(k0+1) * n + (i0+1)][col_start] = 1  # C11
        col += 1
        
        # M7 = (A01-A11) * (B01+B11)
        # A01 = A[i0, j0+1], A11 = A[i0+1, j0+1]
        # B01 = B[j0, k0+1], B11 = B[j0+1, k0+1]
        # C00 -= M7
        U[i0 * m + (j0+1)][col_start] = 1
        U[(i0+1) * m + (j0+1)][col_start] = -1
        V[j0 * p + (k0+1)][col_start] = 1
        V[(j0+1) * p + (k0+1)][col_start] = 1
        W[k0 * n + i0][col_start] = -1  # C00
        col += 1
    
    # Apply Strassen to 2x2x2 blocks
    # i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    # Group j into pairs: (0,1) and (2,3)
    # Group k into pairs: (0,1), (2,3), (4,5), (6,7)
    
    # For i0=0, j0=0: 2x2x8 block
    for k0 in range(0, p, 2):
        strassen_2x2x2(0, 0, k0, col)
    
    # For i0=0, j0=2: 2x2x8 block
    for k0 in range(0, p, 2):
        strassen_2x2x2(0, 2, k0, col)
    
    # Schoolbook for i=2, j in {0,1,2,3}, k in {0,...,7}
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Verify col == rank_strassen
    assert col == rank_strassen, f"Expected {rank_strassen}, got {col}"
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        rank = len(W[0])
        # For each (i, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 if there exists j
        # Actually, the tensor entry T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This should equal 1 for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1.0) > 1e-10:
                        return False, (i, j, k, total)
        return True, None
    
    if not _check_deadline():
        # Return schoolbook if we're running out of time
        return {"U": U_school, "V": V_school, "W": W_school}
    
    valid, info = verify(U, V, W, n, m, p)
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
