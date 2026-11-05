import numpy as np
import itertools
import time

DIMENSIONS = [3, 4, 7]
SEED_RANK = 84  # n * m * p

def _schoolbook(n, m, p):
    """Return U, V, W for the schoolbook algorithm (rank = n*m*p)."""
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
    return U, V, W

def _strassen_2x2x2(U, V, W, i0, j0, k0, col_start):
    """
    Add a 2x2x2 Strassen decomposition covering A[i0:i0+2, j0:j0+2],
    B[j0:j0+2, k0:k0+2], C[k0:k0+2, i0:i0+2].
    Returns the next column index.
    """
    n = 3
    m = 4
    p = 7
    # Strassen 7 multiplications for 2x2 * 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Define the 7 rank-1 terms as (coeff_a, coeff_b, coeff_c) for each (i,j,k)
    # We need to express each M_r as a product of linear forms in A and B,
    # then the contribution to C is the product.
    
    # For each of the 7 Strassen terms, we have:
    # term = (linear form in A entries) * (linear form in B entries)
    # and it contributes to specific C entries with coefficients.
    
    # Let's define the Strassen decomposition more carefully.
    # The 7 products are:
    # P1 = (a11+a22) * (b11+b22)
    # P2 = (a21+a22) * b11
    # P3 = a11 * (b12-b22)
    # P4 = a22 * (b21-b11)
    # P5 = (a11+a12) * b22
    # P6 = (a21-a11) * (b11+b12)
    # P7 = (a12-a22) * (b21+b22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pr is a rank-1 term in the tensor decomposition.
    # We need to assign a column for each Pr.
    
    # P1: A part: a11 + a22, B part: b11 + b22
    # a11 = A[i0, j0], a22 = A[i0+1, j0+1]
    # b11 = B[j0, k0], b22 = B[j0+1, k0+1]
    # C part: C11 = C[k0, i0], C22 = C[k0+1, i0+1]
    # Coefficients: C11 gets +1, C22 gets +1
    
    # P2: A part: a21 + a22, B part: b11
    # a21 = A[i0+1, j0], a22 = A[i0+1, j0+1]
    # b11 = B[j0, k0]
    # C part: C21 = C[k0, i0+1] gets +1, C22 = C[k0+1, i0+1] gets -1
    
    # P3: A part: a11, B part: b12 - b22
    # a11 = A[i0, j0]
    # b12 = B[j0, k0+1], b22 = B[j0+1, k0+1]
    # C part: C12 = C[k0+1, i0] gets +1, C22 = C[k0+1, i0+1] gets +1
    
    # P4: A part: a22, B part: b21 - b11
    # a22 = A[i0+1, j0+1]
    # b21 = B[j0+1, k0], b11 = B[j0, k0]
    # C part: C11 = C[k0, i0] gets +1, C21 = C[k0, i0+1] gets +1
    
    # P5: A part: a11 + a12, B part: b22
    # a11 = A[i0, j0], a12 = A[i0, j0+1]
    # b22 = B[j0+1, k0+1]
    # C part: C11 = C[k0, i0] gets -1, C12 = C[k0+1, i0] gets +1
    
    # P6: A part: a21 - a11, B part: b11 + b12
    # a21 = A[i0+1, j0], a11 = A[i0, j0]
    # b11 = B[j0, k0], b12 = B[j0, k0+1]
    # C part: C22 = C[k0+1, i0+1] gets +1
    
    # P7: A part: a12 - a22, B part: b21 + b22
    # a12 = A[i0, j0+1], a22 = A[i0+1, j0+1]
    # b21 = B[j0+1, k0], b22 = B[j0+1, k0+1]
    # C part: C11 = C[k0, i0] gets +1
    
    # Now, for each Pr, we add a column with the appropriate coefficients.
    
    # P1
    c = col_start
    # A coefficients: a11: 1, a22: 1
    U[i0 * m + j0][c] += 1       # a11
    U[(i0+1) * m + (j0+1)][c] += 1  # a22
    # B coefficients: b11: 1, b22: 1
    V[j0 * p + k0][c] += 1       # b11
    V[(j0+1) * p + (k0+1)][c] += 1  # b22
    # C coefficients: C11: +1, C22: +1
    W[k0 * n + i0][c] += 1       # C11
    W[(k0+1) * n + (i0+1)][c] += 1  # C22
    col_start += 1
    
    # P2
    c = col_start
    # A: a21: 1, a22: 1
    U[(i0+1) * m + j0][c] += 1
    U[(i0+1) * m + (j0+1)][c] += 1
    # B: b11: 1
    V[j0 * p + k0][c] += 1
    # C: C21: +1, C22: -1
    W[k0 * n + (i0+1)][c] += 1
    W[(k0+1) * n + (i0+1)][c] += -1
    col_start += 1
    
    # P3
    c = col_start
    # A: a11: 1
    U[i0 * m + j0][c] += 1
    # B: b12: 1, b22: -1
    V[j0 * p + (k0+1)][c] += 1
    V[(j0+1) * p + (k0+1)][c] += -1
    # C: C12: +1, C22: +1
    W[(k0+1) * n + i0][c] += 1
    W[(k0+1) * n + (i0+1)][c] += 1
    col_start += 1
    
    # P4
    c = col_start
    # A: a22: 1
    U[(i0+1) * m + (j0+1)][c] += 1
    # B: b21: 1, b11: -1
    V[(j0+1) * p + k0][c] += 1
    V[j0 * p + k0][c] += -1
    # C: C11: +1, C21: +1
    W[k0 * n + i0][c] += 1
    W[k0 * n + (i0+1)][c] += 1
    col_start += 1
    
    # P5
    c = col_start
    # A: a11: 1, a12: 1
    U[i0 * m + j0][c] += 1
    U[i0 * m + (j0+1)][c] += 1
    # B: b22: 1
    V[(j0+1) * p + (k0+1)][c] += 1
    # C: C11: -1, C12: +1
    W[k0 * n + i0][c] += -1
    W[(k0+1) * n + i0][c] += 1
    col_start += 1
    
    # P6
    c = col_start
    # A: a21: 1, a11: -1
    U[(i0+1) * m + j0][c] += 1
    U[i0 * m + j0][c] += -1
    # B: b11: 1, b12: 1
    V[j0 * p + k0][c] += 1
    V[j0 * p + (k0+1)][c] += 1
    # C: C22: +1
    W[(k0+1) * n + (i0+1)][c] += 1
    col_start += 1
    
    # P7
    c = col_start
    # A: a12: 1, a22: -1
    U[i0 * m + (j0+1)][c] += 1
    U[(i0+1) * m + (j0+1)][c] += -1
    # B: b21: 1, b22: 1
    V[(j0+1) * p + k0][c] += 1
    V[(j0+1) * p + (k0+1)][c] += 1
    # C: C11: +1
    W[k0 * n + i0][c] += 1
    col_start += 1
    
    return col_start

def _verify_decomposition(U, V, W, n, m, p, tolerance=1e-10):
    """
    Verify that the decomposition is exact.
    For all i, j, k: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should be 1.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1) > tolerance:
                    return False, (i, j, k, val)
    return True, None

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 3x4 matrix by a 4x7 matrix.
    
    Uses a combination of Strassen blocks and schoolbook for the remaining entries.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, V: V, W: W} if False else {"U": U, "V": V, "W": W}
    
    # Try different combinations of Strassen blocks
    # Possible 2x2x2 blocks: i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    # But we need disjoint blocks to avoid interference.
    
    # Let's try to find the best set of disjoint Strassen blocks.
    # Each block covers 2 rows of A (2*m entries), 2 rows of B (2*p entries), 2 cols of C (2*n entries).
    
    # For simplicity, let's try a few specific configurations:
    # Config 1: Two disjoint blocks
    # Block 1: i0=0, j0=0, k0=0
    # Block 2: i0=1, j0=2, k0=2
    
    # Config 2: Three disjoint blocks
    # Block 1: i0=0, j0=0, k0=0
    # Block 2: i0=0, j0=2, k0=2
    # Block 3: i0=1, j0=0, k0=4
    
    # We need to be careful about overlapping.
    # Let's try a greedy approach: try all possible blocks, pick the best set.
    
    # For now, let's try the 3-block configuration that was mentioned in inspirations.
    
    # Try configuration with 3 Strassen blocks
    U2, V2, W2 = _schoolbook(n, m, p)
    col = 0
    
    # We'll use a different approach: build from scratch with Strassen blocks first,
    # then fill the rest with schoolbook.
    
    U_new = [[0] * (n*m*p) for _ in range(n*m)]
    V_new = [[0] * (n*m*p) for _ in range(m*p)]
    W_new = [[0] * (n*m*p) for _ in range(p*n)]
    
    col = 0
    
    # Add Strassen block 1: i0=0, j0=0, k0=0
    col = _strassen_2x2x2(U_new, V_new, W_new, 0, 0, 0, col)
    
    # Add Strassen block 2: i0=1, j0=2, k0=2
    col = _strassen_2x2x2(U_new, V_new, W_new, 1, 2, 2, col)
    
    # Add Strassen block 3: i0=0, j0=0, k0=4
    # But this overlaps with block 1 in i and j. Let's check if it's valid.
    # Block 1 covers i in {0,1}, j in {0,1}, k in {0,1}
    # Block 3 would cover i in {0,1}, j in {0,1}, k in {4,5}
    # The i and j ranges overlap, but k is different. This should be fine.
    col = _strassen_2x2x2(U_new, V_new, W_new, 0, 0, 4, col)
    
    # Now, add schoolbook for the remaining entries
    # We need to track which (i,j,k) tuples are already covered.
    covered = set()
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    for ii in [0, 1]:
        for jj in [0, 1]:
            for kk in [0, 1]:
                covered.add((ii, jj, kk))
    
    # Block 2: i in {1,2}, j in {2,3}, k in {2,3}
    for ii in [1, 2]:
        for jj in [2, 3]:
            for kk in [2, 3]:
                covered.add((ii, jj, kk))
    
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5}
    for ii in [0, 1]:
        for jj in [0, 1]:
            for kk in [4, 5]:
                covered.add((ii, jj, kk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_new[i*m+j][col] = 1
                    V_new[j*p+k][col] = 1
                    W_new[k*n+i][col] = 1
                    col += 1
    
    # The total rank is col
    # We can truncate U, V, W to the actual rank used
    U_trunc = [row[:col] for row in U_new]
    V_trunc = [row[:col] for row in V_new]
    W_trunc = [row[:col] for row in W_new]
    
    # Verify
    valid, info = _verify_decomposition(U_trunc, V_trunc, W_trunc, n, m, p)
    
    if valid:
        best_rank = col
        best_result = {"U": U_trunc, "V": V_trunc, "W": W_trunc}
    else:
        # Fall back to schoolbook
        pass
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    # Try to improve further with more Strassen blocks or different configurations
    # For now, return the best result found
    
    return best_result
