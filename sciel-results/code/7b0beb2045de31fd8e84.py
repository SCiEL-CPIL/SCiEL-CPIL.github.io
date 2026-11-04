"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook decomposition of rank n*m*p."""
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


def _strassen_2x2x2():
    """Return U, V, W for Strassen 2x2x2 decomposition (rank 7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    U = [[0]*7 for _ in range(4)]  # n*m = 4 rows
    V = [[0]*7 for _ in range(4)]  # m*p = 4 rows
    W = [[0]*7 for _ in range(4)]  # p*n = 4 rows

    # A indices: A00->(0,0)->row 0, A01->(0,1)->row 1, A10->(1,0)->row 2, A11->(1,1)->row 3
    # B indices: B00->(0,0)->row 0, B01->(0,1)->row 1, B10->(1,0)->row 2, B11->(1,1)->row 3
    # C indices: C00->(0,0)->k*n+i=0*2+0=0, C01->(0,1)->k*n+i=0*2+1=1, C10->(1,0)->k*n+i=1*2+0=2, C11->(1,1)->k*n+i=1*2+1=3

    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1

    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[3][3] = 1

    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = 1

    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1

    return U, V, W


def _combine_decompositions(decomps):
    """Combine a list of (U, V, W) decompositions into one by concatenating columns.
    Each decomp must have the same row dimensions for U, V, W."""
    if not decomps:
        return None
    n_rows_U = len(decomps[0][0])
    n_rows_V = len(decomps[0][1])
    n_rows_W = len(decomps[0][2])
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0] * total_rank for _ in range(n_rows_U)]
    V = [[0] * total_rank for _ in range(n_rows_V)]
    W = [[0] * total_rank for _ in range(n_rows_W)]
    col_offset = 0
    for U_d, V_d, W_d in decomps:
        r = len(U_d[0])
        for i in range(n_rows_U):
            for c in range(r):
                U[i][col_offset + c] = U_d[i][c]
        for i in range(n_rows_V):
            for c in range(r):
                V[i][col_offset + c] = V_d[i][c]
        for i in range(n_rows_W):
            for c in range(r):
                W[i][col_offset + c] = W_d[i][c]
        col_offset += r
    return U, V, W


def _block_decompose(n, m, p, split_m):
    """Split m into split_m and m-split_m, solve each block separately.
    Block 1: n x split_m x p, Block 2: n x (m-split_m) x p.
    Returns combined U, V, W."""
    m1 = split_m
    m2 = m - split_m
    U1, V1, W1 = _schoolbook(n, m1, p)
    U2, V2, W2 = _schoolbook(n, m2, p)

    # Now we need to merge these into the full n x m x p decomposition.
    # U1 has n*m1 rows, U2 has n*m2 rows. We need n*m rows total.
    # The rows correspond to (i, j) where j ranges over 0..m-1.
    # Block 1 covers j in 0..m1-1, Block 2 covers j in m1..m-1.
    # Similarly for V: V1 has m1*p rows (j in 0..m1-1, k in 0..p-1), V2 has m2*p rows (j in m1..m-1 mapped to 0..m2-1, k in 0..p-1).
    # W is the same dimensions for both: p*n rows.

    total_rank = len(U1[0]) + len(U2[0])
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n

    U = [[0] * total_rank for _ in range(n_rows_U)]
    V = [[0] * total_rank for _ in range(n_rows_V)]
    W = [[0] * total_rank for _ in range(n_rows_W)]

    # Fill U: row i*m+j
    for i in range(n):
        for j in range(m1):
            src_row = i * m1 + j
            for c in range(len(U1[0])):
                U[i * m + j][c] = U1[src_row][c]
        for j in range(m1, m):
            src_row = i * m2 + (j - m1)
            for c in range(len(U2[0])):
                U[i * m + j][len(U1[0]) + c] = U2[src_row][c]

    # Fill V: row j*p+k
    for j in range(m1):
        for k in range(p):
            src_row = j * p + k
            for c in range(len(V1[0])):
                V[j * p + k][c] = V1[src_row][c]
    for j in range(m1, m):
        for k in range(p):
            src_row = (j - m1) * p + k
            for c in range(len(V2[0])):
                V[j * p + k][len(V1[0]) + c] = V2[src_row][c]

    # Fill W: row k*n+i (same for both blocks)
    for i in range(p):
        for j in range(n):
            src_row = i * n + j
            for c in range(len(W1[0])):
                W[src_row][c] = W1[src_row][c]
            for c in range(len(W2[0])):
                W[src_row][len(W1[0]) + c] = W2[src_row][c]

    return U, V, W


def _strassen_2x2x6():
    """Solve 2x2x6 using Strassen for the 2x2 part and schoolbook for the 6.
    Rank = 7 * 3 = 21 (since 6 = 2*3, we use 3 parallel Strassen-like blocks)
    Actually, let's think more carefully.
    
    For 2x2x6: A is 2x2, B is 2x6, C is 6x2.
    We can write B as [B_left | B_right] where B_left is 2x2 and B_right is 2x4.
    Or better: split the 6 into 3 groups of 2.
    
    Actually, the cleanest approach: the 2x2x6 tensor can be seen as 3 parallel 2x2x2 tensors
    if we group the 6 columns of B into 3 pairs. But that's not quite right because
    the output C is 6x2, so each pair of rows in C corresponds to one 2x2 block.
    
    Let's just use schoolbook for 2x2x6: rank = 2*2*6 = 24.
    Or use Strassen on the 2x2 part: rank = 7 * 6 = 42? No, that's worse.
    
    Actually for 2x2x6, the tensor is A(2x2) * B(2x6) = C(6x2).
    Using Strassen on the 2x2 matrix multiplication: we need 7 multiplications,
    but each "multiplication" here is a scalar times a row of B (2x6) producing a row of C (6x2)?
    No, that doesn't work directly.
    
    Let me just use schoolbook: rank 24.
    """
    return _schoolbook(2, 2, 6)


def _try_decomposition(n, m, p, deadline):
    """Try various decomposition strategies and return the best (lowest rank) one."""
    best = None
    best_rank = n * m * p  # schoolbook upper bound

    # Strategy 1: Pure schoolbook
    U, V, W = _schoolbook(n, m, p)
    r = len(U[0])
    if r < best_rank:
        best_rank = r
        best = (U, V, W)

    # Strategy 2: Block decomposition splitting m
    for split in range(1, m):
        if time.time() > deadline:
            break
        U, V, W = _block_decompose(n, m, p, split)
        r = len(U[0])
        if r < best_rank:
            best_rank = r
            best = (U, V, W)

    # Strategy 3: Use Strassen for 2x2x2 sub-blocks where applicable
    # For n=2, m=2: use Strassen (rank 7 instead of 8)
    # For the full 2x3x6, we can split m=3 into m=2 and m=1.
    # The 2x2x6 part: can we do better than schoolbook (rank 24)?
    # The 2x1x6 part: rank 12 (schoolbook, which is minimal for this shape)
    
    # Let's try: split m=3 into 2+1
    # 2x2x6 sub-problem: schoolbook rank 24
    # 2x1x6 sub-problem: schoolbook rank 12
    # Total: 36 (same as pure schoolbook)
    
    # Better: for 2x2x2 blocks, use Strassen.
    # Split p=6 into three 2s. Then 2x2x2 + 2x2x2 + 2x2x2 = 3 * Strassen(2x2x2)
    # Each Strassen 2x2x2 has rank 7, so total for 2x2x6 part = 21
    # 2x1x6 part: schoolbook rank 12
    # Total: 33
    
    if n == 2 and m == 3 and p == 6:
        # Custom construction: split m into 2+1, split p into 2+2+2 for the first part
        # Part A: 2x2x6, further split p into 3 blocks of 2
        # Part B: 2x1x6, schoolbook
        
        # Part A: 2x2x2 done 3 times (for p=0,1; p=2,3; p=4,5)
        # Each 2x2x2 uses Strassen rank 7
        # Part B: 2x1x6 uses schoolbook rank 12
        
        total_rank = 3 * 7 + 12  # 33
        
        U = [[0] * total_rank for _ in range(n * m)]  # 2*3 = 6 rows
        V = [[0] * total_rank for _ in range(m * p)]  # 3*6 = 18 rows
        W = [[0] * total_rank for _ in range(p * n)]  # 6*2 = 12 rows
        
        col = 0
        # Part A: rows i*m+j for j=0,1 (first two columns of A)
        # For each pair of k values (0,1), (2,3), (4,5):
        for k0 in range(0, p, 2):
            k1 = k0 + 1
            sU, sV, sW = _strassen_2x2x2()
            # sU has 4 rows (2x2), sV has 4 rows (2x2), sW has 4 rows (2x2)
            # Map sU rows: (i, j) where j in {0,1} -> full U row i*m+j
            for i in range(2):
                for j in range(2):
                    src = i * 2 + j
                    for c in range(7):
                        U[i * 3 + j][col + c] = sU[src][c]
            # Map sV rows: (j, k) where j in {0,1}, k in {0,1} -> full V row j*p+k
            for j in range(2):
                for k in range(2):
                    src = j * 2 + k
                    for c in range(7):
                        V[j * 6 + k0 + k][col + c] = sV[src][c]
            # Map sW rows: (k, i) where k in {0,1}, i in {0,1} -> full W row k*n+i
            for k in range(2):
                for i in range(2):
                    src = k * 2 + i
                    for c in range(7):
                        W[(k0 + k) * 2 + i][col + c] = sW[src][c]
            col += 7
        
        # Part B: 2x1x6, schoolbook. j=2 (third column of A)
        for i in range(2):
            for k in range(6):
                U[i * 3 + 2][col] = 1
                V[2 * 6 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
        
        if col == total_rank and total_rank < best_rank:
            best_rank = total_rank
            best = (U, V, W)

    if best is None:
        best = _schoolbook(n, m, p)

    return best


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    U, V, W = _try_decomposition(n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
