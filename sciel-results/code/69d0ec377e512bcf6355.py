"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition for n×m×p matrix mult.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must equal 1 for all valid i,j,k."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2_2x2():
    """Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) with entries in {0, 1, -1}."""
    # Strassen's 7 products for 2x2 matrix mult:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U maps A entries (indexed 0=A11, 1=A12, 2=A21, 3=A22) to coefficients
    # V maps B entries (indexed 0=B11, 1=B12, 2=B21, 3=B22) to coefficients
    # W maps C entries (indexed 0=C11, 1=C12, 2=C21, 3=C22) to coefficients
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: A11+A22, B11+B22
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 += M1, C22 += M1
    
    # M2: A21+A22, B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C21 += M2
    W[3][1] = -1  # C22 -= M2
    
    # M3: A11, B12-B22
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C12 += M3
    W[3][2] = 1  # C22 += M3
    
    # M4: A22, B21-B11
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C11 += M4
    W[2][3] = 1  # C21 += M4
    
    # M5: A11+A12, B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C11 -= M5
    W[1][4] = 1  # C12 += M5
    
    # M6: A21-A11, B11+B12
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += M6
    
    # M7: A12-A22, B21+B22
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 += M7
    
    return U, V, W

def _build_hybrid(n, m, p, deadline):
    """
    Build a decomposition that uses Strassen for 2x2x2 sub-blocks where possible,
    and schoolbook for the remaining terms.
    
    For 3x4x5:
    - We can apply Strassen to pairs of (i, j, k) indices where we have 2x2x2 sub-problems.
    - The middle dimension m=4 can be split into two 2x2 blocks: j in {0,1} and j in {2,3}.
    - For each such block, we need 2x2x2 decompositions for each (i, k) pair.
    
    Strategy:
    - For i in {0,1}, j in {0,1}, k in {0,1}: use Strassen (saves 1 per 2x2x2)
    - For i in {0,1}, j in {2,3}, k in {2,3}: use Strassen (saves 1 per 2x2x2)  
    - For i in {1,2}, j in {0,1}, k in {3,4}: use Strassen (saves 1 per 2x2x2)
    - For i in {1,2}, j in {2,3}, k in {0,1}: use Strassen (saves 1 per 2x2x2)
    - Everything else: schoolbook
    
    Each 2x2x2 Strassen saves 1 rank term (7 instead of 8).
    4 Strassen blocks save 4 total, giving rank 60 - 4 = 56.
    """
    # Start with schoolbook rank
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the Strassen blocks: (i_range, j_range, k_range)
    # Each is a 2x2x2 sub-problem
    strassen_blocks = [
        # (i_start, j_start, k_start) - 2x2x2 block
        (0, 0, 0),   # i in {0,1}, j in {0,1}, k in {0,1}
        (0, 2, 2),   # i in {0,1}, j in {2,3}, k in {2,3}
        (1, 0, 3),   # i in {1,2}, j in {0,1}, k in {3,4}
        (1, 2, 0),   # i in {1,2}, j in {2,3}, k in {0,1}
    ]
    
    # Mark which (i,j,k) triples are handled by Strassen
    strassen_set = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    strassen_set.add((i0 + di, j0 + dj, k0 + dk))
    
    # Get Strassen matrices
    S_U, S_V, S_W = _strassen_2x2_2x2()
    
    # Assign column offsets for Strassen blocks
    column = 0
    block_offsets = []
    for (i0, j0, k0) in strassen_blocks:
        block_offsets.append(column)
        column += 7  # Strassen uses 7 terms
    
    # Fill Strassen blocks
    for (i0, j0, k0), offset in zip(strassen_blocks, block_offsets):
        for di in range(2):
            for dj in range(2):
                # A index: (i0+di, j0+dj) -> row (i0+di)*m + (j0+dj)
                a_row = (i0 + di) * m + (j0 + dj)
                # U coefficients from Strassen
                for c in range(7):
                    U[a_row][offset + c] = S_U[di * 2 + dj][c]
        
        for dj in range(2):
            for dk in range(2):
                # B index: (j0+dj, k0+dk) -> row (j0+dj)*p + (k0+dk)
                b_row = (j0 + dj) * p + (k0 + dk)
                for c in range(7):
                    V[b_row][offset + c] = S_V[dj * 2 + dk][c]
        
        for dk in range(2):
            for di in range(2):
                # C index: (k0+dk, i0+di) -> row (k0+dk)*n + (i0+di)
                c_row = (k0 + dk) * n + (i0 + di)
                for c in range(7):
                    W[c_row][offset + c] = S_W[dk * 2 + di][c]
    
    # Fill remaining (i,j,k) triples with schoolbook (1 term each)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_set:
                    continue
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try the hybrid Strassen + schoolbook approach
    result = _build_hybrid(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _build_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
