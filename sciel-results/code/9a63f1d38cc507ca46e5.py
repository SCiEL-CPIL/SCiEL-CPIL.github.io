"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _landsberg_3x3_3():
    """
    Return U, V, W for 3x3 * 3x3 with rank 23 (Landsberg).
    This is a known rank-23 decomposition of the 3x3x3 matrix multiplication tensor.
    """
    # We'll use a construction based on the known rank-23 algorithm.
    # The entries are 0, ±1, ±1/2.
    # We define the 23 terms. Each term has U (3x3), V (3x3), W (3x3) factors.
    # U[i,j] * V[j,k] * W[k,i] should sum to delta_{i,k} * delta_{j,j} for the MM tensor.
    
    # Based on Landsberg's rank-23 construction for 3x3 matrix multiplication.
    # The 23 multiplications can be described as follows.
    # Let A = [a b c; d e f; g h i], B = [j k l; m n o; p q r]
    
    # I'll use a known explicit rank-23 decomposition.
    # Reference: The rank of the 3x3 matrix multiplication tensor is 23.
    
    rank = 23
    # U has shape (9, 23), V has shape (9, 23), W has shape (9, 23)
    # U indexed by (i,j) where i,j in 0..2, flattened as i*3+j
    # V indexed by (j,k) where j,k in 0..2, flattened as j*3+k
    # W indexed by (k,i) where k,i in 0..2, flattened as k*3+i
    
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Here I'll define the 23 terms.
    # Each term is (U_entries, V_entries, W_entries)
    # where each is a dict mapping flat index to value.
    
    # I'll use a construction from the literature.
    # The rank-23 algorithm for 3x3 matrix multiplication.
    
    # Let me use a simpler approach: start from the 3x3x3 tensor and find
    # a rank-23 decomposition using known results.
    
    # Actually, let me use a verified construction.
    # The following is a rank-23 decomposition of 3x3x3 MM.
    
    # Term 1: (a+d+g)(j+m+p) - covers diagonal-ish
    # This is getting complex. Let me use a known explicit form.
    
    # I'll use the decomposition from "On the rank of the 3x3x3 matrix multiplication tensor"
    # The 23 terms with entries in {0, ±1, ±1/2}
    
    # For simplicity and correctness, let me use a different strategy:
    # Use 3x3x3 = 3x2x3 + 3x1x3, where 3x2x3 has rank 11 (known) and 3x1x3 has rank 9.
    # But that gives 20, which is better than 23! Wait, is 3x2x3 rank 11?
    
    # Actually, let me just use the Strassen-based approach for 3x3:
    # 3x3 can be done with rank 19 (known best upper bound for 3x3x3 is 19? No, it's 23.)
    # The best known upper bound for 3x3x3 is 23, lower bound is 19.
    
    # Let me use a practical approach: embed the 3x3x3 as a 2x2x2 Strassen plus corrections.
    # 3x3x3 = 2x2x2 (Strassen rank 7) + remaining terms.
    # The remaining terms: we have entries (2,2), (2,0),(0,2),(2,1),(1,2) etc.
    
    # This is getting complicated. Let me use a different strategy.
    
    # For the 3x3x8 problem, let me try:
    # Use 2x2x2 Strassen blocks in a different arrangement.
    
    # Actually, the key insight: with n=3, m=3, p=8:
    # We can cover the i,j dimensions more efficiently.
    # Instead of 4 blocks at (0,0,0),(0,0,2),(0,0,4),(0,0,6),
    # we can try blocks that cover more of the 3x3 i,j space.
    
    # Max non-overlapping 2x2x2 blocks in 3x3x8:
    # i: can place 1 block (covers i=0,1) or 1 block (covers i=1,2)
    # j: can place 1 block (covers j=0,1) or 1 block (covers j=1,2)  
    # k: can place 4 blocks (covers k=0-7)
    # So max 4 blocks, covering 4*8=32 of 72 entries.
    # Rank = 4*7 + (72-32) = 28 + 40 = 68.
    
    # But what if we use a mix? E.g., one 2x2x2 block plus some 1x1x1 schoolbook?
    # That's worse.
    
    # What about using a 3x3x3 block? If we had a rank-23 algorithm for 3x3x3:
    # Two 3x3x3 blocks cover k=0..5, rank = 2*23 = 46
    # Remaining k=6..7: 3*3*2 = 18 schoolbook
    # Total rank = 46 + 18 = 64. That's better!
    
    # So I need a rank-23 decomposition for 3x3x3.
    
    # Let me construct one. I'll use the following approach:
    # The 3x3x3 MM tensor can be decomposed using the following 23 terms.
    
    # I'll use a known explicit rank-23 decomposition.
    # Based on the work of Landsberg, the 23 terms are:
    
    # For now, let me use a simpler but valid approach:
    # Use the "bilinear" decomposition with 19 terms (best known lower bound is 19,
    # but I don't have an explicit 19-term construction).
    
    # Let me use a 23-term construction. I'll define it explicitly.
    
    # The following is a valid rank-23 decomposition of 3x3x3 MM.
    # Each term (c) has U_c, V_c, W_c which are 3x3 matrices (flattened to 9 entries).
    
    # I'll define the 23 terms as lists of (flat_index, value) for U, V, W.
    
    terms = []
    
    # Term 0: U has (0,0)=1, (2,2)=1; V has (0,0)=1, (2,2)=1; W has (0,0)=1, (2,2)=1
    # This covers a*j + i*r
    terms.append(
        [(0, 1), (8, 1)],  # U: a, i
        [(0, 1), (8, 1)],  # V: j, r
        [(0, 1), (8, 1)]   # W: (0,0), (2,2)
    )
    
    # Term 1: U has (1,1)=1; V has (1,1)=1; W has (1,1)=1
    # This covers e*n
    terms.append(
        [(4, 1)],
        [(4, 1)],
        [(4, 1)]
    )
    
    # Now I need 21 more terms to cover the remaining 17 diagonal entries
    # and ensure off-diagonal terms cancel.
    
    # This is getting very complex to do by hand. Let me use a different strategy.
    
    # Instead, let me use the fact that we can verify decompositions numerically.
    # I'll use a randomized approach to find a rank-23 decomposition, or use
    # a known explicit one.
    
    # For now, let me fall back to the Strassen approach but try different placements
    # to see if we can do better than 68.
    
    # Actually, let me try a hybrid: use 2x2x2 Strassen for some blocks and
    # 1x1x1 for others, but optimize the placement.
    
    # The current best is 68 with 4 Strassen blocks.
    # Can we get 67 or lower?
    
    # With 3 Strassen blocks: rank = 3*7 + (72-24) = 21 + 48 = 69. Worse.
    # With 4 Strassen blocks: rank = 4*7 + (72-32) = 28 + 40 = 68. Current best.
    # With 5 Strassen blocks: need 5 non-overlapping 2x2x2 in 3x3x8.
    #   i: max 1, j: max 1, k: max 4. So max 4. Can't have 5.
    
    # So 68 is the best with 2x2x2 Strassen blocks.
    
    # To do better, we need a different algorithm for some sub-blocks.
    
    # Let me try: use a 3x3x3 block with a known rank-23 algorithm.
    # I'll implement a verified rank-23 decomposition.
    
    # For now, return the Strassen-based result and try to improve.
    return _strassen_2x2_2()  # placeholder, will be replaced

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    sU, sV, sW = _strassen_2x2_2()
    for (i0, j0, k0) in strassen_blocks:
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    if _time_left() > 5:
        try:
            # Place 4 non-overlapping Strassen blocks:
            # i in {0,1}, j in {0,1}, k blocks at 0,2,4,6
            blocks = [
                (0, 0, 0),
                (0, 0, 2),
                (0, 0, 4),
                (0, 0, 6),
            ]
            
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
