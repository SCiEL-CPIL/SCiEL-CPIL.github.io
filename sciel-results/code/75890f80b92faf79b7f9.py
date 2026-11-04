import numpy as np
import time

DIMENSIONS = [4, 5, 5]
DEADLINE = 145.0  # Safety margin before 150s kill

def _strassen_2x2x2_embed(n, m, p, i0, j0, k0):
    """
    Returns U, V, W for a 2x2x2 block at offset (i0, j0, k0) in full n x m x p space.
    Uses Strassen's algorithm (rank 7).
    """
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    
    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i
    
    # M1 = (A00+A11)(B00+B11)
    U[ui(i0, j0), 0] = 1
    U[ui(i1, j1), 0] = 1
    V[vj(j0, k0), 0] = 1
    V[vj(j1, k1), 0] = 1
    W[wk(k0, i0), 0] = 1
    W[wk(k1, i1), 0] = 1
    
    # M2 = (A10+A11)B00
    U[ui(i1, j0), 1] = 1
    U[ui(i1, j1), 1] = 1
    V[vj(j0, k0), 1] = 1
    W[wk(k0, i1), 1] = 1
    W[wk(k1, i1), 1] = -1
    
    # M3 = A00(B01-B11)
    U[ui(i0, j0), 2] = 1
    V[vj(j0, k1), 2] = 1
    V[vj(j1, k1), 2] = -1
    W[wk(k1, i0), 2] = 1
    W[wk(k1, i1), 2] = 1
    
    # M4 = A11(B10-B00)
    U[ui(i1, j1), 3] = 1
    V[vj(j1, k0), 3] = 1
    V[vj(j0, k0), 3] = -1
    W[wk(k0, i0), 3] = 1
    W[wk(k0, i1), 3] = 1
    
    # M5 = (A00+A01)B11
    U[ui(i0, j0), 4] = 1
    U[ui(i0, j1), 4] = 1
    V[vj(j1, k1), 4] = 1
    W[wk(k0, i0), 4] = -1
    W[wk(k1, i0), 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[ui(i1, j0), 5] = 1
    U[ui(i0, j0), 5] = -1
    V[vj(j0, k0), 5] = 1
    V[vj(j0, k1), 5] = 1
    W[wk(k1, i1), 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[ui(i0, j1), 6] = 1
    U[ui(i1, j1), 6] = -1
    V[vj(j1, k0), 6] = 1
    V[vj(j1, k1), 6] = 1
    W[wk(k0, i0), 6] = 1
    
    return U, V, W

def _schoolbook_embed(n, m, p, i0, j0, k0, bi, bj, bk):
    """
    Returns U, V, W for a schoolbook bi x bj x bk block at offset (i0, j0, k0).
    """
    rank = bi * bj * bk
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(bi):
        for j in range(bj):
            for k in range(bk):
                U[(i0 + i) * m + (j0 + j), col] = 1
                V[(j0 + j) * p + (k0 + k), col] = 1
                W[(k0 + k) * n + (i0 + i), col] = 1
                col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Strategy: Decompose 4x5x5 as:
    # C = A*B where A is 4x5, B is 5x5, C is 4x5
    # 
    # Partition A = [A1 | A2] with A1: 4x4, A2: 4x1
    # Partition B = [B11; B21] with B11: 4x5, B21: 1x5
    # 
    # C = A1*B11 + A2*B21
    # 
    # Further partition B11 = [B11a | B11b] with B11a: 4x4, B11b: 4x1
    # C = A1*B11a + A1*B11b + A2*B21
    #
    # A1*B11a: 4x4 x 4x4 -> use 2x2 Strassen tiling (4 blocks x rank 7 = 28)
    # A1*B11b: 4x4 x 4x1 -> schoolbook (4*4*1 = 16)
    # A2*B21: 4x1 x 1x5 -> schoolbook (4*1*5 = 20)
    #
    # Total rank: 28 + 16 + 20 = 64
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: 4x4 x 4x4 block at (i0=0, j0=0, k0=0) using 2x2 Strassen tiling
    # 4 Strassen 2x2x2 blocks:
    # (0,0,0), (0,0,2), (0,2,0), (0,2,2) in (i,j,k) offset space
    strassen_offsets = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2),
    ]
    for i0, j0, k0 in strassen_offsets:
        U_s, V_s, W_s = _strassen_2x2x2_embed(n, m, p, i0, j0, k0)
        U_parts.append(U_s)
        V_parts.append(V_s)
        W_parts.append(W_s)
    
    # Part 2: 4x4 x 4x1 block at (i0=0, j0=4, k0=0) -> schoolbook
    # A1 is rows 0-3, cols 0-3 of A; B11b is rows 0-3, col 4 of B; output cols 0-3
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 0, 4, 0, 4, 1, 4)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Part 3: 4x1 x 1x5 block at (i0=0, j0=4, k0=0) -> schoolbook
    # A2 is rows 0-3, col 4 of A; B21 is row 4, cols 0-4 of B; output all cols 0-4
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 0, 4, 0, 4, 1, 5)
    # Wait, this overlaps with Part 2! j0=4, but Part 2 also uses j0=4 for cols 0-3.
    # Let me reconsider.
    
    # Actually, A1*B11b uses A1 (cols 0-3) times B11b (row 4, but that's wrong)
    # Let me re-think the partition.
    
    # A is 4x5: rows i=0..3, cols j=0..4
    # B is 5x5: rows j=0..4, cols k=0..4
    # C is 4x5: rows i=0..3, cols k=0..4
    #
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # Split j into {0,1,2,3} and {4}:
    # C = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    #
    # Term 1: A1 (4x4) * B1 (4x5) where A1=A[:,0:4], B1=B[0:4,:]
    # Term 2: A2 (4x1) * B2 (1x5) where A2=A[:,4], B2=B[4,:]
    #
    # For Term 1: split B1 into B1a (4x4) and B1b (4x1)
    # B1a = B[0:4, 0:4], B1b = B[0:4, 4]
    # Term 1a: A1 (4x4) * B1a (4x4) -> 4x4x4
    # Term 1b: A1 (4x4) * B1b (4x1) -> 4x4x1
    #
    # Term 1a: 4x4x4 using 2x2 Strassen -> rank 28
    # Term 1b: 4x4x1 schoolbook -> rank 16
    # Term 2: 4x1x5 schoolbook -> rank 20
    #
    # Total: 28 + 16 + 20 = 64
    
    # Reset and redo properly
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Term 1a: A1(4x4) * B1a(4x4) -> C(4x4)
    # i in 0..3, j in 0..3, k in 0..3
    # Use 2x2 Strassen tiling: 4 blocks of 2x2x2
    strassen_offsets = [
        (0, 0, 0),  # i:0-1, j:0-1, k:0-1
        (0, 0, 2),  # i:0-1, j:0-1, k:2-3
        (0, 2, 0),  # i:0-1, j:2-3, k:0-1
        (0, 2, 2),  # i:0-1, j:2-3, k:2-3
    ]
    # Wait, that's only covering i:0-1. I need i:0-3.
    # 2x2 tiling of 4x4x4: 4 blocks in each dimension = 64 blocks? No.
    # 4x4x4 split into 2x2x2 blocks: 2*2*2 = 8 blocks, each 2x2x2.
    # 8 blocks x 7 = 56. That's not 28.
    #
    # I made an error. 4x4x4 with 2x2 tiling gives 8 Strassen blocks = 56 rank.
    # Schoolbook would be 64. So saving is only 8.
    
    # Let me reconsider. Is there a better way for 4x4x4?
    # Known: rank(4x4x4) <= 49 (Laderman-based or other constructions)
    # But for simplicity, let's use 8x7=56.
    
    # Actually, let me just use the straightforward decomposition:
    # Total = 56 + 16 + 20 = 92. Same as before!
    
    # The issue is that 4x4x4 via 2x2 Strassen gives 56, not 28.
    # I confused myself.
    
    # Let me try a different approach: use the 3x3x3 Laderman block.
    # 4x5x5 = 3x5x5 + 1x5x5
    # 3x5x5 = 3x4x5 + 3x1x5
    # 3x4x5 = 3x4x4 + 3x4x1
    # 3x4x4 = 3x3x4 + 1x3x4 = (3x3x3 + 3x3x1) + 1x3x4
    # 3x3x3: Laderman rank 23
    # 3x3x1: schoolbook 9
    # 1x3x4: schoolbook 12
    # 3x4x1: schoolbook 12
    # 3x1x5: schoolbook 15
    # 1x5x5: schoolbook 25
    # Total: 23+9+12+12+15+25 = 96. Worse.
    
    # Let me just go with the 8-Strassen-block approach for the 4x4x4 part:
    # 8 x 7 = 56 for 4x4x4
    # + 16 for 4x4x1
    # + 20 for 4x1x5
    # = 92. Same as current.
    
    # OK so the current approach is already optimal within this family.
    # Let me just output the current approach cleanly.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # 8 Strassen 2x2x2 blocks for the 4x4x4 subproblem (i:0-3, j:0-3, k:0-3)
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                U_s, V_s, W_s = _strassen_2x2x2_embed(n, m, p, i0, j0, k0)
                U_parts.append(U_s)
                V_parts.append(V_s)
                W_parts.append(W_s)
    
    # Schoolbook for 4x4x1 (i:0-3, j:0-3, k:4)
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 0, 0, 4, 4, 4, 1)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Schoolbook for 4x1x5 (i:0-3, j:4, k:0-4)
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 0, 4, 0, 4, 1, 5)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    U_combined = np.hstack(U_parts)
    V_combined = np.hstack(V_parts)
    W_combined = np.hstack(W_parts)
    
    rank = U_combined.shape[1]
    
    # Verify
    if _verify(U_combined.tolist(), V_combined.tolist(), W_combined.tolist(), n, m, p):
        return {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
    
    # Fallback: pure schoolbook
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
