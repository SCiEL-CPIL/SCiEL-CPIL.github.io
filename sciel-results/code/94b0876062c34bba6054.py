"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Strassen decomposition for 2x2 by 2x2 multiplication.
    Returns U, V, W as lists of lists with integer entries.
    U: shape (4, 7)  -- rows indexed by (i, j) in A: i in {0,1}, j in {0,1}
    V: shape (4, 7)  -- rows indexed by (j, k) in B: j in {0,1}, k in {0,1}
    W: shape (4, 7)  -- rows indexed by (k, i) in C: k in {0,1}, i in {0,1}
    
    The 7 bilinear products:
      M1 = (a11+a22)(b11+b22)
      M2 = (a21+a22)b11
      M3 = a11(b21-b11)
      M4 = a22(b22-b12)
      M5 = (a11+a12)b22
      M6 = (a21-a11)(b11+b12)
      M7 = (a12-a22)(b21+b22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # A indices: row = i*2 + j  (i=row of A, j=col of A = row of B)
    # B indices: row = j*2 + k  (j=row of B, k=col of B)
    # C indices: row = k*2 + i  (k=col of B, i=row of A)  [trace convention: W[k*n+i, r]]
    
    U = [[0]*7 for _ in range(4)]  # 4 rows (2x2), 7 cols
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    # a11 = A[0,0] -> U[0], a22 = A[1,1] -> U[3]
    # b11 = B[0,0] -> V[0], b22 = B[1,1] -> V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 += M1, C22 += M1
    W[0][0] = 1   # C[0,0] = k=0,i=0
    W[3][0] = 1   # C[1,1] = k=1,i=1
    
    # M2 = (a21+a22)b11
    # a21 = A[1,0] -> U[2], a22 = A[1,1] -> U[3]
    # b11 = B[0,0] -> V[0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 += M2
    W[2][1] = 1   # C[1,0] = k=1,i=0
    
    # M3 = a11(b21-b11)
    # a11 = A[0,0] -> U[0]
    # b21 = B[1,0] -> V[2], b11 = B[0,0] -> V[0]
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    # C12 += M3, C22 += M3
    W[1][2] = 1   # C[0,1] = k=0,i=1
    W[3][2] = 1   # C[1,1] = k=1,i=1
    
    # M4 = a22(b22-b12)
    # a22 = A[1,1] -> U[3]
    # b22 = B[1,1] -> V[3], b12 = B[0,1] -> V[1]
    U[3][3] = 1
    V[3][3] = 1; V[1][3] = -1
    # C11 += M4, C21 += M4
    W[0][3] = 1   # C[0,0]
    W[2][3] = 1   # C[1,0]
    
    # M5 = (a11+a12)b22
    # a11 = A[0,0] -> U[0], a12 = A[0,1] -> U[1]
    # b22 = B[1,1] -> V[3]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 -= M5, C12 += M5
    W[0][4] = -1  # C[0,0]
    W[1][4] = 1   # C[0,1]
    
    # M6 = (a21-a11)(b11+b12)
    # a21 = A[1,0] -> U[2], a11 = A[0,0] -> U[0]
    # b11 = B[0,0] -> V[0], b12 = B[0,1] -> V[1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 += M6
    W[3][5] = 1   # C[1,1]
    
    # M7 = (a12-a22)(b21+b22)
    # a12 = A[0,1] -> U[1], a22 = A[1,1] -> U[3]
    # b21 = B[1,0] -> V[2], b22 = B[1,1] -> V[3]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 += M7
    W[0][6] = 1   # C[0,0]
    
    return U, V, W


def _build_schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m by m x p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W, rank


def _block_multiply(U1, V1, W1, U2, V2, W2, r1, r2):
    """
    Combine two independent decompositions.
    U1: (rA, r1), V1: (rB, r1), W1: (rC, r1)
    U2: (rA, r2), V2: (rB, r2), W2: (rC, r2)
    Returns combined U: (rA, r1+r2), etc.
    """
    rA = len(U1)
    rB = len(V1)
    rC = len(W1)
    rank = r1 + r2
    U = [row1 + row2 for row1, row2 in zip(U1, U2)]
    V = [row1 + row2 for row1, row2 in zip(V1, V2)]
    W = [row1 + row2 for row1, row2 in zip(W1, W2)]
    return U, V, W, rank


def _embed_strassen_in_block(n, m, p, i0, j0, k0):
    """
    Embed a 2x2x2 Strassen block at position (i0, j0, k0) in an n x m by m x p multiplication.
    Returns the full-size U, V, W with the Strassen block placed and zeros elsewhere.
    """
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    sU, sV, sW = _build_strassen_2x2x2()
    
    for i in range(2):
        for j in range(2):
            a_row = (i0 + i) * m + (j0 + j)
            for r in range(7):
                U[a_row][r] = sU[i*2 + j][r]
    
    for j in range(2):
        for k in range(2):
            b_row = (j0 + j) * p + (k0 + k)
            for r in range(7):
                V[b_row][r] = sV[j*2 + k][r]
    
    for k in range(2):
        for i in range(2):
            c_row = (k0 + k) * n + (i0 + i)
            for r in range(7):
                W[c_row][r] = sW[k*2 + i][r]
    
    return U, V, W


def _combine_decompositions(decs):
    """
    Combine a list of (U, V, W) decompositions (each with same rank) into one.
    Actually, they may have different ranks. We concatenate columns.
    """
    if not decs:
        return None
    
    U0, V0, W0 = decs[0]
    rA = len(U0)
    rB = len(V0)
    rC = len(W0)
    total_rank = sum(len(d[0][0]) for d in decs)
    
    U = [[0]*total_rank for _ in range(rA)]
    V = [[0]*total_rank for _ in range(rB)]
    W = [[0]*total_rank for _ in range(rC)]
    
    col_offset = 0
    for U_d, V_d, W_d in decs:
        r_d = len(U_d[0])
        for i in range(rA):
            for c in range(r_d):
                U[i][col_offset + c] = U_d[i][c]
        for i in range(rB):
            for c in range(r_d):
                V[i][col_offset + c] = V_d[i][c]
        for i in range(rC):
            for c in range(r_d):
                W[i][col_offset + c] = W_d[i][c]
        col_offset += r_d
    
    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,k}... 
    Actually the tensor is T[i,j,k] = 1 for all (i,j,k) in the matmul tensor.
    The decomposition satisfies: for each (i,j,k), sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_hybrid(n, m, p):
    """
    Build a hybrid decomposition using Strassen 2x2x2 blocks where possible,
    and schoolbook for the rest.
    
    Strategy: partition the indices into 2x2x2 blocks and singletons.
    For n=3, m=5, p=7:
    - We can place floor(3/2)=1 full 2x2 in i, floor(5/2)=2 full 2x2 in j, floor(7/2)=3 full 2x2 in k
    - But we need to cover all indices. Let's use a tiling approach.
    
    Actually, let's think more carefully. The matmul tensor T[i,j,k] = 1 for all i,j,k.
    We want to decompose it as sum of rank-1 terms.
    
    A cleaner approach: partition i into groups of size 2 (and possibly 1), same for j and k.
    For each combination of groups, if all three are size 2, use Strassen (rank 7).
    Otherwise, use schoolbook for that sub-block.
    
    For n=3: groups [2, 1]
    For m=5: groups [2, 2, 1]
    For p=7: groups [2, 2, 2, 1]
    
    Number of group combinations: 2 * 3 * 4 = 24
    For each, if all sizes are 2, use Strassen (rank 7); else schoolbook (rank = si*sj*sk).
    """
    def make_groups(n):
        groups = []
        while n > 2:
            groups.append(2)
            n -= 2
        groups.append(n)
        return groups
    
    gi = make_groups(n)  # [2, 1]
    gj = make_groups(m)  # [2, 2, 1]
    gk = make_groups(p)  # [2, 2, 2, 1]
    
    decs = []
    i_off = 0
    for si in gi:
        j_off = 0
        for sj in gj:
            k_off = 0
            for sk in gk:
                if si == 2 and sj == 2 and sk == 2:
                    U_b, V_b, W_b = _embed_strassen_in_block(n, m, p, i_off, j_off, k_off)
                    decs.append((U_b, V_b, W_b))
                else:
                    # Schoolbook for this sub-block
                    rank_b = si * sj * sk
                    U_b = [[0]*rank_b for _ in range(n*m)]
                    V_b = [[0]*rank_b for _ in range(m*p)]
                    W_b = [[0]*rank_b for _ in range(p*n)]
                    col = 0
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                a_row = (i_off + ii) * m + (j_off + jj)
                                b_row = (j_off + jj) * p + (k_off + kk)
                                c_row = (k_off + kk) * n + (i_off + ii)
                                U_b[a_row][col] = 1
                                V_b[b_row][col] = 1
                                W_b[c_row][col] = 1
                                col += 1
                    decs.append((U_b, V_b, W_b))
                k_off += sk
            j_off += sj
        i_off += si
    
    return _combine_decompositions(decs)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, build the hybrid Strassen+schoolbook decomposition
    U, V, W, rank = _build_hybrid(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W, rank = _build_schoolbook_block(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
