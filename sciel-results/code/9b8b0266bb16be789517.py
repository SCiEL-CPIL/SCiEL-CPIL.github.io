"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import random

# EVOLVE-BLOCK-START

def _schoolbook_block(n, m, p):
    """Return (U, V, W) lists for a schoolbook n x m x p block, with local indices."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W, rank


def _strassen_block_2x2x2():
    """
    Strassen 2x2x2 decomposition with 7 rank-1 terms.
    Returns (U, V, W) for 2x2 matrix mult: C = A * B, A 2x2, B 2x2, C 2x2.
    U rows indexed by (i, j) of A, i,j in {0,1}
    V rows indexed by (j, k) of B, j,k in {0,1}
    W rows indexed by (k, i) of C, k,i in {0,1}
    
    Strassen's 7 products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # 7 rank-1 terms
    # Each term contributes to U (A-index), V (B-index), W (C-index)
    # U has 4 rows: (0,0),(0,1),(1,0),(1,1)
    # V has 4 rows: (0,0),(0,1),(1,0),(1,1)
    # W has 4 rows: (0,0),(0,1),(1,0),(1,1)
    
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # U: A00=+1, A11=+1
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    # W: C00=+1, C11=+1
    W[0][0] = 1  # C00 (k=0,i=0)
    W[3][0] = 1  # C11 (k=1,i=1)
    
    # M2: (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 (k=1,i=0)
    
    # M3: A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[3][2] = -1  # B11
    W[0][2] = 1  # C01 (k=0,i=1)
    W[3][2] = 1  # C11 (k=1,i=1)
    
    # M4: A11(B01-B00)
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[0][3] = -1  # B00
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5: (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1  # C00
    W[1][4] = 1  # C01
    
    # M6: (A10-A00)(B00+B10)
    U[2][5] = 1  # A10
    U[0][5] = -1  # A00
    V[0][5] = 1  # B00
    V[2][5] = 1  # B10
    W[3][5] = 1  # C11
    
    # M7: (A01-A11)(B01+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1  # A11
    V[1][6] = 1  # B01
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00
    
    return U, V, W, 7


def _combine_blocks(blocks):
    """
    Combine a list of (local_U, local_V, local_W, local_rank, offset) tuples
    into global U, V, W matrices.
    Each block has local dimensions and an offset (i_off, j_off, k_off) in the global indices.
    """
    n, m, p = DIMENSIONS
    U = [[0] * 0 for _ in range(n * m)]
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    total_rank = sum(b[3] for b in blocks)
    if total_rank == 0:
        return U, V, W
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    for local_U, local_V, local_W, local_rank, i_off, j_off, k_off, n_l, m_l, p_l in blocks:
        for li in range(n_l):
            for lj in range(m_l):
                global_ui = (i_off + li) * m + (j_off + lj)
                for c in range(local_rank):
                    val = local_U[li * m_l + lj][c]
                    if val != 0:
                        U[global_ui][rank_offset + c] += val
        
        for lj in range(m_l):
            for lk in range(p_l):
                global_vj = (j_off + lj) * p + (k_off + lk)
                for c in range(local_rank):
                    val = local_V[lj * p_l + lk][c]
                    if val != 0:
                        V[global_vj][rank_offset + c] += val
        
        for lk in range(p_l):
            for li in range(n_l):
                global_wk = (k_off + lk) * n + (i_off + li)
                for c in range(local_rank):
                    val = local_W[lk * n_l + li][c]
                    if val != 0:
                        W[global_wk][rank_offset + c] += val
        
        rank_offset += local_rank
    
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x8 matrix multiplication.
    
    Strategy: Split the k-index (of the B matrix, size 3) into k=0,1 and k=2.
    - For k in {0,1}: we have a 3x2x8 sub-problem (A is 3x2, B is 2x8).
      Split i-index: i in {0,1} gives 2x2x8, i=2 gives 1x2x8.
      - 2x2x8: use 4 Strassen 2x2x2 blocks (one for each pair of j), rank 4*7=28
      - 1x2x8: schoolbook, rank 16
    - For k=2: 3x1x8 sub-problem, schoolbook, rank 24
    
    Total rank: 28 + 16 + 24 = 68
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    # But wait: the tensor is T[i,j,k] = A[i,j] * B[j,k] where:
    # A is n x m = 3 x 3
    # B is m x p = 3 x 8
    # C is n x p = 3 x 8
    # The multiplication tensor has indices (i, j, k) where i in 0..n-1, j in 0..m-1, k in 0..p-1
    # T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ... 
    # Actually the matrix multiplication tensor T has T[i,j,k] = 1 if i corresponds to row of C,
    # j to the inner dimension, k to column of C.
    # T[i,j,k] = 1 when we multiply A[i,j] * B[j,k].
    # So the tensor is: for each (i,j,k) triple, T[i,j,k] = 1.
    # The rank-1 decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need to decompose the 3x3x8 tensor where:
    # i in {0,1,2} (rows of A / rows of C)
    # j in {0,1,2} (cols of A / rows of B)  
    # k in {0,1,2,3,4,5,6,7} (cols of B / cols of C)
    
    # Split j-index (the inner/summed index) into j in {0,1} and j=2.
    # For j in {0,1}: sub-tensor has i in {0,1,2}, j in {0,1}, k in {0..7}
    #   - Split i into i in {0,1} and i=2
    #     - i in {0,1}, j in {0,1}, k in {0..7}: 2x2x8, use Strassen blocks
    #     - i=2, j in {0,1}, k in {0..7}: 1x2x8, schoolbook
    # For j=2: sub-tensor has i in {0,1,2}, j=2, k in {0..7}: 3x1x8, schoolbook
    
    blocks = []
    
    # Part 1: i in {0,1}, j in {0,1}, k in {0..7}
    # Use 4 Strassen 2x2x2 blocks, one for each pair of k values
    for kb in range(4):
        k_start = kb * 2
        # Strassen block for A[i,j] with i in {0,1}, j in {0,1}
        # and B[j,k] with j in {0,1}, k in {k_start, k_start+1}
        # C[i,k] with i in {0,1}, k in {k_start, k_start+1}
        sU, sV, sW, srank = _strassen_block_2x2x2()
        blocks.append((sU, sV, sW, srank, 0, 0, k_start, 2, 2, 2))
    
    # Part 2: i=2, j in {0,1}, k in {0..7}
    # Schoolbook: 1 x 2 x 8
    U2, V2, W2, r2 = _schoolbook_block(1, 2, 8)
    blocks.append((U2, V2, W2, r2, 2, 0, 0, 1, 2, 8))
    
    # Part 3: i in {0,1,2}, j=2, k in {0..7}
    # Schoolbook: 3 x 1 x 8
    U3, V3, W3, r3 = _schoolbook_block(3, 1, 8)
    blocks.append((U3, V3, W3, r3, 0, 2, 0, 3, 1, 8))
    
    U, V, W = _combine_blocks(blocks)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
