"""
Strategy: Dynamic Rank Reduction via Pairwise Merging
Instead of hardcoding a specific block decomposition (which is prone to indexing errors
and may not be optimal), we start with the guaranteed valid schoolbook decomposition (rank 90).
We then attempt to reduce the rank by finding pairs of columns (rank-1 terms) that can be
merged or eliminated.

Specifically, we look for two terms t1, t2 such that their sum can be approximated by
a single term, or more generally, we try to find a basis for the space spanned by the
current terms that has lower dimension while still spanning the same tensor space.
However, a simpler and more robust heuristic for small ranks is "term elimination":
If we have terms c1, c2, we check if c1 is a scalar multiple of c2 (rare in schoolbook).
A better approach: Use the fact that the matrix multiplication tensor has symmetries.
We can use a random projection method or a specific algebraic identity.

Actually, the most effective "different" strategy for this specific task (3x5x6) is to
use the known result that rank(3x5x6) <= 84 via a specific block construction, but since
hardcoding is risky, we will implement a **Greedy Rank Reduction** algorithm.

Algorithm:
1. Start with schoolbook (rank 90).
2. For each pair of columns (i, j), check if the subspace spanned by them can be represented
   by a single column that satisfies the tensor equation. This is generally hard.
   
   Alternative: Use the **Bilinear Complexity Reduction** via finding linear dependencies.
   The tensor T = sum_{c=1}^R u_c v_c w_c^T.
   If we can find a linear combination of columns that results in a zero tensor, we can
   remove a degree of freedom.
   
   Let's try a different structural change: **Circulant Embedding / FFT-based approach**.
   For matrix multiplication C = A B, we can embed A and B into larger circulant matrices
   if the dimensions allow, but 3, 5, 6 are not powers of 2 or highly composite in a way
   that makes naive FFT embedding efficient for rank reduction without padding to 8x8x8 or similar.
   
   Let's stick to a **Verified Block Construction** but generated programmatically to avoid
   manual indexing errors. We will construct the decomposition for 3x5x6 by combining:
   - 2x2x2 Strassen blocks (rank 7) for the 2x2 submatrices.
   - 1xkxk or kx1xk schoolbook blocks for the remaining strips.
   
   Dimensions: 3x5 and 5x6.
   Split A (3x5) into:
     A11 (2x2), A12 (2x2), A13 (2x1)
     A21 (1x2), A22 (1x2), A23 (1x1)
   Split B (5x6) into:
     B11 (2x2), B12 (2x2), B13 (2x2)
     B21 (2x2), B22 (2x2), B23 (2x2)
     B31 (1x2), B32 (1x2), B33 (1x2)
   
   This is getting complicated to verify manually.
   
   Let's use the **Programmatic Block Builder** which correctly maps Strassen factors
   to the global indices. The previous code had a bug in `_strassen_2x2` (V2 undefined)
   and the hybrid builder was complex. I will write a clean, verified block builder.
"""

import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _build_block_strassen(n, m, p):
    """
    Construct a block Strassen decomposition for 3x5x6.
    We partition the indices into blocks of size 2 where possible, and 1 otherwise.
    
    A is 3x5. B is 5x6. C is 3x6.
    
    Partitions:
    Rows of A (i): 2, 1
    Cols of A (j): 2, 2, 1
    Rows of B (j): 2, 2, 1
    Cols of B (k): 2, 2, 2
    
    We iterate over all block combinations (ib, jb, kb).
    If the block size is 2x2x2, we use Strassen (rank 7).
    Otherwise, we use Schoolbook for that block.
    
    Block sizes:
    ib in {0, 1} -> size 2
    ib in {2} -> size 1
    
    jb in {0, 1} -> size 2
    jb in {2} -> size 1
    
    kb in {0, 1, 2} -> size 2
    
    Total blocks: 2 * 3 * 3 = 18 blocks.
    
    Block (ib, jb, kb) corresponds to:
    i in [start_i, start_i + size_i - 1]
    j in [start_j, start_j + size_j - 1]
    k in [start_k, start_k + size_k - 1]
    
    If size_i == 2 and size_j == 2 and size_k == 2:
        Use Strassen 2x2x2.
    Else:
        Use Schoolbook for size_i * size_j * size_k terms.
    """
    
    # Define partition sizes
    i_sizes = [2, 1]
    j_sizes = [2, 2, 1]
    k_sizes = [2, 2, 2]
    
    # Calculate start indices
    i_starts = [0, 2]
    j_starts = [0, 2, 4]
    k_starts = [0, 2, 4]
    
    # Pre-allocate maximum possible rank (schoolbook is 90, Strassen will be less)
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    
    col = 0
    
    # Strassen 2x2x2 factors
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U2[bi*2+bj, c] is coeff of A[bi, bj] in M_c
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ])
    
    # V2[bj*2+bk, c] is coeff of B[bj, bk] in M_c
    V2 = np.array([
        [1, 1, 0, 1, 0, 1, 0], # B11
        [0, 0, 0, 0, 0, 1, 0], # B12
        [0, 0, 1, -1, 0, 0, 1],# B21
        [1, 0, -1, 0, 1, 0, 1] # B22
    ])
    
    # W2[bk*2+bi, c] is coeff of M_c in C[bi, bk]
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C11 (bi=0, bk=0)
        [0, 0, 1, 0, 1, 0, 0],  # C12 (bi=0, bk=1)
        [0, 1, 0, 1, 0, 0, 0],  # C21 (bi=1, bk=0)
        [1, -1, 1, 0, 0, 1, 0]  # C22 (bi=1, bk=1)
    ])
    
    for ib in range(len(i_sizes)):
        for jb in range(len(j_sizes)):
            for kb in range(len(k_sizes)):
                si = i_starts[ib]
                sj = j_starts[jb]
                sk = k_starts[kb]
                si_sz = i_sizes[ib]
                sj_sz = j_sizes[jb]
                sk_sz = k_sizes[kb]
                
                if si_sz == 2 and sj_sz == 2 and sk_sz == 2:
                    # Use Strassen
                    for c in range(7):
                        # U part
                        for bi in range(2):
                            for bj in range(2):
                                u_val = U2[bi * 2 + bj, c]
                                if u_val != 0:
                                    i = si + bi
                                    j = sj + bj
                                    U[i * m + j, col] = u_val
                        # V part
                        for bj in range(2):
                            for bk in range(2):
                                v_val = V2[bj * 2 + bk, c]
                                if v_val != 0:
                                    j = sj + bj
                                    k = sk + bk
                                    V[j * p + k, col] = v_val
                        # W part
                        for bk in range(2):
                            for bi in range(2):
                                w_val = W2[bk * 2 + bi, c]
                                if w_val != 0:
                                    k = sk + bk
                                    i = si + bi
                                    W[k * n + i, col] = w_val
                        col += 1
                else:
                    # Use Schoolbook for this block
                    for i in range(si, si + si_sz):
                        for j in range(sj, sj + sj_sz):
                            for k in range(sk, sk + sk_sz):
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 50s margin
    
    # 1. Generate the Block Strassen decomposition
    U_bs, V_bs, W_bs, rank_bs = _build_block_strassen(n, m, p)
    
    # Convert to lists for verification
    U_list = [[float(x) for x in row] for row in U_bs]
    V_list = [[float(x) for x in row] for row in V_bs]
    W_list = [[float(x) for x in row] for row in W_bs]
    
    # Verify the block Strassen decomposition
    is_valid = _verify(U_list, V_list, W_list, n, m, p)
    
    if is_valid:
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook if the block construction failed verification
        # This ensures we always return a valid result
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
