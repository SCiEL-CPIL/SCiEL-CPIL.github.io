"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Strassen's 7 multiplications for 2x2x2
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M1 + M2
    # C22 = M4 + M6 - M2 - M3
    # where:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U: (4, 7) for A entries indexed [i*2+j, r]
    # V: (4, 7) for B entries indexed [j*2+k, r]  
    # W: (4, 7) for C entries indexed [k*2+i, r]
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1  # C11 += M1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21 += M2
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1; V[3, 2] = -1  # B12, -B22
    W[1, 2] = 1  # C12 += M3
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1; V[0, 3] = -1  # B21, -B11
    W[0, 3] = 1  # C11 += M4
    W[3, 3] = 1  # C22 += M4
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1  # B22
    W[1, 4] = 1  # C12 += M5
    W[0, 4] = -1  # C11 -= M5
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1  # A21, -A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1  # C22 += M6
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1  # A12, -A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = -1  # C11 -= M7
    W[2, 6] = 1  # C21 += M7
    
    return U.tolist(), V.tolist(), W.tolist()


def _schoolbook_block(n, m, p):
    """Return U, V, W for schoolbook n×m × m×p decomposition."""
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


def _combine_decompositions(blocks):
    """
    Combine block decompositions into full matrices.
    blocks: list of (i_start, i_end, j_start, j_end, k_start, k_end, U_block, V_block, W_block)
    where i is row index in A/C, j is col in A / row in B, k is col in B / col in C
    """
    n, m, p = DIMENSIONS
    total_rank = sum(len(W_block[0]) for _, _, _, _, _, _, _, _, W_block in blocks)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for (i_s, i_e, j_s, j_e, k_s, k_e, U_b, V_b, W_b) in blocks:
        rank_b = len(W_b[0])
        for r in range(rank_b):
            for i in range(i_s, i_e):
                for j in range(j_s, j_e):
                    U[i * m + j][col_offset + r] = U_b[(i - i_s) * (j_e - j_s) + (j - j_s)][r]
            for j in range(j_s, j_e):
                for k in range(k_s, k_e):
                    V[j * p + k][col_offset + r] = V_b[(j - j_s) * (k_e - k_s) + (k - k_s)][r]
            for k in range(k_s, k_e):
                for i in range(i_s, i_e):
                    W[k * n + i][col_offset + r] = W_b[(k - k_s) * (i_e - i_s) + (i - i_s)][r]
        col_offset += rank_b
    
    return U, V, W


def _verify(U, V, W):
    """Verify the decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                expected = 1.0
                err = abs(val - expected)
                max_err = max(max_err, err)
    return max_err < 1e-9


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try different block partitions
    # Split n=3 into (2,1) or (1,2)
    # Split m=5 into (2,3) or (3,2)  
    # Split p=5 into (2,3) or (3,2)
    
    n_splits = [(2, 1), (1, 2)]
    m_splits = [(2, 3), (3, 2)]
    p_splits = [(2, 3), (3, 2)]
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, get schoolbook as fallback
    U_school, V_school, W_school = _schoolbook_block(n, m, p)
    best_result = (U_school, V_school, W_school, best_rank)
    
    # Try each combination of splits
    for n_split in n_splits:
        for m_split in m_splits:
            for p_split in p_splits:
                if time.time() > deadline:
                    break
                
                n1, n2 = n_split
                m1, m2 = m_split
                p1, p2 = p_split
                
                # We have 8 sub-blocks for A, B, C
                # A blocks: (n1×m1), (n1×m2), (n2×m1), (n2×m2)
                # B blocks: (m1×p1), (m1×p2), (m2×p1), (m2×p2)
                # C blocks: (n1×p1), (n1×p2), (n2×p1), (n2×p2)
                
                # The multiplication: C_block[i,j] = sum_l A_block[i,l] * B_block[l,j]
                # where i is row index (0 or 1), j is col index (0 or 1), l is mid (0 or 1)
                
                # For each of the 8 "tensor components" (i,j,l), we need a decomposition
                # of the sub-problem: (n_i × m_l) * (m_l × p_j) -> (n_i × p_j)
                # But these are coupled through the shared rank space.
                
                # Simpler approach: use Strassen on 2x2x2 blocks where all three dims are 2
                # Otherwise use schoolbook
                
                # Let's just do schoolbook on each block and sum ranks
                # This won't beat 75, so we need Strassen.
                
                # Better: identify which sub-blocks are 2x2x2 and use Strassen there.
                
                blocks = []
                
                # For each combination of (row_block, mid_block, col_block):
                # row_block in {0,1}, mid_block in {0,1}, col_block in {0,1}
                # The sub-problem is: A[row_block, mid_block] * B[mid_block, col_block] -> C[row_block, col_block]
                # Dimensions: (n_rb × m_mb) * (m_mb × p_cb) -> (n_rb × p_cb)
                
                # We can't independently decompose each because they share the rank space.
                # Instead, let's build the full decomposition by concatenating.
                
                # Actually, the correct approach: the full tensor is a sum over all (rb, mb, cb)
                # of the sub-tensors. Each sub-tensor has rank = rank of that sub-problem.
                # Total rank = sum of ranks of all 8 sub-problems.
                
                # For a sub-problem of size (a×b) * (b×c) -> (a×c):
                # If a=b=c=2, use Strassen: rank 7
                # Otherwise, use schoolbook: rank a*b*c
                # But we can also use other known decompositions.
                
                sub_ranks = []
                for rb in range(2):
                    for mb in range(2):
                        for cb in range(2):
                            a = n_split[rb]
                            b = m_split[mb]
                            c = p_split[cb]
                            if a == 2 and b == 2 and c == 2:
                                sub_ranks.append(7)
                            else:
                                sub_ranks.append(a * b * c)
                
                total_rank = sum(sub_ranks)
                
                if total_rank < best_rank:
                    # Build the full decomposition
                    blocks = []
                    for rb in range(2):
                        for mb in range(2):
                            for cb in range(2):
                                a = n_split[rb]
                                b = m_split[mb]
                                c = p_split[cb]
                                
                                i_s = n_split[0] if rb == 1 else 0
                                i_e = i_s + a
                                j_s = m_split[0] if mb == 1 else 0
                                j_e = j_s + b
                                k_s = p_split[0] if cb == 1 else 0
                                k_e = k_s + c
                                
                                if a == 2 and b == 2 and c == 2:
                                    U_b, V_b, W_b = _strassen_2x2()
                                else:
                                    U_b, V_b, W_b = _schoolbook_block(a, b, c)
                                
                                blocks.append((i_s, i_e, j_s, j_e, k_s, k_e, U_b, V_b, W_b))
                    
                    U_full, V_full, W_full = _combine_decompositions(blocks)
                    
                    # Verify
                    if _verify(U_full, V_full, W_full):
                        best_result = (U_full, V_full, W_full, total_rank)
                        best_rank = total_rank
                
                if time.time() > deadline:
                    break
    
    if time.time() > deadline:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Try more sophisticated approaches: use Strassen on larger 2x2x2-like structures
    # by grouping blocks
    
    # Alternative: try all possible ways to partition into 2x2x2 Strassen blocks
    # This is complex; for now, return best found
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
