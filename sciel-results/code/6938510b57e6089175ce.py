"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Build schoolbook factor matrices for (n,m,p)."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return (U, V, W) for Strassen 2x2x2 multiplication, rank 7."""
    # U has 2*2=4 rows, V has 2*2=4 rows, W has 2*2=4 rows
    # Rows of U: index (i,j) -> i*2+j
    # Rows of V: index (j,k) -> j*2+k
    # Rows of W: index (k,i) -> k*2+i
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # U row (i,j):
    # (0,0)=0: M1:+1, M3:+1, M5:+1, M6:-1, M7:-1
    # (0,1)=1: M3:+1, M5:+1, M6:-1, M7:-1
    # (1,0)=2: M1:+1, M2:+1, M6:+1
    # (1,1)=3: M1:+1, M2:+1, M4:+1, M7:+1
    
    U[0][0] = 1.0; U[0][2] = 1.0; U[0][4] = 1.0; U[0][5] = -1.0; U[0][6] = -1.0
    U[1][2] = 1.0; U[1][4] = 1.0; U[1][5] = -1.0; U[1][6] = -1.0
    U[2][0] = 1.0; U[2][1] = 1.0; U[2][5] = 1.0
    U[3][0] = 1.0; U[3][1] = 1.0; U[3][3] = 1.0; U[3][6] = 1.0
    
    # V row (j,k):
    # (0,0)=0: M1:+1, M2:+1, M6:+1
    # (0,1)=1: M1:+1, M4:-1, M6:+1
    # (1,0)=2: M3:+1, M5:+1, M7:+1
    # (1,1)=3: M1:+1, M4:+1, M5:+1, M7:+1
    
    V[0][0] = 1.0; V[0][1] = 1.0; V[0][5] = 1.0
    V[1][0] = 1.0; V[1][3] = -1.0; V[1][5] = 1.0
    V[2][2] = 1.0; V[2][4] = 1.0; V[2][6] = 1.0
    V[3][0] = 1.0; V[3][3] = 1.0; V[3][4] = 1.0; V[3][6] = 1.0
    
    # W row (k,i):
    # (0,0)=0 (C11): M1:+1, M4:+1, M5:-1, M7:-1
    # (0,1)=1 (C12): M3:+1, M5:+1
    # (1,0)=2 (C21): M2:+1, M4:+1
    # (1,1)=3 (C22): M1:+1, M2:-1, M3:+1, M7:+1
    
    W[0][0] = 1.0; W[0][3] = 1.0; W[0][4] = -1.0; W[0][6] = -1.0
    W[1][2] = 1.0; W[1][4] = 1.0
    W[2][1] = 1.0; W[2][3] = 1.0
    W[3][0] = 1.0; W[3][1] = -1.0; W[3][2] = 1.0; W[3][6] = 1.0
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta(i,j,k) for all i,j,k."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if nm > 0 else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i, j, k) == (0, 0, 0) else 0.0
                # Compute sum over c
                u_vals = U_arr[i * m + j, :]
                v_vals = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                total = float(np.dot(np.multiply(u_vals, v_vals), w_val))
                if abs(total - expected) > 1e-6:
                    return False
    return True


def construct():
    t0 = time.time()
    DEADLINE = 120.0
    MARGIN = 10.0
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try block decomposition: split dimensions into blocks
    # For 3x5x5, try splitting 5 into 2+3
    # Blocks: (2,2,2), (2,2,3), (2,3,2), (2,3,3), (3,2,2), (3,2,3), (3,3,2), (3,3,3)
    # Each block (bi, bj, bk) contributes rank bi*bj*bk (schoolbook) or less (Strassen for 2x2x2)
    
    def _block_rank(bi, bj, bk):
        """Get the minimal known rank for a bi x bj x bk block."""
        if bi == 2 and bj == 2 and bk == 2:
            return 7
        return bi * bj * bk
    
    def _build_block_decomposition(splits):
        """Build a decomposition using the given splits for each dimension.
        splits = [list of block sizes for dim0, list for dim1, list for dim2]
        """
        s_n, s_m, s_p = splits
        # Compute offsets
        n_off = [0]
        for s in s_n:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in s_m:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in s_p:
            p_off.append(p_off[-1] + s)
        
        total_rank = 0
        for bi, bj, bk in zip(s_n, s_m, s_p):
            # For each triple of block indices
            pass
        
        # Actually, we need to decompose into rectangular blocks where each block
        # uses rows/cols from specific ranges of i, j, k
        # Let's use a simpler approach: split each dimension independently
        # and use schoolbook/strassen for each rectangular sub-problem
        
        # Block decomposition: for each (a, b, c) where a in range(len(s_n)), 
        # b in range(len(s_m)), c in range(len(s_p)):
        # The sub-problem is s_n[a] x s_m[b] x s_p[c]
        # But this only works if we're doing a tensor product structure...
        # Actually for matrix multiplication, if we partition the matrix sizes,
        # the multiplication decomposes into a sum over block columns/rows.
        
        # Let me think more carefully.
        # A is n x m, B is m x p, C = A*B is n x p.
        # If we partition m into m1 + m2, then C = A[:, :m1] @ B[:m1, :] + A[:, m1:] @ B[m1:, :]
        # This is a sum of two matrix multiplications, so rank = rank1 + rank2.
        # Similarly for partitioning n or p.
        
        # So a valid decomposition strategy:
        # Partition m into segments m_1 + m_2 + ... + m_t
        # Then rank = sum_r rank(n, m_r, p)
        
        # Or partition n into n_1 + n_2 + ...
        # Then rank = sum_r rank(n_r, m, p)
        
        # Or partition p into p_1 + p_2 + ...
        # Then rank = sum_r rank(n, m, p_r)
        
        # The best simple improvement: partition the 5 dimension.
        # For n=3, m=5, p=5:
        # Option 1: partition m: (2,3) -> rank(3,2,5) + rank(3,3,5)
        #   rank(3,2,5) = 30, rank(3,3,5) = 45, total = 75 (same as schoolbook)
        #   Unless we can do better on sub-blocks.
        #   rank(3,2,5): can we partition p? (2,3) -> rank(3,2,2)+rank(3,2,3) = 12+18=30
        #   rank(3,3,5): partition p? (2,3) -> rank(3,3,2)+rank(3,3,3) = 18+27=45
        #   Total still 75.
        
        # Option 2: Use Strassen on 2x2x2 blocks within a partition.
        # If we partition m into (2,3) and p into (2,3):
        # We get 4 sub-problems:
        #   (3, 2, 2), (3, 2, 3), (3, 3, 2), (3, 3, 3)
        # Wait, that's not right. If we partition BOTH m and p, the structure is different.
        
        # Let me reconsider. If we partition m = m1 + m2:
        # C = A1 @ B1 + A2 @ B2
        # where A1 is n x m1, B1 is m1 x p, A2 is n x m2, B2 is m2 x p.
        # rank = rank(n, m1, p) + rank(n, m2, p)
        
        # For rank(3, 2, 5): we can further partition p = 2 + 3:
        # rank(3, 2, 5) = rank(3, 2, 2) + rank(3, 2, 3)
        # rank(3, 2, 2): schoolbook = 12. Can we do better?
        #   This is multiplying 3x2 by 2x2. 
        #   C (3x2) = A (3x2) @ B (2x2)
        #   Each row of C is a linear combination of rows of B weighted by A's row.
        #   Schoolbook: 3*2*2 = 12.
        #   Can we share? Row i of C: C_i0 = A_i0*B_00 + A_i1*B_10, C_i1 = A_i0*B_01 + A_i1*B_11
        #   For each i, we need 4 mults. No sharing between different i rows since A_i's are independent.
        #   So 12 seems minimal for this sub-block.
        #   
        #   Actually wait - Strassen applies to 2x2x2. Here we have 3x2x2.
        #   We can think of it as 3 copies of 1x2x2 (rank 4 each) = 12.
        #   Or 2 copies of 3x1x1... no that doesn't help.
        #   Or use the 2x2x2 Strassen on the last two dimensions?
        #   A is 3x2, B is 2x2. Think of A as 3 rows, each a 2-vector.
        #   For each row, multiply 2x1 by 2x2 = 2x2 result. That's rank 4 per row (schoolbook) or...
        #   Actually 2x1 times 2x2: result is 2x2. 
        #   c0 = a0*b0 + a1*b2, c1 = a0*b1 + a1*b3 (where b0,b1,b2,b3 are B entries)
        #   That's 4 mults, no Strassen-like savings since it's a 2x1x2 problem.
        
        # So the simple partitioning doesn't help beyond 75.
        
        # The key insight: we need a genuinely better algorithm, not just partitioning.
        # Let me try a different approach: use a known good decomposition for 3x5x5 or build one.
        
        # Actually, let me try a random search approach to find a lower-rank decomposition.
        # But that's very hard for tensor rank.
        
        # Alternative: use the fact that 3x5x5 can be embedded in a larger structure.
        # Or use bilinear complexity results.
        
        # For now, let me try: can we reduce by using half-integer or other values?
        # The schoolbook is already exact. The question is whether a lower rank exists.
        
        # Let me try a specific construction:
        # Split the 5x5 multiplication into overlapping 2x2 and 3x3 blocks.
        # This is getting complex. Let me just return schoolbook for now and add
        # a simple optimization attempt.
        
        return None  # Signal that this approach didn't yield an improvement
    
    # Simple optimization: try to find a rank-74 decomposition via manual construction
    # Based on the analysis: 3x5x5 can be decomposed as:
    # - Partition m into 2+3
    # - For the 3x2x5 part: use schoolbook (rank 30)
    # - For the 3x3x5 part: use schoolbook (rank 45)
    # Total: 75 (no improvement)
    
    # Let me try a different partition:
    # Partition n into 1+2:
    # rank(1,5,5) + rank(2,5,5) = 25 + 50 = 75 (no improvement)
    
    # The real gain comes from Strassen on 2x2x2 blocks.
    # To use Strassen, we need to identify 2x2x2 substructures.
    
    # Here's an idea: pad the 3x5x5 problem to use Strassen more effectively.
    # But padding increases rank, so that's counterproductive.
    
    # Let me try to construct a rank-74 decomposition explicitly.
    # Idea: use Strassen for one 2x2x2 block and schoolbook for the rest.
    
    # Partition: n=3, m=5, p=5
    # Break into blocks:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2, use Strassen (rank 7)
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3,4} -> 2x2x3, schoolbook (rank 12)
    # Block 3: i in {0,1}, j in {2,3,4}, k in {0,1} -> 2x3x2, schoolbook (rank 12)
    # Block 4: i in {0,1}, j in {2,3,4}, k in {2,3,4} -> 2x3x3, schoolbook (rank 18)
    # Block 5: i in {2}, j in {0,1}, k in {0,1} -> 1x2x2, schoolbook (rank 4)
    # Block 6: i in {2}, j in {0,1}, k in {2,3,4} -> 1x2x3, schoolbook (rank 6)
    # Block 7: i in {2}, j in {2,3,4}, k in {0,1} -> 1x3x2, schoolbook (rank 6)
    # Block 8: i in {2}, j in {2,3,4}, k in {2,3,4} -> 1x3x3, schoolbook (rank 9)
    
    # Total: 7 + 12 + 12 + 18 + 4 + 6 + 6 + 9 = 74
    
    # Now I need to verify this decomposition is valid.
    # The issue: when we partition all three dimensions independently, the sub-blocks
    # don't simply add up because the matrix multiplication involves sums over j.
    # 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # If we partition i, j, k independently, each (i,j,k) triple belongs to exactly one block.
    # The schoolbook decomposition assigns one term per (i,j,k) triple.
    # For the 2x2x2 block, we use Strassen which gives 7 terms instead of 8.
    # The other blocks use schoolbook.
    # This should be valid because the tensor is the sum of its (i,j,k) components,
    # and we're just finding a shorter representation for the 2x2x2 sub-tensor.
    
    # Let me construct this explicitly.
    
    def _build_74_decomposition():
        """Build the 74-rank decomposition using Strassen on the 2x2x2 block."""
        n, m, p = 3, 5, 5
        
        # Define block ranges
        # i: [0,1] and [2]
        # j: [0,1] and [2,3,4]  
        # k: [0,1] and [2,3,4]
        
        i_blocks = [range(0, 2), range(2, 3)]
        j_blocks = [range(0, 2), range(2, 5)]
        k_blocks = [range(0, 2), range(2, 5)]
        
        # Compute total rank
        ranks = []
        for ib in range(2):
            for jb in range(2):
                for kb in range(2):
                    ni = len(i_blocks[ib])
                    nj = len(j_blocks[jb])
                    nk = len(k_blocks[kb])
                    if ni == 2 and nj == 2 and nk == 2:
                        ranks.append(7)
                    else:
                        ranks.append(ni * nj * nk)
        
        total_rank = sum(ranks)
        assert total_rank == 74, f"Expected 74, got {total_rank}"
        
        # Build U, V, W
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ib in range(2):
            for jb in range(2):
                for kb in range(2):
                    i_list = list(i_blocks[ib])
                    j_list = list(j_blocks[jb])
                    k_list = list(k_blocks[kb])
                    ni, nj, nk = len(i_list), len(j_list), len(k_list)
                    
                    if ni == 2 and nj == 2 and nk == 2:
                        # Use Strassen
                        sU, sV, sW = _strassen_2x2x2()
                        # Map Strassen indices to global indices
                        for c in range(7):
                            # U rows: (i, j) where i is 0 or 1 in local, j is 0 or 1 in local
                            for li in range(2):
                                for lj in range(2):
                                    gi = i_list[li]
                                    gj = j_list[lj]
                                    U[gi * m + gj][col + c] += sU[li * 2 + lj][c]
                            # V rows: (j, k) where j is 0 or 1 in local, k is 0 or 1 in local
                            for lj in range(2):
                                for lk in range(2):
                                    gj = j_list[lj]
                                    gk = k_list[lk]
                                    V[gj * p + gk][col + c] += sV[lj * 2 + lk][c]
                            # W rows: (k, i) where k is 0 or 1 in local, i is 0 or 1 in local
                            for lk in range(2):
                                for li in range(2):
                                    gk = k_list[lk]
                                    gi = i_list[li]
                                    W[gk * n + gi][col + c] += sW[lk * 2 + li][c]
                        col += 7
                    else:
                        # Schoolbook for this block
                        for li in range(ni):
                            for lj in range(nj):
                                for lk in range(nk):
                                    gi = i_list[li]
                                    gj = j_list[lj]
                                    gk = k_list[lk]
                                    U[gi * m + gj][col] = 1.0
                                    V[gj * p + gk][col] = 1.0
                                    W[gk * n + gi][col] = 1.0
                                    col += 1
        
        assert col == total_rank
        return U, V, W, total_rank
    
    # Build the 74-rank decomposition
    U74, V74, W74, r74 = _build_74_decomposition()
    
    # Verify
    if _verify(U74, V74, W74, n, m, p):
        best_rank = r74
        best_U, best_V, best_W = U74, V74, W74
    else:
        # Fall back to schoolbook if verification fails
        U, V, W = _schoolbook(n, m, p)
        best_rank = n * m * p
        best_U, best_V, best_W = U, V, W
    
    # Check time and return best result
    elapsed = time.time() - t0
    if elapsed < DEADLINE - MARGIN:
        # Could try further optimization, but 74 is already good
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
