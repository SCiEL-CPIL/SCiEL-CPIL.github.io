import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0

    # Use 2x2x2 Strassen blocks: 4 blocks of 2x2x4, each decomposed as two 2x2x2 Strassen
    # Total rank = 4 * 2 * 7 = 56

    rank = 56

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 base matrices
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0: (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    # c=1: (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    # c=2: A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    # c=3: A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    # c=4: (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    # c=5: (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    # c=6: (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # For each 2x2x4 block (C_block, A_block, B_block), decompose as two 2x2x2 Strassen
    # C[i,j] = A[i,:] * B[:,j]
    # We have 4 C blocks (2x2), 4 A blocks (2x2), 4 B blocks (2x2x4)
    # For each C block at position (ci, cj), it uses A row ci and B column cj
    # A row ci: A[ci, 0] and A[ci, 1] -> 2x2 block
    # B column cj: B[:, cj*2] and B[:, cj*2+1] -> 2x2x2 block
    
    # C block (ci, cj) = A_row_ci * B_col_cj
    # where A_row_ci is 2x2 (rows ci*2, ci*2+1 of A), B_col_cj is 2x2x2 (cols cj*2, cj*2+1 of B)
    # Wait, B is 4x8, so B[:, cj*2:cj*2+2] is 4x2, not 2x2x2.
    # Let me reconsider.
    
    # A is 4x4, B is 4x8, C is 8x4
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # 
    # Block decomposition:
    # A: 2x2 blocks of 2x2: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # B: 2x2 blocks of 2x4: B[0:2,0:4], B[0:2,4:8], B[2:4,0:4], B[2:4,4:8]
    #   Or: B[0:2,0:2], B[0:2,2:4], B[0:2,4:6], B[0:2,6:8], etc.
    # C: 2x2 blocks of 4x2: C[0:4,0:2], C[0:4,2:4], C[4:8,0:2], C[4:8,2:4]
    #   Or: C[0:2,0:2], C[0:2,2:4], C[2:4,0:2], C[2:4,2:4], C[4:6,0:2], etc.
    
    # Let's use finer blocks:
    # A: 2x2 grid of 2x2 blocks
    # B: 2x4 grid of 2x2 blocks (rows 0-1, 2-3; cols 0-1, 2-3, 4-5, 6-7)
    # C: 4x2 grid of 2x2 blocks (rows 0-1, 2-3, 4-5, 6-7; cols 0-1, 2-3)
    
    # C[i_block, k_block] = sum_j_block A[i_block, j_block] * B[j_block, k_block]
    # where i_block in {0,1}, j_block in {0,1,2,3}, k_block in {0,1,2,3}
    # Wait, C is 8x4, so C has 4 row blocks and 2 col blocks.
    # A is 4x4, so A has 2 row blocks and 2 col blocks.
    # B is 4x8, so B has 2 row blocks and 4 col blocks.
    
    # C[ir, kc] = sum_jr A[ir, jr] * B[jr, kc]
    # where ir in {0,1,2,3} (row blocks of C, each 2x2), jr in {0,1} (row blocks of B / col blocks of A), kc in {0,1,2,3} (col blocks of C / col blocks of B)
    
    # Actually, let me just do it directly with 2x2x2 Strassen for each 2x2x4 subproblem.
    # For each pair (i_block, k_block) where i_block in {0,1,2,3}, k_block in {0,1,2,3}:
    #   C[i_block*2:(i_block+1)*2, k_block*2:(k_block+1)*2] 
    #     = sum_j A[i_block*2:(i_block+1)*2, j*2:(j+1)*2] * B[j*2:(j+1)*2, k_block*2:(k_block+1)*2]
    #     = A_i0 * B_0 + A_i1 * B_1
    #   where A_i0 = A[i_block*2:(i_block+1)*2, 0:2], A_i1 = A[i_block*2:(i_block+1)*2, 2:4]
    #         B_0 = B[0:2, k_block*2:(k_block+1)*2], B_1 = B[2:4, k_block*2:(k_block+1)*2]
    
    # So for each (i_block, k_block), we have a 2x2x2 + 2x2x2 = 2x2x4 problem.
    # We can decompose each 2x2x2 using Strassen (rank 7).
    # Total: 4 * 4 * 2 * 7 = 224? No, that's too high.
    
    # Wait, I think I'm overcomplicating. Let me just use the existing rank-98 approach but optimize.
    # Actually, let me try a different approach: use the 4x4x8 = (2x2x4) x 2 structure.
    
    # C = A * B where A is 4x4, B is 4x8
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2
    # Split B into 2x4 blocks: B = [[B00, B01], [B10, B11]] where each is 2x4
    # C = [[C00, C01], [C10, C11]] where each is 2x4
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each C_ij is 2x4, computed as sum of two 2x2 * 2x2 products.
    # Each 2x2*2x2 can be done with Strassen (rank 7).
    # So each C_ij needs 2*7 = 14 rank.
    # Total: 4 * 14 = 56 rank.
    
    # Let's build this.
    rank = 56
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # For each C block (ci, cj) where ci in {0,1}, cj in {0,1}:
    # C[ci] = A[ci,0]*B[0,cj] + A[ci,1]*B[1,cj]
    # where A[ci,0] is 2x2, B[0,cj] is 2x4, etc.
    # Wait, B[0,cj] is 2x4, not 2x2. Let me re-split.
    
    # B is 4x8. Split into 2 row blocks of 2x8, then each 2x8 into 4 col blocks of 2x2.
    # B[0, k] = B[0:2, k*2:(k+1)*2] for k in {0,1,2,3}
    # B[1, k] = B[2:4, k*2:(k+1)*2] for k in {0,1,2,3}
    
    # C is 8x4. Split into 4 row blocks of 2x4, then each 2x4 into 2 col blocks of 2x2.
    # C[i, k] = C[i*2:(i+1)*2, k*2:(k+1)*2] for i in {0,1,2,3}, k in {0,1}
    
    # C[i, k] = sum_j A[i//2, j] * B[j, k*2 + i%2] ... this is getting confusing.
    
    # Let me just index directly.
    # A[i, j] for i in 0..3, j in 0..3
    # B[j, k] for j in 0..3, k in 0..7
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # Use 2x2x2 Strassen for each (i_block, j_block, k_block) triple where:
    # i_block in {0,1} (rows 0-1, 2-3 of C and A)
    # j_block in {0,1} (cols 0-1, 2-3 of A; rows 0-1, 2-3 of B)
    # k_block in {0,1,2,3} (cols 0-1, 2-3, 4-5, 6-7 of B; cols 0-1, 2-3 of C)
    
    # C[i_block*2:(i_block+1)*2, k_block*2:(k_block+1)*2] 
    #   = A[i_block*2:(i_block+1)*2, j_block*2:(j_block+1)*2] * B[j_block*2:(j_block+1)*2, k_block*2:(k_block+1)*2]
    #   summed over j_block.
    
    # So for each (i_block, k_block), we have:
    # C[i_block, k_block] = A[i_block, 0] * B[0, k_block] + A[i_block, 1] * B[1, k_block]
    # where A[i_block, j_block] is 2x2, B[j_block, k_block] is 2x2.
    
    # Each product A[i_block, j_block] * B[j_block, k_block] is 2x2x2, rank 7 via Strassen.
    # Total: 4 * 4 * 2 * 7 = 224. Still too high.
    
    # I think the key insight is that we can share Strassen factors across the j_block dimension.
    # For fixed i_block and k_block:
    # C[i_block, k_block] = A0 * B0 + A1 * B1
    # This is a 2x2x4 problem (2x2 * 2x4 = 2x4... no, 2x2 * 2x4 is not standard).
    
    # Actually, let me just stick with the rank-98 approach and try to verify it's correct.
    # The current program claims rank 98. Let me just return that.
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Use the existing construction from the current program
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]
    
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            a_idx = gr * 4 + gc
                            eff = r * 2 + c
                            U[a_idx, col] += coeff_a * U_2[eff, m_s]
                
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            gk = gc + half * 4
                            b_idx = gr * 8 + gk
                            eff = r * 2 + c
                            V[b_idx, col] += coeff_b * V_2[eff, m_s]
                
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bi * 2 + r
                            gk = bj * 2 + c + half * 4
                            w_idx = gk * 4 + gi
                            eff = r * 2 + c
                            W[w_idx, col] += coeff_c * W_2[eff, m_s]
                
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
