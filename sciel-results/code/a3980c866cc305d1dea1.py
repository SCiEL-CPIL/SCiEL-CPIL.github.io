"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Deadline guard: 140s safety margin within 150s limit
    start_time = time.time()
    deadline = start_time + 140
    
    # Build schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 80
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), rank_schoolbook
    
    # Helper: verify a decomposition
    def verify(U_try, V_try, W_try, tol=1e-9):
        n, m, p = DIMENSIONS
        # Check on-support entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_try[a, :], V_try[b, :] * W_try[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check off-support entries: exhaustive check
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i1, j1 = divmod(a, m)
                    j2, k2 = divmod(b, p)
                    i2, k1 = divmod(c, n)
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        continue
                    val = np.dot(U_try[a, :], V_try[b, :] * W_try[c, :])
                    if abs(val) > tol:
                        return False
        return True
    
    # Build rank-49 Strassen decomposition for 4x4x4
    # Using 2x2 block Strassen: each 2x2 block multiplication has rank 7
    # 4 blocks * 7 = 28? No, that's not right.
    # Actually for 4x4x4, we can view it as 2x2 blocks of 2x2 matrices.
    # Each 2x2x2 has rank 7 (Strassen).
    # 4 such multiplications: 4*7 = 28? But that only gives rank 28 for 4x4x4.
    # Wait, that's not correct either. Let me think again.
    
    # For 4x4x4 matrix multiplication:
    # C = A * B where A, B, C are 4x4
    # View as 2x2 blocks: A = [[A11, A12], [A21, A22]] with 2x2 blocks
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each Cij is a sum of two 2x2x2 multiplications.
    # We can use Strassen's algorithm on the 2x2 block structure.
    # The 7 Strassen products for 2x2 blocks:
    # M1 = (A11 - A21)(B21 - B22)
    # M2 = (A11 + A12)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B11 + B22)
    # M5 = (A12 - A22)(B11 + B21)
    # M6 = (A11 + A21)(B22 - B21)
    # M7 = (A21 - A11)(B11 + B12)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M3
    # C21 = M1 + M6
    # C22 = M3 + M4 - M2 + M5
    
    # But each Mi is a 2x2x2 tensor multiplication (rank 7 each).
    # So total rank = 7 * 7 = 49 for the 4x4x4 case.
    
    # Strassen 2x2x2 decomposition (rank 7):
    # For multiplying 2x2 matrices X * Y = Z:
    # M1 = (X11 - X21)(Y21 - Y22)
    # M2 = (X11 + X12)Y22
    # M3 = X11(Y11 - Y22)
    # M4 = X22(Y11 + Y22)
    # M5 = (X12 - X22)(Y11 + Y21)
    # M6 = (X11 + X21)(Y22 - Y21)
    # M7 = (X21 - X11)(Y11 + Y12)
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M2 + M3
    # Z21 = M1 + M6
    # Z22 = M3 + M4 - M2 + M5
    
    # Define the 7 Strassen factors for 2x2x2
    # U factors: for each of 7 products, which entries of X are used and with what sign
    # V factors: for each of 7 products, which entries of Y are used and with what sign
    # W factors: for each of 7 products, which entries of Z are used and with what sign
    
    # Indexing for 2x2: row*2 + col
    # X entries: X00=0, X01=1, X10=2, X11=3
    # Y entries: Y00=0, Y01=1, Y10=2, Y11=3
    # Z entries: Z00=0, Z01=1, Z10=2, Z11=3
    
    # M1 = (X11 - X21)(Y21 - Y22)
    # U: X11=+1, X21=-1
    # V: Y21=-1, Y22=+1
    # W: Z11=+1, Z21=+1
    
    # M2 = (X11 + X12)Y22
    # U: X11=+1, X12=+1
    # V: Y22=+1
    # W: Z12=+1
    
    # M3 = X11(Y11 - Y22)
    # U: X11=+1
    # V: Y11=+1, Y22=-1
    # W: Z12=+1, Z22=+1
    
    # M4 = X22(Y11 + Y22)
    # U: X22=+1
    # V: Y11=+1, Y22=+1
    # W: Z11=+1, Z22=+1
    
    # M5 = (X12 - X22)(Y11 + Y21)
    # U: X12=+1, X22=-1
    # V: Y11=+1, Y21=+1
    # W: Z11=-1, Z22=+1
    
    # M6 = (X11 + X21)(Y22 - Y21)
    # U: X11=+1, X21=+1
    # V: Y22=+1, Y21=-1
    # W: Z21=+1
    
    # M7 = (X21 - X11)(Y11 + Y12)
    # U: X21=+1, X11=-1
    # V: Y11=+1, Y12=+1
    # W: Z11=+1
    
    # Now build the full 4x4x4 rank-49 decomposition
    # The 4x4 matrix has blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    # C = [[C11, C12], [C21, C22]] where each Ck is 2x2
    
    # Block indices for 4x4:
    # A: rows 0-3, cols 0-3
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # Similarly for B and C
    
    # For each of the 7 Strassen products at the block level:
    # Product r (r=0..6) computes Mi = (linear combo of A blocks) * (linear combo of B blocks)
    # Each Mi is a 2x2x2 multiplication, which itself has rank 7
    
    # So for each block-level product r, and each 2x2-level product s (s=0..6):
    # We have a term that contributes to the final C
    
    # Let's define the block-level Strassen factors:
    # For product r, U_block[r] maps 4x4 A entries to the "X" of the 2x2 block mult
    # V_block[r] maps 4x4 B entries to the "Y" of the 2x2 block mult
    # W_block[r] maps 4x4 C entries to the "Z" of the 2x2 block mult
    
    # Actually, let me think of this differently.
    # The 4x4x4 tensor can be decomposed as a Kronecker product of two 2x2x2 tensors.
    # T_4x4x4 = T_2x2x2 ⊗ T_2x2x2 (approximately, with the right structure)
    
    # More precisely, if we have:
    # T_2x2x2[i,j,k] = sum_c U2[i,c]*V2[j,c]*W2[k,c] for the 2x2x2 multiplication tensor
    # Then T_4x4x4 can be built using the 7 Strassen products at two levels.
    
    # Let me just directly construct it.
    
    # U4: shape (16, 49)
    # V4: shape (16, 49)
    # W4: shape (16, 49)
    
    # Index in U4: a = row*4 + col (for 4x4 A matrix)
    # Index in V4: b = row*4 + col (for 4x4 B matrix)
    # Index in W4: c = row*4 + col (for 4x4 C matrix)
    
    # For each block-level product r (0..6) and 2x2-level product s (0..6):
    # col = r * 7 + s
    
    # Define the 2x2 Strassen factors
    # U2: shape (4, 7), V2: shape (4, 7), W2: shape (4, 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 (s=0): U: X11=+1, X21=-1; V: Y21=-1, Y22=+1; W: Z11=+1, Z21=+1
    U2[3, 0] = 1   # X11 (index 3 = row1,col1)
    U2[2, 0] = -1  # X21 (index 2 = row1,col0)
    V2[2, 0] = -1  # Y21
    V2[3, 0] = 1   # Y22
    W2[3, 0] = 1   # Z11
    W2[2, 0] = 1   # Z21
    
    # M2 (s=1): U: X11=+1, X12=+1; V: Y22=+1; W: Z12=+1
    U2[3, 1] = 1
    U2[1, 1] = 1   # X12 (index 1 = row0,col1)
    V2[3, 1] = 1
    W2[1, 1] = 1   # Z12 (index 1 = row0,col1)
    
    # M3 (s=2): U: X11=+1; V: Y11=+1, Y22=-1; W: Z12=+1, Z22=+1
    U2[3, 2] = 1
    V2[0, 2] = 1   # Y11
    V2[3, 2] = -1
    W2[1, 2] = 1   # Z12
    W2[3, 2] = 1   # Z22
    
    # M4 (s=3): U: X22=+1; V: Y11=+1, Y22=+1; W: Z11=+1, Z22=+1
    U2[3, 3] = 1
    V2[0, 3] = 1
    V2[3, 3] = 1
    W2[3, 3] = 1
    W2[3, 3] += 1  # Wait, Z22 is index 3
    
    # Let me redo this more carefully.
    # 2x2 indexing: row*2 + col
    # X00=0, X01=1, X10=2, X11=3
    # Y00=0, Y01=1, Y10=2, Y11=3
    # Z00=0, Z01=1, Z10=2, Z11=3
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # s=0: M1 = (X11 - X21)(Y21 - Y22), contributes to Z11 and Z21
    U2[3, 0] = 1   # X11
    U2[2, 0] = -1  # X21
    V2[2, 0] = -1  # Y21
    V2[3, 0] = 1   # Y22
    W2[3, 0] = 1   # Z11
    W2[2, 0] = 1   # Z21
    
    # s=1: M2 = (X11 + X12)Y22, contributes to Z12
    U2[3, 1] = 1   # X11
    U2[1, 1] = 1   # X12
    V2[3, 1] = 1   # Y22
    W2[1, 1] = 1   # Z12
    
    # s=2: M3 = X11(Y11 - Y22), contributes to Z12 and Z22
    U2[3, 2] = 1   # X11
    V2[0, 2] = 1   # Y11
    V2[3, 2] = -1  # Y22
    W2[1, 2] = 1   # Z12
    W2[3, 2] = 1   # Z22
    
    # s=3: M4 = X22(Y11 + Y22), contributes to Z11 and Z22
    U2[3, 3] = 1   # X22
    V2[0, 3] = 1   # Y11
    V2[3, 3] = 1   # Y22
    W2[3, 3] = 1   # Z11
    W2[3, 3] += 1  # Z22 -> this is wrong, Z11 and Z22 are different indices
    
    # I'm making errors. Let me restart the 2x2x2 Strassen definition.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen 2x2x2:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M3
    # C21 = M1 + M6
    # C22 = M3 + M4 - M2 + M5
    
    # M1 = (A11 - A21)(B21 - B22)
    # M2 = (A11 + A12)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B11 + B22)
    # M5 = (A12 - A22)(B11 + B21)
    # M6 = (A11 + A21)(B22 - B21)
    # M7 = (A21 - A11)(B11 + B12)
    
    # Using 0-based indexing: A00=0, A01=1, A10=2, A11=3
    # B00=0, B01=1, B10=2, B11=3
    # C00=0, C01=1, C10=2, C11=3
    
    # s=0: M1
    U2[3, 0] = 1    # A11
    U2[2, 0] = -1   # A21
    V2[2, 0] = -1   # B21
    V2[3, 0] = 1    # B22
    W2[0, 0] = 1    # C11
    W2[2, 0] = 1    # C21
    
    # s=1: M2
    U2[3, 1] = 1    # A11
    U2[1, 1] = 1    # A12
    V2[3, 1] = 1    # B22
    W2[1, 1] = 1    # C12
    
    # s=2: M3
    U2[3, 2] = 1    # A11
    V2[0, 2] = 1    # B11
    V2[3, 2] = -1   # B22
    W2[1, 2] = 1    # C12
    W2[3, 2] = 1    # C22
    
    # s=3: M4
    U2[3, 3] = 1    # A22
    V2[0, 3] = 1    # B11
    V2[3, 3] = 1    # B22
    W2[0, 3] = 1    # C11
    W2[3, 3] = 1    # C22
    
    # s=4: M5
    U2[1, 4] = 1    # A12
    U2[3, 4] = -1   # A22
    V2[0, 4] = 1    # B11
    V2[2, 4] = 1    # B21
    W2[0, 4] = -1   # C11
    W2[3, 4] = 1    # C22
    
    # s=5: M6
    U2[3, 5] = 1    # A11
    U2[2, 5] = 1    # A21
    V2[3, 5] = 1    # B22
    V2[2, 5] = -1   # B21
    W2[2, 5] = 1    # C21
    
    # s=6: M7
    U2[2, 6] = 1    # A21
    U2[3, 6] = -1   # A11
    V2[0, 6] = 1    # B11
    V2[1, 6] = 1    # B12
    W2[0, 6] = 1    # C11
    
    # Now build the 4x4x4 rank-49 decomposition
    # The 4x4 matrix A has blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    # For the block-level Strassen, we need to map 4x4 indices to 2x2 block indices.
    # 4x4 index (r, c) -> block (r//2, c//2) and within-block (r%2, c%2)
    
    # The block-level "X" matrix is 2x2, where X[i,j] is a 2x2 block.
    # When we do U2[x_idx, s] * X[x_idx], we're scaling a 2x2 block.
    # So for the 4x4 U matrix:
    # U4[a, col] where a = r*4 + c
    # col = r_block * 7 + s
    # U4[a, col] = U2[x_block_idx, s] if (r//2, c//2) maps to x_block_idx
    
    # Actually, let me think of it as:
    # The block-level product r computes: Mi = (sum_x U2_block[x,r] * A_block[x]) * (sum_y V2_block[y,r] * B_block[y])
    # where A_block[x] is a 2x2 matrix.
    # Then each Mi is a 2x2x2 multiplication, which has its own rank-7 decomposition.
    
    # So for block product r and 2x2 product s:
    # col = r * 7 + s
    # The contribution to A entries:
    # A_block[x] has entries at 4x4 positions:
    # x=0 (A11): rows 0-1, cols 0-1 -> 4x4 indices 0,1,4,5
    # x=1 (A12): rows 0-1, cols 2-3 -> 4x4 indices 2,3,6,7
    # x=2 (A21): rows 2-3, cols 0-1 -> 4x4 indices 8,9,12,13
    # x=3 (A22): rows 2-3, cols 2-3 -> 4x4 indices 10,11,14,15
    
    # For a given block x and 2x2 index u (0..3), the 4x4 index is:
    # block_row = x // 2, block_col = x % 2
    # u_row = u // 2, u_col = u % 2
    # 4x4_row = block_row * 2 + u_row
    # 4x4_col = block_col * 2 + u_col
    # 4x4_idx = 4x4_row * 4 + 4x4_col
    
    # Similarly for B and C.
    
    # Block-level Strassen factors (same as U2, V2, W2 but for blocks)
    # Actually, the block-level factors are the same as the 2x2 Strassen factors.
    # U2_block = U2, V2_block = V2, W2_block = W2
    
    # Now build U4, V4, W4:
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r in range(7):  # block-level product
        for s in range(7):  # 2x2-level product
            col = r * 7 + s
            
            # For each non-zero entry in U2[:, s]:
            for x in range(4):  # block index
                if U2[x, s] != 0:
                    # This block x is scaled by U2[x, s]
                    # For each 2x2 entry within block x:
                    for u in range(4):  # 2x2 index within block
                        # 4x4 index for A
                        block_row = x // 2
                        block_col = x % 2
                        u_row = u // 2
                        u_col = u % 2
                        a_row = block_row * 2 + u_row
                        a_col = block_col * 2 + u_col
                        a_idx = a_row * 4 + a_col
                        U4[a_idx, col] += U2[x, s] * U2[u, s]  # Wait, this isn't right either
                        
    # I think I'm overcomplicating this. Let me use a different approach.
    # The 4x4x4 multiplication tensor T[i,j,k] where i,j,k are 4x4 matrix indices.
    # T[i,j,k] = 1 if the corresponding A[i], B[j], C[k] are "compatible" in the sense that
    # A's row-block and B's col-block match, and C's row/col blocks are determined.
    
    # Actually, the standard approach: 
    # C = A * B
    # C[r1, c1] = sum_r A[r1, r] * B[r, c1]
    
    # For 4x4x4, we can write:
    # C = sum_{r=0}^{48} U[:,r] * V[:,r] * W[:,r] (outer products)
    # where the product means: C[i,k] += U[i,r] * V[k,r] * W[r,r]... no, that's not right.
    
    # The tensor decomposition is:
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where T[i,j,k] is the matrix multiplication tensor.
    # T[i,j,k] = 1 if i, j, k form a valid "path" in the multiplication, 0 otherwise.
    # Specifically, if we write i = (i_row, i_col) for A, j = (j_row, j_col) for B, k = (k_row, k_col) for C:
    # T[i,j,k] = 1 if i_col == j_row and i_row == k_row and j_col == k_col
    
    # For the Kronecker product structure:
    # T_4x4x4 = T_2x2x2 ⊗ T_2x2x2 (in a specific sense)
    
    # Let me just verify that the rank-49 decomposition works by constructing it properly.
    
    # I'll use the fact that 4x4x4 multiplication can be done with 49 multiplications
    # using Strassen's algorithm recursively.
    
    # Let me try a different approach: directly construct the rank-49 factors.
    
    # The key insight: if we have a rank-r decomposition of 2x2x2,
    # we can get a rank-r^2 decomposition of 4x4x4 by taking the Kronecker product.
    
    # T_4x4x4[i,j,k] where i,j,k are in {0..15}
    # i = i_hi * 4 + i_lo, j = j_hi * 4 + j_lo, k = k_hi * 4 + k_lo
    # T_4x4x4[i,j,k] = T_2x2x2[i_hi, j_hi, k_hi] * T_2x2x2[i_lo, j_lo, k_lo]
    # where T_2x2x2 is the 2x2x2 multiplication tensor.
    
    # Wait, that's not quite right. Let me think about the block structure.
    
    # A[4x4] = [[A11, A12], [A21, A22]] where each is 2x2
    # B[4x4] = [[B11, B12], [B21, B22]]
    # C[4x4] = [[C11, C12], [C21, C22]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each of these is a 2x2x2 multiplication tensor.
    
    # So T_4x4x4 can be written as a sum of 4 2x2x2 tensors, but they share the same A and B blocks.
    # This is where Strassen helps: instead of 4 separate 2x2x2 multiplications (rank 7 each = 28 total),
    # we use Strassen to combine them into 7 "combined" 2x2x2 multiplications (rank 7 each = 49 total).
    
    # Hmm, 7 * 7 = 49, but we only need 4 2x2x2 multiplications. So 4*7 = 28 should suffice?
    # But that's not how Strassen works. Strassen reduces the number of multiplications from 4 to 7 for 2x2x2.
    # For 4x4x4, we have 4 block multiplications, each of which is 2x2x2.
    # Using Strassen on the block level: we need 7 "combined" products, each of which is a 2x2x2 multiplication.
    # So total rank = 7 * 7 = 49.
    
    # OK so my original understanding was correct. Let me just implement it carefully.
    
    # I'll define the block-level Strassen factors and the 2x2-level Strassen factors,
    # and combine them.
    
    # Block-level: 2x2 blocks, using Strassen (rank 7)
    # 2x2-level: 2x2 matrices within blocks, using Strassen (rank 7)
    
    # For block product r (0..6) and 2x2 product s (0..6):
    # col = r * 7 + s
    
    # The A factor: U4[a, col]
    # a is a 4x4 index (0..15)
    # a = block_a * 4 + u_a where block_a is 0..3 (which 2x2 block) and u_a is 0..3 (position within block)
    # block_a = (block_row_a, block_col_a)
    # u_a = (u_row_a, u_col_a)
    # 4x4 index = (block_row_a * 2 + u_row_a) * 4 + (block_col_a * 2 + u_col_a)
    
    # U4[a, col] = U_block[block_a, r] * U_2x2[u_a, s]
    # where U_block is the block-level Strassen U factor (4x7)
    # and U_2x2 is the 2x2-level Strassen U factor (4x7)
    
    # Similarly for V and W.
    
    # The block-level Strassen factors are the same as U2, V2, W2 defined above.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            for block_a in range(4):
                for u_a in range(4):
                    if U2[block_a, r] != 0 and U2[u_a, s] != 0:
                        # Compute 4x4 index
                        block_row_a = block_a // 2
                        block_col_a = block_a % 2
                        u_row_a = u_a // 2
                        u_col_a = u_a % 2
                        a_row = block_row_a * 2 + u_row_a
                        a_col = block_col_a * 2 + u_col_a
                        a_idx = a_row * 4 + a_col
                        U4[a_idx, col] = U2[block_a, r] * U2[u_a, s]
            
            for block_b in range(4):
                for u_b in range(4):
                    if V2[block_b, r] != 0 and V2[u_b, s] != 0:
                        block_row_b = block_b // 2
                        block_col_b = block_b % 2
                        u_row_b = u_b // 2
                        u_col_b = u_b % 2
                        b_row = block_row_b * 2 + u_row_b
                        b_col = block_col_b * 2 + u_col_b
                        b_idx = b_row * 4 + b_col
                        V4[b_idx, col] = V2[block_b, r] * V2[u_b, s]
            
            for block_c in range(4):
                for u_c in range(4):
                    if W2[block_c, r] != 0 and W2[u_c, s] != 0:
                        block_row_c = block_c // 2
                        block_col_c = block_c % 2
                        u_row_c = u_c // 2
                        u_col_c = u_c % 2
                        c_row = block_row_c * 2 + u_row_c
                        c_col = block_col_c * 2 + u_col_c
                        c_idx = c_row * 4 + c_col
                        W4[c_idx, col] = W2[block_c, r] * W2[u_c, s]
    
    # Now extend to 4x4x5
    # C[4x5] = A[4x4] * B[4x5]
    # Split B into B1[4x4] and B2[4x1]
    # C = A*B1 + A*B2
    # A*B1: use rank-49 Strassen (above)
    # A*B2: use schoolbook rank 16
    
    # Total rank: 49 + 16 = 65
    
    # For the 4x4x1 part:
    # C[:, 4] = A * B[:, 4]
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    # This is a 4x4x1 multiplication, which has rank 16 (schoolbook).
    
    # U_441: shape (16, 16)
    # V_441: shape (4, 16)  -- wait, B is 4x5, so B[:, 4] is 4x1
    # Actually, V should be shape (m*p, rank) = (4*5, 65)
    # For the 4x4x1 part, V entries correspond to B[j, 4] for j=0..3
    
    # Let me build the full 65-rank decomposition:
    # First 49 columns: Strassen for A*B1 where B1 = B[:, 0:4]
    # Last 16 columns: schoolbook for A*B2 where B2 = B[:, 4:5]
    
    U65 = np.zeros((16, 65))
    V65 = np.zeros((20, 65))  # m*p = 4*5 = 20
    W65 = np.zeros((20, 65))  # p*n = 5*4 = 20
    
    # Copy the 4x4x4 Strassen part
    U65[:, :49] = U4
    # V4 is 16x49, corresponds to B[:, 0:4]
    # B index: b = row*5 + col for 4x5 matrix
    # For B[:, 0:4]: b = row*5 + col where col=0..3
    # V4[b_idx, col] where b_idx = row*4 + col (for 4x4)
    # Need to map to 4x5 indexing: b_45 = row*5 + col
    for b_idx in range(16):
        b_row = b_idx // 4
        b_col = b_idx % 4
        b_45 = b_row * 5 + b_col
        V65[b_45, :49] = V4[b_idx, :49]
    
    # W4 is 16x49, corresponds to C[:, 0:4]
    # C index: c = row*4 + col for 4x4
    # For C[:, 0:4]: c = row*4 + col where col=0..3
    # W65 index: c_54 = col*4 + row for 5x4 (p*n = 5*4)
    # Wait, the convention is W[k*n+i, r] where k is the output column, i is the output row
    # So W index = k*4 + i where k=0..4 (output col), i=0..3 (output row)
    # For C[:, 0:4]: k=0..3, i=0..3
    # C 4x4 index: c = i*4 + k (row i, col k)
    # W65 index: k*4 + i
    for c_idx in range(16):
        c_row = c_idx // 4  # i
        c_col = c_idx % 4   # k
        w_idx = c_col * 4 + c_row
        W65[w_idx, :49] = W4[c_idx, :49]
    
    # Now add the 4x4x1 schoolbook part for columns 49..64
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    # For each (i, j): one column
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U65[i * 4 + j, col] = 1  # A[i, j]
            V65[j * 5 + 4, col] = 1  # B[j, 4]
            W65[4 * 4 + i, col] = 1  # C[i, 4] -> k=4, i=i -> index = 4*4 + i = 16 + i
    
    # Verify the 65-rank decomposition
    if verify(U65, V65, W65):
        best_U, best_V, best_W, best_rank = U65.copy(), V65.copy(), W65.copy(), 65
    else:
        # Debug: check which entries fail
        pass
    
    # If we have time, try to find a lower-rank decomposition
    if time.time() < deadline - 30:
        # Try rank 64, 63, etc. with random search
        for test_rank in range(64, 40, -1):
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                # Initialize with the 65-rank decomposition, zero out one column
                U_try = U65.copy()
                V_try = V65.copy()
                W_try = W65.copy()
                
                # Remove the last column and try to repair
                U_try[:, -1] = 0
                V_try[:, -1] = 0
                W_try[:, -1] = 0
                
                # Try random perturbation of the last column
                rng = np.random.RandomState(attempt)
                U_try[:, -1] = rng.randn(16) * 0.1
                V_try[:, -1] = rng.randn(20) * 0.1
                W_try[:, -1] = rng.randn(20) * 0.1
                
                if verify(U_try, V_try, W_try):
                    best_U, best_V, best_W, best_rank = U_try.copy(), V_try.copy(), W_try.copy(), test_rank
                    found = True
                    break
            
            if found:
                break
    
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    return result
# EVOLVE-BLOCK-END
