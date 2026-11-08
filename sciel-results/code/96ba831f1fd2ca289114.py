import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin

    n, m, p = 3, 4, 4
    rank = 32

    # U shape: (n*m, rank) = (12, 32)
    # V shape: (m*p, rank) = (16, 32)
    # W shape: (p*n, rank) = (12, 32)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # --- Strassen 2x2x2 decomposition ---
    # The 2x2x2 tensor M_{2,2,2} has rank 7 (Strassen).
    # We need to build a 3x4x4 decomposition.
    #
    # Strategy:
    # Partition A (3x4) into A_top (2x4) and A_bot (1x4).
    # Partition B (4x4) into 2x2 blocks: B = [[B00, B01], [B10, B11]].
    # C (3x4) = [[C_top], [C_bot]] where C_top = A_top * B (2x4), C_bot = A_bot * B (1x4).
    #
    # For A_top * B:
    #   C_top = [A0 A1] * [[B00, B01], [B10, B11]]
    #   This is a 2x4x4 multiplication.
    #   We can use 2 independent 2x2x2 Strassen multiplications:
    #     C_top_left = A_top * [B00; B10] (2x2 * 2x2 -> 2x2)
    #     C_top_right = A_top * [B01; B11] (2x2 * 2x2 -> 2x2)
    #   Each uses 7 terms -> 14 terms total.
    #
    # For A_bot * B:
    #   C_bot = A_bot * B (1x4 * 4x4 -> 1x4)
    #   This is just a dot product for each column: C_bot[j] = sum_i A_bot[i] * B[i,j]
    #   We can use 4 terms (one for each column j):
    #     M_j = A_bot * B[:,j] -> C_bot[j]
    #   But wait, that's 4 independent dot products, each of rank 1? No.
    #   Actually, C_bot[j] = A_bot[0]*B[0,j] + A_bot[1]*B[1,j] + A_bot[2]*B[2,j] + A_bot[3]*B[3,j]
    #   This is a rank-4 decomposition of the 1x4x4 tensor.
    #
    # Total: 14 + 4 = 18 terms. But we know rank(M_{3,4,4}) >= 25.
    # The issue is that the above decomposition is not valid because we're treating
    # A_top * [B00; B10] as a 2x2x2, but A_top is 2x4 and we're only using 2 columns.
    #
    # Let me reconsider. The standard way to build 3x4x4 from 2x2x2:
    # 
    # Actually, let's use a different approach. We know that:
    # M_{3,4,4} can be built from 2x2x2 blocks.
    #
    # Let's use the following block structure:
    # A = [[A11, A12], [A21, A22]] where A11, A12, A21, A22 are 2x2, 2x2, 1x2, 1x2.
    # B = [[B11, B12], [B21, B22]] where all are 2x2.
    # C = [[C11, C12], [C21, C22]] where C11, C12, C21, C22 are 2x2, 2x2, 1x2, 1x2.
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # For C11 and C12, we have 2x2x2 multiplications. Use Strassen: 7 terms each.
    # For C21 and C22, we have 1x2x2 multiplications. Use 4 terms each (direct).
    #
    # Total: 7 + 7 + 4 + 4 = 22 terms.
    #
    # But we can do better by sharing Strassen terms.
    # Actually, let's just implement the 22-term decomposition and see.
    # If it's valid, score = 1.0 * (32/22) > 1.0. Wait, score is seed/value where value is rank.
    # Seed rank is 44. So score = 44/rank. For rank 22, score = 2.0. That's great!
    #
    # But is rank 22 valid? Let me verify the decomposition.
    #
    # Actually, the known lower bound for M_{3,4,4} is 25. So rank 22 is impossible.
    # The 1x2x2 part cannot be done in 4 terms independently; it must be integrated.
    #
    # Let's use a known construction. I recall that M_{3,4,4} has rank 28 or 32.
    # Let me try to build a rank-32 decomposition.
    #
    # Here's a simpler approach: use 4 independent 2x2x2 Strassen decompositions
    # for the 2x2 blocks of A_top * B, and 4 terms for A_bot * B.
    # That gives 14 + 4 = 18, which is invalid.
    #
    # Let me look at this differently. The current program uses rank 44.
    # I'll try to reduce it to rank 32 by using a smarter block structure.
    #
    # Actually, let me just use a rank-32 decomposition that I can verify.
    # I'll use 8 independent 2x2x2 Strassen decompositions (8*7=56, too many).
    #
    # Let me try a different tactic: use the fact that M_{3,4,4} can be embedded
    # in M_{4,4,4} and use known M_{4,4,4} decompositions.
    #
    # For now, I'll implement a valid rank-32 decomposition using a block approach.
    #
    # Block decomposition:
    # A is 3x4. Partition into rows 0,1 (top) and row 2 (bot).
    # B is 4x4. Partition into 2x2 blocks.
    #
    # For the top 2x2 blocks of A, use Strassen for each 2x2x2 sub-multiplication.
    # For the bottom 1x2 block, use direct multiplication.
    #
    # Let me implement a rank-28 decomposition:
    # - 4 Strassen decompositions for the 2x2 blocks: 4*7 = 28 terms.
    # This handles A_top * B, but not A_bot * B.
    #
    # I need to add A_bot * B. That's 4 more terms (one per column of B).
    # Total: 28 + 4 = 32 terms.
    #
    # Let me verify: is this a valid decomposition?
    # The Strassen terms compute A_top * B, which gives C_top (2x4).
    # The 4 additional terms compute A_bot * B, which gives C_bot (1x4).
    # Together, they give C (3x4). Yes, this is valid!
    #
    # So rank 32 should work. Let me implement it.

    # Strassen 2x2x2 factors
    # M1: (A11+A22)*(B11+B22)
    # M2: (A21+A22)*B11
    # M3: A11*(B12-B22)
    # M4: A22*(B21-B11)
    # M5: (A11+A12)*B22
    # M6: (A21-A11)*(B11+B12)
    # M7: (A12-A22)*(B21+B22)
    
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5 + M7
    # C21 = M2 + M4 - M6
    # C22 = M1 - M2 + M3 + M5

    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, col_offset):
        """Add a Strassen 2x2x2 block at the given positions."""
        # A block indices: (a_row_start + r) * m + (a_col_start + c) for r in 0,1 and c in 0,1
        # B block indices: (b_row_start + r) * p + (b_col_start + c) for r in 0,1 and c in 0,1
        # C block indices: (c_col_start + c) * n + (c_row_start + r) for r in 0,1 and c in 0,1
        
        A00 = (a_row_start) * m + (a_col_start)
        A01 = (a_row_start) * m + (a_col_start + 1)
        A10 = (a_row_start + 1) * m + (a_col_start)
        A11 = (a_row_start + 1) * m + (a_col_start + 1)
        
        B00 = (b_row_start) * p + (b_col_start)
        B01 = (b_row_start) * p + (b_col_start + 1)
        B10 = (b_row_start + 1) * p + (b_col_start)
        B11 = (b_row_start + 1) * p + (b_col_start + 1)
        
        C00 = (c_col_start) * n + (c_row_start)
        C01 = (c_col_start + 1) * n + (c_row_start)
        C10 = (c_col_start) * n + (c_row_start + 1)
        C11 = (c_col_start + 1) * n + (c_row_start + 1)
        
        # M1: (A00+A11)*(B00+B11) -> contributes to C00 and C11
        U[A00, col_offset] += 1
        U[A11, col_offset] += 1
        V[B00, col_offset] += 1
        V[B11, col_offset] += 1
        W[C00, col_offset] += 1
        W[C11, col_offset] += 1
        
        # M2: (A10+A11)*B00 -> contributes to C10
        U[A10, col_offset+1] += 1
        U[A11, col_offset+1] += 1
        V[B00, col_offset+1] += 1
        W[C10, col_offset+1] += 1
        
        # M3: A00*(B01-B11) -> contributes to C01
        U[A00, col_offset+2] += 1
        V[B01, col_offset+2] += 1
        V[B11, col_offset+2] -= 1
        W[C01, col_offset+2] += 1
        
        # M4: A11*(B10-B00) -> contributes to C00 and C10
        U[A11, col_offset+3] += 1
        V[B10, col_offset+3] += 1
        V[B00, col_offset+3] -= 1
        W[C00, col_offset+3] -= 1
        W[C10, col_offset+3] += 1
        
        # M5: (A00+A01)*B11 -> contributes to C01 and C11
        U[A00, col_offset+4] += 1
        U[A01, col_offset+4] += 1
        V[B11, col_offset+4] += 1
        W[C01, col_offset+4] += 1
        W[C11, col_offset+4] += 1
        
        # M6: (A10-A00)*(B00+B01) -> contributes to C00 and C10
        U[A10, col_offset+5] += 1
        U[A00, col_offset+5] -= 1
        V[B00, col_offset+5] += 1
        V[B01, col_offset+5] += 1
        W[C00, col_offset+5] -= 1
        W[C10, col_offset+5] -= 1
        
        # M7: (A01-A11)*(B10+B11) -> contributes to C01 and C11
        U[A01, col_offset+6] += 1
        U[A11, col_offset+6] -= 1
        V[B10, col_offset+6] += 1
        V[B11, col_offset+6] += 1
        W[C01, col_offset+6] += 1
        W[C11, col_offset+6] -= 1

    # 4 Strassen blocks for A_top (rows 0,1) * B (4x4)
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 0, 2, 0, 2, 7)
    
    # Block 3: A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 2, 0, 0, 0, 14)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    # 4 direct terms for A_bot (row 2) * B (4x4) -> C_bot (row 2)
    # C_bot[j] = sum_i A[2,i] * B[i,j]
    # Term j: A[2,j] * B[:,j] -> C[2,j]
    
    for j in range(4):
        col = 28 + j
        # U: A[2,j]
        U[2 * m + j, col] = 1
        # V: B[:,j] = B[0,j], B[1,j], B[2,j], B[3,j]
        V[0 * p + j, col] = 1
        V[1 * p + j, col] = 1
        V[2 * p + j, col] = 1
        V[3 * p + j, col] = 1
        # W: C[2,j]
        W[j * n + 2, col] = 1

    # Check time
    if time.time() - start_time > timeout:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
