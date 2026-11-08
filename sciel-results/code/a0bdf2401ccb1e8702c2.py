"""Matrix multiplication 4x4x4 using Strassen-style block decomposition."""
import time
import random

DIMENSIONS = [4, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _construct_strassen_4x4():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using two levels of Strassen's algorithm.
    
    Level 1: Treat 4x4 as 2x2 blocks of 2x2 matrices.
    Level 2: Apply Strassen to each 2x2 block multiplication (rank 7).
    
    The 4x4 matrix M is partitioned as:
        M = [[A, B],
             [C, D]]
    where A, B, C, D are 2x2 blocks.
    
    Result C = M1 * M2 where M1, M2 are also 4x4 with 2x2 blocks.
    The 2x2 block result is computed via Strassen (7 multiplications).
    
    Each Strassen multiplication of 2x2 matrices requires rank 7.
    For the 4x4 case, we have 7 "virtual" 2x2 multiplications, each of rank 7.
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen's 7 intermediate products for 2x2 block multiplication:
    # Given blocks [[A,B],[C,D]] * [[E,F],[G,H]]:
    # M1 = (A+B)*(E+H)
    # M2 = (C+D)*E
    # M3 = A*(F-H)
    # M4 = D*(G-C)
    # M5 = (A+D)*(E+H)
    # M6 = (B-C)*(G+H)
    # M7 = (A-C)*(E+F)
    # Result blocks:
    # P11 = M1 + M4 - M5 + M7
    # P12 = M3 + M5
    # P21 = M2 + M5
    # P22 = M1 - M2 + M3 + M6
    
    # For 4x4 matrices, we partition into 2x2 blocks of size 2x2.
    # Let the input matrices be X (4x4) and Y (4x4), result Z = X*Y (4x4).
    # X blocks: X11 (rows 0-1, cols 0-1), X12 (rows 0-1, cols 2-3)
    #            X21 (rows 2-3, cols 0-1), X22 (rows 2-3, cols 2-3)
    # Y blocks: Y11 (rows 0-1, cols 0-1), Y12 (rows 0-1, cols 2-3)
    #            Y21 (rows 2-3, cols 0-1), Y22 (rows 2-3, cols 2-3)
    # Z blocks: Z11, Z12, Z21, Z22 similarly.
    
    # Strassen intermediates for the 2x2 block-level multiplication:
    # S1 = X11 + X12  (top row blocks sum)
    # S2 = X21 + X22  (bottom row blocks sum)
    # S3 = Y11 + Y22  (diagonal blocks sum)
    # S4 = Y21 - Y11  (lower-left minus upper-left)
    # S5 = Y12 - Y22  (upper-right minus lower-right)
    # S6 = X11 + X22  (diagonal blocks sum)
    # S7 = X21 - X11  (lower-left minus upper-left)
    # S8 = X12 - X22  (upper-right minus lower-right)
    
    # M1 = S1 * S3 = (X11+X12)*(Y11+Y22)
    # M2 = S2 * Y11 = (X21+X22)*Y11
    # M3 = X11 * S5 = X11*(Y12-Y22)
    # M4 = X22 * S4 = X22*(Y21-Y11)
    # M5 = S6 * S3 = (X11+X22)*(Y11+Y22)
    # M6 = S8 * (Y21+Y22) = (X12-X22)*(Y21+Y22)
    # M7 = S7 * (Y11+Y12) = (X21-X11)*(Y11+Y12)
    
    # Each Mi is a 2x2 matrix product, so each can be done with Strassen rank 7.
    
    # Now we need to express the full 4x4 multiplication as a tensor decomposition.
    # The tensor T[i,j,k] = delta_{i+k, j} for matrix multiplication (i=row of X, j=col of Y, k=shared).
    # Actually T_{i,j,k} such that Z[i,j] = sum_k X[i,k] * Y[k,j]
    
    # For the rank-49 decomposition, each rank-1 term corresponds to one of the 49
    # scalar multiplications in the two-level Strassen.
    
    # Let's build U, V, W matrices of shape (16, 49), (16, 49), (16, 49).
    # U[i, r] corresponds to coefficient of X[i, k_r] 
    # V[k, r] corresponds to coefficient of Y[k, j_r]
    # W[j, r] corresponds to contribution to Z[i_r, j]
    
    # Actually the convention is:
    # U has shape (n*m, rank) = (16, 49): U[i*m + j, r] is the coefficient for X[i,j] in rank-r term
    # V has shape (m*p, rank) = (16, 49): V[j*p + k, r] is the coefficient for Y[j,k] in rank-r term
    # W has shape (p*n, rank) = (16, 49): W[k*n + i, r] is the coefficient for Z[i,k] in rank-r term
    # Wait, let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n+i, r] gives coefficient for Z[i, k] in rank-r term.
    
    # The decomposition says: Z[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * X[i,j]) * (sum_l V[j*p+l, r] * Y[j,l])
    # Wait, that's not quite right. Let me think again.
    
    # Standard tensor decomposition for matmul:
    # T_{i,j,k} = sum_r U_{i,r} * V_{j,r} * W_{k,r}
    # where T is the matmul tensor with T_{i,j,k} = delta_{i+k, j} (for Z[i,k] = sum_j X[i,j] Y[j,k])
    # 
    # But here the interface uses:
    # U: shape (n*m, rank), indexed as U[i*m+j, r]
    # V: shape (m*p, rank), indexed as V[j*p+k, r]  
    # W: shape (p*n, rank), indexed as W[k*n+i, r]
    #
    # This suggests a different grouping. Let me interpret:
    # The matmul is: Z[i, k] = sum_j X[i, j] * Y[j, k]
    # 
    # A rank-1 term contributes: W[k*n+i, r] * (linear combo of X entries) * (linear combo of Y entries)
    # 
    # So U[i*m+j, r] is the coefficient of X[i,j] in the r-th rank-1 term's X-part.
    # V[j*p+k, r] is the coefficient of Y[j,k] in the r-th rank-1 term's Y-part.
    # W[k*n+i, r] is the coefficient for Z[i,k] in the r-th rank-1 term.
    
    # Let me build this directly using the two-level Strassen structure.
    
    # For a 2x2 Strassen, the rank-7 decomposition of 2x2 matmul:
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M5
    # Z22 = M1 - M2 + M3 + M6
    #
    # where M1 = (A+B)(E+H), M2 = (C+D)E, M3 = A(F-H), M4 = D(G-C),
    # M5 = (A+D)(E+H), M6 = (B-C)(G+H), M7 = (A-C)(E+F)
    
    # For the 4x4 case with 2x2 blocks, we have 7 block-level multiplications,
    # each of which is a 2x2 matmul that itself uses rank 7.
    
    # Let me define the block-level Strassen intermediates and their 2x2 Strassen decompositions.
    
    # Block indices: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # X blocks: X0 (rows 0-1, cols 0-1), X1 (rows 0-1, cols 2-3), X2 (rows 2-3, cols 0-1), X3 (rows 2-3, cols 2-3)
    # Y blocks: Y0 (rows 0-1, cols 0-1), Y1 (rows 0-1, cols 2-3), Y2 (rows 2-3, cols 0-1), Y3 (rows 2-3, cols 2-3)
    # Z blocks: Z0, Z1, Z2, Z3 similarly
    
    # Block-level Strassen:
    # M1 = (X0+X1)*(Y0+Y3)
    # M2 = (X2+X3)*Y0
    # M3 = X0*(Y1-Y3)
    # M4 = X3*(Y2-Y0)
    # M5 = (X0+X3)*(Y0+Y3)
    # M6 = (X1-X3)*(Y2+Y3)
    # M7 = (X2-X0)*(Y0+Y1)
    
    # Z0 = M1 + M4 - M5 + M7
    # Z1 = M3 + M5
    # Z2 = M2 + M5
    # Z3 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 block product. Each 2x2 block product uses rank 7.
    # So total rank = 7 * 7 = 49.
    
    # For each block-level intermediate Mm (m=1..7), it's a product of two 2x2 blocks.
    # Let Mm = Am * Bm where Am and Bm are 2x2 blocks (possibly sums/differences of X/Y blocks).
    
    # For each such 2x2 product, we apply Strassen with rank 7.
    
    # Let me define for each m in 1..7:
    #   Am: a 2x2 matrix that is a linear combination of X blocks
    #   Bm: a 2x2 matrix that is a linear combination of Y blocks
    
    # The 2x2 Strassen for Am * Bm:
    # Let Am = [[a00, a01], [a10, a11]], Bm = [[b00, b01], [b10, b11]]
    # S1 = a00 + a01, S2 = a10 + a11, S3 = b00 + b11, S4 = b10 - b00, S5 = b01 - b11
    # S6 = a00 + a11, S7 = a10 - a00, S8 = a01 - a11
    # Mm_1 = S1*S3, Mm_2 = S2*b00, Mm_3 = a00*S5, Mm_4 = a11*S4
    # Mm_5 = S6*S3, Mm_6 = S8*(b10+b11), Mm_7 = S7*(b00+b01)
    # Result:
    # R00 = Mm_1 + Mm_4 - Mm_5 + Mm_7
    # R01 = Mm_3 + Mm_5
    # R10 = Mm_2 + Mm_5
    # R11 = Mm_1 - Mm_2 + Mm_3 + Mm_6
    
    # Each Mm_j is a scalar product of two linear combinations of 2x2 entries.
    # Each such scalar product corresponds to one rank-1 term in the overall decomposition.
    
    # Let me now construct U, V, W.
    
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    # Helper: get flat index for X[i,j] -> i*m+j = i*4+j
    # Helper: get flat index for Y[j,k] -> j*p+k = j*4+k
    # Helper: get flat index for Z[i,k] -> k*n+i = k*4+i
    
    # Define block-level intermediates
    # For each m in range(7):
    #   Am is a 2x2 block (linear combo of X blocks)
    #   Bm is a 2x2 block (linear combo of Y blocks)
    
    # Let me define the 2x2 entries of each Am and Bm.
    # X block b has entries at positions:
    #   Block 0: X[0,0], X[0,1], X[1,0], X[1,1] -> flat indices 0, 1, 4, 5
    #   Block 1: X[0,2], X[0,3], X[1,2], X[1,3] -> flat indices 2, 3, 6, 7
    #   Block 2: X[2,0], X[2,1], X[3,0], X[3,1] -> flat indices 8, 9, 12, 13
    #   Block 3: X[2,2], X[2,3], X[3,2], X[3,3] -> flat indices 10, 11, 14, 15
    
    # Y block b has entries at positions:
    #   Block 0: Y[0,0], Y[0,1], Y[1,0], Y[1,1] -> flat indices 0, 1, 4, 5
    #   Block 1: Y[0,2], Y[0,3], Y[1,2], Y[1,3] -> flat indices 2, 3, 6, 7
    #   Block 2: Y[2,0], Y[2,1], Y[3,0], Y[3,1] -> flat indices 8, 9, 12, 13
    #   Block 3: Y[2,2], Y[2,3], Y[3,2], Y[3,3] -> flat indices 10, 11, 14, 15
    
    # Z block b has entries at positions (using W[k*n+i, r] convention, so Z[i,k]):
    #   Block 0: Z[0,0], Z[0,1], Z[1,0], Z[1,1] -> W flat indices: 0*4+0=0, 1*4+0=4, 0*4+1=1, 1*4+1=5
    #     Wait, W[k*n+i, r] where k is column of Z, i is row of Z.
    #     Z[i,k] -> W[k*4+i, r]
    #     Z[0,0] -> W[0, r], Z[0,1] -> W[4, r], Z[1,0] -> W[1, r], Z[1,1] -> W[5, r]
    #   Block 1: Z[0,2], Z[0,3], Z[1,2], Z[1,3] -> W[8, r], W[12, r], W[9, r], W[13, r]
    #   Block 2: Z[2,0], Z[2,1], Z[3,0], Z[3,1] -> W[2, r], W[6, r], W[3, r], W[7, r]
    #   Block 3: Z[2,2], Z[2,3], Z[3,2], Z[3,3] -> W[10, r], W[14, r], W[11, r], W[15, r]
    
    # Let me define the block-level Am and Bm as 2x2 arrays of coefficients.
    # Each entry is a dict mapping (block, local_row, local_col) -> coefficient
    
    # Actually, let me just directly compute the coefficients.
    
    # For each block-level intermediate m (0-indexed: 0..6), define:
    #   Am: 2x2 array where Am[r][c] is a linear combination of X entries
    #   Bm: 2x2 array where Bm[r][c] is a linear combination of Y entries
    
    # Let me represent each 2x2 block entry as a list of (flat_index, coeff) pairs.
    
    def block_entries_X(block):
        """Return the 4 flat indices for X block, in order [0,0],[0,1],[1,0],[1,1]."""
        base_r = block // 2 * 2  # 0 or 2
        base_c = block % 2 * 2   # 0 or 2
        return [
            (base_r + 0) * 4 + (base_c + 0),  # [0,0]
            (base_r + 0) * 4 + (base_c + 1),  # [0,1]
            (base_r + 1) * 4 + (base_c + 0),  # [1,0]
            (base_r + 1) * 4 + (base_c + 1),  # [1,1]
        ]
    
    def block_entries_Y(block):
        """Return the 4 flat indices for Y block, in order [0,0],[0,1],[1,0],[1,1]."""
        base_r = block // 2 * 2  # 0 or 2
        base_c = block % 2 * 2   # 0 or 2
        return [
            (base_r + 0) * 4 + (base_c + 0),  # [0,0]
            (base_r + 0) * 4 + (base_c + 1),  # [0,1]
            (base_r + 1) * 4 + (base_c + 0),  # [1,0]
            (base_r + 1) * 4 + (base_c + 1),  # [1,1]
        ]
    
    def block_entries_Z(block):
        """Return the 4 W flat indices for Z block, in order Z[0,0],Z[0,1],Z[1,0],Z[1,1]."""
        base_r = block // 2 * 2  # 0 or 2
        base_c = block % 2 * 2   # 0 or 2
        # Z[i,k] -> W[k*4+i, r]
        return [
            (base_c + 0) * 4 + (base_r + 0),  # Z[0,0] -> W[0*4+0]
            (base_c + 1) * 4 + (base_r + 0),  # Z[0,1] -> W[1*4+0]
            (base_c + 0) * 4 + (base_r + 1),  # Z[1,0] -> W[0*4+1]
            (base_c + 1) * 4 + (base_r + 1),  # Z[1,1] -> W[1*4+1]
        ]
    
    # Block-level Strassen intermediates (0-indexed m=0..6):
    # m=0: M1 = (X0+X1)*(Y0+Y3)
    # m=1: M2 = (X2+X3)*Y0
    # m=2: M3 = X0*(Y1-Y3)
    # m=3: M4 = X3*(Y2-Y0)
    # m=4: M5 = (X0+X3)*(Y0+Y3)
    # m=5: M6 = (X1-X3)*(Y2+Y3)
    # m=6: M7 = (X2-X0)*(Y0+Y1)
    
    # For each m, define Am as 2x2 where Am[r][c] = sum of coeff * X_block_entry
    # and Bm as 2x2 where Bm[r][c] = sum of coeff * Y_block_entry
    
    # Let me define Am and Bm as 2x2 arrays of lists of (block, local_pos, coeff).
    # local_pos: 0=[0,0], 1=[0,1], 2=[1,0], 3=[1,1]
    
    # m=0: Am = X0+X1, Bm = Y0+Y3
    # Am[r][c] = X0[r][c] + X1[r][c]
    # Bm[r][c] = Y0[r][c] + Y3[r][c]
    
    # I'll define a helper to add block contributions.
    
    def make_block_2x2(terms):
        """
        terms: list of (block, coeff)
        Returns 2x2 list where each entry is a list of (flat_index, coeff)
        """
        result = [[[] for _ in range(2)] for _ in range(2)]
        for block, coeff in terms:
            entries = block_entries_X(block) if coeff > 0 else block_entries_X(block)
            # Actually I need to know if it's X or Y. Let me pass a flag.
            pass
        return result
    
    # This is getting complex. Let me just directly build U, V, W by iterating
    # through the 49 rank-1 terms.
    
    # For each block-level m (0..6) and each Strassen sub-term s (0..6):
    #   rank index r = m * 7 + s
    
    # The X-part of term (m, s) is a linear combination of X entries.
    # The Y-part of term (m, s) is a linear combination of Y entries.
    # The Z-part contributes to specific Z entries with specific coefficients.
    
    # Let me define the Strassen sub-terms for a generic 2x2 product A*B:
    # s=0: S1*S3 where S1=A00+A01, S3=B00+B11
    # s=1: S2*B00 where S2=A10+A11
    # s=2: A00*S5 where S5=B01-B11
    # s=3: A11*S4 where S4=B10-B00
    # s=4: S6*S3 where S6=A00+A11
    # s=5: S8*(B10+B11) where S8=A01-A11
    # s=6: S7*(B00+B01) where S7=A10-A00
    
    # For each sub-term, the X-coefficients (for A entries) and Y-coefficients (for B entries)
    # and the Z-contribution (which result entries it affects and with what coefficient).
    
    # Strassen result:
    # R00 = M1 + M4 - M5 + M7  (s=0 + s=3 - s=4 + s=6)
    # R01 = M3 + M5            (s=2 + s=4)
    # R10 = M2 + M5            (s=1 + s=4)
    # R11 = M1 - M2 + M3 + M6  (s=0 - s=1 + s=2 + s=5)
    
    # So for sub-term s, the Z-contribution to the 2x2 result block is:
    # s=0: R00 += 1, R11 += 1
    # s=1: R10 += 1, R11 -= 1
    # s=2: R01 += 1, R11 += 1
    # s=3: R00 += 1
    # s=4: R00 -= 1, R01 += 1, R10 += 1
    # s=5: R11 += 1
    # s=6: R00 += 1
    
    z_contrib = {
        0: [(0, 1), (3, 1)],      # R00, R11
        1: [(2, 1), (3, -1)],     # R10, R11
        2: [(1, 1), (3, 1)],      # R01, R11
        3: [(0, 1)],              # R00
        4: [(0, -1), (1, 1), (2, 1)],  # R00, R01, R10
        5: [(3, 1)],              # R11
        6: [(0, 1)],              # R00
    }
    
    # Now for each block-level m, I need to define Am and Bm as 2x2 matrices
    # where each entry is a linear combination of X/Y block entries.
    
    # Let me define Am and Bm as 2x2 arrays where each entry is a dict: {X_block: coeff}
    # Then I'll expand to flat indices.
    
    # m=0: Am = X0 + X1, Bm = Y0 + Y3
    # Am[0][0] = X0[0][0] + X1[0][0], Am[0][1] = X0[0][1] + X1[0][1], etc.
    
    # Let me define for each m the X-block contributions to Am[0][0], Am[0][1], Am[1][0], Am[1][1]
    # and similarly for Bm.
    
    # I'll represent as: Am_coeffs[m][r][c] = [(block, coeff), ...]
    
    Am_coeffs = [
        # m=0: (X0+X1)
        [[[(0,1),(1,1)], [(0,1),(1,1)]],
         [[(0,1),(1,1)], [(0,1),(1,1)]]],
        # m=1: (X2+X3)
        [[[(2,1),(3,1)], [(2,1),(3,1)]],
         [[(2,1),(3,1)], [(2,1),(3,1)]]],
        # m=2: X0
        [[[(0,1)], [(0,1)]],
         [[(0,1)], [(0,1)]]],
        # m=3: X3
        [[[(3,1)], [(3,1)]],
         [[(3,1)], [(3,1)]]],
        # m=4: (X0+X3)
        [[[(0,1),(3,1)], [(0,1),(3,1)]],
         [[(0,1),(3,1)], [(0,1),(3,1)]]],
        # m=5: (X1-X3)
        [[[(1,1),(3,-1)], [(1,1),(3,-1)]],
         [[(1,1),(3,-1)], [(1,1),(3,-1)]]],
        # m=6: (X2-X0)
        [[[(2,1),(0,-1)], [(2,1),(0,-1)]],
         [[(2,1),(0,-1)], [(2,1),(0,-1)]]],
    ]
    
    Bm_coeffs = [
        # m=0: (Y0+Y3)
        [[[(0,1),(3,1)], [(0,1),(3,1)]],
         [[(0,1),(3,1)], [(0,1),(3,1)]]],
        # m=1: Y0
        [[[(0,1)], [(0,1)]],
         [[(0,1)], [(0,1)]]],
        # m=2: (Y1-Y3)
        [[[(1,1),(3,-1)], [(1,1),(3,-1)]],
         [[(1,1),(3,-1)], [(1,1),(3,-1)]]],
        # m=3: (Y2-Y0)
        [[[(2,1),(0,-1)], [(2,1),(0,-1)]],
         [[(2,1),(0,-1)], [(2,1),(0,-1)]]],
        # m=4: (Y0+Y3)
        [[[(0,1),(3,1)], [(0,1),(3,1)]],
         [[(0,1),(3,1)], [(0,1),(3,1)]]],
        # m=5: (Y2+Y3)
        [[[(2,1),(3,1)], [(2,1),(3,1)]],
         [[(2,1),(3,1)], [(2,1),(3,1)]]],
        # m=6: (Y0+Y1)
        [[[(0,1),(1,1)], [(0,1),(1,1)]],
         [[(0,1),(1,1)], [(0,1),(1,1)]]],
    ]
    
    # Now for each m and s, build the rank-1 term.
    
    for m in range(7):
        for s in range(7):
            r = m * 7 + s
            
            # Determine X-coefficients for this term.
            # The X-part depends on s:
            # s=0: S1*S3 -> X-part is S1 = Am[0][0]+Am[0][1]
            # s=1: S2*B00 -> X-part is S2 = Am[1][0]+Am[1][1]
            # s=2: A00*S5 -> X-part is Am[0][0]
            # s=3: A11*S4 -> X-part is Am[1][1]
            # s=4: S6*S3 -> X-part is S6 = Am[0][0]+Am[1][1]
            # s=5: S8*(B10+B11) -> X-part is S8 = Am[0][1]-Am[1][1]
            # s=6: S7*(B00+B01) -> X-part is S7 = Am[1][0]-Am[0][0]
            
            if s == 0:
                # S1 = Am[0][0] + Am[0][1]
                x_terms = []
                for block, coeff in Am_coeffs[m][0][0]:
                    x_terms.append((block, coeff))
                for block, coeff in Am_coeffs[m][0][1]:
                    x_terms.append((block, coeff))
            elif s == 1:
                # S2 = Am[1][0] + Am[1][1]
                x_terms = []
                for block, coeff in Am_coeffs[m][1][0]:
                    x_terms.append((block, coeff))
                for block, coeff in Am_coeffs[m][1][1]:
                    x_terms.append((block, coeff))
            elif s == 2:
                # Am[0][0]
                x_terms = Am_coeffs[m][0][0]
            elif s == 3:
                # Am[1][1]
                x_terms = Am_coeffs[m][1][1]
            elif s == 4:
                # S6 = Am[0][0] + Am[1][1]
                x_terms = []
                for block, coeff in Am_coeffs[m][0][0]:
                    x_terms.append((block, coeff))
                for block, coeff in Am_coeffs[m][1][1]:
                    x_terms.append((block, coeff))
            elif s == 5:
                # S8 = Am[0][1] - Am[1][1]
                x_terms = []
                for block, coeff in Am_coeffs[m][0][1]:
                    x_terms.append((block, coeff))
                for block, coeff in Am_coeffs[m][1][1]:
                    x_terms.append((block, -coeff))
            elif s == 6:
                # S7 = Am[1][0] - Am[0][0]
                x_terms = []
                for block, coeff in Am_coeffs[m][1][0]:
                    x_terms.append((block, coeff))
                for block, coeff in Am_coeffs[m][0][0]:
                    x_terms.append((block, -coeff))
            
            # Determine Y-coefficients for this term.
            # s=0: S1*S3 -> Y-part is S3 = Bm[0][0]+Bm[1][1]
            # s=1: S2*B00 -> Y-part is Bm[0][0]
            # s=2: A00*S5 -> Y-part is S5 = Bm[0][1]-Bm[1][1]
            # s=3: A11*S4 -> Y-part is S4 = Bm[1][0]-Bm[0][0]
            # s=4: S6*S3 -> Y-part is S3 = Bm[0][0]+Bm[1][1]
            # s=5: S8*(B10+B11) -> Y-part is Bm[1][0]+Bm[1][1]
            # s=6: S7*(B00+B01) -> Y-part is Bm[0][0]+Bm[0][1]
            
            if s == 0:
                # S3 = Bm[0][0] + Bm[1][1]
                y_terms = []
                for block, coeff in Bm_coeffs[m][0][0]:
                    y_terms.append((block, coeff))
                for block, coeff in Bm_coeffs[m][1][1]:
                    y_terms.append((block, coeff))
            elif s == 1:
                # Bm[0][0]
                y_terms = Bm_coeffs[m][0][0]
            elif s == 2:
                # S5 = Bm[0][1] - Bm[1][1]
                y_terms = []
                for block, coeff in Bm_coeffs[m][0][1]:
                    y_terms.append((block, coeff))
                for block, coeff in Bm_coeffs[m][1][1]:
                    y_terms.append((block, -coeff))
            elif s == 3:
                # S4 = Bm[1][0] - Bm[0][0]
                y_terms = []
                for block, coeff in Bm_coeffs[m][1][0]:
                    y_terms.append((block, coeff))
                for block, coeff in Bm_coeffs[m][0][0]:
                    y_terms.append((block, -coeff))
            elif s == 4:
                # S3 = Bm[0][0] + Bm[1][1]
                y_terms = []
                for block, coeff in Bm_coeffs[m][0][0]:
                    y_terms.append((block, coeff))
                for block, coeff in Bm_coeffs[m][1][1]:
                    y_terms.append((block, coeff))
            elif s == 5:
                # Bm[1][0] + Bm[1][1]
                y_terms = []
                for block, coeff in Bm_coeffs[m][1][0]:
                    y_terms.append((block, coeff))
                for block, coeff in Bm_coeffs[m][1][1]:
                    y_terms.append((block, coeff))
            elif s == 6:
                # Bm[0][0] + Bm[0][1]
                y_terms = []
                for block, coeff in Bm_coeffs[m][0][0]:
                    y_terms.append((block, coeff))
                for block, coeff in Bm_coeffs[m][0][1]:
                    y_terms.append((block, coeff))
            
            # Now expand x_terms to flat indices and fill U.
            # x_terms is a list of (block, coeff)
            # For each (block, coeff), add coeff to U[flat_idx, r] for each entry in the block.
            # But wait, the X-part is a scalar (sum of X entries with coefficients),
            # so U[i*m+j, r] should be the coefficient of X[i,j].
            
            # Actually, for a rank-1 term, U[i*m+j, r] is the coefficient such that
            # the X-part is sum_{i,j} U[i*m+j, r] * X[i,j].
            
            # So for each (block, coeff) in x_terms, for each of the 4 entries in the block,
            # U[flat_idx, r] += coeff.
            
            for block, coeff in x_terms:
                entries = block_entries_X(block)
                for flat_idx in entries:
                    U[flat_idx][r] += coeff
            
            # Similarly for V.
            for block, coeff in y_terms:
                entries = block_entries_Y(block)
                for flat_idx in entries:
                    V[flat_idx][r] += coeff
            
            # Now fill W.
            # The Z-contribution for sub-term s within block-level m.
            # The result block of Mm is placed at a specific position in the 4x4 Z.
            
            # Block-level result:
            # Z0 = M1 + M4 - M5 + M7  (m=0,3,4,6)
            # Z1 = M3 + M5            (m=2,4)
            # Z2 = M2 + M5            (m=1,4)
            # Z3 = M1 - M2 + M3 + M6  (m=0,1,2,5)
            
            # So for block-level m, the coefficient in the result is:
            z_block_coeff = {
                0: {0: 1, 3: 1},   # M1 contributes to Z0 and Z3
                1: {2: 1, 3: -1},  # M2 contributes to Z2 and Z3
                2: {1: 1, 3: 1},   # M3 contributes to Z1 and Z3
                3: {0: 1},         # M4 contributes to Z0
                4: {0: -1, 1: 1, 2: 1},  # M5 contributes to Z0, Z1, Z2
                5: {3: 1},         # M6 contributes to Z3
                6: {0: 1},         # M7 contributes to Z0
            }
            
            # For sub-term s within block m, the 2x2 result entries affected are z_contrib[s].
            # Each (local_pos, local_coeff) in z_contrib[s] means the local 2x2 result entry
            # local_pos gets local_coeff * (this rank-1 term's value).
            
            # The block-level coefficient for m is z_block_coeff[m][result_block].
            
            # So for each (result_block, block_coeff) in z_block_coeff[m].items():
            #   for each (local_pos, local_coeff) in z_contrib[s]:
            #     W[W_flat_idx, r] += block_coeff * local_coeff
            
            for result_block, block_coeff in z_block_coeff[m].items():
                for local_pos, local_coeff in z_contrib[s]:
                    # local_pos: 0=R00, 1=R01, 2=R10, 3=R11
                    # Map to Z flat index.
                    z_entries = block_entries_Z(result_block)
                    W_flat_idx = z_entries[local_pos]
                    W[W_flat_idx][r] += block_coeff * local_coeff
    
    return {"U": U, "V": V, "W": W}


def construct():
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
        return {"U": U, "V": V, "W": W}
    
    try:
        return _construct_strassen_4x4()
    except Exception:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
