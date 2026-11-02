"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    Uses block decomposition with Strassen for 2x2x2 blocks and Strassen+Naive for 2x2x3 blocks.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # Strategy: decompose into 2x2 blocks
    # A is 4x4, B is 4x5
    # Split A into 2x2 blocks of 2x2 matrices: A = [[A11, A12], [A21, A22]]
    # Split B into 2x2 blocks of 2x2 and 2x3 matrices: B = [[B11, B12], [B21, B22]]
    # where B11, B12, B21 are 2x2 and B22 is 2x3 (or we split differently)
    
    # Actually, let's think about this more carefully.
    # A: 4x4, B: 4x5, C: 4x5
    # Split rows of A and C into 2+2, columns of A and rows of B into 2+2, columns of B into 2+3
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where B11, B12, B21 are 2x2 and B22 is 2x3
    # C = [[C11, C12], [C21, C22]] where C11, C12, C21 are 2x2 and C22 is 2x3
    
    # C11 = A11*B11 + A12*B21 (2x2 * 2x2 + 2x2 * 2x2 = 2x2)
    # C12 = A11*B12 + A12*B22 (2x2 * 2x2 + 2x2 * 2x3 = 2x2 + 2x3? No...)
    
    # Wait, let me reconsider. B is 4x5. If I split rows of B into 2+2, then:
    # B = [B_top; B_bottom] where B_top is 2x5 and B_bottom is 2x5
    # A = [A_left, A_right] where A_left is 4x2 and A_right is 4x2
    # C = A * B = A_left * B_top + A_right * B_bottom (4x5 + 4x5 = 4x5)
    
    # Now split further:
    # A_left = [A11; A21] where A11, A21 are 2x2
    # A_right = [A12; A22] where A12, A22 are 2x2
    # B_top = [B11, B12] where B11 is 2x2 and B12 is 2x3
    # B_bottom = [B21, B22] where B21 is 2x2 and B22 is 2x3
    
    # C = A_left * B_top + A_right * B_bottom
    #   = [A11; A21] * [B11, B12] + [A12; A22] * [B21, B22]
    #   = [A11*B11 + A12*B21, A11*B12 + A12*B22;
    #      A21*B11 + A22*B21, A21*B12 + A22*B22]
    
    # So:
    # C11 (2x2) = A11*B11 + A12*B21 (both 2x2*2x2)
    # C12 (2x3) = A11*B12 + A12*B22 (2x2*2x3 + 2x2*2x3)
    # C21 (2x2) = A21*B11 + A22*B21 (both 2x2*2x2)
    # C22 (2x3) = A21*B12 + A22*B22 (2x2*2x3 + 2x2*2x3)
    
    # For the 2x2*2x2 multiplications (C11, C21), use Strassen (rank 7 each)
    # For the 2x2*2x3 multiplications (C12, C22), we need a decomposition
    
    # For 2x2 * 2x3: 
    # X (2x2) * Y (2x3) = Z (2x3)
    # We can split Y into [Y1, Y2] where Y1 is 2x1 and Y2 is 2x2
    # X*Y = [X*Y1, X*Y2]
    # X*Y1: 2x2 * 2x1 = 2x1 (rank 4 schoolbook, or can we do better?)
    # X*Y2: 2x2 * 2x2 = 2x2 (rank 7 Strassen)
    # Total for 2x2*2x3: 4 + 7 = 11? Or maybe there's a better way.
    
    # Actually, let's think of 2x2 * 2x3 differently.
    # Let X = [[x11, x12], [x21, x22]], Y = [[y11, y12, y13], [y21, y22, y23]]
    # Z = X*Y has entries:
    # Z11 = x11*y11 + x12*y21
    # Z12 = x11*y12 + x12*y22
    # Z13 = x11*y13 + x12*y23
    # Z21 = x21*y11 + x22*y21
    # Z22 = x21*y12 + x22*y22
    # Z23 = x21*y13 + x22*y23
    
    # This is like two 2x1 multiplications (Z11,Z21) and two 2x2 multiplications (Z12,Z13) and (Z22,Z23)
    # Wait no, it's:
    # Column 1 of Z: X * [y11; y21] (2x2 * 2x1)
    # Column 2 of Z: X * [y12; y22] (2x2 * 2x1)
    # Column 3 of Z: X * [y13; y23] (2x2 * 2x1)
    
    # Each 2x2 * 2x1 multiplication has rank 4 (schoolbook) or possibly less?
    # Actually 2x2 * 2x1: result is 2x1. The tensor is of size 2*2*1 = 4.
    # Schoolbook rank is 4. Can we do better? For 2x2*2x1, I believe rank 4 is minimal.
    
    # So 2x2 * 2x3 via 3 separate 2x2*2x1 multiplications: rank 3*4 = 12
    # Or via 1 2x2*2x1 + 1 2x2*2x2: rank 4 + 7 = 11
    
    # Let's use the 4+7=11 approach for 2x2*2x3.
    
    # Total rank:
    # C11: Strassen 2x2*2x2 = 7
    # C12: 2x2*2x3 = 11 (4 + 7)
    # C21: Strassen 2x2*2x2 = 7
    # C22: 2x2*2x3 = 11 (4 + 7)
    # Total: 7 + 11 + 7 + 11 = 36
    
    # But wait, this might not be optimal. Let me reconsider.
    
    # Actually, for the 2x2*2x3 case, let me think again.
    # X (2x2) * Y (2x3):
    # We can write Y = [Y1, Y2] where Y1 is 2x2 and Y2 is 2x1
    # X*Y = [X*Y1, X*Y2]
    # X*Y1: 2x2*2x2, rank 7 (Strassen)
    # X*Y2: 2x2*2x1, rank 4
    # Total: 11
    
    # Alternatively, we could try to find a direct decomposition.
    # The tensor for 2x2*2x3 has dimension 2*2*3 = 12.
    # I'm not sure of the exact minimal rank, but 11 seems reasonable.
    
    # Let me build this decomposition.
    
    rank_total = 36  # 7 + 11 + 7 + 11
    U = np.zeros((n*m, rank_total), dtype=float)
    V = np.zeros((m*p, rank_total), dtype=float)
    W = np.zeros((p*n, rank_total), dtype=float)
    
    col = 0
    
    # Helper: add Strassen 2x2*2x2 decomposition
    # U indices: (i, j) where i in {0,1} for row block, j in {0,1} for col block
    # V indices: (j, k) where j in {0,1} for row block, k in {0,1} for col block  
    # W indices: (k, i) where k in {0,1} for col block, i in {0,1} for row block
    # 
    # Strassen for 2x2*2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(U_base, V_base, W_base, U_row_offset, U_col_offset, 
                       V_row_offset, V_col_offset, W_row_offset, W_col_offset, 
                       U_dim_i, U_dim_j, V_dim_j, V_dim_k, W_dim_k, W_dim_i,
                       start_col):
        """
        Add Strassen 2x2*2x2 decomposition.
        U_base: U matrix, V_base: V matrix, W_base: W matrix
        U_row_offset, U_col_offset: offset in U for the 2x2 block
        V_row_offset, V_col_offset: offset in V for the 2x2 block
        W_row_offset, W_col_offset: offset in W for the 2x2 block
        U_dim_i: size of i dimension in U (should be 2)
        U_dim_j: size of j dimension in U (should be 2)
        V_dim_j: size of j dimension in V (should be 2)
        V_dim_k: size of k dimension in V (should be 2)
        W_dim_k: size of k dimension in W (should be 2)
        W_dim_i: size of i dimension in W (should be 2)
        start_col: starting column index
        Returns: end column index
        """
        col_idx = start_col
        # Strassen factors
        # M1 = (A11+A22)*(B11+B22)
        U[U_row_offset + 0, U_col_offset + 0, col_idx] = 1
        U[U_row_offset + 1, U_col_offset + 1, col_idx] = 1
        V[V_row_offset + 0, V_col_offset + 0, col_idx] = 1
        V[V_row_offset + 1, V_col_offset + 1, col_idx] = 1
        W[W_row_offset + 0, W_col_offset + 0, col_idx] = 1
        W[W_row_offset + 0, W_col_offset + 1, col_idx] = 1
        W[W_row_offset + 1, W_col_offset + 0, col_idx] = 1
        W[W_row_offset + 1, W_col_offset + 1, col_idx] = 1
        col_idx += 1
        
        # M2 = (A21+A22)*B11
        U[U_row_offset + 1, U_col_offset + 0, col_idx] = 1
        U[U_row_offset + 1, U_col_offset + 1, col_idx] = 1
        V[V_row_offset + 0, V_col_offset + 0, col_idx] = 1
        W[W_row_offset + 1, W_col_offset + 0, col_idx] = 1
        col_idx += 1
        
        # M3 = A11*(B12-B22)
        U[U_row_offset + 0, U_col_offset + 0, col_idx] = 1
        V[V_row_offset + 0, V_col_offset + 1, col_idx] = 1
        V[V_row_offset + 1, V_col_offset + 1, col_idx] = -1
        W[W_row_offset + 0, W_col_offset + 1, col_idx] = 1
        col_idx += 1
        
        # M4 = A22*(B21-B11)
        U[U_row_offset + 1, U_col_offset + 1, col_idx] = 1
        V[V_row_offset + 1, V_col_offset + 0, col_idx] = 1
        V[V_row_offset + 0, V_col_offset + 0, col_idx] = -1
        W[W_row_offset + 1, W_col_offset + 0, col_idx] = 1
        col_idx += 1
        
        # M5 = (A11+A12)*B22
        U[U_row_offset + 0, U_col_offset + 0, col_idx] = 1
        U[U_row_offset + 0, U_col_offset + 1, col_idx] = 1
        V[V_row_offset + 1, V_col_offset + 1, col_idx] = 1
        W[W_row_offset + 0, W_col_offset + 0, col_idx] = -1
        W[W_row_offset + 1, W_col_offset + 1, col_idx] = 1
        col_idx += 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[U_row_offset + 1, U_col_offset + 0, col_idx] = 1
        U[U_row_offset + 0, U_col_offset + 0, col_idx] = -1
        V[V_row_offset + 0, V_col_offset + 0, col_idx] = 1
        V[V_row_offset + 0, V_col_offset + 1, col_idx] = 1
        W[W_row_offset + 1, W_col_offset + 1, col_idx] = 1
        col_idx += 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[U_row_offset + 0, U_col_offset + 1, col_idx] = 1
        U[U_row_offset + 1, U_col_offset + 1, col_idx] = -1
        V[V_row_offset + 1, V_col_offset + 0, col_idx] = 1
        V[V_row_offset + 1, V_col_offset + 1, col_idx] = 1
        W[W_row_offset + 0, W_col_offset + 0, col_idx] = 1
        col_idx += 1
        
        return col_idx
    
    # Helper: add 2x2*2x1 decomposition (schoolbook, rank 4)
    def naive_2x2x1(U_base, V_base, W_base, U_row_offset, U_col_offset,
                    V_row_offset, V_col_offset, W_row_offset, W_col_offset,
                    start_col):
        """
        Add naive 2x2*2x1 decomposition.
        U: 2x2 block, V: 2x1 block, W: 2x1 block
        """
        col_idx = start_col
        for i in range(2):
            for j in range(2):
                U[U_row_offset + i, U_col_offset + j, col_idx] = 1
                V[V_row_offset + j, V_col_offset + 0, col_idx] = 1
                W[W_row_offset + 0, W_col_offset + i, col_idx] = 1
                col_idx += 1
        return col_idx
    
    # Helper: add 2x2*2x3 decomposition (1 2x2*2x2 Strassen + 1 2x2*2x1 naive)
    def decompose_2x2x2x3(U_base, V_base, W_base, U_row_offset, U_col_offset,
                          V_row_offset, V_col_offset, W_row_offset, W_col_offset,
                          start_col):
        """
        Add 2x2*2x3 decomposition.
        U: 2x2, V: 2x3, W: 2x3
        Split V into V1 (2x2) and V2 (2x1)
        W1 = U*V1 (2x2, Strassen)
        W2 = U*V2 (2x1, naive)
        """
        # First: U * V1 (2x2 * 2x2 -> 2x2) using Strassen
        col_idx = strassen_2x2x2(U_base, V_base, W_base,
                                 U_row_offset, U_col_offset,
                                 V_row_offset, V_col_offset,
                                 W_row_offset, W_col_offset,
                                 2, 2, 2, 2, 2, 2,
                                 start_col)
        # Second: U * V2 (2x2 * 2x1 -> 2x1) using naive
        col_idx = naive_2x2x1(U_base, V_base, W_base,
                              U_row_offset, U_col_offset,
                              V_row_offset + 0, V_col_offset + 2,
                              W_row_offset + 0, W_col_offset + 2,
                              col_idx)
        return col_idx
    
    # Now build the full decomposition
    # A is 4x4: split into 2x2 blocks
    # A = [[A11, A12], [A21, A22]]
    # U index for A: row i (0-3), col j (0-3) -> U[i, j]
    # 
    # B is 4x5: split rows into 2+2, cols into 2+3
    # B = [[B11, B12], [B21, B22]]
    # B11: rows 0-1, cols 0-1 (2x2)
    # B12: rows 0-1, cols 2-4 (2x3)
    # B21: rows 2-3, cols 0-1 (2x2)
    # B22: rows 2-3, cols 2-4 (2x3)
    # V index for B: row j (0-3), col k (0-4) -> V[j, k]
    
    # C is 4x5: split rows into 2+2, cols into 2+3
    # C = [[C11, C12], [C21, C22]]
    # C11: rows 0-1, cols 0-1 (2x2)
    # C12: rows 0-1, cols 2-4 (2x3)
    # C21: rows 2-3, cols 0-1 (2x2)
    # C22: rows 2-3, cols 2-4 (2x3)
    # W index for C: row k (0-4), col i (0-3) -> W[k, i]
    # Note: W[k*n+i, r] means W index is k*4+i
    
    # C11 = A11*B11 + A12*B21
    # A11: U rows 0-1, cols 0-1
    # B11: V rows 0-1, cols 0-1
    # A12: U rows 0-1, cols 2-3
    # B21: V rows 2-3, cols 0-1
    # C11: W rows 0-1 (k=0,1), cols 0-1 (i=0,1) -> W indices 0-3
    
    # For C11 = A11*B11 + A12*B21:
    # This is a sum of two 2x2*2x2 products.
    # We need to combine them.
    
    # Actually, let me reconsider the approach. The issue is that C11 = A11*B11 + A12*B21
    # involves two separate matrix products. We can't directly apply Strassen to this sum.
    
    # Let me think differently. We have:
    # C = A * B where A is 4x4 and B is 4x5.
    # 
    # Let's use a different blocking strategy.
    # Split A into left 2 cols and right 2 cols: A = [AL, AR] where AL, AR are 4x2
    # Split B into top 2 rows and bottom 2 rows: B = [BT; BB] where BT, BB are 2x5
    # C = AL * BT + AR * BB (4x5 + 4x5)
    
    # Now AL is 4x2, BT is 2x5. AL*BT is 4x5.
    # AR is 4x2, BB is 2x5. AR*BB is 4x5.
    
    # For AL*BT (4x2 * 2x5 = 4x5):
    # Split AL into top 2 rows and bottom 2 rows: AL = [AL1; AL2] where AL1, AL2 are 2x2
    # Split BT into left 2 cols and right 3 cols: BT = [BT1, BT2] where BT1 is 2x2, BT2 is 2x3
    # AL*BT = [AL1*BT1 + AL1*BT2, AL2*BT1 + AL2*BT2]
    #        = [AL1*BT1, AL1*BT2; AL2*BT1, AL2*BT2]
    # AL1*BT1: 2x2*2x2 = 2x2, rank 7 (Strassen)
    # AL1*BT2: 2x2*2x3 = 2x3, rank 11
    # AL2*BT1: 2x2*2x2 = 2x2, rank 7 (Strassen)
    # AL2*BT2: 2x2*2x3 = 2x3, rank 11
    # Total for AL*BT: 7 + 11 + 7 + 11 = 36
    
    # Similarly for AR*BB: 36
    
    # Total: 72. That's worse than my earlier estimate of 36.
    
    # Let me go back to the original blocking.
    # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11, B12], [B21, B22]] where B11, B21 are 2x2, B12, B22 are 2x3
    # C = [[C11, C12], [C21, C22]] where C11, C21 are 2x2, C12, C22 are 2x3
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For C11 = A11*B11 + A12*B21:
    # This is the sum of two 2x2 matrix products.
    # We can view this as: [A11, A12] * [B11; B21] where [A11, A12] is 2x4 and [B11; B21] is 4x2
    # So it's a 2x4 * 4x2 = 2x2 multiplication.
    # Rank of 2x4*4x2? Schoolbook is 2*4*2 = 16.
    # Can we do better? This is equivalent to computing a 2x2 matrix where each entry is a dot product of length 4.
    # I don't know the exact minimal rank, but let's use schoolbook for now: rank 16.
    
    # Hmm, this is getting complicated. Let me try a different approach.
    
    # Let me just use the block structure and compute each block product separately.
    
    # For C11 = A11*B11 + A12*B21:
    # A11*B11: 2x2*2x2, rank 7
    # A12*B21: 2x2*2x2, rank 7
    # Total for C11: 14
    
    # For C12 = A11*B12 + A12*B22:
    # A11*B12: 2x2*2x3, rank 11
    # A12*B22: 2x2*2x3, rank 11
    # Total for C12: 22
    
    # For C21 = A21*B11 + A22*B21:
    # A21*B11: 2x2*2x2, rank 7
    # A22*B21: 2x2*2x2, rank 7
    # Total for C21: 14
    
    # For C22 = A21*B12 + A22*B22:
    # A21*B12: 2x2*2x3, rank 11
    # A22*B22: 2x2*2x3, rank 11
    # Total for C22: 22
    
    # Grand total: 14 + 22 + 14 + 22 = 72
    
    # That's still 72. Let me see if I can do better.
    
    # Actually, wait. The knowledge says "Block decomposition using Strassen (rank 7) for 2x2x2 blocks and Strassen+Naive (rank 9) for 2x2x3 blocks reduces rank from 80 to 64"
    
    # So Strassen+Naive for 2x2x3 has rank 9, not 11!
    # Let me reconsider.
    
    # For 2x2 * 2x3 with rank 9:
    # Maybe there's a cleverer decomposition.
    # X (2x2) * Y (2x3) = Z (2x3)
    # 
    # Let's try:
    # M1 = (x11+x21) * (y11+y21)  -- but this doesn't quite work for the 2x3 case
    
    # Actually, let me think about it as:
    # Z = X * Y where X is 2x2 and Y is 2x3
    # Z11 = x11*y11 + x12*y21
    # Z12 = x11*y12 + x12*y22
    # Z13 = x11*y13 + x12*y23
    # Z21 = x21*y11 + x22*y21
    # Z22 = x21*y12 + x22*y22
    # Z23 = x21*y13 + x22*y23
    
    # This can be seen as:
    # Z = [x11, x12; x21, x22] * [y11, y12, y13; y21, y22, y23]
    
    # One approach: use Strassen-like ideas.
    # Let's define:
    # M1 = (x11+x21)*(y11+y21)
    # M2 = (x11+x21)*(y12+y22)
    # M3 = (x11+x21)*(y13+y23)
    # M4 = x11*(y12-y22)
    # M5 = x11*(y13-y23)
    # M6 = x21*(y11-y12)  -- hmm, this is getting messy
    
    # Let me try a different factorization.
    # Actually, the knowledge says rank 9 for 2x2x3. Let me just use that.
    
    # For now, let me assume:
    # 2x2*2x2: rank 7 (Strassen)
    # 2x2*2x3: rank 9
    
    # Total:
    # C11: 7 + 7 = 14
    # C12: 9 + 9 = 18
    # C21: 7 + 7 = 14
    # C22: 9 + 9 = 18
    # Total: 14 + 18 + 14 + 18 = 64
    
    # That matches the knowledge! Rank 64.
    
    # Now I need to implement the 2x2*2x3 decomposition with rank 9.
    
    # Let me figure out a rank-9 decomposition for 2x2*2x3.
    # X = [[a, b], [c, d]], Y = [[e, f, g], [h, i, j]]
    # Z = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    # I'll use a combination of Strassen and naive.
    # Split Y into Y1 (2x2) and Y2 (2x1):
    # Y1 = [[e, f], [h, i]], Y2 = [[g], [j]]
    # Z1 = X*Y1 (2x2, Strassen rank 7)
    # Z2 = X*Y2 (2x1, naive rank 4)
    # Total: 7 + 4 = 11. But we want 9.
    
    # So there must be a better way. Let me think...
    
    # Actually, maybe I can share some factors.
    # 
    # Let me try:
    # M1 = (a+c)*(e+h)
    # M2 = (a+c)*(f+i)
    # M3 = (a+c)*(g+j)
    # M4 = a*(f-i)
    # M5 = a*(g-j)
    # M6 = c*(e-f)
    # M7 = c*(g-j)  -- wait, I'm reusing M5's factor
    # M8 = (b-d)*(e+h)
    # M9 = (b-d)*(f+i)
    
    # Let me verify:
    # Z11 = ae+bh
    # From M1, M6, M8: M1 = ae+af+ce+ci, M6 = ce-cf, M8 = be+bf-de-di
    # This doesn't seem to work cleanly.
    
    # Let me just use the 11-rank decomposition for now (7+4) and accept rank 72.
    # Or better, let me try to find the rank-9 decomposition.
    
    # Actually, let me just implement what I can and get a valid answer.
    # I'll use the straightforward block decomposition.
    
    # Let me restart with a cleaner implementation.
    
    # Reset
    rank_total = 64
    U = np.zeros((n*m, rank_total), dtype=float)
    V = np.zeros((m*p, rank_total), dtype=float)
    W = np.zeros((p*n, rank_total), dtype=float)
    
    col = 0
    
    # For simplicity, let me use the 11-rank (7+4) decomposition for 2x2*2x3
    # Total rank will be 14 + 22 + 14 + 22 = 72
    
    rank_total = 72
    U = np.zeros((n*m, rank_total), dtype=float)
    V = np.zeros((m*p, rank_total), dtype=float)
    W = np.zeros((p*n, rank_total), dtype=float)
    
    col = 0
    
    # Strassen 2x2*2x2 helper
    def strassen(U_mat, V_mat, W_mat, u_r, u_c, v_r, v_c, w_r, w_c, start):
        """u_r, u_c: row/col offset in U for 2x2 block
           v_r, v_c: row/col offset in V for 2x2 block
           w_r, w_c: row/col offset in W for 2x2 block (w is indexed as [k*n+i])
           Returns end col.
        """
        c = start
        # M1 = (A11+A22)*(B11+B22)
        U_mat[u_r+0, u_c+0, c] += 1
        U_mat[u_r+1, u_c+1, c] += 1
        V_mat[v_r+0, v_c+0, c] += 1
        V_mat[v_r+1, v_c+1, c] += 1
        W_mat[w_r+0, w_c+0, c] += 1
        W_mat[w_r+0, w_c+1, c] += 1
        W_mat[w_r+1, w_c+0, c] += 1
        W_mat[w_r+1, w_c+1, c] += 1
        c += 1
        # M2 = (A21+A22)*B11
        U_mat[u_r+1, u_c+0, c] += 1
        U_mat[u_r+1, u_c+1, c] += 1
        V_mat[v_r+0, v_c+0, c] += 1
        W_mat[w_r+1, w_c+0, c] += 1
        c += 1
        # M3 = A11*(B12-B22)
        U_mat[u_r+0, u_c+0, c] += 1
        V_mat[v_r+0, v_c+1, c] += 1
        V_mat[v_r+1, v_c+1, c] += -1
        W_mat[w_r+0, w_c+1, c] += 1
        c += 1
        # M4 = A22*(B21-B11)
        U_mat[u_r+1, u_c+1, c] += 1
        V_mat[v_r+1, v_c+0, c] += 1
        V_mat[v_r+0, v_c+0, c] += -1
        W_mat[w_r+1, w_c+0, c] += 1
        c += 1
        # M5 = (A11+A12)*B22
        U_mat[u_r+0, u_c+0, c] += 1
        U_mat[u_r+0, u_c+1, c] += 1
        V_mat[v_r+1, v_c+1, c] += 1
        W_mat[w_r+0, w_c+0, c] += -1
        W_mat[w_r+1, w_c+1, c] += 1
        c += 1
        # M6 = (A21-A11)*(B11+B12)
        U_mat[u_r+1, u_c+0, c] += 1
        U_mat[u_r+0, u_c+0, c] += -1
        V_mat[v_r+0, v_c+0, c] += 1
        V_mat[v_r+0, v_c+1, c] += 1
        W_mat[w_r+1, w_c+1, c] += 1
        c += 1
        # M7 = (A12-A22)*(B21+B22)
        U_mat[u_r+0, u_c+1, c] += 1
        U_mat[u_r+1, u_c+1, c] += -1
        V_mat[v_r+1, v_c+0, c] += 1
        V_mat[v_r+1, v_c+1, c] += 1
        W_mat[w_r+0, w_c+0, c] += 1
        c += 1
        return c
    
    # Naive 2x2*2x1 helper
    def naive_2x2x1(U_mat, V_mat, W_mat, u_r, u_c, v_r, v_c, w_r, w_c, start):
        """u_r, u_c: row/col offset in U for 2x2 block
           v_r, v_c: row/col offset in V for 2x1 block
           w_r, w_c: row/col offset in W for 2x1 block
        """
        c = start
        for i in range(2):
            for j in range(2):
                U_mat[u_r+i, u_c+j, c] += 1
                V_mat[v_r+j, v_c+0, c] += 1
                W_mat[w_r+0, w_c+i, c] += 1
                c += 1
        return c
    
    # 2x2*2x3 helper (Strassen for 2x2*2x2 part + naive for 2x2*2x1 part)
    def decomp_2x2x2x3(U_mat, V_mat, W_mat, u_r, u_c, v_r, v_c, w_r, w_c, start):
        """u_r, u_c: row/col offset in U for 2x2 block
           v_r, v_c: row/col offset in V for 2x3 block
           w_r, w_c: row/col offset in W for 2x3 block
        """
        c = strassen(U_mat, V_mat, W_mat, u_r, u_c, v_r, v_c, w_r, w_c, start)
        c = naive_2x2x1(U_mat, V_mat, W_mat, u_r, u_c, v_r, v_c+2, w_r, w_c+2, c)
        return c
    
    # Now build the decomposition
    # A: 4x4, U indexed as [i*4+j] where i=row, j=col
    # B: 4x5, V indexed as [j*5+k] where j=row, k=col
    # C: 4x5, W indexed as [k*4+i] where k=row, i=col (per trace convention W[k*n+i,r])
    
    # C11 (rows 0-1, cols 0-1) = A11*B11 + A12*B21
    # A11: U rows 0-1, cols 0-1 -> u_r=0, u_c=0
    # B11: V rows 0-1, cols 0-1 -> v_r=0, v_c=0
    # C11: W rows 0-1 (k=0,1), cols 0-1 (i=0,1) -> w_r=0, w_c=0
    col = strassen(U, V, W, 0, 0, 0, 0, 0, 0, col)  # A11*B11
    col = strassen(U, V, W, 0, 2, 2, 0, 0, 0, col)  # A12*B21
    
    # C12 (rows 0-1, cols 2-4) = A11*B12 + A12*B22
    # A11: u_r=0, u_c=0
    # B12: V rows 0-1, cols 2-4 -> v_r=0, v_c=2
    # C12: W rows 2-4 (k=2,3,4), cols 0-1 (i=0,1) -> w_r=2, w_c=0
    col = decomp_2x2x2x3(U, V, W, 0, 0, 0, 2, 2, 0, col)  # A11*B12
    col = decomp_2x2x2x3(U, V, W, 0, 2, 2, 2, 2, 0, col)  # A12*B22
    
    # C21 (rows 2-3, cols 0-1) = A21*B11 + A22*B21
    # A21: U rows 2-3, cols 0-1 -> u_r=2, u_c=0
    # B11: v_r=0, v_c=0
    # C21: W rows 0-1 (k=0,1), cols 2-3 (i=2,3) -> w_r=0, w_c=2
    col = strassen(U, V, W, 2, 0, 0, 0, 0, 2, col)  # A21*B11
    col = strassen(U, V, W, 2, 2, 2, 0, 0, 2, col)  # A22*B21
    
    # C22 (rows 2-3, cols 2-4) = A21*B12 + A22*B22
    # A21: u_r=2, u_c=0
    # B12: v_r=0, v_c=2
    # C22: W rows 2-4 (k=2,3,4), cols 2-3 (i=2,3) -> w_r=2, w_c=2
    col = decomp_2x2x2x3(U, V, W, 2, 0, 0, 2, 2, 2, col)  # A21*B12
    col = decomp_2x2x2x3(U, V, W, 2, 2, 2, 2, 2, 2, col)  # A22*B22
    
    # Verify col == rank_total
    assert col == rank_total, f"Column mismatch: {col} != {rank_total}"
    
    # Verify the decomposition
    # For random A, B, check that sum_r U[:,r] * V[:,r] * W[:,r] = A*B
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank_total):
        u_vec = U[:, r]  # shape (n*m,)
        v_vec = V[:, r]  # shape (m*p,)
        w_vec = W[:, r]  # shape (p*n,)
        
        # u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        # Reshape
        U_block = u_vec.reshape(n, m)
        V_block = v_vec.reshape(m, p)
        W_block = w_vec.reshape(p, n)
        
        # Contribution to C[i,k] = sum_j U_block[i,j] * V_block[j,k] * W_block[k,i]
        # This is not a standard matrix product. Let me compute it directly.
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i, k] += U_block[i, j] * V_block[j, k] * W_block[k, i]
    
    # Check
    max_err = np.max(np.abs(C_computed - C_test))
    if max_err > 1e-10:
        # Fallback to schoolbook
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
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
