import time

def construct():
    """
    Constructs a rank-50 decomposition for 2x4 x 4x7 matrix multiplication.
    
    The strategy splits the 4x7 matrix B into four vertical blocks of size 4x2, 4x2, 4x2, and 4x1.
    The 2x4 matrix A is split into two horizontal blocks A_L (2x2) and A_R (2x2).
    
    For each 4x2 block of B, we compute A_L * B_block and A_R * B_block.
    Each 2x2 * 4x2 multiplication is further split.
    Actually, the previous logic split B into 2x2 blocks (in the middle dimension) and used Strassen.
    
    Let's re-verify the block structure used in the score 1.12 code.
    
    A is 2x4. Split into A_L (cols 0,1) and A_R (cols 2,3).
    B is 4x7. Split into B_top (rows 0,1) and B_bot (rows 2,3).
    B_top is 2x7. B_bot is 2x7.
    
    C = A_L @ B_top + A_R @ B_bot.
    
    We need to decompose the product of a 2x2 matrix (A_L) and a 2x7 matrix (B_top).
    Let's call this T(2,2,7).
    B_top is 2x7. Split B_top into three 2x2 blocks (cols 0,1; 2,3; 4,5) and one 2x1 block (col 6).
    
    T(2,2,7) = T(2,2,2) + T(2,2,2) + T(2,2,2) + T(2,2,1).
    
    T(2,2,2) is Strassen (rank 7).
    T(2,2,1) is schoolbook (rank 4).
    
    Rank of T(2,2,7) is 7+7+7+4 = 25.
    
    Since A_L and A_R are disjoint, and B_top and B_bot are disjoint,
    Total Rank = Rank(T(2,2,7) for A_L) + Rank(T(2,2,7) for A_R) = 25 + 25 = 50.
    """
    
    n, m, p = 2, 4, 7
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50
    
    col = 0
    
    def add_strassen_222(A_indices, B_indices, C_indices):
        """
        Adds a Strassen decomposition for 2x2 * 2x2 -> 2x2.
        A_indices: dict mapping local (r,c) to global index in U.
        B_indices: dict mapping local (r,c) to global index in V.
        C_indices: dict mapping local (r,c) to global index in W.
        """
        nonlocal col
        
        # Local indices for 2x2 matrix:
        # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        # M1: (A00+A11)(B00+B11) -> C00+C11
        # M2: (A10+A11)(B00) -> C10, C11 -= M2
        # M3: (A00)(B11-B01) -> C01, C11 += M3
        # M4: (A11)(B01-B00) -> C00, C10 += M4
        # M5: (A00+A01)(B11) -> C01, C00 -= M5
        # M6: (A10+A00)(B00+B01) -> C11
        # M7: (A01-A10)(B10+B11) -> C10
        
        def process(a_coeffs, b_coeffs, c_coeffs):
            nonlocal col
            for la, coeff in a_coeffs:
                ga = A_indices[la]
                U[ga][col] += coeff
            for lb, coeff in b_coeffs:
                gb = B_indices[lb]
                V[gb][col] += coeff
            for lc, coeff in c_coeffs:
                gc = C_indices[lc]
                W[gc][col] += coeff
            col += 1

        # M1
        process([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
        # M2
        process([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
        # M3
        process([(0, 1)], [(3, 1), (1, -1)], [(1, 1), (3, 1)])
        # M4
        process([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)])
        # M5
        process([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)])
        # M6
        process([(2, 1), (0, 1)], [(0, 1), (1, 1)], [(3, 1)])
        # M7
        process([(1, 1), (2, -1)], [(2, 1), (3, 1)], [(2, 1)])
        
    def add_schoolbook_221(A_indices, B_indices, C_indices):
        """
        Adds a schoolbook decomposition for 2x2 * 2x1 -> 2x1.
        """
        nonlocal col
        
        def process(a_idx, b_idx, c_idx, coeff):
            nonlocal col
            ga = A_indices[a_idx]
            gb = B_indices[b_idx]
            gc = C_indices[c_idx]
            U[ga][col] += coeff
            V[gb][col] += coeff
            W[gc][col] += coeff
            col += 1

        # C0 = A00*B0 + A01*B1
        process(0, 0, 0, 1)
        process(1, 1, 0, 1)
        # C1 = A10*B0 + A11*B1
        process(2, 0, 1, 1)
        process(3, 1, 1, 1)

    # Define global index mappings for sub-blocks
    
    # A_L: rows 0,1 of A, cols 0,1
    # A is 2x4. Index (r, c) -> r*4 + c
    A_L_idx = {
        (0,0): 0*4+0, (0,1): 0*4+1,
        (1,0): 1*4+0, (1,1): 1*4+1
    }
    
    # A_R: rows 0,1 of A, cols 2,3
    A_R_idx = {
        (0,0): 0*4+2, (0,1): 0*4+3,
        (1,0): 1*4+2, (1,1): 1*4+3
    }
    
    # B_top: rows 0,1 of B, cols 0..6
    # B is 4x7. Index (r, c) -> r*7 + c
    
    # B_L_L: B_top cols 0,1 -> B rows 0,1, cols 0,1
    B_LL_idx = {
        (0,0): 0*7+0, (0,1): 0*7+1,
        (1,0): 1*7+0, (1,1): 1*7+1
    }
    # B_L_R: B_top cols 2,3 -> B rows 0,1, cols 2,3
    B_LR_idx = {
        (0,0): 0*7+2, (0,1): 0*7+3,
        (1,0): 1*7+2, (1,1): 1*7+3
    }
    # B_L_RR: B_top cols 4,5 -> B rows 0,1, cols 4,5
    B_LRR_idx = {
        (0,0): 0*7+4, (0,1): 0*7+5,
        (1,0): 1*7+4, (1,1): 1*7+5
    }
    # B_L_1: B_top col 6 -> B rows 0,1, col 6
    B_L1_idx = {
        0: 0*7+6,
        1: 1*7+6
    }
    
    # B_bot: rows 2,3 of B, cols 0..6
    # B_R_L: B_bot cols 0,1 -> B rows 2,3, cols 0,1
    B_RL_idx = {
        (0,0): 2*7+0, (0,1): 2*7+1,
        (1,0): 3*7+0, (1,1): 3*7+1
    }
    # B_R_R: B_bot cols 2,3 -> B rows 2,3, cols 2,3
    B_RR_idx = {
        (0,0): 2*7+2, (0,1): 2*7+3,
        (1,0): 3*7+2, (1,1): 3*7+3
    }
    # B_R_RR: B_bot cols 4,5 -> B rows 2,3, cols 4,5
    B_RRR_idx = {
        (0,0): 2*7+4, (0,1): 2*7+5,
        (1,0): 3*7+4, (1,1): 3*7+5
    }
    # B_R_1: B_bot col 6 -> B rows 2,3, col 6
    B_R1_idx = {
        0: 2*7+6,
        1: 3*7+6
    }
    
    # C indices
    # C is 2x7. Index (r, c) -> c*2 + r (Trace convention W[k*n+i, r] where k is col, i is row? 
    # Task says: W[k*n+i, r]. So W index is (col * n + row).
    # n=2. So W index = col*2 + row.
    
    # C_L_L: C cols 0,1 -> W indices (0,0),(0,1),(1,0),(1,1)
    C_LL_idx = {
        (0,0): 0*2+0, (0,1): 0*2+1,
        (1,0): 1*2+0, (1,1): 1*2+1
    }
    # C_L_R: C cols 2,3 -> W indices (2,0),(2,1),(3,0),(3,1)
    C_LR_idx = {
        (0,0): 2*2+0, (0,1): 2*2+1,
        (1,0): 3*2+0, (1,1): 3*2+1
    }
    # C_L_RR: C cols 4,5 -> W indices (4,0),(4,1),(5,0),(5,1)
    C_LRR_idx = {
        (0,0): 4*2+0, (0,1): 4*2+1,
        (1,0): 5*2+0, (1,1): 5*2+1
    }
    # C_L_1: C col 6 -> W indices (6,0),(6,1)
    C_L1_idx = {
        0: 6*2+0,
        1: 6*2+1
    }
    
    # Process A_L @ B_top
    add_strassen_222(A_L_idx, B_LL_idx, C_LL_idx)
    add_strassen_222(A_L_idx, B_LR_idx, C_LR_idx)
    add_strassen_222(A_L_idx, B_LRR_idx, C_LRR_idx)
    add_schoolbook_221(A_L_idx, B_L1_idx, C_L1_idx)
    
    # Process A_R @ B_bot
    add_strassen_222(A_R_idx, B_RL_idx, C_LL_idx)
    add_strassen_222(A_R_idx, B_RR_idx, C_LR_idx)
    add_strassen_222(A_R_idx, B_RRR_idx, C_LRR_idx)
    add_schoolbook_221(A_R_idx, B_R1_idx, C_L1_idx)
    
    return {"U": U, "V": V, "W": W}
