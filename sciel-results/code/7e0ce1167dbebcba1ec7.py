"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # Strategy: Split A into two 2x4 horizontal strips.
    # A_top = A[0:2, :] (2x4), A_bot = A[2:4, :] (2x4)
    # C_top = A_top * B (2x6), C_bot = A_bot * B (2x6)
    # 
    # For each 2x4x6 sub-problem, use Strassen-style decomposition.
    # Split B into two 4x3 blocks: B_left (4x3), B_right (4x3)
    # C_strip_left = A_strip * B_left (2x4x3)
    # C_strip_right = A_strip * B_right (2x4x3)
    #
    # For 2x4x3: split A_strip into two 2x2 blocks, B_block into two 2x3 blocks
    # A_strip = [A1, A2] where A1, A2 are 2x2
    # B_block = [B1; B2] where B1, B2 are 2x3
    # C = A1*B1 + A2*B2 (2x3)
    #
    # For 2x2x3: use Strassen-like decomposition with rank 6:
    # M1 = (A1+A2)(B1+B2)
    # M2 = A1*(B2-B1)  
    # M3 = A2*(B1+B2)
    # C = (M1 + M3 - M2)/2 ... let me verify
    # M1 = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # M2 = A1*B2 - A1*B1
    # M3 = A2*B1 + A2*B2
    # M1 - M2 + M3 = A1*B1 + A1*B2 + A2*B1 + A2*B2 - A1*B2 + A1*B1 + A2*B1 + A2*B2
    #              = 2*A1*B1 + 2*A2*B1 + 2*A2*B2
    # That's not right. Let me use:
    # M1 = (A1+A2)(B1+B2)
    # M2 = (A1-A2)(B1-B2)
    # M1 - M2 = (A1*B1 + A1*B2 + A2*B1 + A2*B2) - (A1*B1 - A1*B2 - A2*B1 + A2*B2)
    #          = 2*A1*B2 + 2*A2*B1
    # Still not what we want.
    #
    # Let me just use schoolbook for 2x2x3: rank = 2*2*3 = 12
    # For each 2x4x3: rank = 2 * 12 = 24
    # For each 2x4x6: rank = 2 * 24 = 48
    # Total: 2 * 48 = 96 (same as schoolbook)
    #
    # Let me try a different approach. Use Strassen for 2x2x2 parts.
    # For 2x2x2: rank 7 (Strassen)
    # For 2x2x3: decompose into 2x2x2 + 2x2x1
    # 2x2x1: rank = 2*2*1 = 4 (schoolbook)
    # So 2x2x3: rank = 7 + 4 = 11
    # For 2x4x3: rank = 2 * 11 = 22
    # For 2x4x6: rank = 2 * 22 = 44
    # Total: 2 * 44 = 88
    #
    # Let me implement the 2x2x2 Strassen decomposition carefully.
    # 
    # Strassen for 2x2x2:
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A*B
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # In tensor form, each Mi is a bilinear form:
    # M1 = (a11+a22)(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
    # This contributes to C11 with coeff 1, C22 with coeff 1
    #
    # Let me build the full decomposition systematically.
    
    start_time = time.time()
    deadline = start_time + 100
    
    # Build 2x2x2 Strassen decomposition
    # Returns: list of (U_contrib, V_contrib, W_contrib) for each rank-1 term
    # U is indexed by (a_row, a_col) -> flat index a_row*2 + a_col
    # V is indexed by (b_row, b_col) -> flat index b_row*2 + b_col  
    # W is indexed by (c_row, c_col) -> flat index c_row*2 + c_col
    
    def strassen_2x2x2():
        """Return list of (u, v, w) where u, v, w are dicts mapping indices to coefficients"""
        terms = []
        
        # M1 = (a11+a22)(b11+b22)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}  # C11 and C22 both get +1
        terms.append((u1, v1, w1))
        
        # M2 = (a21+a22)b11
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(1,0): 1}  # C21 gets +1
        terms.append((u2, v2, w2))
        
        # M3 = a11(b12-b22)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(0,1): 1, (1,1): 1}  # C12 gets +1, C22 gets +1
        terms.append((u3, v3, w3))
        
        # M4 = a22(b21-b11)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (1,0): 1}  # C11 gets +1, C21 gets +1
        terms.append((u4, v4, w4))
        
        # M5 = (a11+a12)b22
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (0,1): 1}  # C11 gets -1, C12 gets +1
        terms.append((u5, v5, w5))
        
        # M6 = (a21-a11)(b11+b12)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}  # C22 gets +1
        terms.append((u6, v6, w6))
        
        # M7 = (a12-a22)(b21+b22)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}  # C11 gets +1
        terms.append((u7, v7, w7))
        
        return terms
    
    # Build 2x2x1 schoolbook decomposition
    # A is 2x2, B is 2x1, C is 2x1
    def schoolbook_2x2x1():
        """Return list of (u, v, w) for 2x2x1 multiplication"""
        terms = []
        for i in range(2):
            for j in range(2):
                u = {(i, j): 1}
                v = {(j, 0): 1}
                w = {(i, 0): 1}
                terms.append((u, v, w))
        return terms
    
    # Build 2x2x3 decomposition: 2x2x2 (Strassen) + 2x2x1 (schoolbook)
    def decompose_2x2x3():
        """Return list of (u, v, w) for 2x2x3 multiplication"""
        terms = []
        
        # First 2 columns: use Strassen 2x2x2
        strassen_terms = strassen_2x2x2()
        for u, v, w in strassen_terms:
            # u: (a_row, a_col) -> coeff
            # v: (b_row, b_col) -> coeff, b_col in {0,1}
            # w: (c_row, c_col) -> coeff, c_col in {0,1}
            terms.append((u, v, w))
        
        # Third column: use schoolbook 2x2x1
        schoolbook_terms = schoolbook_2x2x1()
        for u, v, w in schoolbook_terms:
            # u: (a_row, a_col) -> coeff
            # v: (b_row, 0) -> coeff, but we need b_col = 2
            # w: (c_row, 0) -> coeff, but we need c_col = 2
            v_new = {(b_row, b_col + 1): coeff for (b_row, b_col), coeff in v.items()}
            w_new = {(c_row, c_col + 1): coeff for (c_row, c_col), coeff in w.items()}
            terms.append((u, v_new, w_new))
        
        return terms
    
    # Build 2x4x3 decomposition: two 2x2x3 sub-problems
    def decompose_2x4x3():
        """Return list of (u, v, w) for 2x4x3 multiplication"""
        terms = []
        
        # Split A (2x4) into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
        # Split B (4x3) into B1 (2x3, rows 0-1) and B2 (2x3, rows 2-3)
        # C = A1*B1 + A2*B2
        
        # For A1*B1: use 2x2x3
        terms_1 = decompose_2x2x3()
        for u, v, w in terms_1:
            # u: (a_row, a_col) where a_col in {0,1}
            # v: (b_row, b_col) where b_row in {0,1}, b_col in {0,1,2}
            # w: (c_row, c_col) where c_row in {0,1}, c_col in {0,1,2}
            # These are already in the right coordinate system for the first half
            terms.append((u, v, w))
        
        # For A2*B2: use 2x2x3 with shifted indices
        terms_2 = decompose_2x2x3()
        for u, v, w in terms_2:
            # Shift a_col by 2, b_row by 2
            u_new = {(a_row, a_col + 2): coeff for (a_row, a_col), coeff in u.items()}
            v_new = {(b_row + 2, b_col): coeff for (b_row, b_col), coeff in v.items()}
            terms.append((u_new, v_new, w))
        
        return terms
    
    # Build 2x4x6 decomposition: two 2x4x3 sub-problems
    def decompose_2x4x6():
        """Return list of (u, v, w) for 2x4x6 multiplication"""
        terms = []
        
        # Split B (4x6) into B1 (4x3, cols 0-2) and B2 (4x3, cols 3-5)
        # C = A*B1 + A*B2
        
        # For A*B1: use 2x4x3
        terms_1 = decompose_2x4x3()
        for u, v, w in terms_1:
            terms.append((u, v, w))
        
        # For A*B2: use 2x4x3 with shifted b_col
        terms_2 = decompose_2x4x3()
        for u, v, w in terms_2:
            v_new = {(b_row, b_col + 3): coeff for (b_row, b_col), coeff in v.items()}
            w_new = {(c_row, c_col + 3): coeff for (c_row, c_col), coeff in w.items()}
            terms.append((u, v_new, w_new))
        
        return terms
    
    # Build full 4x4x6 decomposition
    # Split A into A_top (2x4, rows 0-1) and A_bot (2x4, rows 2-3)
    # C_top = A_top * B (2x6), C_bot = A_bot * B (2x6)
    
    all_terms = []
    
    # Top strip: A_top (rows 0-1) * B
    terms_top = decompose_2x4x6()
    for u, v, w in terms_top:
        # u: (a_row, a_col) where a_row in {0,1}
        # v: (b_row, b_col) where b_row in {0,1,2,3}, b_col in {0..5}
        # w: (c_row, c_col) where c_row in {0,1}, c_col in {0..5}
        all_terms.append((u, v, w))
    
    # Bottom strip: A_bot (rows 2-3) * B
    terms_bot = decompose_2x4x6()
    for u, v, w in terms_bot:
        # Shift a_row and c_row by 2
        u_new = {(a_row + 2, a_col): coeff for (a_row, a_col), coeff in u.items()}
        w_new = {(c_row + 2, c_col): coeff for (c_row, c_col), coeff in w.items()}
        all_terms.append((u_new, v, w_new))
    
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 24 x rank
    W = [[0] * rank for _ in range(p * n)]  # 24 x rank
    
    for r, (u, v, w) in enumerate(all_terms):
        for (a_row, a_col), coeff in u.items():
            U[a_row * m + a_col][r] = coeff
        for (b_row, b_col), coeff in v.items():
            V[b_row * p + b_col][r] = coeff
        for (c_row, c_col), coeff in w.items():
            W[c_col * n + c_row][r] = coeff
    
    # Verify the decomposition
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        error += abs(val - 1.0)
        
        if error > 1e-9:
            # Fall back to schoolbook
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
