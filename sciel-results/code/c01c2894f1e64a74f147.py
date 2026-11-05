"""Matrix multiplication 6x6x6 via recursive Strassen (7 level-1 terms, schoolbook 3x3 leaves)."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _mul3(U, V):
    """Multiply 3x3 matrices U, V using schoolbook.
    U, V are 9-element lists (row-major). Returns 9-element list."""
    out = [0] * 9
    for i in range(3):
        for k in range(3):
            if U[i * 3 + k] == 0:
                continue
            for j in range(3):
                out[i * 3 + j] += U[i * 3 + k] * V[k * 3 + j]
    return out

def _strassen2(A, B):
    """Multiply 2x2 matrices A, B (4-element lists) using Strassen. Returns 4-element list."""
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    a, b, c, d = A
    e, f, g, h = B
    M1 = _mul3([a, 0, 0, 0, b, 0, 0, 0, 0], [e, f, 0, g, h, 0, 0, 0, 0])
    M2 = _mul3([0, b, 0, 0, d, 0, 0, 0, 0], [e, f, 0, g, h, 0, 0, 0, 0])
    M3 = _mul3([a, 0, 0, c, 0, 0, 0, 0, 0], [g, h, 0, 0, 0, 0, 0, 0, 0])
    M4 = _mul3([0, 0, 0, 0, d, 0, 0, 0, 0], [g, h, 0, 0, 0, 0, 0, 0, 0])
    M5 = _mul3([a, 0, 0, 0, 0, 0, 0, 0, 0], [e, 0, 0, 0, 0, 0, 0, 0, 0])
    M6 = _mul3([0, 0, 0, 0, d, 0, 0, 0, 0], [0, f, 0, 0, 0, 0, 0, 0, 0])
    M7 = _mul3([0, b, 0, c, 0, 0, 0, 0, 0], [e, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Actually let me do this more carefully with scalar operations
    # For 2x2 Strassen with scalar leaves:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+d)(f-h)
    # M7 = (a-b)(e+g)
    # P11 = M1+M4-M5+M7
    # P12 = M3+M5
    # P21 = M2+M4
    # P22 = M1-M2+M3+M6
    
    # But our leaves are 3x3 matrices, so we need matrix multiplication for the products
    # Let me redo: each "scalar" is actually a 3x3 matrix
    # a, b, c, d, e, f, g, h are each 3x3 matrices (9 elements)
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+d)(f-h)
    # M7 = (a-b)(e+g)
    
    def add3(X, Y):
        return [X[i] + Y[i] for i in range(9)]
    
    def sub3(X, Y):
        return [X[i] - Y[i] for i in range(9)]
    
    a, b, c, d = A[0:9], A[9:18], A[18:27], A[27:36]
    e, f, g, h = B[0:9], B[9:18], B[18:27], B[27:36]
    
    M1 = _mul3(add3(a, d), add3(e, h))
    M2 = _mul3(add3(c, d), e)
    M3 = _mul3(a, sub3(f, h))
    M4 = _mul3(d, sub3(g, e))
    M5 = _mul3(add3(a, b), h)
    M6 = _mul3(add3(c, d), sub3(f, h))
    M7 = _mul3(sub3(a, b), add3(e, g))
    
    P11 = [M1[i] + M4[i] - M5[i] + M7[i] for i in range(9)]
    P12 = [M3[i] + M5[i] for i in range(9)]
    P21 = [M2[i] + M4[i] for i in range(9)]
    P22 = [M1[i] - M2[i] + M3[i] + M6[i] for i in range(9)]
    
    return P11 + P12 + P21 + P22

def _strassen4(A, B):
    """Multiply 4x4 matrices A, B (16-element lists) using Strassen. Returns 16-element list.
    Each leaf is a 2x2 matrix (4 elements), so we use _strassen2 for products."""
    def add4(X, Y):
        return [X[i] + Y[i] for i in range(4)]
    def sub4(X, Y):
        return [X[i] - Y[i] for i in range(4)]
    
    a, b, c, d = A[0:4], A[4:8], A[8:12], A[12:16]
    e, f, g, h = B[0:4], B[4:8], B[8:12], B[12:16]
    
    M1 = _strassen2(add4(a, d), add4(e, h))
    M2 = _strassen2(add4(c, d), e)
    M3 = _strassen2(a, sub4(f, h))
    M4 = _strassen2(d, sub4(g, e))
    M5 = _strassen2(add4(a, b), h)
    M6 = _strassen2(add4(c, d), sub4(f, h))
    M7 = _strassen2(sub4(a, b), add4(e, g))
    
    P11 = [M1[i] + M4[i] - M5[i] + M7[i] for i in range(4)]
    P12 = [M3[i] + M5[i] for i in range(4)]
    P21 = [M2[i] + M4[i] for i in range(4)]
    P22 = [M1[i] - M2[i] + M3[i] + M6[i] for i in range(4)]
    
    return P11 + P12 + P21 + P22

def _strassen6(A, B):
    """Multiply 6x6 matrices A, B (36-element lists) using Strassen. Returns 36-element list.
    Each leaf is a 3x3 matrix (9 elements), so we use _mul3 for products."""
    def add9(X, Y):
        return [X[i] + Y[i] for i in range(9)]
    def sub9(X, Y):
        return [X[i] - Y[i] for i in range(9)]
    
    a, b, c, d = A[0:9], A[9:18], A[18:27], A[27:36]
    e, f, g, h = B[0:9], B[9:18], B[18:27], B[27:36]
    
    M1 = _mul3(add9(a, d), add9(e, h))
    M2 = _mul3(add9(c, d), e)
    M3 = _mul3(a, sub9(f, h))
    M4 = _mul3(d, sub9(g, e))
    M5 = _mul3(add9(a, b), h)
    M6 = _mul3(add9(c, d), sub9(f, h))
    M7 = _mul3(sub9(a, b), add9(e, g))
    
    P11 = [M1[i] + M4[i] - M5[i] + M7[i] for i in range(9)]
    P12 = [M3[i] + M5[i] for i in range(9)]
    P21 = [M2[i] + M4[i] for i in range(9)]
    P22 = [M1[i] - M2[i] + M3[i] + M6[i] for i in range(9)]
    
    return P11 + P12 + P21 + P22

def construct():
    """Construct rank-189 tensor decomposition for 6x6x6 matrix multiplication using recursive Strassen."""
    n, m, p = DIMENSIONS  # 6, 6, 6
    rank = 189  # 7 * 27
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen decomposition:
    # Split 6x6 into four 3x3 blocks:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[P11, P12], [P21, P22]]
    # where:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+d)(f-h)
    # M7 = (a-b)(e+g)
    # P11 = M1+M4-M5+M7
    # P12 = M3+M5
    # P21 = M2+M4
    # P22 = M1-M2+M3+M6
    
    # Each Mj is a 3x3 matrix product, which has rank 9 (schoolbook)
    # Total rank = 7 * 9 = 63... wait, that's not right.
    
    # Actually, each 3x3 schoolbook multiplication has rank 27 (3*3*3)
    # So total rank = 7 * 27 = 189. Correct.
    
    # Let me define the Strassen terms more carefully.
    # For each Strassen term Mj, we have a 3x3 matrix multiplication.
    # The 3x3 multiplication tensor has rank 27 with the schoolbook decomposition.
    
    # Let's define helper to add a schoolbook 3x3 multiplication to U, V, W
    # at a given offset in the rank dimension.
    
    def add_schoolbook_3x3(U, V, W, rank_offset, A_rows, B_cols, C_rows, C_cols, sign=1):
        """Add a signed schoolbook 3x3 matrix multiplication.
        A_rows: list of 9 indices (row-major) into U for the A matrix
        B_cols: list of 9 indices (row-major) into V for the B matrix  
        C_rows: list of 9 indices (row-major) into W for the C matrix
        C_cols: list of 9 indices (row-major) into W for the C matrix
        """
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    col = rank_offset + ii * 3 + jj * 3 + kk
                    # A element at (ii, jj)
                    U[A_rows[ii * 3 + jj]][col] += sign
                    # B element at (jj, kk)
                    V[B_cols[jj * 3 + kk]][col] += sign
                    # C element at (ii, kk)
                    W[C_rows[ii * 3 + kk]][col] += sign
    
    # Now define the block structure for 6x6 matrices.
    # A is 6x6, indexed as A[i,j] for i,j in 0..5
    # Split into 3x3 blocks:
    # a = A[0:3, 0:3], b = A[0:3, 3:6]
    # c = A[3:6, 0:3], d = A[3:6, 3:6]
    
    # In the U matrix (36 x rank), row index = i * 6 + j for A[i,j]
    # In the V matrix (36 x rank), row index = j * 6 + k for B[j,k]
    # In the W matrix (36 x rank), row index = k * 6 + i for C[k,i] (trace convention)
    
    # Block a: A[0:3, 0:3] -> rows 0,1,2,6,7,8,12,13,14 in U
    # Block b: A[0:3, 3:6] -> rows 3,4,5,9,10,11,15,16,17 in U
    # Block c: A[3:6, 0:3] -> rows 18,19,20,24,25,26,30,31,32 in U
    # Block d: A[3:6, 3:6] -> rows 21,22,23,27,28,29,33,34,35 in U
    
    # Block e: B[0:3, 0:3] -> rows 0,1,2,6,7,8,12,13,14 in V
    # Block f: B[0:3, 3:6] -> rows 3,4,5,9,10,11,15,16,17 in V
    # Block g: B[3:6, 0:3] -> rows 18,19,20,24,25,26,30,31,32 in V
    # Block h: B[3:6, 3:6] -> rows 21,22,23,27,28,29,33,34,35 in V
    
    # Block P11: C[0:3, 0:3] -> W rows: k*6+i for k in 0:3, i in 0:3 -> 0,1,2,6,7,8,12,13,14
    # Block P12: C[0:3, 3:6] -> W rows: k*6+i for k in 0:3, i in 3:6 -> 3,4,5,9,10,11,15,16,17
    # Block P21: C[3:6, 0:3] -> W rows: k*6+i for k in 3:6, i in 0:3 -> 18,19,20,24,25,26,30,31,32
    # Block P22: C[3:6, 3:6] -> W rows: k*6+i for k in 3:6, i in 3:6 -> 21,22,23,27,28,29,33,34,35
    
    # Wait, I need to be more careful. Let me define the row indices for each 3x3 block.
    
    def block_rows(offset_i, offset_j, size=3):
        """Get row indices for a size x size block starting at (offset_i, offset_j) in a 6x6 matrix."""
        rows = []
        for ii in range(size):
            for jj in range(size):
                rows.append((offset_i + ii) * 6 + (offset_j + jj))
        return rows
    
    # U rows for A blocks
    a_rows = block_rows(0, 0)  # A[0:3, 0:3]
    b_rows = block_rows(0, 3)  # A[0:3, 3:6]
    c_rows = block_rows(3, 0)  # A[3:6, 0:3]
    d_rows = block_rows(3, 3)  # A[3:6, 3:6]
    
    # V rows for B blocks
    e_rows = block_rows(0, 0)  # B[0:3, 0:3]
    f_rows = block_rows(0, 3)  # B[0:3, 3:6]
    g_rows = block_rows(3, 0)  # B[3:6, 0:3]
    h_rows = block_rows(3, 3)  # B[3:6, 3:6]
    
    # W rows for C blocks (trace convention: W[k*n+i])
    # C[i,j] -> W[j*6+i]
    P11_rows = block_rows(0, 0)  # C[0:3, 0:3] -> W[j*6+i] for j in 0:3, i in 0:3
    P12_rows = block_rows(3, 0)  # C[3:6, 0:3] -> W[j*6+i] for j in 3:6, i in 0:3
    # Wait, P12 is C[0:3, 3:6], so i in 0:3, j in 3:6
    # W[j*6+i] for j in 3:6, i in 0:3
    P12_rows = [j * 6 + i for j in range(3, 6) for i in range(0, 3)]
    P21_rows = [j * 6 + i for j in range(0, 3) for i in range(3, 6)]  # C[3:6, 0:3], i in 3:6, j in 0:3
    P22_rows = [j * 6 + i for j in range(3, 6) for i in range(3, 6)]  # C[3:6, 3:6], i in 3:6, j in 3:6
    
    # Hmm, let me reconsider. The W matrix has rows indexed by (k, i) where k is the column of C and i is the row of C.
    # W[k*6 + i] corresponds to C[i, k].
    
    # P11 = C[0:3, 0:3]: i in 0:3, k in 0:3 -> W[k*6+i] for k in 0:3, i in 0:3
    P11_W = [k * 6 + i for k in range(3) for i in range(3)]
    # P12 = C[0:3, 3:6]: i in 0:3, k in 3:6 -> W[k*6+i] for k in 3:6, i in 0:3
    P12_W = [k * 6 + i for k in range(3, 6) for i in range(3)]
    # P21 = C[3:6, 0:3]: i in 3:6, k in 0:3 -> W[k*6+i] for k in 0:3, i in 3:6
    P21_W = [k * 6 + i for k in range(3) for i in range(3, 6)]
    # P22 = C[3:6, 3:6]: i in 3:6, k in 3:6 -> W[k*6+i] for k in 3:6, i in 3:6
    P22_W = [k * 6 + i for k in range(3, 6) for i in range(3, 6)]
    
    # Now, for each Strassen term Mj, we need to express it as a 3x3 matrix multiplication.
    # The A-matrix for Mj is a linear combination of the 3x3 blocks a, b, c, d.
    # The B-matrix for Mj is a linear combination of the 3x3 blocks e, f, g, h.
    # The result contributes to one or more of P11, P12, P21, P22.
    
    # M1 = (a+d)(e+h): A = a+d, B = e+h, contributes to P11 and P22
    # M2 = (c+d)e: A = c+d, B = e, contributes to P21 and P22 (with negative sign)
    # M3 = a(f-h): A = a, B = f-h, contributes to P12 and P22
    # M4 = d(g-e): A = d, B = g-e, contributes to P11, P21, P22
    # M5 = (a+b)h: A = a+b, B = h, contributes to P11 (negative), P12
    # M6 = (c+d)(f-h): A = c+d, B = f-h, contributes to P22
    # M7 = (a-b)(e+g): A = a-b, B = e+g, contributes to P11
    
    # For each term, the A-matrix is a sum of blocks with coefficients +/-1.
    # We can express this as a sum of schoolbook 3x3 multiplications.
    
    # Let me define a helper to add a term where A is a linear combination and B is a linear combination.
    
    def add_term(U, V, W, rank_offset, A_coeffs, B_coeffs, C_target, C_sign):
        """Add a Strassen term.
        A_coeffs: dict mapping block name ('a','b','c','d') to coefficient
        B_coeffs: dict mapping block name ('e','f','g','h') to coefficient
        C_target: which P block this contributes to ('P11','P12','P21','P22')
        C_sign: sign of contribution
        """
        A_blocks = {'a': a_rows, 'b': b_rows, 'c': c_rows, 'd': d_rows}
        B_blocks = {'e': e_rows, 'f': f_rows, 'g': g_rows, 'h': h_rows}
        C_blocks = {'P11': P11_W, 'P12': P12_W, 'P21': P21_W, 'P22': P22_W}
        
        # For each combination of A block and B block with nonzero coefficients
        for a_name, a_coeff in A_coeffs.items():
            for b_name, b_coeff in B_coeffs.items():
                if a_coeff == 0 or b_coeff == 0:
                    continue
                coeff = a_coeff * b_coeff * C_sign
                if coeff == 0:
                    continue
                add_schoolbook_3x3(U, V, W, rank_offset, A_blocks[a_name], B_blocks[b_name], 
                                   C_blocks[C_target], C_blocks[C_target], sign=coeff)
    
    # Now add all 7 Strassen terms
    offset = 0
    
    # M1 = (a+d)(e+h) -> P11 (+1), P22 (+1)
    A1 = {'a': 1, 'd': 1}
    B1 = {'e': 1, 'h': 1}
    add_term(U, V, W, offset, A1, B1, 'P11', 1)
    add_term(U, V, W, offset, A1, B1, 'P22', 1)
    offset += 27
    
    # M2 = (c+d)e -> P21 (+1), P22 (-1)
    A2 = {'c': 1, 'd': 1}
    B2 = {'e': 1}
    add_term(U, V, W, offset, A2, B2, 'P21', 1)
    add_term(U, V, W, offset, A2, B2, 'P22', -1)
    offset += 27
    
    # M3 = a(f-h) -> P12 (+1), P22 (+1)
    A3 = {'a': 1}
    B3 = {'f': 1, 'h': -1}
    add_term(U, V, W, offset, A3, B3, 'P12', 1)
    add_term(U, V, W, offset, A3, B3, 'P22', 1)
    offset += 27
    
    # M4 = d(g-e) -> P11 (+1), P21 (+1), P22 (-1)
    A4 = {'d': 1}
    B4 = {'g': 1, 'e': -1}
    add_term(U, V, W, offset, A4, B4, 'P11', 1)
    add_term(U, V, W, offset, A4, B4, 'P21', 1)
    add_term(U, V, W, offset, A4, B4, 'P22', -1)
    offset += 27
    
    # M5 = (a+b)h -> P11 (-1), P12 (+1)
    A5 = {'a': 1, 'b': 1}
    B5 = {'h': 1}
    add_term(U, V, W, offset, A5, B5, 'P11', -1)
    add_term(U, V, W, offset, A5, B5, 'P12', 1)
    offset += 27
    
    # M6 = (c+d)(f-h) -> P22 (+1)
    A6 = {'c': 1, 'd': 1}
    B6 = {'f': 1, 'h': -1}
    add_term(U, V, W, offset, A6, B6, 'P22', 1)
    offset += 27
    
    # M7 = (a-b)(e+g) -> P11 (+1)
    A7 = {'a': 1, 'b': -1}
    B7 = {'e': 1, 'g': 1}
    add_term(U, V, W, offset, A7, B7, 'P11', 1)
    offset += 27
    
    assert offset == rank, f"Expected rank {rank}, got offset {offset}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
