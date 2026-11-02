import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    def get_u_idx(i, j):
        return i * m + j

    def get_v_idx(j, k):
        return j * p + k

    def get_w_idx(k, i):
        return k * n + i

    # Helper to add 7 Strassen terms for a 2x2x2 multiplication
    # The 2x2 blocks are located at (ai, aj) in A, (bj, bk) in B, and (ci, ck) in C
    # Note: C indices are (row, col). W index is (col, row) flattened.
    # We assume the 2x2 blocks are contiguous 2x2 submatrices.
    def add_strassen_2x2(ai, aj, bj, bk, ci, ck, scale_u=1, scale_v=1, scale_w=1):
        # u indices: (ai, aj), (ai, aj+1), (ai+1, aj), (ai+1, aj+1)
        # v indices: (bj, bk), (bj, bk+1), (bj+1, bk), (bj+1, bk+1)
        # w indices: (ck, ci), (ck, ci+1), (ck+1, ci), (ck+1, ci+1)
        
        # 1. (a+d)(e+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = scale_u
        u[get_u_idx(ai+1, aj+1)] = scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = scale_v
        v[get_v_idx(bj+1, bk+1)] = scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk, ci)] = scale_w
        w[get_w_idx(bk+1, ci+1)] = scale_w
        add_term(u, v, w)
        
        # 2. (c+d)e
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = scale_u
        u[get_u_idx(ai+1, aj+1)] = scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk, ci+1)] = scale_w
        w[get_w_idx(bk+1, ci+1)] = -scale_w
        add_term(u, v, w)
        
        # 3. a(f-h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+1)] = scale_v
        v[get_v_idx(bj+1, bk+1)] = -scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ci)] = scale_w
        w[get_w_idx(bk+1, ci+1)] = scale_w
        add_term(u, v, w)
        
        # 4. d(g-e)
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj+1)] = scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = scale_v
        v[get_v_idx(bj, bk)] = -scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk, ci)] = scale_w
        w[get_w_idx(bk, ci+1)] = scale_w
        add_term(u, v, w)
        
        # 5. (a+b)h
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = scale_u
        u[get_u_idx(ai, aj+1)] = scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+1)] = scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk, ci)] = -scale_w
        w[get_w_idx(bk+1, ci)] = scale_w
        add_term(u, v, w)
        
        # 6. (c-a)(e+f)
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = scale_u
        u[get_u_idx(ai, aj)] = -scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = scale_v
        v[get_v_idx(bj, bk+1)] = scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ci+1)] = scale_w
        add_term(u, v, w)
        
        # 7. (b-d)(g+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj+1)] = scale_u
        u[get_u_idx(ai+1, aj+1)] = -scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = scale_v
        v[get_v_idx(bj+1, bk+1)] = scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk, ci)] = scale_w
        add_term(u, v, w)

    # The 4x4x4 part
    # We decompose the 4x4 multiplication into 7 terms, each of which is a 2x2x2 multiplication.
    # However, the "2x2x2" here is not a simple block multiplication, but a linear combination.
    # The standard Strassen for 4x4 uses 7 products of 2x2 matrices.
    # Let's call the 2x2 blocks of A: A00, A01, A10, A11.
    # And B: B00, B01, B10, B11.
    # The 7 products are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need to expand each Mi into 7 rank-1 terms.
    # This is equivalent to applying the 2x2x2 Strassen decomposition to the "virtual" 2x2 matrices.
    # But the "virtual" matrices are sums/differences of blocks.
    # We can handle this by modifying the U, V, W vectors to include contributions from multiple blocks.
    
    # Let's create a generic function to add 7 terms for a product of two "virtual" 2x2 matrices.
    # The virtual A matrix has entries that are linear combinations of A blocks.
    # The virtual B matrix has entries that are linear combinations of B blocks.
    # The result C block is a 2x2 block of the final 4x4 matrix.
    
    # We define the 7 super-terms by specifying:
    # - The linear combination of A blocks for each of the 4 positions (a,b,c,d) of the virtual 2x2 A.
    # - The linear combination of B blocks for each of the 4 positions (e,f,g,h) of the virtual 2x2 B.
    # - The linear combination of C blocks for each of the 4 positions of the virtual 2x2 C.
    
    # This is getting complicated. Let's use a simpler approach:
    # Just iterate through the 7 super-terms and for each, call a helper that adds 7 terms,
    # where the helper takes a list of (block_idx, coeff) for each of the 4 positions in A, B, and C.
    
    # Block indices:
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
    # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)
    
    # Super-term definitions:
    # M1: A' = A00+A11, B' = B00+B11, C' = C00+C11 (with signs from Strassen formula)
    # Actually, the Strassen formula for C is:
    # C00 = M1 - M3 + M4 - M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    
    # So for M1, it contributes +1 to C00 and +1 to C11.
    # For M2, it contributes +1 to C10 and -1 to C11.
    # Etc.
    
    # Let's define a function to add 7 terms for a super-term.
    # The function will take:
    # a_coeffs: list of 4 lists, each containing (block_start_row, block_start_col, coeff)
    # b_coeffs: list of 4 lists, each containing (block_start_row, block_start_col, coeff)
    # c_coeffs: list of 4 lists, each containing (block_start_row, block_start_col, coeff)
    
    def add_super_strassen(a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs[i] is a list of (row, col, coeff) for the i-th position of the virtual 2x2 A.
        # Position 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1) in the virtual 2x2.
        
        # 1. (a+d)(e+h)
        u = np.zeros(n*m)
        for r, c, coeff in a_coeffs[0]: u[get_u_idx(r, c)] += coeff
        for r, c, coeff in a_coeffs[3]: u[get_u_idx(r, c)] += coeff
        v = np.zeros(m*p)
        for r, c, coeff in b_coeffs[0]: v[get_v_idx(r, c)] += coeff
        for r, c, coeff in b_coeffs[3]: v[get_v_idx(r, c)] += coeff
        w = np.zeros(p*n)
        for r, c, coeff in c_coeffs[0]: w[get_w_idx(c, r)] += coeff
        for r, c, coeff in c_coeffs[3]: w[get_w_idx(c, r)] += coeff
        add_term(u, v, w)
        
        # 2. (c+d)e
        u = np.zeros(n*m)
        for r, c, coeff in a_coeffs[2]: u[get_u_idx(r, c)] += coeff
        for r, c, coeff in a_coeffs[3]: u[get_u_idx(r, c)] += coeff
        v = np.zeros(m*p)
        for r, c, coeff in b_coeffs[0]: v[get_v_idx(r, c)] += coeff
        w = np.zeros(p*n)
        for r, c, coeff in c_coeffs[2]: w[get_w_idx(c, r)] += coeff
        for r, c, coeff in c_coeffs[3]: w[get_w_idx(c, r)] -= coeff
        add_term(u, v, w)
        
        # 3. a(f-h)
        u = np.zeros(n*m)
        for r, c, coeff in a_coeffs[0]: u[get_u_idx(r, c)] += coeff
        v = np.zeros(m*p)
        for r, c, coeff in b_coeffs[1]: v[get_v_idx(r, c)] += coeff
        for r, c, coeff in b_coeffs[3]: v[get_v_idx(r, c)] -= coeff
        w = np.zeros(p*n)
        for r, c, coeff in c_coeffs[1]: w[get_w_idx(c, r)] += coeff
        for r, c, coeff in c_coeffs[3]: w[get_w_idx(c, r)] += coeff
        add_term(u, v, w)
        
        # 4. d(g-e)
        u = np.zeros(n*m)
        for r, c, coeff in a_coeffs[3]: u[get_u_idx(r, c)] += coeff
        v = np.zeros(m*p)
        for r, c, coeff in b_coeffs[2]: v[get_v_idx(r, c)] += coeff
        for r, c, coeff in b_coeffs[0]: v[get_v_idx(r, c)] -= coeff
        w = np.zeros(p*n)
        for r, c, coeff in c_coeffs[0]: w[get_w_idx(c, r)] += coeff
        for r, c, coeff in c_coeffs[2]: w[get_w_idx(c, r)] += coeff
        add_term(u, v, w)
        
        # 5. (a+b)h
        u = np.zeros(n*m)
        for r, c, coeff in a_coeffs[0]: u[get_u_idx(r, c)] += coeff
        for r, c, coeff in a_coeffs[1]: u[get_u_idx(r, c)] += coeff
        v = np.zeros(m*p)
        for r, c, coeff in b_coeffs[3]: v[get_v_idx(r, c)] += coeff
        w = np.zeros(p*n)
        for r, c, coeff in c_coeffs[0]: w[get_w_idx(c, r)] -= coeff
        for r, c, coeff in c_coeffs[2]: w[get_w_idx(c, r)] += coeff
        add_term(u, v, w)
        
        # 6. (c-a)(e+f)
        u = np.zeros(n*m)
        for r, c, coeff in a_coeffs[2]: u[get_u_idx(r, c)] += coeff
        for r, c, coeff in a_coeffs[0]: u[get_u_idx(r, c)] -= coeff
        v = np.zeros(m*p)
        for r, c, coeff in b_coeffs[0]: v[get_v_idx(r, c)] += coeff
        for r, c, coeff in b_coeffs[1]: v[get_v_idx(r, c)] += coeff
        w = np.zeros(p*n)
        for r, c, coeff in c_coeffs[3]: w[get_w_idx(c, r)] += coeff
        add_term(u, v, w)
        
        # 7. (b-d)(g+h)
        u = np.zeros(n*m)
        for r, c, coeff in a_coeffs[1]: u[get_u_idx(r, c)] += coeff
        for r, c, coeff in a_coeffs[3]: u[get_u_idx(r, c)] -= coeff
        v = np.zeros(m*p)
        for r, c, coeff in b_coeffs[2]: v[get_v_idx(r, c)] += coeff
        for r, c, coeff in b_coeffs[3]: v[get_v_idx(r, c)] += coeff
        w = np.zeros(p*n)
        for r, c, coeff in c_coeffs[0]: w[get_w_idx(c, r)] += coeff
        add_term(u, v, w)

    # Define the 7 super-terms
    # A blocks: A00(0,0), A01(0,2), A10(2,0), A11(2,2)
    # B blocks: B00(0,0), B01(0,2), B10(2,0), B11(2,2)
    # C blocks: C00(0,0), C01(0,2), C10(2,0), C11(2,2)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    a1 = [[(0,0,1), (2,2,1)], [(0,1,1), (2,3,1)], [(1,0,1), (3,2,1)], [(1,1,1), (3,3,1)]]
    b1 = [[(0,0,1), (2,2,1)], [(0,1,1), (2,3,1)], [(1,0,1), (3,2,1)], [(1,1,1), (3,3,1)]]
    c1 = [[(0,0,1), (2,2,1)], [(0,1,1), (2,3,1)], [(1,0,1), (3,2,1)], [(1,1,1), (3,3,1)]]
    add_super_strassen(a1, b1, c1)
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    a2 = [[(2,0,1), (2,2,1)], [(2,1,1), (2,3,1)], [(3,0,1), (3,2,1)], [(3,1,1), (3,3,1)]]
    b2 = [[(0,0,1)], [(0,1,1)], [(1,0,1)], [(1,1,1)]]
    c2 = [[(0,0,0)], [(0,1,0)], [(2,0,1), (2,2,-1)], [(2,1,0)], [(3,0,1), (3,2,-1)], [(3,1,0)], [(0,2,0)], [(1,2,0)], [(2,2,-1)], [(3,2,0)], [(0,3,0)], [(1,3,0)], [(2,3,0)], [(3,3,0)]]
    # Wait, c_coeffs needs to be 4 lists for the 4 positions of the virtual 2x2 C.
    # Position 0: C00. Position 1: C01. Position 2: C10. Position 3: C11.
    # M2 contributes to C10 (pos 2) and C11 (pos 3).
    c2 = [[], [], [(2,0,1), (2,2,-1)], [(2,2,-1)]] # This is wrong. C10 is a 2x2 block.
    # The virtual C is 2x2. Its positions are 0,1,2,3.
    # Position 2 is the bottom-left 2x2 block of the 4x4 C, which is C10.
    # Position 3 is the bottom-right 2x2 block of the 4x4 C, which is C11.
    # So for M2, c_coeffs[2] should have the contributions to C10, and c_coeffs[3] to C11.
    # C10 is at (2,0)..(3,1). C11 is at (2,2)..(3,3).
    # So c_coeffs[2] = [(2,0,1), (2,1,1), (3,0,1), (3,1,1)]? No, the coefficients are for the *whole* block.
    # In `add_super_strassen`, `c_coeffs[i]` is a list of (row, col, coeff) for the *individual entries* of the virtual 2x2 C block?
    # No, the virtual C block is 2x2. The entries are C00, C01, C10, C11 of the virtual matrix.
    # But in the global 4x4 matrix, these correspond to 2x2 blocks.
    # So `c_coeffs[0]` corresponds to the top-left entry of the virtual C, which is the top-left 2x2 block of the global C (C00).
    # So `c_coeffs[0]` should contain the indices of the 4 entries of C00.
    # C00 entries: (0,0), (0,1), (1,0), (1,1).
    # So `c_coeffs[0] = [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]`.
    
    # Let's redefine the c_coeffs for M1:
    # M1 contributes to C00 and C11.
    # c_coeffs[0] (C00): [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]
    # c_coeffs[3] (C11): [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]
    # c_coeffs[1] and c_coeffs[2] are empty.
    
    # Let's redefine a_coeffs and b_coeffs similarly.
    # a_coeffs[0] (A00): [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]
    # a_coeffs[3] (A11): [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]
    
    # This is much cleaner.
    
    # Redo M1
    a1 = [[(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [], [], [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]]
    b1 = [[(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [], [], [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]]
    c1 = [[(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [], [], [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]]
    add_super_strassen(a1, b1, c1)
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    a2 = [[], [], [(2,0,1), (2,1,1), (3,0,1), (3,1,1)], [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]]
    b2 = [[(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [], [], []]
    c2 = [[], [], [(2,0,1), (2,1,1), (3,0,1), (3,1,1)], [(2,2,-1), (2,3,-1), (3,2,-1), (3,3,-1)]]
    add_super_strassen(a2, b2, c2)
    
    # M3 = A00(B10-B00) -> C00 -= M3, C10 += M3
    a3 = [[(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [], [], []]
    b3 = [[(0,0,-1), (0,1,-1), (1,0,-1), (1,1,-1)], [], [(2,0,1), (2,1,1), (3,0,1), (3,1,1)], []]
    c3 = [[(0,0,-1), (0,1,-1), (1,0,-1), (1,1,-1)], [], [(2,0,1), (2,1,1), (3,0,1), (3,1,1)], []]
    add_super_strassen(a3, b3, c3)
    
    # M4 = A11(B01-B10) -> C00 += M4, C10 += M4
    a4 = [[], [], [], [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]]
    b4 = [[], [(0,2,1), (0,3,1), (1,2,1), (1,3,1)], [(2,0,-1), (2,1,-1), (3,0,-1), (3,1,-1)], []]
    c4 = [[(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [], [(2,0,1), (2,1,1), (3,0,1), (3,1,1)], []]
    add_super_strassen(a4, b4, c4)
    
    # M5 = (A00+A01)B11 -> C01 += M5, C11 += M5
    a5 = [[(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [(0,2,1), (0,3,1), (1,2,1), (1,3,1)], [], []]
    b5 = [[], [], [], [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]]
    c5 = [[], [(0,2,1), (0,3,1), (1,2,1), (1,3,1)], [], [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]]
    add_super_strassen(a5, b5, c5)
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    a6 = [[(0,0,-1), (0,1,-1), (1,0,-1), (1,1,-1)], [], [(2,0,1), (2,1,1), (3,0,1), (3,1,1)], []]
    b6 = [[(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [(0,2,1), (0,3,1), (1,2,1), (1,3,1)], [], []]
    c6 = [[], [], [], [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]]
    add_super_strassen(a6, b6, c6)
    
    # M7 = (A01-A11)(B10+B11) -> C00 -= M7, C01 += M7
    a7 = [[], [(0,2,1), (0,3,1), (1,2,1), (1,3,1)], [], [(2,2,-1), (2,3,-1), (3,2,-1), (3,3,-1)]]
    b7 = [[], [], [(2,0,1), (2,1,1), (3,0,1), (3,1,1)], [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]]
    c7 = [[(0,0,-1), (0,1,-1), (1,0,-1), (1,1,-1)], [(0,2,1), (0,3,1), (1,2,1), (1,3,1)], [], []]
    add_super_strassen(a7, b7, c7)

    # Add the 16 terms for the 5th column (col 4)
    for i in range(4):
        for j in range(4):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
