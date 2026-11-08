import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 7
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off):
        """
        Standard Strassen for 2x2 matrices.
        A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-e)
        M5 = (a+b)h
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        
        C = [[M1+M4-M5+M7, M3+M5],
             [M2+M4,       M1-M2+M3+M6]]
        """
        
        # M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        add_term(u, v, w)

        # M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        add_term(u, v, w)

        # M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        add_term(u, v, w)

    def strassen_4x4_head():
        """
        Apply Strassen to 4x4x4 multiplication.
        A is 4x4, B is 4x4.
        Split into 2x2 blocks of 2x2 submatrices.
        A = [[A00, A01], [A10, A11]]
        B = [[B00, B01], [B10, B11]]
        
        Use Strassen on the 2x2 block matrix:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1+M4-M5+M7
        C01 = M3+M5
        C10 = M2+M4
        C11 = M1-M2+M3+M6
        
        Each M is a product of 2x2 matrices, which we compute using strassen_2x2_block.
        However, the inputs to strassen_2x2_block are sums/differences of blocks.
        We need to expand each M into 7 terms (from the inner Strassen).
        Total 7*7 = 49 terms.
        """
        
        # Define the 7 outer Strassen products for 4x4x4
        # Each product is of the form (Sum of A blocks) * (Sum of B blocks)
        # We need to distribute the multiplication over the addition.
        # Since multiplication is bilinear, (A00+A11)*B00 = A00*B00 + A11*B00
        # But we want to use Strassen on the 2x2 blocks, so we treat each "block" as a 2x2 matrix.
        
        # Let's define the 7 outer terms as lists of (block_idx, coeff) for A and B.
        # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        outer_terms = [
            # M1: (A00+A11)(B00+B11)
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2: (A10+A11)B00
            ([(2, 1), (3, 1)], [(0, 1)]),
            # M3: A00(B01-B11)
            ([(0, 1)], [(1, 1), (3, -1)]),
            # M4: A11(B10-B00)
            ([(3, 1)], [(2, 1), (0, -1)]),
            # M5: (A00+A01)B11
            ([(0, 1), (1, 1)], [(3, 1)]),
            # M6: (A10-A00)(B00+B01)
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
            # M7: (A01-A11)(B10+B11)
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        ]
        
        # For each outer term, we have a sum of A blocks and a sum of B blocks.
        # The product is a sum of products of individual blocks.
        # Each individual product is a 2x2x2 multiplication, which we do with Strassen (7 terms).
        
        for a_blocks, b_blocks in outer_terms:
            for a_idx, a_coeff in a_blocks:
                for b_idx, b_coeff in b_blocks:
                    # Compute product of A[a_idx] and B[b_idx]
                    # A[a_idx] is at row block a_idx//2, col block a_idx%2
                    # B[b_idx] is at row block b_idx//2, col block b_idx%2
                    a_r = a_idx // 2
                    a_c = a_idx % 2
                    b_r = b_idx // 2
                    b_c = b_idx % 2
                    
                    # The product contributes to C at block (a_r, b_c)
                    # We need to add the 7 Strassen terms for this 2x2x2 multiplication
                    # scaled by a_coeff * b_coeff
                    
                    coeff = a_coeff * b_coeff
                    i_off = a_r * 2
                    j_off = a_c * 2
                    k_off = b_r * 2
                    
                    # We need to generate the 7 Strassen terms for this specific block product
                    # and add them to the global terms list, scaled by coeff.
                    # But strassen_2x2_block adds to the global list directly.
                    # We need a version that returns the terms or allows scaling.
                    
                    # Let's modify strassen_2x2_block to accept a scale factor
                    # Or, we can just call a helper that generates the terms.
                    
                    # For simplicity, let's create a helper that generates the 7 terms
                    # for a 2x2 block product at specific offsets.
                    
                    # Actually, the current strassen_2x2_block function adds to the global list.
                    # We can't easily scale it. Let's refactor.
                    
                    pass # This approach is getting complicated. Let's use the block indices directly.

    def strassen_4x4_head_v2():
        """
        Alternative approach: Directly generate the 49 terms for 4x4x4 Strassen.
        """
        # The 49 terms are generated by combining the 7 outer Strassen steps
        # with the 7 inner Strassen steps.
        
        # Let's define the 7 inner Strassen terms for a 2x2x2 multiplication.
        # These are the same as in strassen_2x2_block, but we'll generate them
        # for specific block positions.
        
        # For each of the 7 outer terms, we have a product of (Sum A blocks) * (Sum B blocks).
        # We expand this into a sum of individual block products.
        # Each block product is a 2x2x2 multiplication, which has 7 Strassen terms.
        
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)]),
            ([(0, 1)], [(1, 1), (3, -1)]),
            ([(3, 1)], [(2, 1), (0, -1)]),
            ([(0, 1), (1, 1)], [(3, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        ]
        
        for a_blocks, b_blocks in outer_terms:
            for a_idx, a_coeff in a_blocks:
                for b_idx, b_coeff in b_blocks:
                    a_r = a_idx // 2
                    a_c = a_idx % 2
                    b_r = b_idx // 2
                    b_c = b_idx % 2
                    coeff = a_coeff * b_coeff
                    i_off = a_r * 2
                    j_off = a_c * 2
                    k_off = b_r * 2
                    
                    # Generate the 7 Strassen terms for this 2x2x2 block product
                    # and add them to the global list, scaled by coeff.
                    
                    # M1
                    u = [0] * (n * m)
                    u[i_off * m + j_off] = coeff
                    u[(i_off+1) * m + (j_off+1)] = coeff
                    v = [0] * (m * p)
                    v[j_off * p + k_off] = 1
                    v[(j_off+1) * p + (k_off+1)] = 1
                    w = [0] * (p * n)
                    w[k_off * n + i_off] = 1
                    w[(k_off+1) * n + (i_off+1)] = 1
                    add_term(u, v, w)

                    # M2
                    u = [0] * (n * m)
                    u[(i_off+1) * m + j_off] = coeff
                    u[(i_off+1) * m + (j_off+1)] = coeff
                    v = [0] * (m * p)
                    v[j_off * p + k_off] = 1
                    w = [0] * (p * n)
                    w[k_off * n + (i_off+1)] = 1
                    w[(k_off+1) * n + (i_off+1)] = -1
                    add_term(u, v, w)

                    # M3
                    u = [0] * (n * m)
                    u[i_off * m + j_off] = coeff
                    v = [0] * (m * p)
                    v[j_off * p + (k_off+1)] = 1
                    v[(j_off+1) * p + (k_off+1)] = -1
                    w = [0] * (p * n)
                    w[(k_off+1) * n + i_off] = 1
                    w[(k_off+1) * n + (i_off+1)] = 1
                    add_term(u, v, w)

                    # M4
                    u = [0] * (n * m)
                    u[(i_off+1) * m + (j_off+1)] = coeff
                    v = [0] * (m * p)
                    v[(j_off+1) * p + k_off] = 1
                    v[j_off * p + k_off] = -1
                    w = [0] * (p * n)
                    w[k_off * n + i_off] = 1
                    w[k_off * n + (i_off+1)] = 1
                    add_term(u, v, w)

                    # M5
                    u = [0] * (n * m)
                    u[i_off * m + j_off] = coeff
                    u[i_off * m + (j_off+1)] = coeff
                    v = [0] * (m * p)
                    v[(j_off+1) * p + (k_off+1)] = 1
                    w = [0] * (p * n)
                    w[(k_off+1) * n + i_off] = 1
                    w[k_off * n + i_off] = -1
                    add_term(u, v, w)

                    # M6
                    u = [0] * (n * m)
                    u[(i_off+1) * m + j_off] = coeff
                    u[i_off * m + j_off] = -coeff
                    v = [0] * (m * p)
                    v[j_off * p + k_off] = 1
                    v[j_off * p + (k_off+1)] = 1
                    w = [0] * (p * n)
                    w[(k_off+1) * n + (i_off+1)] = 1
                    add_term(u, v, w)

                    # M7
                    u = [0] * (n * m)
                    u[i_off * m + (j_off+1)] = coeff
                    u[(i_off+1) * m + (j_off+1)] = -coeff
                    v = [0] * (m * p)
                    v[(j_off+1) * p + k_off] = 1
                    v[(j_off+1) * p + (k_off+1)] = 1
                    w = [0] * (p * n)
                    w[k_off * n + i_off] = 1
                    add_term(u, v, w)

    def strassen_4x4x2_tail():
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    strassen_4x4_head_v2()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()

    if time.time() > deadline - 5:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
