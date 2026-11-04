import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Rank 65 decomposition
    # 49 terms for 4x4x4 (Strassen)
    # 16 terms for 4x4x1 (Schoolbook)
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal current_rank
        if current_rank >= rank:
            return
        U[u_idx][current_rank] += u_val
        V[v_idx][current_rank] += v_val
        W[w_idx][current_rank] += w_val
        current_rank += 1

    # Strassen 2x2 terms
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We need to expand 4x4 Strassen.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B01)  # Note: standard is d(g-f). d=A11, g=B10, f=B01.
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each Pi is a product of two 2x2 matrices.
    # We decompose each Pi into 7 terms using 2x2 Strassen.
    
    # Define the 7 2x2 Strassen terms as (u_coeff, v_coeff, w_coeff) for the 2x2 blocks
    # But we need to map to 4x4 indices.
    
    # Let's define the 7 macro products first.
    # For each macro product, we have an A-side matrix (2x2) and B-side matrix (2x2).
    # Then we apply 2x2 Strassen to that pair.
    
    # 2x2 Strassen decomposition of X*Y into C.
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Terms:
    # T1: (x00+x11)(y00+y11) -> C00 += T1, C11 += T1
    # T2: (x10+x11)y00 -> C10 += T2, C11 -= T2
    # T3: x00(y01-y11) -> C01 += T3, C11 += T3
    # T4: x11(y10-y01) -> C00 -= T4, C10 += T4
    # T5: (x00+x01)y11 -> C00 -= T5, C01 += T5
    # T6: (x10-x00)(y00+y01) -> C11 += T6
    # T7: (x01-x11)(y10+y11) -> C00 += T7
    
    # We need to map x_ij and y_kl to the original A and B indices.
    
    # Macro 1: P1 = (A00+A11)(B00+B11)
    # X = [[A00, A11], [A11, A00]]? No.
    # X is a 2x2 matrix formed from A blocks?
    # No, P1 is a product of two 2x2 matrices: X1 = A00+A11, Y1 = B00+B11.
    # Wait, A00 is a 2x2 matrix. A11 is a 2x2 matrix.
    # So X1 is a 2x2 matrix (sum of two 2x2 matrices).
    # Y1 is a 2x2 matrix.
    # So P1 = X1 * Y1.
    # We decompose X1 * Y1 using 2x2 Strassen.
    
    # Let's define the X and Y matrices for each of the 7 macros.
    # X_s and Y_s are 2x2 matrices of coefficients?
    # No, they are 2x2 blocks of A and B.
    # So X_s is a 2x2 matrix where each entry is a 2x2 block of A?
    # No, that's not right.
    # A is 4x4. A00 is top-left 2x2.
    # X1 = A00 + A11. This is a 2x2 matrix.
    # Y1 = B00 + B11. This is a 2x2 matrix.
    # P1 = X1 * Y1.
    # We decompose P1 into 7 terms.
    # Each term is of the form (linear combo of entries of X1) * (linear combo of entries of Y1).
    # The entries of X1 are entries of A00 and A11.
    # So we can express each term as a product of a linear combo of A entries and a linear combo of B entries.
    # And the result contributes to C blocks.
    
    # Let's write a helper to add a 2x2 Strassen decomposition of X*Y.
    # X and Y are 2x2 matrices represented as 4x4 blocks?
    # No, let's represent X and Y as 2x2 matrices of 2x2 blocks?
    # No, X1 is a 2x2 matrix. Its entries are scalars (from A).
    # So we just need to know which A entries contribute to X1's entries.
    
    # Let's define the 7 macros explicitly with their X and Y blocks.
    # X_s is a 2x2 matrix. Y_s is a 2x2 matrix.
    # X_s[i,j] is a linear combination of A blocks?
    # No, X_s is a 2x2 matrix formed from A blocks.
    # Example: X1 = A00 + A11.
    # X1[0,0] = A00[0,0] + A11[0,0]
    # X1[0,1] = A00[0,1] + A11[0,1]
    # etc.
    
    # So for each macro s, we have a 2x2 matrix X_s and Y_s.
    # We apply 2x2 Strassen to X_s * Y_s.
    # The 7 terms of 2x2 Strassen involve linear combinations of entries of X_s and Y_s.
    # Each entry of X_s is a linear combination of 4 A entries.
    # So each term is a product of a linear combo of 16 A entries and a linear combo of 16 B entries?
    # No, the 2x2 Strassen term is like (x00+x11)(y00+y11).
    # x00 is A00[0,0] + A11[0,0].
    # So the term is (A00[0,0] + A11[0,0] + A00[1,1] + A11[1,1]) * (B00[0,0] + B11[0,0] + B00[1,1] + B11[1,1]).
    # This is a valid term.
    
    # We need to accumulate the coefficients for U, V, W.
    
    # Let's define the 7 macros.
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    
    # Macro 1: X1 = A00 + A11, Y1 = B00 + B11
    # Macro 2: X2 = A10 + A11, Y2 = B00
    # Macro 3: X3 = A00, Y3 = B01 - B11
    # Macro 4: X4 = A11, Y4 = B10 - B01
    # Macro 5: X5 = A00 + A01, Y5 = B11
    # Macro 6: X6 = A10 - A00, Y6 = B00 + B01
    # Macro 7: X7 = A01 - A11, Y7 = B10 + B11
    
    # For each macro, we decompose X_s * Y_s using 2x2 Strassen.
    # The 2x2 Strassen terms are:
    # T1: (x00+x11)(y00+y11) -> C00 += T1, C11 += T1
    # T2: (x10+x11)y00 -> C10 += T2, C11 -= T2
    # T3: x00(y01-y11) -> C01 += T3, C11 += T3
    # T4: x11(y10-y01) -> C00 -= T4, C10 += T4
    # T5: (x00+x01)y11 -> C00 -= T5, C01 += T5
    # T6: (x10-x00)(y00+y01) -> C11 += T6
    # T7: (x01-x11)(y10+y11) -> C00 += T7
    
    # We need to map x_ij and y_kl to A and B entries.
    # x_ij is an entry of the 2x2 matrix X_s.
    # X_s is formed from A blocks.
    # X_s[0,0] corresponds to the top-left entry of the 2x2 matrix X_s.
    # If X_s = A00 + A11, then X_s[0,0] = A00[0,0] + A11[0,0].
    
    # Let's define a function to get the A entries for a given X_s entry.
    # This is getting complex to do generically.
    # Let's hardcode the 49 terms by iterating over the 7 macros and 7 micro terms.
    
    # To simplify, we can just define the 7 macros as lists of (block, sign) for A and B.
    # But X_s is a 2x2 matrix.
    
    # Let's use a different approach.
    # We know the 49 terms of 4x4 Strassen.
    # We can generate them programmatically.
    
    # Define the 7 2x2 Strassen terms as functions of 4x4 blocks.
    
    # This is too error-prone to do manually in the thought process.
    # I will write the code to generate the Strassen terms for 4x4x4.
    
    # 4x4 Strassen:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B01)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each P_s is a 2x2 matrix product.
    # We decompose each P_s into 7 terms.
    
    # Let's define the 2x2 Strassen terms for a generic 2x2 product X*Y.
    # X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # T1 = (x0+x3)(y0+y3)
    # T2 = (x2+x3)y0
    # T3 = x0(y1-y3)
    # T4 = x3(y2-y1)
    # T5 = (x0+x1)y3
    # T6 = (x2-x0)(y0+y1)
    # T7 = (x1-x3)(y2+y3)
    
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # For each macro s, X is a 2x2 matrix of A-blocks, Y is a 2x2 matrix of B-blocks.
    # x0, x1, x2, x3 are 2x2 matrices (blocks of A).
    # y0, y1, y2, y3 are 2x2 matrices (blocks of B).
    
    # We need to expand T1 = (x0+x3)(y0+y3).
    # x0+x3 is a 2x2 matrix. y0+y3 is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # We decompose this product using 2x2 Strassen again?
    # No, T1 is a scalar in the macro Strassen sense?
    # No, P1 is a 2x2 matrix.
    # T1 is a 2x2 matrix.
    # We need to decompose T1 into 7 scalar terms?
    # Yes, because the final rank is 49.
    # So each of the 7 macro terms is decomposed into 7 micro terms.
    
    # So for each macro s, we have X_s and Y_s (2x2 matrices of blocks).
    # We compute the 7 macro terms T1..T7 (each a 2x2 matrix).
    # Then for each T_i, we decompose it into 7 scalar terms using 2x2 Strassen.
    # Wait, T_i is a 2x2 matrix product of two 2x2 matrices (which are sums of blocks).
    # So T_i = Z_i * W_i where Z_i and W_i are 2x2 matrices.
    # We decompose Z_i * W_i into 7 terms.
    
    # This seems correct.
    
    # Let's implement this.
    
    # Define the 7 macros.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # We will represent a 2x2 matrix of blocks as a 2x2 list of block names.
    # Then we will expand the Strassen terms.
    
    # This is still complex.
    # Let's just use the known 49 terms for 4x4x4 Strassen.
    # I will hardcode the 49 terms based on a reference or generate them carefully.
    
    # Actually, I can just use the schoolbook for 4x4x4 (rank 64) and add 16 for 5th column?
    # Rank 80.
    # Strassen 4x4x4 is rank 49.
    # Let's try to implement the 49 terms.
    
    # I will use a simplified Strassen implementation for 4x4.
    
    # The 49 terms are the tensor product of the 7 Strassen terms of 2x2 with themselves.
    # Term (s, t) where s, t in 1..7.
    # U part: from term s of A.
    # V part: from term t of B.
    # W part: from combination of s and t.
    
    # The 2x2 Strassen terms for A:
    # A1 = a+d, A2 = c+d, A3 = a, A4 = d, A5 = a+b, A6 = c-a, A7 = b-d
    # The 2x2 Strassen terms for B:
    # B1 = e+h, B2 = e, B3 = f-h, B4 = g-f, B5 = h, B6 = e+f, B7 = g+h
    
    # The products are A_s * B_t.
    # The result C is a linear combination of these 49 products.
    
    # We need the coefficients for C.
    # C00 = ...
    # This requires the full Strassen recursion.
    
    # Given the time constraint, I will implement the rank 65 decomposition using:
    # 49 terms for 4x4x4 (Strassen)
    # 16 terms for 4x4x1 (Schoolbook)
    
    # I will generate the 49 terms using a recursive Strassen function.
    
    def strassen_2x2_terms():
        # Returns list of (u_coeffs, v_coeffs, w_coeffs) for 2x2
        # u_coeffs: dict mapping (i,j) to coeff
        # v_coeffs: dict mapping (i,j) to coeff
        # w_coeffs: dict mapping (i,j) to coeff
        terms = []
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        terms.append({
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(0,0): 1, (1,1): 1}
        })
        # M2 = (c+d)e
        terms.append({
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(1,0): 1, (1,1): -1}
        })
        # M3 = a(f-h)
        terms.append({
            'u': {(0,0): 1},
            'v': {(0,1): 1, (1,1): -1},
            'w': {(0,1): 1, (1,1): 1}
        })
        # M4 = d(g-f)
        terms.append({
            'u': {(1,1): 1},
            'v': {(1,0): 1, (0,1): -1},
            'w': {(0,0): -1, (1,0): 1}
        })
        # M5 = (a+b)h
        terms.append({
            'u': {(0,0): 1, (0,1): 1},
            'v': {(1,1): 1},
            'w': {(0,0): -1, (0,1): 1}
        })
        # M6 = (c-a)(e+f)
        terms.append({
            'u': {(1,0): 1, (0,0): -1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(1,1): 1}
        })
        # M7 = (b-d)(g+h)
        terms.append({
            'u': {(0,1): 1, (1,1): -1},
            'v': {(1,0): 1, (1,1): 1},
            'w': {(0,0): 1}
        })
        return terms

    # For 4x4, we use the 7 macro terms.
    # Each macro term is a product of two 2x2 matrices (blocks).
    # We decompose each macro term using the 2x2 Strassen terms above.
    
    # Macro 1: (A00+A11)(B00+B11)
    # X = A00+A11, Y = B00+B11
    # We apply 2x2 Strassen to X*Y.
    # The 7 terms of 2x2 Strassen on X*Y will give 7 terms for the 4x4 block product.
    # But X and Y are 2x2 matrices of 2x2 blocks.
    # So the "entries" of X are 2x2 blocks.
    # This means we need to expand the 2x2 Strassen terms on the blocks.
    
    # This is effectively Strassen recursion.
    # I will implement a recursive Strassen function that returns the U, V, W contributions.
    
    def add_strassen_term(u_block, v_block, w_block, offset_u, offset_v, offset_w, scale=1):
        # u_block, v_block, w_block are 2x2 matrices of coefficients
        # offset_u, offset_v, offset_w are the top-left indices in the global matrices
        # This adds one term to the global U, V, W
        pass

    # This is getting too complicated for the time limit.
    # I will use a simpler, known rank 65 decomposition.
    # Or just use schoolbook for 4x4x4 (rank 64) + 16 = 80? No.
    # I will use the 49+16=65 decomposition.
    # I will hardcode the 49 terms for 4x4x4 Strassen.
    
    # Reference: Strassen 4x4x4 rank 49.
    # I will generate them using the recursive structure in code.
    
    # Let's try to implement the recursive Strassen correctly.
    
    def strassen_decompose(n, m, p):
        # Returns list of terms for n x m x p
        # If n=m=p=2, return the 7 terms.
        # If n=m=p=4, use recursion.
        pass

    # Given the complexity, I will provide a working rank 65 solution by combining
    # a pre-computed 4x4x4 Strassen decomposition (rank 49) and schoolbook 4x4x1 (rank 16).
    # I will generate the 49 terms programmatically.
    
    # The 49 terms are the tensor product of the 7 terms of 2x2 Strassen.
    # Term (s, t) corresponds to U_s * V_t.
    # The W part is determined by the Strassen combination.
    
    # I will use a library-like approach to generate the 49 terms.
    
    # For now, I will return the schoolbook rank 80 if I can't get 65 working quickly.
    # But 65 is much better.
    
    # Let's try to implement the 49 terms.
    
    # The 7 2x2 Strassen terms are defined above.
    # For 4x4, we have 7 macro terms.
    # Each macro term is a product of two 2x2 matrices.
    # We decompose each macro term into 7 micro terms.
    
    # I will implement this in the code.
    
    return {"U": U, "V": V, "W": W}

# The code above is a sketch. I will write the full code in the final answer.
