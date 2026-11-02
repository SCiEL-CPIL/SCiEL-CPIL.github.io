import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # We view A (4x4) as a 2x2 block matrix of 2x2 blocks.
    # We view B (4x7) as having rows split into two groups of 2: B_top (2x7), B_bottom (2x7).
    # C (4x7) has C_top (2x7), C_bottom (2x7).
    #
    # C_top = A00 * B_top + A01 * B_bottom
    # C_bottom = A10 * B_top + A11 * B_bottom
    #
    # Using Strassen's algorithm for the 2x2 block structure:
    # M1 = (A00 + A11) * (B_top + B_bottom)
    # M2 = (A10 + A11) * B_top
    # M3 = A00 * (B_bottom - B_top)
    # M4 = A11 * (B_top - B_bottom)
    # M5 = (A00 + A01) * B_bottom
    # M6 = (A10 - A00) * (B_top + B_bottom)
    # M7 = (A01 - A11) * (B_top + B_bottom)
    #
    # Each Mi is a 2x7 matrix computed by schoolbook multiplication (rank 14 each).
    # Total rank = 7 * 14 = 98.
    #
    # The tensor decomposition: for each Strassen term t (0..6) and each schoolbook
    # index (i,j,k) in the 2x2x7 multiplication, we have a rank-1 tensor.
    
    # Strassen coefficients for A blocks (2x2 structure: indices 0,1,2,3 = A00,A01,A10,A11)
    # Each entry is (coeff, block_index) or we can use the standard Strassen form.
    # Let's use the standard 7-term Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B00)
    # M4 = A11(B00-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B11)
    # M7 = (A01-A11)(B11+B00)
    
    # In our case, B00 = B_top, B11 = B_bottom (since we're multiplying 2x2 blocks with 2x7)
    # Wait, I need to be careful. Let me redefine.
    
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    # This is NOT a 2x2 block matrix for B in the standard sense.
    # 
    # Actually, for Strassen, we need B to also be a 2x2 block matrix.
    # Let's pad B to 4x4 by adding zeros? No, that changes the problem.
    #
    # Alternative: just do 7 separate 2x2x7 multiplications using Strassen-like ideas.
    # But 2x2x7 is small.
    
    # Let me just implement the rank-98 solution directly:
    # For each of the 7 Strassen terms, and for each of the 14 schoolbook indices
    # in the 2x2x7 multiplication, create a rank-1 tensor.
    
    # Strassen terms: list of (a_coeffs, b_coeffs) where a_coeffs is dict {block: sign}
    # and b_coeffs is dict {block: sign}
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B "blocks": 0=B_top, 1=B_bottom
    
    strassen_terms = [
        # M1 = (A00+A11)(B_top+B_bottom)
        ({0: 1, 3: 1}, {0: 1, 1: 1}),
        # M2 = (A10+A11)B_top
        ({2: 1, 3: 1}, {0: 1}),
        # M3 = A00(B_bottom - B_top)
        ({0: 1}, {1: 1, 0: -1}),
        # M4 = A11(B_top - B_bottom)
        ({3: 1}, {0: 1, 1: -1}),
        # M5 = (A00+A01)B_bottom
        ({0: 1, 1: 1}, {1: 1}),
        # M6 = (A10-A00)(B_top+B_bottom)
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # M7 = (A01-A11)(B_top+B_bottom)
        ({1: 1, 3: -1}, {0: 1, 1: 1}),
    ]
    
    # C reconstruction:
    # C_top = M1 + M3 - M5 + M7  (2x7)
    # C_bottom = M2 + M4 + M6 - M1  (2x7)
    # Wait, let me verify with standard Strassen:
    # C00 = M1 + M3 - M5 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M4 + M6
    # But our C is not 2x2 blocks of 2x2, it's 2x7.
    # C_top corresponds to C00 and C01 combined? No.
    #
    # Let me reconsider. The output C is 4x7. We split it into C_top (rows 0-1) and C_bottom (rows 2-3).
    # C_top[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0..6
    # C_bottom[i,k] = sum_j A[i,j] * B[j,k] for i=2,3 and k=0..6
    #
    # Using the block structure:
    # C_top = A00 * B_top + A01 * B_bottom
    # C_bottom = A10 * B_top + A11 * B_bottom
    #
    # This is a 2x2 matrix multiplication where the "entries" are 2x7 matrices.
    # So we can apply Strassen to this 2x2 structure.
    #
    # Let X00 = A00, X01 = A01, X10 = A10, X11 = A11 (each 2x2)
    # Let Y00 = B_top, Y10 = B_bottom (each 2x7)
    # Wait, this isn't symmetric. B is not a 2x2 block matrix of 2x7 blocks in the same way.
    #
    # I think the cleanest approach is:
    # View the multiplication as: for each column k of B (7 columns), we do a 4x4 x 4x1 multiplication.
    # Use Strassen for 4x4x4 (rank 49) for each of the 7 columns? That gives rank 49*7=343, worse.
    #
    # Let me just go with the rank-98 approach: 7 Strassen terms, each involving 2x2x7 schoolbook.
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 28 x 98
    W = [[0] * rank for _ in range(p * n)]  # 28 x 98
    
    # A index mapping: block b, local (i,j) -> global index
    # A00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    def a_index(block, i, j):
        # block: 0=A00, 1=A01, 2=A10, 3=A11
        # i,j: 0 or 1 (local row/col in the 2x2 block)
        row = (block // 2) * 2 + i
        col = (block % 2) * 2 + j
        return row * m + col  # m=4
    
    # B index mapping: "block" 0=B_top (rows 0-1), 1=B_bottom (rows 2-3)
    # B_top: rows 0-1, cols 0-6 -> indices 0-6, 7-13
    # B_bottom: rows 2-3, cols 0-6 -> indices 14-20, 21-27
    def b_index(block, i, k):
        # block: 0=B_top, 1=B_bottom
        # i: 0 or 1 (local row)
        # k: 0..6 (column)
        row = block * 2 + i
        return row * p + k  # p=7
    
    # C index mapping: "block" 0=C_top (rows 0-1), 1=C_bottom (rows 2-3)
    def c_index(block, i, k):
        # block: 0=C_top, 1=C_bottom
        # i: 0 or 1 (local row)
        # k: 0..6 (column)
        row = block * 2 + i
        return k * n + row  # n=4, trace convention: W[k*n+i, r]
    
    # For each Strassen term, and for each schoolbook index in 2x2x7:
    # The schoolbook multiplication of (2x2) * (2x7) -> (2x7) has 2*2*7=28 rank-1 tensors.
    # But we're using 7 terms, so total 7*28=196? No, that's wrong.
    #
    # Actually, each Strassen term Mi is a 2x7 matrix. To compute Mi = Xi * Yi where
    # Xi is 2x2 and Yi is 2x7, we need 2*2*7=28 rank-1 tensors for schoolbook.
    # But we only have 98 total, so 98/7=14 per term. That means we're not using schoolbook for each term.
    #
    # I think I'm overcomplicating this. Let me just use the known rank-98 decomposition.
    # 
    # The rank-98 decomposition for 4x4x7 can be constructed as:
    # For each of the 7 Strassen terms, use a rank-14 decomposition for 2x2x7.
    # 7 * 14 = 98.
    #
    # For 2x2x7 schoolbook: for i in 0..1, j in 0..1, k in 0..6:
    #   U[i*2+j] = 1, V[j*7+k] = 1, W[k*2+i] = 1
    # That's 2*2*7=28 tensors, not 14.
    #
    # So rank-98 is NOT achievable with this naive approach. The current schoolbook is 112.
    # 98 < 112, so there must be a better structure.
    #
    # Let me just return the schoolbook rank-112 solution for safety.
    
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
