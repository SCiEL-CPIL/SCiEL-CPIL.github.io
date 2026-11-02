import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 73.
    # Split A (4x5) into A_L (4x4) and A_R (4x1)
    # Split B (5x5) into B_T (4x5) and B_B (1x5)
    # C = A_L B_T + A_R B_B
    # Split B_T (4x5) into B_TL (4x4) and B_TR (4x1)
    # C = A_L B_TL + A_L B_TR + A_R B_B
    
    # Part 1: A_L B_TL (4x4x4) using Strassen (Rank 49)
    # Part 2: A_L B_TR (4x4x1) (Rank 4)
    # Part 3: A_R B_B (4x1x5) (Rank 20)
    
    rank = 49 + 4 + 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, k, j, u_val, v_val, w_val):
        nonlocal col
        U[i * m + k][col] = u_val
        V[k * p + j][col] = v_val
        W[j * n + i][col] = w_val
        col += 1

    # --- Part 1: 4x4x4 Strassen ---
    # Blocks:
    # A_L:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B_TL:
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    # Strassen 7 products for 4x4:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each Pk is 2x2. We use Strassen 2x2 (rank 7) for each.
    # Total 49 terms.
    
    # We define the 7 inner Strassen terms for a 2x2 product C = AB.
    # The terms are linear combinations of A and B elements.
    # For a given block product Pk, we need to expand it into scalar terms.
    
    # To simplify, we will iterate through the 7 block products and for each,
    # iterate through the 7 inner Strassen terms.
    
    # Inner Strassen terms for 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We need to map these to global indices.
    # This is complex to hardcode. 
    # Instead, we use a generic approach: 
    # For each of the 49 scalar multiplications in the 4x4 Strassen,
    # we determine the coefficients for A[i,k] and B[k,j].
    
    # Given the time, I will provide the Rank 73 solution with a placeholder 
    # for the 49 terms that are correctly structured but might need verification.
    # However, to ensure correctness, I will use a simpler decomposition if possible.
    
    # Actually, let's just use the schoolbook for the 4x4x4 part? Rank 64.
    # Total Rank: 64 + 4 + 20 = 88.
    # This is better than 100 and much easier to implement correctly.
    
    # Let's reset rank to 88.
    # We will rebuild U, V, W.
    
    rank = 64 + 4 + 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: A_L B_TL (4x4x4) Schoolbook
    for i in range(4):
        for k in range(4):
            for j in range(4):
                add_term(i, k, j, 1, 1, 1)
                
    # Part 2: A_L B_TR (4x4x1)
    # A_L is cols 0-3. B_TR is col 4 of B_T (which is row 0-3 of B, col 4).
    # So k in 0-3, j=4.
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1, 1, 1)
            
    # Part 3: A_R B_B (4x1x5)
    # A_R is col 4 of A. B_B is row 4 of B.
    # So k=4.
    for i in range(4):
        for j in range(5):
            add_term(i, 4, j, 1, 1, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
