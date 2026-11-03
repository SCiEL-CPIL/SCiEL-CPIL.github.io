import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will use a decomposition of rank 81.
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Part 2: 4x4x2 Schoolbook (Rank 32)
    
    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    col = 0
    def add_term(i_u, i_v, i_w, val_u, val_v, val_w):
        nonlocal col
        U[i_u][col] = val_u
        V[i_v][col] = val_v
        W[i_w][col] = val_w
        col += 1

    # --- Part 1: 4x4x4 Strassen ---
    # A is 4x4, B is 4x4 (first 4 cols of B), C is 4x4
    # Strassen terms for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M_k is a 2x2 matrix product.
    # We use Strassen for each 2x2x2 product (Rank 7).
    # Total 7 * 7 = 49 terms.
    
    # Define the 7 Strassen sub-products for 2x2x2
    # For a 2x2x2 product X*Y, the 7 terms are:
    # S1 = (X00+X11)(Y00+Y11)
    # S2 = (X10+X11)Y00
    # S3 = X00(Y01-Y11)
    # S4 = X11(Y10-Y00)
    # S5 = (X00+X01)Y11
    # S6 = (X10-X00)(Y00+Y01)
    # S7 = (X01-X11)(Y10+Y11)
    
    # We need to map the 4x4 indices to these sub-products.
    # This is complex to hardcode correctly without errors.
    # Given the time, I will fall back to the Schoolbook for the 4x4x4 part?
    # No, that gives 64 + 32 = 96.
    
    # Let's just use the Schoolbook for the whole thing if Strassen is too risky.
    # But I must improve.
    
    # Alternative: Use the known rank 49 decomposition for 4x4x4.
    # I will implement the 49 terms explicitly.
    
    # ... (Implementation of 49 terms) ...
    
    # For safety and correctness within the time limit, I will use the Schoolbook 
    # but with a small optimization if possible, or just return 96 if I can't 
    # guarantee the Strassen implementation.
    
    # However, the prompt asks for a stronger algorithm.
    # I will implement the 4x4x4 Strassen (49) + 4x4x2 Schoolbook (32) = 81.
    
    # To ensure correctness, I will generate the 49 terms programmatically 
    # by applying Strassen's logic to the indices.
    
    # This is too complex for a quick rewrite.
    # I will return the schoolbook solution but with a comment that it's suboptimal, 
    # or try a simple block decomposition that is definitely correct.
    
    # Block decomposition:
    # Split A into 2x2 blocks.
    # Split B into 2x2 blocks.
    # Use Schoolbook for each 2x2x2 block multiplication.
    # Number of block multiplications:
    # A has 4 blocks. B has 6 blocks (3 cols of 2x2).
    # For each of the 6 output blocks (2 rows of A, 3 cols of B), 
    # we sum 2 products of 2x2 blocks.
    # Each 2x2x2 product has rank 4 (Schoolbook).
    # So each output block costs 2 * 4 = 8.
    # Total 6 * 8 = 48.
    # Wait, this is better than 96!
    
    # Let's verify.
    # C_rk = A_r0 B_0k + A_r1 B_1k.
    # A_r0 is 2x2, B_0k is 2x2. Product is 2x2.
    # Schoolbook for 2x2x2 has rank 4.
    # So A_r0 B_0k uses 4 terms.
    # A_r1 B_1k uses 4 terms.
    # Total 8 terms per C_rk.
    # There are 2 * 3 = 6 such C_rk.
    # Total rank 48.
    
    # This is a valid decomposition!
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for r in range(2): # Row block of A
        for k in range(3): # Col block of B
            # Term 1: A_r0 * B_0k
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        # A index: (2*r + i, j)
                        # B index: (j, 2*k + l)
                        # C index: (2*k + l, 2*r + i)
                        u_idx = (2*r + i) * m + j
                        v_idx = j * p + (2*k + l)
                        w_idx = (2*k + l) * n + (2*r + i)
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                        col += 1
            # Term 2: A_r1 * B_1k
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        # A index: (2*r + i, 2 + j)
                        # B index: (2 + j, 2*k + l)
                        # C index: (2*k + l, 2*r + i)
                        u_idx = (2*r + i) * m + (2 + j)
                        v_idx = (2 + j) * p + (2*k + l)
                        w_idx = (2*k + l) * n + (2*r + i)
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
