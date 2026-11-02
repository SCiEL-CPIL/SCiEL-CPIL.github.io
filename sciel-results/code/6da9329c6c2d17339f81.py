"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Blockwise decomposition:
    # A is 2x3, B is 3x8. Split B's rows into 3 blocks of 2 columns each: B = [B0, B1, B2] each 3x2.
    # C = [A B0, A B1, A B2], each 2x2.
    # Use rank-3 algorithm for 2x1x2 (A_left 2x1, B_left 1x2).
    # Actually, let's think more carefully.
    
    # A is 2x3. Let's split A into columns: A = [A0, A1, A2] where each Ai is 2x1.
    # B is 3x8. Let's split B into rows: B = [B0; B1; B2] where each Bi is 1x8.
    # C = A0 B0 + A1 B1 + A2 B2, sum of 3 rank-(2*1*8)=16 terms = 48 total. That's schoolbook.
    
    # Better: split B's columns into 4 blocks of 2: B = [B0 B1 B2 B3] each 3x2.
    # C = [A B0, A B1, A B2, A B3] each 2x2.
    # Need to compute 2x3 * 3x2 = 2x2 product.
    
    # For 2x3 * 3x2, we can use:
    # Let A = [[a0, a1, a2], [b0, b1, b2]], B = [[c0, c1], [d0, d1], [e0, e1]]
    # C00 = a0 c0 + a1 d0 + a2 e0
    # C01 = a0 c1 + a1 d1 + a2 e1
    # C10 = b0 c0 + b1 d0 + b2 e0
    # C11 = b0 c1 + b1 d1 + b2 e1
    
    # We can use a rank-7 algorithm? No, the tensor rank of 2x3x2 multiplication...
    # Actually, let's just use the schoolbook for 2x3*3x2 which is 12 terms per block, 4 blocks = 48.
    
    # Alternative: Use Strassen-like for the 3-dimensional part.
    # Let's try a different approach: view as (2x3) * (3x8).
    # We can write A as 2x3 and use the fact that we need 3 multiplications per output column.
    # Total 3*8 = 24 scalar multiplications? No, that's not right either.
    
    # Let me just implement the schoolbook and return it, but with a deadline guard.
    import time
    start = time.time()
    n, m, p = DIMENSIONS
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
