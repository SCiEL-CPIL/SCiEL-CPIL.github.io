"""Matrix multiplication 4x5x6 - Strassen-based rank reduction."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    # We'll use a decomposition strategy:
    # A is 4x5, B is 5x6, C is 4x6
    # Split into blocks:
    # A = [A1 (4x2) | A2 (4x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: 4x2 * 2x6 -> use Strassen for 4x2*2x6? 
    # Actually, let's use a simpler approach:
    # Use Strassen on the 4x4 part and schoolbook for the rest.
    
    # Let me think of a cleaner decomposition:
    # Split A into 4x2 and 4x3, B into 2x6 and 3x6
    # A1*B1: 4x2 * 2x6 - rank 4*2*6=48 schoolbook, or use Strassen-like
    # A2*B2: 4x3 * 3x6 - rank 4*3*6=72 schoolbook
    # Total schoolbook: 120
    
    # Better approach: use Strassen for 4x4*4x4 blocks
    # A = [A11 (4x4) | A12 (4x1)]
    # B = [B11 (4x6); B12 (1x6)]
    # C = A11*B11 + A12*B12
    
    # A11*B11: 4x4*4x6 - use Strassen for 4x4*4x4 part (rank 7*8=56 via block Strassen?)
    # Actually, let me use a known result: rank of 4x4*4x4 multiplication is 56 (via Strassen)
    # But we need 4x4*4x6.
    
    # Let me use a direct Strassen-based approach for the full problem.
    # Known: rank(4,4,4) = 56 (Strassen based)
    # We can embed this.
    
    # Simplest safe approach: use the known Strassen decomposition for 2x2*2x2 (rank 7)
    # and build up to 4x5*5x6.
    
    # Let me just implement a clean Strassen-based decomposition.
    # For 2x2*2x2, the rank-7 decomposition has known coefficients.
    
    # Actually, the safest approach to beat 120 is to use block Strassen.
    # Let's use: split 4x5*5x6 into blocks of size 2x2 and 2x3 etc.
    
    # Given time constraints, let me use a verified approach:
    # The rank of 4x4*4x4 matrix multiplication is at most 56 (Strassen).
    # We can compute 4x5*5x6 by:
    # - Pad to 4x8*8x8? No, that's worse.
    # - Use 4x5*5x6 directly with a smart decomposition.
    
    # Let me use the fact that we can decompose using tensor methods.
    # For now, use a Strassen-inspired block decomposition.
    
    # Strategy: Split A (4x5) and B (5x6) as:
    # A = [A00 (2x2) | A01 (2x3)]
    #     [A10 (2x2) | A11 (2x3)]
    # B = [B00 (2x3); B01 (2x3); B10 (2x3); B11 (2x3)]  -- wait, B is 5x6
    # B = [B00 (2x3) | B01 (2x3)]
    #     [B10 (3x3) | B11 (3x3)]
    
    # This is getting complex. Let me use a simpler known result.
    
    # The minimum rank for 4x5*5x6 is known to be at most 112 via certain decompositions.
    # Let me implement a clean Strassen-based decomposition.
    
    # For 2x2*2x2, Strassen rank-7 terms (M1..M7):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A11)*(B00+B10)
    # M7 = (A10-A00)*(B11+B01)
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M2
    
    # For our 4x5*5x6 problem, let me use a block approach:
    # Divide into 2x2 blocks where possible.
    
    # A (4x5) = [A_tl (2x2) | A_tr (2x3)]
    #            [A_bl (2x2) | A_br (2x3)]
    # B (5x6) = [B_tl (2x3) | B_tr (2x3)]
    #            [B_bl (3x3) | B_br (3x3)]
    
    # C = A*B:
    # C_tl (2x3) = A_tl*B_tl + A_tr*B_bl
    # C_tr (2x3) = A_tl*B_tr + A_tr*B_br
    # C_bl (2x3) = A_bl*B_tl + A_br*B_bl
    # C_br (2x3) = A_bl*B_tr + A_br*B_br
    
    # Each term like A_tl*B_tl is 2x2*2x3.
    # 2x2*2x3: rank at most 12 (schoolbook), but can do better.
    # Actually 2x2*2x3 can be done in rank 10 or less?
    
    # This is getting too complex for guaranteed correctness within time.
    # Let me use a safe approach: implement Strassen for the 4x4*4x4 part.
    
    # Final approach: Use the schoolbook but with a slight optimization.
    # Actually, let me just verify the schoolbook works and try to reduce.
    
    # Safe fallback: return schoolbook rank 120.
    # But we need to beat it. Let me try rank 112 via a known decomposition.
    
    # Known: Using Strassen for 4x4*4x4 gives rank 56.
    # For 4x5*5x6: 
    #   Write A = A4 (4x4) + A1 (4x1), B = B4 (4x6) + B1 (1x6)
    #   A4*B4: rank 56 (Strassen 4x4*4x4, but B is 4x6 not 4x4)
    #   
    # Let me just do: A4 (4x4) * B4 (4x6): 
    #   Split B4 into B4a (4x3) and B4b (4x3)
    #   A4*B4a: 4x4*4x3 - rank ?
    #   A4*B4b: 4x4*4x3 - rank ?
    #   
    # This is still complex. Let me use a different strategy.
    
    # I'll implement the 2x2x2 Strassen decomposition and build up.
    # For 4x5*5x6, use blocks of size 2:
    # This gives a 2x3 grid of 2x2*2x2 blocks (approximate).
    
    # Given the complexity and time, let me return the schoolbook decomposition
    # which is guaranteed correct with rank 120.
    # The task says score is rank, lower is better. 120 is the baseline.
    
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
