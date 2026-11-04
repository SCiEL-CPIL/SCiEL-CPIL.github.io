"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will decompose M_{2,4,7} into M_{2,4,4} and M_{2,4,3}.
    # M_{2,4,4} can be decomposed using two M_{2,2,4} blocks.
    # M_{2,2,4} can be decomposed using two M_{2,2,2} blocks.
    # M_{2,2,2} has a known rank-7 Winograd decomposition.
    # So M_{2,4,4} rank = 4 * 7 = 28.
    
    # M_{2,4,3} can be decomposed using two M_{2,2,3} blocks.
    # M_{2,2,3} can be decomposed using one M_{2,2,2} and one M_{2,2,1}.
    # M_{2,2,2} rank 7.
    # M_{2,2,1} rank 4 (schoolbook: 2*2*1=4).
    # So M_{2,2,3} rank = 7 + 4 = 11.
    # So M_{2,4,3} rank = 2 * 11 = 22.
    
    # Total rank = 28 + 22 = 50.
    # This matches the Winograd improvement mentioned in the comments of the seed.
    
    # Let's define the base decompositions.
    
    # 1. Winograd 2x2x2 (Rank 7)
    # U_w, V_w, W_w are 4x7, 4x7, 4x7
    # Rows of U correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # Rows of V correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    # Rows of W correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    
    U_222 = [
        [1, 0, 0, 1, 0, 1, 0], # A00
        [0, 1, 0, 1, 0, 0, 1], # A01
        [0, 0, 1, 0, 1, 1, 0], # A10
        [1, 0, 0, 0, 1, 0, 1]  # A11
    ]
    V_222 = [
        [1, 0, 0, 1, 0, 0, 0], # B00
        [0, 0, 1, 0, 0, 0, 1], # B01
        [0, 1, 0, 0, 0, 1, 0], # B10
        [1, 0, 0, 0, 1, 0, 0]  # B11
    ]
    W_222 = [
        [1, 0, 0, 0, 0, 0, 0], # C00 = p1
        [0, 1, 0, 0, 0, 0, 0], # C01 = p2? No.
        [0, 0, 1, 0, 0, 0, 0], # C10 = p3?
        [0, 0, 0, 1, 0, 0, 0]  # C11 = p4?
    ]
    
    # The W matrix above is a guess. I need the correct W for Winograd 2x2x2.
    # Let's derive it or find a reliable source in my head.
    # p1 = (a00+a11)(b00+b11)
    # p2 = a01 b10
    # p3 = a10 b01
    # p4 = (a00+a01) b00
    # p5 = (a11+a10) b11
    # p6 = (a00-a10) b10
    # p7 = (a11-a01) b01
    
    # C00 = a00 b00 + a01 b10
    # C01 = a00 b01 + a01 b11
    # C10 = a10 b00 + a11 b10
    # C11 = a10 b01 + a11 b11
    
    # Let's express C in terms of p.
    # C00 = a00 b00 + a01 b10
    # p4 = a00 b00 + a01 b00
    # p2 = a01 b10
    # p6 = a00 b10 - a10 b10
    # p1 = a00 b00 + a00 b11 + a11 b00 + a11 b11
    # This is getting complicated to invert mentally.
    
    # Alternative: Use a known valid decomposition for 2x2x2 with half-integers or integers.
    # There is a simple rank 7 decomposition.
    # Let's use the one from "Matrix Multiplication with Fewer Multiplications".
    
    # Actually, I can just use the Schoolbook for 2x2x2 (Rank 8) to be safe?
    # If I use Rank 8 for 2x2x2:
    # M_244 rank = 4 * 8 = 32.
    # M_243 rank = 2 * (8 + 4) = 24.
    # Total = 56. This is the seed.
    
    # I need Rank 7 for 2x2x2 to get 50.
    # Let's try to construct the W matrix correctly.
    # From p definitions:
    # a00 b00 = p4 - a01 b00
    # This is not linear in a's and b's independently.
    
    # Let's look at the columns of U, V, W.
    # Col 1: U=[1,0,0,1], V=[1,0,0,1], W=[?]. p1 = (a00+a11)(b00+b11).
    # This term contributes to C00, C01, C10, C11.
    # C00 gets a00 b00 + a00 b11 + a11 b00 + a11 b11.
    # C01 gets a00 b01 + a00 b11 + a11 b01 + a11 b11.
    # C10 gets a10 b00 + a10 b11 + a11 b00 + a11 b11.
    # C11 gets a10 b01 + a10 b11 + a11 b01 + a11 b11.
    
    # This manual derivation is error-prone.
    # I will implement the Schoolbook solution which is guaranteed correct, 
    # and then attempt to replace 2x2x2 blocks with Winograd if I can find the matrices.
    # Since I cannot guarantee the Winograd matrices, I will stick to the Schoolbook 
    # but structure it clearly. The score will be 1.0 (same as seed).
    # However, the directive asks to improve.
    
    # Is there a simpler improvement?
    # No, without the correct Winograd matrices, I cannot go below 56.
    
    # I will return the Schoolbook solution.
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
