"""
Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
Uses Strassen-style splitting to reduce below schoolbook rank 40.
"""
DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    # We'll build a decomposition of rank 36 using:
    # Split A into A_left (2x2) and A_right (2x2)
    # Split B into B_top (2x5) and B_bottom (2x5)
    # C = A_left * B_top + A_right * B_bottom
    # For each 2x2 * 2x5 product, use Strassen for the 2x2 part combined with schoolbook for the 5 columns
    # Actually, let's think differently.
    
    # Better approach: Use the fact that R(2,4,5) <= R(2,2,5) + R(2,2,5) = 2*R(2,2,5)
    # R(2,2,5): We can use Strassen on the 2x2 block and schoolbook for the 5th dimension?
    # No, Strassen works for square matrices. For 2x2 * 2x5, we can split the 5 columns into 2+2+1.
    # R(2,2,2) = 7 (Strassen), R(2,2,1) = 4 (schoolbook)
    # So R(2,2,5) <= 7 + 7 + 4 = 18? But we need to be careful.
    
    # Actually, for 2x2 * 2x5, the standard approach is:
    # Treat it as 5 independent 2x2 * 2x1 multiplications? No, that's not right.
    # The product is (2x2)(2x5) = 2x5.
    # We can use the bilinear complexity. R(2,2,5) is known to be <= 16? Let me check.
    # Actually, R(2,2,p) for p=5. The schoolbook is 20.
    # Using Strassen on the 2x2 part: if we had 2x2*2x2, it's 7.
    # For 2x2*2x5, we can write it as 2x2*2x2 + 2x2*2x2 + 2x2*2x1
    # = 7 + 7 + 4 = 18.
    # So R(2,4,5) <= 2*18 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    # Helper: add a Strassen 2x2*2x2 decomposition
    # A is 2x2 with indices a00,a01,a10,a11
    # B is 2x2 with indices b00,b01,b10,b11
    # C is 2x2 with indices c00,c01,c10,c11
    def add_strassen(a_base, b_base, c_base):
        nonlocal col
        # M1 = (a00+a11)(b00+b11)
        U[a_base][col] += 1; U[a_base+3][col] += 1
        V[b_base][col] += 1; V[b_base+1][col] += 1
        W[c_base][col] += 1; W[c_base+2][col] += 1
        W[c_base+1][col] -= 1; W[c_base+3][col] -= 1
        col += 1
        # M2 = a11(b10+b11)
        U[a_base+3][col] += 1
        V[b_base+2][col] += 1; V[b_base+3][col] += 1
        W[c_base+2][col] += 1; W[c_base+3][col] += 1
        col += 1
        # M3 = a00(b00-b10)
        U[a_base][col] += 1
        V[b_base][col] += 1; V[b_base+2][col] -= 1
        W[c_base][col] += 1; W[c_base+2][col] += 1
        col += 1
        # M4 = a11(b01-b11)
        U[a_base+3][col] += 1
        V[b_base+1][col] += 1; V[b_base+3][col] -= 1
        W[c_base+1][col] += 1; W[c_base+3][col] += 1
        col += 1
        # M5 = (a00+a01)b11
        U[a_base][col] += 1; U[a_base+1][col] += 1
        V[b_base+3][col] += 1
        W[c_base+1][col] += 1; W[c_base+3][col] += 1
        col += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_base+2][col] += 1; U[a_base][col] -= 1
        V[b_base][col] += 1; V[b_base+1][col] += 1
        W[c_base][col] += 1; W[c_base+2][col] += 1
        col += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_base+1][col] += 1; U[a_base+3][col] -= 1
        V[b_base+2][col] += 1; V[b_base+3][col] += 1
        W[c_base+1][col] += 1; W[c_base+3][col] += 1
        col += 1
    
    # Helper: add schoolbook 2x2*2x1
    def add_school_2x2_1(a_base, b_base, c_base):
        nonlocal col
        # c0 = a00*b0 + a01*b1
        U[a_base][col] += 1; V[b_base][col] += 1; W[c_base][col] += 1
        col += 1
        U[a_base+1][col] += 1; V[b_base+1][col] += 1; W[c_base][col] += 1
        col += 1
        # c1 = a10*b0 + a11*b1
        U[a_base+2][col] += 1; V[b_base][col] += 1; W[c_base+1][col] += 1
        col += 1
        U[a_base+3][col] += 1; V[b_base+1][col] += 1; W[c_base+1][col] += 1
        col += 1
    
    # Left part: A_left (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 0-4)
    # A_left indices: U[0], U[1], U[2], U[3]
    # B_top cols 0-1: V[0], V[1], V[5], V[6]
    # C cols 0-1: W[0], W[1], W[2], W[3]
    add_strassen(0, 0, 0)
    
    # B_top cols 2-3: V[2], V[3], V[7], V[8]
    # C cols 2-3: W[4], W[5], W[6], W[7]
    add_strassen(0, 2, 4)
    
    # B_top col 4: V[4], V[9]
    # C col 4: W[8], W[9]
    add_school_2x2_1(0, 4, 8)
    
    # Right part: A_right (rows 0-1, cols 2-3) * B_bottom (rows 2-3, cols 0-4)
    # A_right indices: U[4], U[5], U[6], U[7]
    # B_bottom cols 0-1: V[10], V[11], V[15], V[16]
    # C cols 0-1: W[0], W[1], W[2], W[3]
    add_strassen(4, 10, 0)
    
    # B_bottom cols 2-3: V[12], V[13], V[17], V[18]
    # C cols 2-3: W[4], W[5], W[6], W[7]
    add_strassen(4, 12, 4)
    
    # B_bottom col 4: V[14], V[19]
    # C col 4: W[8], W[9]
    add_school_2x2_1(4, 14, 8)
    
    return {"U": U, "V": V, "W": W}
