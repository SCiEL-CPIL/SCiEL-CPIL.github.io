"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    
    Uses the Winograd minimum rank 7 decomposition for 2x2 matrix multiplication
    to achieve a lower rank than schoolbook (128).
    
    Strategy: 
    - Split the 4x4 matrix A into four 2x2 blocks
    - Split the 4x8 matrix B into four 2x4 blocks
    - Use Winograd rank-7 for each 2x2 * 2x2 subproduct
    - Each 2x2 * 2x4 product can be decomposed into two 2x2 * 2x2 products
    - Total: 4 (A blocks) * 4 (B blocks) * 2 (sub-products) * 7 (Winograd rank) = 224
    Wait, that's worse than 128. Let me reconsider.
    
    Actually, for 4x4 * 4x8:
    - We can view this as 8 separate 4x4 * 4x1 multiplications
    - Or better: use Strassen-like recursion
    
    Let me use a different approach: 
    - Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2
    - Split B into 2x2 blocks: B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] where each is 2x2
    
    Then C = A*B has 2x2 blocks:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C13 = A11*B13 + A12*B23
    C14 = A11*B14 + A12*B24
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    C23 = A21*B13 + A22*B23
    C24 = A21*B14 + A22*B24
    
    Each 2x2*2x2 product can be done with rank 7 (Winograd).
    So each Cij requires 2 such products, total 8*2*7 = 112.
    
    But we can share some computations. Let me use the known Winograd 2x2 decomposition.
    
    Winograd 2x2 rank-7:
    For A = [[a, b], [c, d]] and B = [[e, f], [g, h]]:
    m1 = a * e
    m2 = b * f  
    m3 = c * g
    m4 = d * h
    m5 = (a+b) * (e+f)
    m6 = (c+d) * (g+h)
    m7 = (a+c) * (e-g)
    
    C[0,0] = m1 + m5 - m7
    C[0,1] = m2 + m5
    C[1,0] = m3 + m6
    C[1,1] = m4 + m6
    
    Wait, let me verify:
    C[0,0] should be ae + cg
    m1 + m5 - m7 = ae + (a+b)(e+f) - (a+c)(e-g)
    = ae + ae + af + be + bf - ae + ag - ce + cg
    = ae + af + be + bf + ag - ce + cg
    That's not right.
    
    Let me look up the correct Winograd 2x2 decomposition.
    
    Correct Winograd 2x2 (rank 7):
    p1 = (a+b) * (e+h)
    p2 = (c+d) * (e+f)
    p3 = a * (f-h)
    p4 = d * (g-f)
    p5 = (a-c) * (e+g)
    p6 = (b+d) * (g+h)
    p7 = (b+c) * (e+f) -- wait, this doesn't look right either.
    
    Let me use a verified formula. The standard Winograd 2x2 is:
    m1 = (a+b)(e+h)
    m2 = (c+d)(e+f)
    m3 = a(f-h)
    m4 = d(g-f)
    m5 = (a-c)(e+g)
    m6 = (b+d)(g+h)
    m7 = (b+c)(e+f) -- no, this is getting confused.
    
    Let me just use the known fact that Winograd 2x2 has rank 7 and construct it properly.
    
    Actually, I'll use a simpler approach: just use schoolbook for now with rank 128,
    which is correct. Improving to Winograd would require careful verification.
    
    For 4x4 * 4x8, schoolbook rank is 4*4*8 = 128.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    rank = n * m * p  # 128
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 128
    V = [[0] * rank for _ in range(m * p)]  # 32 x 128
    W = [[0] * rank for _ in range(p * n)]  # 32 x 128
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
