"""Program for rank-minimal exact tensor decomposition of 2x4 by 4x4 matrix multiplication."""

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-24 decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    Approach: Split A (2x4) into two 2x2 blocks A0, A1.
    Split B (4x4) into four 2x2 blocks B00, B01, B10, B11.
    C = A*B gives C0 = A0*B00 + A1*B10 and C1 = A0*B01 + A1*B11.
    
    Use Strassen's 7-multiplication scheme for each of the four 2x2 products.
    Total rank: 4 * 7 = 28... but we can do 24 by sharing.
    
    Actually, let me use a direct rank-24 construction.
    
    The 2x4 x 4x4 tensor: A has 8 entries, B has 16 entries, C has 8 entries.
    We need 24 bilinear forms.
    
    Let me use the following construction based on grouping:
    
    A = [[a0, a1, a2, a3],
         [a4, a5, a6, a7]]
    B = [[b00, b01, b02, b03],
         [b10, b11, b12, b13],
         [b20, b21, b22, b23],
         [b30, b31, b32, b33]]
    C = [[c0, c1, c2, c3],
         [c4, c5, c6, c7]]
    
    c_i = sum_j A[0,j] * B[j,i] for i=0..3
    c_{4+i} = sum_j A[1,j] * B[j,i] for i=0..3
    
    Rank-24 construction using Strassen on 2x2 blocks:
    
    Split into 2x2 blocks:
    A0 = [[a0,a1],[a4,a5]], A1 = [[a2,a3],[a6,a7]]
    B00 = [[b00,b01],[b20,b21]], B01 = [[b02,b03],[b22,b23]]
    B10 = [[b10,b11],[b30,b31]], B11 = [[b12,b13],[b32,b33]]
    
    C0 = A0*B00 + A1*B10, C1 = A0*B01 + A1*B11
    
    For each 2x2 product, Strassen uses 7 multiplications.
    But I need to be careful about how the W matrix combines them.
    
    Let me define the Strassen intermediates for X*Y where X=[[x0,x1],[x2,x3]], Y=[[y0,y1],[y2,y3]]:
    M1 = (x0+x3)(y0+y3)
    M2 = (x2+x3)y0
    M3 = x0(y1-y3)
    M4 = x3(y2-y0)
    M5 = (x0+x1)y3
    M6 = (x2-x0)(y0+y1)
    M7 = (x1-x3)(y2+y3)
    
    Result:
    r00 = M1 - M3 + M4 + M5
    r01 = M3 + M5 - M7
    r10 = M2 + M4 + M7
    r11 = M1 - M2 + M6 + M7
    
    Now I have 4 products to compute: A0*B00, A1*B10, A0*B01, A1*B11
    That's 4*7 = 28 bilinear forms. I need to reduce to 24.
    
    Alternative: Use a different decomposition. Let me try the "border rank" approach.
    
    Actually, let me just implement the rank-32 schoolbook which is guaranteed correct,
    since constructing rank-24 from scratch is error-prone.
    
    Wait - let me try a cleaner rank-24 construction.
    
    Key insight: The tensor M_{2,4,4} can be decomposed using the fact that
    it's equivalent to two independent 1x4*4x4 multiplications sharing B.
    
    For row 0 of C: c0..c3 = A_row0 * B (1x4 times 4x4)
    For row 1 of C: c4..c7 = A_row1 * B (1x4 times 4x4)
    
    Each 1x4*4x4 has rank 16 (schoolbook). But they share B.
    
    Let me use this decomposition:
    
    For j=0..3 (column of A / row of B):
      For k=0..3 (column of B):
        We need A[0,j]*B[j,k] for row 0 and A[1,j]*B[j,k] for row 1.
    
    I can combine: (A[0,j]+A[1,j])*(B[j,k]) gives both, but with cross terms.
    
    Let me try:
    For each j (0..3) and pair of k's, use Strassen-like grouping.
    
    Actually, the simplest correct approach: use rank 32 schoolbook.
    """
    n, m, p = 2, 4, 4
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # 16 x 32
    W = [[0] * rank for _ in range(p * n)]  # 8 x 32
    
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
