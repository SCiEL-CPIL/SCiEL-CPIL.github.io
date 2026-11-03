"""Program for finding rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x3x8 matrix multiplication tensor using a block-Strassen approach.
    
    Strategy:
    - Split the 3x8 matrix B into two 3x4 blocks: B1 (cols 0-3), B2 (cols 4-7)
    - C1 = A * B1 (2x4), C2 = A * B2 (2x4)
    - Decompose each 2x3x4 multiplication optimally
    
    For 2x3x4:
    - Split B (3x4) into B_top (2x4) and B_bottom (1x4)
    - Split A (2x3) into A_left (2x2) and A_right (2x1)
    - C = A_left * B_top + A_right * B_bottom
    
    For A_left (2x2) * B_top (2x4):
    - Use Strassen-like decomposition:
      - Split B_top into two 2x2 blocks: B_tl (2x2), B_tr (2x2)
      - A_left * B_tl: 2x2x2 (rank 7 with Strassen)
      - A_left * B_tr: 2x2x2 (rank 7 with Strassen)
      Total: 14
    
    For A_right (2x1) * B_bottom (1x4):
    - This is 2x1x4, which requires 2*1*4 = 8 multiplications (schoolbook is optimal for this small case)
    
    Total rank per 2x3x4 block: 14 + 8 = 22
    Total rank for 2x3x8: 2 * 22 = 44
    
    Wait, let me reconsider. Let's use a better decomposition.
    
    Alternative: Use the known result that R(2,3,4) <= 17.
    Actually, let's use a direct construction.
    
    For 2x3x4:
    A = [[a11, a12, a13],
         [a21, a22, a23]]
    B = [[b11, b12, b13, b14],
         [b21, b22, b23, b24],
         [b31, b32, b33, b34]]
    C = A * B (2x4)
    
    C11 = a11*b11 + a12*b21 + a13*b31
    C12 = a11*b12 + a12*b22 + a13*b32
    C13 = a11*b13 + a12*b23 + a13*b33
    C14 = a11*b14 + a12*b24 + a13*b34
    C21 = a21*b11 + a22*b21 + a23*b31
    C22 = a21*b12 + a22*b22 + a23*b32
    C23 = a21*b13 + a22*b23 + a23*b33
    C24 = a21*b14 + a22*b24 + a23*b34
    
    Schoolbook: 2*3*4 = 24
    
    Let's try to reduce this. We can group terms.
    
    Actually, let me just implement a known lower-bound construction.
    The best known upper bound for R(2,3,4) is 17 or 18.
    Let me use a simpler approach: just use the schoolbook for now to ensure correctness,
    then we can improve.
    
    For now, let's use rank 48 (schoolbook) as a baseline, but structure it properly.
    """
    n, m, p = 2, 3, 8
    rank = n * m * p  # 48
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
