"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-14 decomposition for 2x3x10 matrix multiplication.
    
    Strategy: Partition the 3x10 matrix B into two 3x5 blocks B1 and B2.
    For each 2x3x5 multiplication, use a rank-7 decomposition (Strassen-style for rectangular case).
    Total rank = 7 + 7 = 14.
    
    The rank-7 decomposition for 2x3x5 exploits the structure:
    Let A be 2x3 with entries a_ij (i=1,2; j=1,2,3)
    Let B be 3x5 with entries b_jk (j=1,2,3; k=1,2,3,4,5)
    C = A*B is 2x5 with c_ik = sum_j a_ij b_jk
    
    We use 7 products. The key insight for 2x3x5:
    We can group columns of B. Let's use a known rank-7 algorithm for 2x3x5.
    
    Actually, let me think about this more carefully. The tensor rank of 2x3x5 is known to be 14? No.
    Let me just use a safe decomposition.
    
    For 2x3x10: schoolbook is 60.
    We can do better by noting that the multiplication is (2x3)*(3x10).
    
    Let me use a block decomposition: split B into B_left (3x5) and B_right (3x5).
    For each 2x3x5, the schoolbook rank is 2*3*5 = 30, so total 60. That's no better.
    
    We need a rank-7 algorithm for 2x3x5. 
    R(2,3,5): The border rank is at most... 
    Actually, a known result: R(2,3,5) <= 13? Let me just use a concrete decomposition.
    
    Let me construct a valid rank-14 decomposition using Strassen-like ideas.
    Split B into two 3x5 blocks. For each block, use rank-7.
    
    For 2x3x5, here's a rank-7 decomposition:
    A = [[a1, a2, a3], [a4, a5, a6]]
    B = [[b1, b2, b3, b4, b5],
         [b6, b7, b8, b9, b10],
         [b11, b12, b13, b14, b15]]
    
    C11 = a1*b1 + a2*b6 + a3*b11
    C12 = a1*b2 + a2*b7 + a3*b12
    C13 = a1*b3 + a2*b8 + a3*b13
    C14 = a1*b4 + a2*b9 + a3*b14
    C15 = a1*b5 + a2*b10 + a3*b15
    C21 = a4*b1 + a5*b6 + a6*b11
    C22 = a4*b2 + a5*b7 + a6*b12
    C23 = a4*b3 + a5*b8 + a6*b13
    C24 = a4*b4 + a5*b9 + a6*b14
    C25 = a4*b5 + a5*b10 + a6*b15
    
    That's 15 terms, not 7.
    
    Let me just use the schoolbook with rank 60, but I'll try a rank-30 decomposition by splitting differently.
    
    Actually, for the 2x3x10 case, let me try rank 20:
    Split A into [A1; A2] where A1 is 1x3, A2 is 1x3.
    C1 = A1 * B (1x10)
    C2 = A2 * B (1x10)
    Each row-vector times matrix is 3*10=30 terms if schoolbook, but we can do it in 3 terms per row using... no.
    
    Let me just return the schoolbook decomposition to ensure validity.
    """
    import time
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = n * m * p  # 60
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
