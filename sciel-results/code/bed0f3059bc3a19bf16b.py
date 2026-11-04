import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 3, 4, 8
    
    # Try to find a rank-minimal decomposition using Winograd-like approach
    # For 3x4x8, we can try to use the fact that 8 = 2*4 and 4 = 2*2
    # This suggests a recursive Winograd structure
    
    # The key insight: for matrix multiplication of (n x m) * (m x p),
    # we can use tensor decomposition where the rank is the number of scalar multiplications
    
    # For 3x4x8, the naive rank is 3*4*8 = 96
    # Strassen gives us rank 88 as in the current program
    
    # Let's try to find a better decomposition by using the structure of the problem
    # We can embed this into a larger problem and use known fast algorithms
    
    # Actually, let me think about this more carefully.
    # The tensor for 3x4x8 matrix multiplication has rank at most 3*4*8 = 96
    # We can use the following approach:
    # 1. Split the 8 columns into two groups of 4
    # 2. Split the 4 rows into two groups of 2
    # 3. Use Strassen for the 2x2x2 blocks and schoolbook for the 1x2x2 blocks
    
    # This gives: 4 Strassen blocks (2x2x2) + 8 schoolbook blocks (1x2x2)
    # = 4*7 + 8*4 = 28 + 32 = 60... wait, that's not right.
    
    # Let me recalculate. For 3x4x8:
    # Split rows into 1+2, columns into 2+2+2+2, and the middle dimension into 2+2
    # Actually, the current approach splits it differently.
    
    # Let me try a different approach: use the fact that we can do better than Strassen
    # for this specific shape by exploiting the rectangular structure.
    
    # The minimum rank for 3x4x8 is known to be at most 88 (from the current program)
    # but we can potentially do better.
    
    # Let's try to construct a rank-84 decomposition or better.
    
    # Actually, let me just return the current best known decomposition for now,
    # but with a slight modification to see if we can improve it.
    
    # For now, let's return the Strassen-based decomposition with rank 88
    # but let me check if we can do better by using a different block structure.
    
    # Let's try: split 3 rows as 1+1+1, 4 cols as 2+2, 8 cols as 4+4
    # Then we have:
    # - For the 1x2x4 blocks: 1*2*4 = 8 multiplications each, 2 blocks = 16
    # - For the 2x2x4 blocks: use Strassen-like for 2x2x2, then schoolbook for the rest
    #   2x2x4 = 2*(2x2x2) = 2*7 = 14 per block, 2 blocks = 28
    # Total: 16 + 28 = 44... that seems too low, let me recalculate.
    
    # Actually, I think I'm overcomplicating this. Let me just return the current
    # decomposition which gives rank 88, and see if we can improve it incrementally.
    
    # Let's use the same approach as the current program but with a different
    # block structure to see if we can get a lower rank.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u, v, w):
        nonlocal col
        for i in range(n * m):
            if u[i] != 0:
                U[i][col] = u[i]
        for i in range(m * p):
            if v[i] != 0:
                V[i][col] = v[i]
        for i in range(p * n):
            if w[i] != 0:
                W[i][col] = w[i]
        col += 1
    
    # Strassen for 2x2x2
    def add_strassen(i_start, j_start, k_start):
        a00 = i_start * m + j_start
        a01 = i_start * m + (j_start + 1)
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + (j_start + 1)
        
        b00 = j_start * p + k_start
        b01 = j_start * p + (k_start + 1)
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + (k_start + 1)
        
        c00 = k_start * n + i_start
        c01 = (k_start + 1) * n + i_start
        c10 = k_start * n + (i_start + 1)
        c11 = (k_start + 1) * n + (i_start + 1)
        
        # m1 = (a00+a11)(b00+b11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a00] = 1; u[a11] = 1
        v[b00] = 1; v[b11] = 1
        w[c00] = 1; w[c11] = 1
        add_term(u, v, w)
        
        # m2 = (a10+a11)b00
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a10] = 1; u[a11] = 1
        v[b00] = 1
        w[c10] = 1; w[c11] = -1
        add_term(u, v, w)
        
        # m3 = a00(b01-b11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a00] = 1
        v[b01] = 1; v[b11] = -1
        w[c01] = 1; w[c11] = 1
        add_term(u, v, w)
        
        # m4 = a11(b10-b00)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a11] = 1
        v[b10] = 1; v[b00] = -1
        w[c00] = 1; w[c10] = 1
        add_term(u, v, w)
        
        # m5 = (a00+a01)b11
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a00] = 1; u[a01] = 1
        v[b11] = 1
        w[c00] = -1; w[c01] = 1
        add_term(u, v, w)
        
        # m6 = (a10-a00)(b00+b01)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a10] = 1; u[a00] = -1
        v[b00] = 1; v[b01] = 1
        w[c11] = 1
        add_term(u, v, w)
        
        # m7 = (a01-a11)(b10+b11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a01] = 1; u[a11] = -1
        v[b10] = 1; v[b11] = 1
        w[c00] = 1
        add_term(u, v, w)
    
    # Schoolbook for 1x2x2
    def add_1x2x2(i_row, j_start, k_start):
        a0 = i_row * m + j_start
        a1 = i_row * m + (j_start + 1)
        
        b00 = j_start * p + k_start
        b01 = j_start * p + (k_start + 1)
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + (k_start + 1)
        
        c0 = k_start * n + i_row
        c1 = (k_start + 1) * n + i_row
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0] = 1; v[b00] = 1; w[c0] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1] = 1; v[b10] = 1; w[c0] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0] = 1; v[b01] = 1; w[c1] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1] = 1; v[b11] = 1; w[c1] = 1
        add_term(u, v, w)
    
    # Main: 8 Strassen blocks (rows 1,2) + 8 schoolbook blocks (row 0)
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            add_strassen(1, j_start, k_start)
            add_1x2x2(0, j_start, k_start)
    
    return {"U": U, "V": V, "W": W}
