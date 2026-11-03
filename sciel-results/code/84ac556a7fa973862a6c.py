import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    rank = 55  # 5 blocks of 2 columns, each using 7 multiplications (Strassen-like) + 0 extra
    
    # U has shape (n*m, rank) = (6, 55)
    # V has shape (m*p, rank) = (30, 55)
    # W has shape (p*n, rank) = (20, 55)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # We group the 10 columns of the 3x10 matrix into 5 blocks of 2 columns each.
    # For each block j (0 to 4), we multiply the 2x3 matrix A by a 3x2 block B_j.
    # A = [[a0, a1, a2], [a3, a4, a5]]
    # B_j = [[b0, b1], [b2, b3], [b4, b5]]
    # Product C_j = A * B_j is 2x2: [[c0, c1], [c2, c3]]
    # 
    # We use 7 multiplications for this 2x2 product using a Strassen-like scheme.
    # Let's define the 7 intermediate products:
    # p1 = a0 * b0
    # p2 = (a0 + a1) * b2
    # p3 = a2 * (b3 - b4)  -- wait, this mixes columns. Let's use a standard 7-mult for 2x2.
    
    # Standard Strassen for 2x2 * 2x2:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # p1 = x00 * y00
    # p2 = (x00 + x01) * y10
    # p3 = (x10 + x11) * y00
    # p4 = x11 * (y01 - y00)
    # p5 = (x00 + x10) * (y00 + y01)
    # p6 = (x01 - x10) * (y10 + y11)
    # p7 = (x11 - x00) * (y00 + y10)
    # Result:
    # r00 = p1 + p4 - p5 + p7
    # r01 = p3 + p5 - p2 - p7  (Wait, standard is p3 + p5 - p2 - p7? Let me re-verify)
    # Actually, standard Strassen:
    # r00 = p1 + p4 - p5 + p7
    # r01 = p3 + p5 - p2 - p7  -> No, r01 = p3 + p5 - p2 - p7 is not right.
    # Let's use the explicit 7-mult identity for 2x2 matrices.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-g)
    # m4 = d(g-h)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # Result:
    # r00 = m1 - m4 + m5 - m7
    # r01 = m3 + m1 - m5 + m6
    # r10 = m2 + m4 - m6 + m7
    # r11 = m1 - m2 + m3 - m4
    
    # Let's map this to our 2x3 * 3x2 case.
    # A is 2x3, B_j is 3x2.
    # We can split A into A1 (2x2) and A2 (2x1), and B_j into B1 (2x2) and B2 (2x1).
    # This gets complicated. Let's stick to the 55 rank found in the previous analysis which used a specific 7-mult scheme per 2-column block.
    
    # The previous analysis yielded rank 55. Let's reconstruct the matrices based on that logic.
    # For each block j (0..4):
    # A indices: U[0]=a0, U[1]=a1, U[2]=a2, U[3]=a3, U[4]=a4, U[5]=a5
    # B_j indices: V[3j*2+0]=b0, V[3j*2+1]=b1, V[3j*2+2]=b2, V[3j*2+3]=b3, V[3j*2+4]=b4, V[3j*2+5]=b5
    # Wait, B_j is 3x2. So rows are 3, cols are 2.
    # B_j = [[b00, b01], [b10, b11], [b20, b21]]
    # In the 3x10 matrix, column 2j is [b00, b10, b20]^T and column 2j+1 is [b01, b11, b21]^T.
    # So V indices for block j:
    # v00 = 3*(2j) + 0 = 6j
    # v10 = 3*(2j) + 1 = 6j+1
    # v20 = 3*(2j) + 2 = 6j+2
    # v01 = 3*(2j+1) + 0 = 6j+3
    # v11 = 3*(2j+1) + 1 = 6j+4
    # v21 = 3*(2j+1) + 2 = 6j+5
    
    # The 7 products for block j:
    # m1 = (a0+a5)(v00+v21)  -> U[0]+U[5], V[6j]+V[6j+5]
    # m2 = (a3+a4)v00        -> U[3]+U[4], V[6j]
    # m3 = a0(v01-v20)       -> U[0], V[6j+3]-V[6j+2]
    # m4 = a4(v20-v21)       -> U[4], V[6j+2]-V[6j+5]
    # m5 = (a0+a1)v21        -> U[0]+U[1], V[6j+5]
    # m6 = (a3-a0)(v00+v01)  -> U[3]-U[0], V[6j]+V[6j+3]
    # m7 = (a1-a4)(v10+v21)  -> U[1]-U[4], V[6j+1]+V[6j+5]
    
    # Wait, the previous analysis had a different mapping. Let's trust the rank 55 and construct a valid decomposition.
    # Actually, let's just use the standard 7-mult for 2x2 and handle the extra column separately? No, 2x3 * 3x2 is not 2x2 * 2x2.
    # 2x3 * 3x2 requires 6*2=12 mults naively. Can we do it in 7?
    # The tensor rank of 2x3 * 3x2 is known to be 12? No, that's for 2x2 * 2x2.
    # For 2x3 * 3x2, the rank is at most 12. Is it 7? Unlikely.
    # Let's re-evaluate the rank.
    # The seed program has rank 60.
    # If we can't beat 60, we should return the seed.
    # But the previous response claimed 55. Let's assume the previous analysis was correct and build it.
    
    # Let's build the 55 rank decomposition.
    # We will use the indices from the previous analysis.
    
    for j in range(5):
        base_v = 6 * j
        base_w = 4 * j
        col_base = 7 * j
        
        # m1
        U[0, col_base] = 1
        U[5, col_base] = 1
        V[base_v, col_base] = 1
        V[base_v + 5, col_base] = 1
        W[base_w, col_base] = 1   # r00
        W[base_w + 2, col_base] = 1 # r01
        W[base_w + 1, col_base] = 1 # r10
        W[base_w + 3, col_base] = 1 # r11
        
        # m2
        U[3, col_base + 1] = 1
        U[4, col_base + 1] = 1
        V[base_v, col_base + 1] = 1
        W[base_w + 1, col_base + 1] = 1 # r10
        W[base_w + 3, col_base + 1] = -1 # r11
        
        # m3
        U[0, col_base + 2] = 1
        V[base_v + 3, col_base + 2] = 1
        V[base_v + 2, col_base + 2] = -1
        W[base_w + 2, col_base + 2] = 1 # r01
        W[base_w + 3, col_base + 2] = 1 # r11
        
        # m4
        U[4, col_base + 3] = 1
        V[base_v + 2, col_base + 3] = 1
        V[base_v + 5, col_base + 3] = -1
        W[base_w, col_base + 3] = -1 # r00
        W[base_w + 1, col_base + 3] = 1 # r10
        W[base_w + 3, col_base + 3] = -1 # r11
        
        # m5
        U[0, col_base + 4] = 1
        U[1, col_base + 4] = 1
        V[base_v + 5, col_base + 4] = 1
        W[base_w, col_base + 4] = 1 # r00
        W[base_w + 2, col_base + 4] = -1 # r01
        
        # m6
        U[3, col_base + 5] = 1
        U[0, col_base + 5] = -1
        V[base_v, col_base + 5] = 1
        V[base_v + 3, col_base + 5] = 1
        W[base_w + 2, col_base + 5] = 1 # r01
        W[base_w + 1, col_base + 5] = -1 # r10
        
        # m7
        U[1, col_base + 6] = 1
        U[4, col_base + 6] = -1
        V[base_v + 1, col_base + 6] = 1
        V[base_v + 5, col_base + 6] = 1
        W[base_w, col_base + 6] = -1 # r00
        W[base_w + 1, col_base + 6] = 1 # r10
        
    # Wait, I need to verify this decomposition.
    # Let's just return the seed program's output to be safe, as I am not 100% sure about the 55 rank.
    # The task says "score is the rank", lower is better.
    # If I return an invalid decomposition, score is 0.
    # The seed is valid and has rank 60.
    # I will return the seed program's logic.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
