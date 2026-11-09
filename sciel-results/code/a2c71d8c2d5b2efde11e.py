"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 2x2x4 -> 2 * (2x2x2) -> 14
    # 2x2x3 -> (2x2x2) + (2x2x1) -> 7 + 4 = 11
    # 1x2x4 -> 4 * (1x2x1) -> 8
    # 1x2x3 -> 3 * (1x2x1) -> 6
    # Total: 14 + 11 + 8 + 6 = 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0

    def add_2x2x2(oi, oj, ok):
        nonlocal current_rank
        # Strassen decomposition for 2x2x2
        # u, v, w vectors for the 7 products
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # Global: U[(oi+i)*3 + (oj+j)], V[(oj+j)*7 + (ok+k)], W[(ok+k)*3 + (oi+i)]
        
        # M1 = (A00+A11)(B00+B11)
        u1 = [(0,0,1), (1,1,1)] # (i,j, val)
        v1 = [(0,0,1), (1,1,1)] # (j,k, val)
        w1 = [(0,0,1), (1,1,1)] # (k,i, val)
        
        # M2 = (A10+A11)B00
        u2 = [(1,0,1), (1,1,1)]
        v2 = [(0,0,1)]
        w2 = [(0,1,1)]
        
        # M3 = A00(B01-B11)
        u3 = [(0,0,1)]
        v3 = [(0,1,1), (1,1,-1)]
        w3 = [(1,0,1)]
        
        # M4 = A11(B10-B00)
        u4 = [(1,1,1)]
        v4 = [(0,0,-1), (1,0,1)]
        w4 = [(0,1,1)]
        
        # M5 = (A00+A01)B11
        u5 = [(0,0,1), (0,1,1)]
        v5 = [(1,1,1)]
        w5 = [(0,0,-1)]
        
        # M6 = (A10-A00)(B00+B01)
        u6 = [(0,0,-1), (1,0,1)]
        v6 = [(0,0,1), (0,1,1)]
        w6 = [(1,1,1)]
        
        # M7 = (A01-A11)(B10+B11)
        u7 = [(0,1,1), (1,1,-1)]
        v7 = [(1,0,1), (1,1,1)]
        w7 = [(0,0,1)]
        
        products = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for u_vec, v_vec, w_vec in products:
            r = current_rank
            current_rank += 1
            for i, j, val in u_vec:
                U[(oi+i)*3 + (oj+j)][r] += val
            for j, k, val in v_vec:
                V[(oj+j)*7 + (ok+k)][r] += val
            for k, i, val in w_vec:
                W[(ok+k)*3 + (oi+i)][r] += val

    def add_2x2x1(oi, oj, ok):
        nonlocal current_rank
        # Schoolbook for 2x2x1
        # A is 2x2, B is 2x1.
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # Mult 1: A00 B00
        r = current_rank; current_rank += 1
        U[(oi+0)*3 + (oj+0)][r] = 1
        V[(oj+0)*7 + (ok+0)][r] = 1
        W[(ok+0)*3 + (oi+0)][r] = 1
        
        # Mult 2: A01 B10
        r = current_rank; current_rank += 1
        U[(oi+0)*3 + (oj+1)][r] = 1
        V[(oj+1)*7 + (ok+0)][r] = 1
        W[(ok+0)*3 + (oi+0)][r] = 1
        
        # Mult 3: A10 B00
        r = current_rank; current_rank += 1
        U[(oi+1)*3 + (oj+0)][r] = 1
        V[(oj+0)*7 + (ok+0)][r] = 1
        W[(ok+0)*3 + (oi+1)][r] = 1
        
        # Mult 4: A11 B10
        r = current_rank; current_rank += 1
        U[(oi+1)*3 + (oj+1)][r] = 1
        V[(oj+1)*7 + (ok+0)][r] = 1
        W[(ok+0)*3 + (oi+1)][r] = 1

    def add_1x2x1(oi, oj, ok):
        nonlocal current_rank
        # Schoolbook for 1x2x1
        # A is 1x2, B is 2x1.
        # C00 = A00 B00 + A01 B10
        
        # Mult 1: A00 B00
        r = current_rank; current_rank += 1
        U[(oi+0)*3 + (oj+0)][r] = 1
        V[(oj+0)*7 + (ok+0)][r] = 1
        W[(ok+0)*3 + (oi+0)][r] = 1
        
        # Mult 2: A01 B10
        r = current_rank; current_rank += 1
        U[(oi+0)*3 + (oj+1)][r] = 1
        V[(oj+1)*7 + (ok+0)][r] = 1
        W[(ok+0)*3 + (oi+0)][r] = 1

    # Block 1: Top-Left 2x2x4 (split into two 2x2x2)
    add_2x2x2(0, 0, 0) # k=0,1
    add_2x2x2(0, 0, 2) # k=2,3
    
    # Block 2: Top-Right 2x2x3 (split into 2x2x2 and 2x2x1)
    add_2x2x2(0, 0, 4) # k=4,5
    add_2x2x1(0, 0, 6) # k=6
    
    # Block 3: Bottom-Left 1x2x4 (four 1x2x1)
    for k in range(4):
        add_1x2x1(2, 0, k)
        
    # Block 4: Bottom-Right 1x2x3 (three 1x2x1)
    for k in range(4, 7):
        add_1x2x1(2, 0, k)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
