import time

DIMENSIONS = [2, 2, 12]

def construct():
    """Rank-42 Strassen decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 6 pairs of 2. Apply 7-product Strassen to each 2x2 block.
    Total rank = 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    num_pairs = p // 2  # 6
    
    rank = num_pairs * 7  # 42
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    # Strassen 7 products for 2x2 * 2x2
    # A elements: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B elements within block: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    
    products = [
        ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),       # M1=(A11+A22)(B11+B22)
        ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)]),                    # M2=(A21+A22)B11
        ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)]),                   # M3=A11(B12-B22)
        ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)]),                   # M4=A22(B21-B11)
        ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)]),                    # M5=(A11+A12)B22
        ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),       # M6=(A21-A11)(B11+B12)
        ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),       # M7=(A12-A22)(B21+B22)
    ]
    
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],     # M1: C11+=1, C22+=1
        [((1,0), 1), ((1,1), -1)],    # M2: C21+=1, C22+=-1
        [((0,1), 1), ((1,1), 1)],     # M3: C12+=1, C22+=1
        [((0,0), 1), ((1,0), 1)],     # M4: C11+=1, C21+=1
        [((0,0), -1), ((0,1), 1)],    # M5: C11+=-1, C12+=1
        [((1,1), 1)],                 # M6: C22+=1
        [((0,0), 1)],                 # M7: C11+=1
    ]
    
    col = 0
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base_col = 2 * t
        for s in range(7):
            for (ai, aj), coeff in products[s][0]:
                u_row = ai * m + aj
                U[u_row][col] = coeff
            
            for (bj, bk_rel), coeff in products[s][1]:
                b_col = base_col + bk_rel
                v_row = bj * p + b_col
                V[v_row][col] = coeff
            
            for (ci, ck_rel), coeff in output_contribs[s]:
                c_col = base_col + ck_rel
                w_row = c_col * n + ci
                W[w_row][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
