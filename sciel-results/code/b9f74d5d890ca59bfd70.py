import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 2, 10
    rank = 35
    
    nU = n * m  # 4
    nV = m * p  # 20
    nW = p * n  # 20
    
    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]
    
    # Verified terms from the successful run
    # Order: (U_vec, V_vec, W_vec)
    # U_vec: [a00, a01, a10, a11]
    # V_vec: [b00, b01, b10, b11]
    # W_vec: [c00, c10, c01, c11]
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for block in range(p // 2):
        for a_vec, b_vec, c_vec in terms:
            # U
            for idx in range(nU):
                U[idx][col] = a_vec[idx]
            
            # V
            # B local indices 0,1,2,3 map to:
            # 0: 0*p + 2*block
            # 1: 0*p + 2*block + 1
            # 2: 1*p + 2*block
            # 3: 1*p + 2*block + 1
            V[0 * p + 2 * block][col] = b_vec[0]
            V[0 * p + 2 * block + 1][col] = b_vec[1]
            V[1 * p + 2 * block][col] = b_vec[2]
            V[1 * p + 2 * block + 1][col] = b_vec[3]
            
            # W
            # C local indices 0,1,2,3 map to:
            # 0 (c00): (2*block)*n + 0
            # 1 (c10): (2*block)*n + 1
            # 2 (c01): (2*block+1)*n + 0
            # 3 (c11): (2*block+1)*n + 1
            W[2 * block * n + 0][col] = c_vec[0]
            W[2 * block * n + 1][col] = c_vec[1]
            W[(2 * block + 1) * n + 0][col] = c_vec[2]
            W[(2 * block + 1) * n + 1][col] = c_vec[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
