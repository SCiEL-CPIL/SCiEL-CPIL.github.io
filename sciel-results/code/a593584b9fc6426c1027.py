import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # Rank 42 using Strassen on 6 blocks of 2x2x2
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 4 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 24 x 42
    
    # Strassen vectors for 2x2x2
    # u, v, w are lists of 4 elements corresponding to indices 0,1,2,3
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    U_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, -1, 0],
        [0, 1, 0, -1]
    ]
    
    V_vecs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2 # 6 blocks
    
    for b in range(num_blocks):
        # Map local indices 0,1,2,3 to global V indices
        # Local 0: B[0, 2b] -> 0*12 + 2b
        # Local 1: B[0, 2b+1] -> 0*12 + 2b+1
        # Local 2: B[1, 2b] -> 1*12 + 2b
        # Local 3: B[1, 2b+1] -> 1*12 + 2b+1
        v_map = [
            0 * p + 2 * b,
            0 * p + 2 * b + 1,
            1 * p + 2 * b,
            1 * p + 2 * b + 1
        ]
        
        # Map local indices 0,1,2,3 to global W indices
        # Local 0: C[0, 2b] -> (2b)*2 + 0
        # Local 1: C[0, 2b+1] -> (2b+1)*2 + 0
        # Local 2: C[1, 2b] -> (2b)*2 + 1
        # Local 3: C[1, 2b+1] -> (2b+1)*2 + 1
        w_map = [
            (2 * b) * n + 0,
            (2 * b + 1) * n + 0,
            (2 * b) * n + 1,
            (2 * b + 1) * n + 1
        ]
        
        for s in range(7):
            col = b * 7 + s
            
            # Fill U
            for i in range(4):
                U[i][col] = U_vecs[s][i]
                
            # Fill V
            for i in range(4):
                val = V_vecs[s][i]
                if val != 0:
                    V[v_map[i]][col] = val
                    
            # Fill W
            for i in range(4):
                val = W_vecs[s][i]
                if val != 0:
                    W[w_map[i]][col] = val
                    
    return {"U": U, "V": V, "W": W}
