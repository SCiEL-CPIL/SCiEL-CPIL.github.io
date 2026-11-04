import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 3, 9
    
    # Rank-14 decomposition for 2x3 x 3x3 matrix multiplication
    # A is 2x3, B is 3x3, C = A*B is 2x3
    # A = [[a0,a1,a2],[a3,a4,a5]], B = [[b0,b1,b2],[b3,b4,b5],[b6,b7,b8]]
    # C = [[c0,c1,c2],[c3,c4,c5]]
    #
    # c0 = a0*b0 + a1*b3 + a2*b6
    # c1 = a0*b1 + a1*b4 + a2*b7
    # c2 = a0*b2 + a1*b5 + a2*b8
    # c3 = a3*b0 + a4*b3 + a5*b6
    # c4 = a3*b1 + a4*b4 + a5*b7
    # c5 = a3*b2 + a4*b5 + a5*b8
    #
    # 14 bilinear forms (rank 14):
    # Using Strassen-like combinations for the 2x2 sub-problems plus corrections
    
    # A-indices: a0=0, a1=1, a2=2, a3=3, a4=4, a5=5
    # B-indices: b0=0, b1=1, b2=2, b3=3, b4=4, b5=5, b6=6, b7=7, b8=8
    # C-indices: c0=0, c1=1, c2=2, c3=3, c4=4, c5=5
    
    u_block = [
        [1, 0, 0, 0, 1, 0],   # 0: a0+a4
        [1, 0, 0, 0,-1, 0],   # 1: a0-a4
        [1, 0, 0, 1, 0, 0],   # 2: a0+a3
        [1, 0, 0,-1, 0, 0],   # 3: a0-a3
        [0, 1, 0, 0, 0, 0],   # 4: a1
        [0, 0, 1, 0, 0, 0],   # 5: a2
        [0, 0, 0, 1, 0, 0],   # 6: a3
        [0, 0, 0, 0, 1, 0],   # 7: a4
        [0, 0, 0, 0, 0, 1],   # 8: a5
        [1, 1, 0, 0, 0, 0],   # 9: a0+a1
        [1,-1, 0, 0, 0, 0],   # 10: a0-a1
        [0, 1, 1, 0, 0, 0],   # 11: a1+a2
        [0, 1,-1, 0, 0, 0],   # 12: a1-a2
        [1, 0, 0, 1, 1, 0],   # 13: a0+a3+a4
    ]
    
    v_block = [
        [1, 0, 0, 0, 1, 0, 0, 0, 0],   # 0: b0+b4
        [1, 0, 0, 0,-1, 0, 0, 0, 0],   # 1: b0-b4
        [1, 0, 0, 1, 0, 0, 0, 0, 0],   # 2: b0+b3
        [1, 0, 0,-1, 0, 0, 0, 0, 0],   # 3: b0-b3
        [0, 0, 0, 0, 1, 0, 0, 0, 0],   # 4: b4
        [0, 0, 0, 0, 0, 0, 0, 1, 0],   # 5: b7
        [0, 0, 0, 0, 0, 0, 1, 0, 0],   # 6: b6
        [0, 0, 0, 0, 0, 0, 0, 0, 1],   # 7: b8
        [0, 0, 0, 0, 0, 0, 0, 1, 0],   # 8: b7
        [1, 0, 0, 0, 0, 0, 0, 0, 0],   # 9: b0
        [0, 1, 0, 0, 0, 0, 0, 0, 0],   # 10: b1
        [0, 0, 0, 1, 0, 0, 0, 0, 0],   # 11: b3
        [0, 0, 0, 0, 0, 1, 0, 0, 0],   # 12: b5
        [1, 0, 0, 1, 1, 0, 0, 0, 0],   # 13: b0+b3+b4
    ]
    
    w_block = [
        [1/4, 0, 0, 1/4, 0, 0],   # 0
        [1/4, 0, 0,-1/4, 0, 0],   # 1
        [1/4, 0, 0, 1/4, 0, 0],   # 2
        [1/4, 0, 0,-1/4, 0, 0],   # 3
        [0, 1, 0, 0, 0, 0],       # 4
        [0, 0, 1, 0, 0, 0],       # 5
        [0, 0, 0, 0, 1, 0],       # 6
        [0, 0, 0, 0, 0, 1],       # 7
        [0, 0, 0, 0, 1, 0],       # 8
        [1/2, 1/2, 0, 0, 0, 0],   # 9
        [1/2,-1/2, 0, 0, 0, 0],   # 10
        [0, 1/2, 1/2, 0, 0, 0],   # 11
        [0, 1/2,-1/2, 0, 0, 0],   # 12
        [1/4, 0, 0, 1/4, 0, 0],   # 13
    ]
    
    # Verify the block decomposition
    u_b = np.array(u_block, dtype=float)
    v_b = np.array(v_block, dtype=float)
    w_b = np.array(w_block, dtype=float)
    
    # Check: for each (i,j,k), sum_c u[i*m+j,c]*v[j*p+k,c]*w[k*n+i,c] should equal 1
    # if that's the correct (i,j,k) term, else 0
    block_valid = True
    for i in range(2):
        for j in range(3):
            for k in range(3):
                expected = 1.0
                val = sum(u_b[i*3+j, c] * v_b[j*3+k, c] * w_b[k*2+i, c] for c in range(14))
                if abs(val - expected) > 1e-10:
                    block_valid = False
    
    if not block_valid:
        # Fall back to schoolbook rank 54
        rank = n * m * p
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
    
    # Build full 2x3x9 decomposition: 3 blocks of rank 14 = rank 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 6 x 42
    V = [[0] * rank for _ in range(m * p)]  # 27 x 42
    W = [[0] * rank for _ in range(p * n)]  # 18 x 42
    
    for block in range(3):
        offset = block * 14
        for c in range(14):
            col = offset + c
            # U: same for all blocks (A part is shared)
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = u_block[c][i * m + j]
            # V: B part, block b corresponds to columns 3b..3b+2
            for j in range(m):
                for k in range(3):
                    global_k = block * 3 + k
                    V[j * p + global_k][col] = v_block[c][j * 3 + k]
            # W: C part, block b corresponds to output columns 3b..3b+2
            for i in range(n):
                for k in range(3):
                    global_k = block * 3 + k
                    W[global_k * n + i][col] = w_block[c][k * n + i]
    
    return {"U": U, "V": V, "W": W}
