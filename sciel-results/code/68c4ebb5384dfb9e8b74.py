import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 56 decomposition: 8 independent 2x2 multiplications via Strassen (rank 7 each)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    u_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    v_vecs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U[li][col] = u_vecs[s][li]
                    
            v_indices = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V[v_indices[li]][col] = v_vecs[s][li]
                    
            w_indices = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W[w_indices[li]][col] = w_vecs[s][li]
    
    # Verify correctness: T[i,k,j] = sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r] should be 1
    U_np = [[float(x) for x in row] for row in U]
    V_np = [[float(x) for x in row] for row in V]
    W_np = [[float(x) for x in row] for row in W]
    
    max_err = 0.0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+k][r] * V_np[k*p+j][r] * W_np[j*n+i][r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to naive rank 64
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = i * m * p + k * p + j
                    U[i*m+k][idx] = 1
                    V[k*p+j][idx] = 1
                    W[j*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    elapsed = time.time() - start_time
    if elapsed > 130:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
