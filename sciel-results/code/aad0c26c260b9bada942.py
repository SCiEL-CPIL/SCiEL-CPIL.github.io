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
    
    # Strassen vectors for 2x2x2 multiplication
    # Indices 0..3 correspond to the 4 elements of the 2x2 matrix
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
    
    num_blocks = p // 2
    for r in range(num_blocks):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # Fill U (shared across blocks, same vectors)
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U[li][col] = u_vecs[s][li]
                    
            # Fill V (block specific)
            # V row index is k * p + j.
            # Block r corresponds to columns 2*r and 2*r+1 of B.
            # So indices are k * p + 2*r + offset.
            v_indices = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V[v_indices[li]][col] = v_vecs[s][li]
                    
            # Fill W (block specific)
            # W row index is j * n + i.
            # Block r corresponds to columns 2*r and 2*r+1 of C.
            # So indices are (2*r+offset) * 2 + i.
            w_indices = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W[w_indices[li]][col] = w_vecs[s][li]
    
    # Quick verification to ensure correctness
    # T[i,k,j] = sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r] should be 1
    max_err = 0.0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r_idx in range(rank):
                    val += U[i*m+k][r_idx] * V[k*p+j][r_idx] * W[j*n+i][r_idx]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to naive rank 64 if verification fails
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
    if elapsed > 140:
        # Return immediately if close to limit
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
