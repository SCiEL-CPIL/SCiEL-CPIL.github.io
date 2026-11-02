import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Rank-56 decomposition: 8 independent 2x2x2 blocks, each using Strassen (rank 7)
    # The 2x2x16 multiplication is a direct sum of 8 copies of 2x2x2 multiplication
    # (since B is 2x16 = 2x(2*8), we get 8 independent 2x2 blocks)
    # Each 2x2x2 block has tensor rank 7 (Strassen), total rank = 8*7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (32, 56)
    
    # Strassen vectors for 2x2 matrix multiplication (rank 7)
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    u_vecs = [
        [1, 0, 0, 1],   # M1: a + d
        [0, 0, 1, 1],   # M2: c + d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a + b
        [-1, 0, 1, 0],  # M6: c - a
        [0, 1, 0, -1]   # M7: b - d
    ]
    v_vecs = [
        [1, 0, 0, 1],   # M1: e + h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f - h
        [-1, 0, 1, 0],  # M4: g - e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e + f
        [0, 0, 1, 1]    # M7: g + h
    ]
    w_vecs = [
        [1, 0, 0, 1],   # M1: c00 + c11
        [0, 0, 1, -1],  # M2: c10 - c11
        [0, 1, 0, 1],   # M3: c01 + c11
        [1, 0, 1, 0],   # M4: c00 + c10
        [-1, 1, 0, 0],  # M5: -c00 + c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 0]    # M7: c00
    ]
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # U: A indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for li in range(4):
                val = u_vecs[s][li]
                if val != 0:
                    U[li][col] = val
                    
            # V: B indices for block r
            # li=0 -> B[0, 2r] -> V[2r]
            # li=1 -> B[0, 2r+1] -> V[2r+1]
            # li=2 -> B[1, 2r] -> V[16+2r]
            # li=3 -> B[1, 2r+1] -> V[16+2r+1]
            v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                val = v_vecs[s][li]
                if val != 0:
                    V[v_idx[li]][col] = val
                    
            # W: C indices for block r
            # W index = k*n + i where k is column, i is row
            # li=0 -> C[0, 2r] -> W[(2r)*2 + 0] = W[4r]
            # li=1 -> C[0, 2r+1] -> W[(2r+1)*2 + 0] = W[4r+2]
            # li=2 -> C[1, 2r] -> W[(2r)*2 + 1] = W[4r+1]
            # li=3 -> C[1, 2r+1] -> W[(2r+1)*2 + 1] = W[4r+3]
            w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                val = w_vecs[s][li]
                if val != 0:
                    W[w_idx[li]][col] = val
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
