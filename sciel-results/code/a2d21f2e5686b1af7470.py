import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) - index i*m + k for A[i,k]
    # V: (m*p, rank) - index k*p + j for B[k,j]
    # W: (p*n, rank) - index j*n + i for C[i,j]
    
    # Start with pure schoolbook: rank = n*m*p = 42
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Now try to reduce rank using Strassen-like techniques
    # For 2x3x7, we can try to apply Strassen to sub-blocks
    
    # Strategy: For the k-dimension (size 3), we can't do full Strassen (needs even size)
    # But we can try partial optimizations
    
    # Let's try: treat k=0,1 as a Strassen block for j=0,1
    # This saves 1 rank for that 2x2x2 sub-block (4 -> 3)
    
    # Rebuild with Strassen optimization
    rank = 39  # Target rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1
    
    # Strassen block for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # B indices: (0,0)=0, (0,1)=1, (1,0)=7, (1,1)=8
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (A00+A11)(B00+B11)
    add_term([0, 4], [0, 8], [0, 3])
    
    # M2 = (A10+A11)B00
    add_term([3, 4], [0], [2, 3])
    W[3][col-1] = -1
    
    # M3 = A00(B10-B00)
    add_term([0], [7, 0], [1, 3])
    V[0][col-1] = -1
    
    # M4 = A11(B01-B11)
    add_term([4], [1, 8], [0, 2])
    V[8][col-1] = -1
    
    # M5 = (A00+A01)B11
    add_term([0, 1], [8], [0, 1])
    W[0][col-1] = -1
    
    # M6 = (A10-A00)(B00+B10)
    add_term([3, 0], [0, 7], [3])
    U[0][col-1] = -1
    
    # M7 = (A01-A11)(B10+B11)
    add_term([1, 4], [7, 8], [0])
    U[4][col-1] = -1
    
    # Now handle remaining terms:
    # 1. k=0,1 with j=2..6 (schoolbook): 2*2*5 = 20 terms
    for k in [0, 1]:
        for j in range(2, p):
            for i in range(n):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                add_term([u_idx], [v_idx], [w_idx])
    
    # 2. k=2 with all j (schoolbook): 2*1*7 = 14 terms
    for j in range(p):
        for i in range(n):
            u_idx = i * m + 2
            v_idx = 2 * p + j
            w_idx = j * n + i
            add_term([u_idx], [v_idx], [w_idx])
    
    # Total: 7 + 20 + 14 = 41. That's more than 39.
    # We need to be smarter.
    
    # Let's try a different approach: use the seed's exact logic
    # The seed scored 1.0769 = 42/39, so rank 39 is valid.
    
    # Rebuild with seed's exact indexing
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term2(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1
    
    def add_strassen_block(k_start):
        # Seed's indexing:
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        add_term2([0, 4], [b00, b11], [c00, c11])
        add_term2([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        add_term2([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        add_term2([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        add_term2([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1
        add_term2([3, 0], [b00, b01], [c11])
        U[0][col-1] = -1
        add_term2([1, 4], [b10, b11], [c00])
        U[4][col-1] = -1
    
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term2([0], [b06], [c06])
    add_term2([1], [b16], [c06])
    add_term2([3], [b06], [c16])
    add_term2([4], [b16], [c16])
    
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        add_term2([2], [b2k], [c0k])
        add_term2([5], [b2k], [c1k])
    
    return {"U": U, "V": V, "W": W}
