import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use a block decomposition.
    # Group columns of B (size m x p = 2 x 11) into blocks of size 2, plus one block of size 1.
    # There are 5 blocks of size 2 and 1 block of size 1.
    # For each block of size 2, we use Strassen's algorithm for 2x2x2 multiplication (rank 7).
    # For the block of size 1, we use the trivial 2x2x1 multiplication (rank 4).
    # Total rank = 5 * 7 + 4 = 39.
    
    # Strassen coefficients for 2x2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U indices: i*m + j (i in 0..1, j in 0..1)
    # V indices: j*p + k (j in 0..1, k in 0..p-1)
    # W indices: k*n + i (k in 0..p-1, i in 0..1)
    
    strassen_U = [
        [1, 0, 1, 0],  # M1: a, M2: c, M3: a, M4: d
        [0, 1, 0, 1],  # M1: d, M2: d, M3: 0, M4: 0
        [1, 0, 0, 1],  # M5: a, M6: c, M7: b
        [0, 1, 1, 0],  # M5: b, M6: 0, M7: 0
        [0, 1, -1, 0], # M6: c, M7: 0
        [0, 0, 1, -1]  # M7: b
    ]
    # Wait, the above matrix is wrong. Let's re-derive carefully.
    # M1: (a+d) -> U col 0: [1, 0, 0, 1] (a at (0,0), d at (1,1))
    # M2: (c+d) -> U col 1: [0, 0, 1, 1] (c at (1,0), d at (1,1))
    # M3: a -> U col 2: [1, 0, 0, 0]
    # M4: d -> U col 3: [0, 0, 0, 1]
    # M5: (a+b) -> U col 4: [1, 1, 0, 0]
    # M6: (c-a) -> U col 5: [-1, 0, 1, 0]
    # M7: (b-d) -> U col 6: [0, 1, 0, -1]
    
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],  # i=0, j=0 (a)
        [0, 0, 0, 0, 1, 0, 1],   # i=0, j=1 (b)
        [0, 1, 0, 0, 0, 1, 0],   # i=1, j=0 (c)
        [1, 1, 0, 1, 0, 0, -1]   # i=1, j=1 (d)
    ]
    
    # V factors for 2x2x2 (local k=0,1):
    # M1: (e+h) -> V col 0: [1, 0, 0, 1] (e at (0,0), h at (1,1))
    # M2: e -> V col 1: [1, 0, 0, 0]
    # M3: (f-h) -> V col 2: [0, 1, 0, -1]
    # M4: (g-e) -> V col 3: [-1, 0, 1, 0]
    # M5: h -> V col 4: [0, 0, 0, 1]
    # M6: (e+f) -> V col 5: [1, 1, 0, 0]
    # M7: (g+h) -> V col 6: [0, 0, 1, 1]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],  # j=0, k=0 (e)
        [0, 0, 1, 0, 0, 1, 0],   # j=0, k=1 (f)
        [0, 0, 0, 1, 0, 0, 1],   # j=1, k=0 (g)
        [1, 0, -1, 0, 1, 0, 1]   # j=1, k=1 (h)
    ]
    
    # W factors for 2x2x2 (local k=0,1, i=0,1):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # W indices: k*n + i.
    # k=0, i=0 -> idx 0 (C00)
    # k=0, i=1 -> idx 1 (C10)
    # k=1, i=0 -> idx 2 (C01)
    # k=1, i=1 -> idx 3 (C11)
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    
    # Trivial 2x2x1 (1 column):
    # C00 = ae
    # C01 = af
    # C10 = ce
    # C11 = cf
    # Wait, 2x2x1 means A is 2x2, B is 2x1. Output C is 2x1.
    # So only 2 outputs: C0, C1.
    # C0 = a*e + b*g
    # C1 = c*e + d*g
    # Rank 4:
    # M1 = a*e
    # M2 = b*g
    # M3 = c*e
    # M4 = d*g
    # C0 = M1 + M2
    # C1 = M3 + M4
    
    trivial_U = [
        [1, 0, 0, 0],  # a
        [0, 1, 0, 0],  # b
        [0, 0, 1, 0],  # c
        [0, 0, 0, 1]   # d
    ]
    trivial_V = [
        [1, 0, 1, 0],  # e
        [0, 1, 0, 1]   # g
    ]
    # W indices: k*n + i. Here k=0 (only 1 col), i in 0..1.
    # idx 0: C0, idx 1: C1
    trivial_W = [
        [1, 1, 0, 0],  # C0
        [0, 0, 1, 1]   # C1
    ]
    
    total_rank = 5 * 7 + 4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    # Process 5 blocks of size 2
    for block_idx in range(5):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        for r in range(7):
            col = rank_offset + r
            
            # U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = strassen_U[i * m + j][r]
            
            # V
            for j in range(m):
                # k0 corresponds to local k=0
                V[j * p + k0][col] = strassen_V[j * 2 + 0][r]
                # k1 corresponds to local k=1
                V[j * p + k1][col] = strassen_V[j * 2 + 1][r]
            
            # W
            # Local k=0 -> global k0
            W[k0 * n + 0][col] = strassen_W[0 * n + 0][r]
            W[k0 * n + 1][col] = strassen_W[0 * n + 1][r]
            # Local k=1 -> global k1
            W[k1 * n + 0][col] = strassen_W[1 * n + 0][r]
            W[k1 * n + 1][col] = strassen_W[1 * n + 1][r]
        
        rank_offset += 7
    
    # Process 1 block of size 1
    k0 = 10
    for r in range(4):
        col = rank_offset + r
        
        # U
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = trivial_U[i * m + j][r]
        
        # V
        for j in range(m):
            V[j * p + k0][col] = trivial_V[j * 1 + 0][r]
        
        # W
        W[k0 * n + 0][col] = trivial_W[0 * n + 0][r]
        W[k0 * n + 1][col] = trivial_W[0 * n + 1][r]
        
        rank_offset += 4
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
