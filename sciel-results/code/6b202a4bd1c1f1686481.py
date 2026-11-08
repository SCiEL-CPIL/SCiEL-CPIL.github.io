import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Part 2 (A2 B2): 2 * 1 * 7 = 14 terms
    # Part 1 (A1 B1):
    #   3 blocks of 2x2x2 using Strassen (7 terms each) = 21 terms
    #   1 block of 2x2x1 using Schoolbook (4 terms) = 4 terms
    # Total Rank = 14 + 21 + 4 = 39
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # --- Part 2: A2 (2x1) * B2 (1x7) ---
    # A2 is column j=2 of A. Indices in U: i*3 + 2 for i in 0,1
    # B2 is row j=2 of B. Indices in V: 2*7 + k for k in 0..6
    # Result C indices: k*2 + i
    
    for i in range(n):
        for k in range(p):
            U[i * m + 2, col] = 1.0
            V[2 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # --- Part 1: A1 (2x2) * B1 (2x7) ---
    # A1 is columns j=0,1 of A. Indices in U: 
    #   A[0,0]->0, A[0,1]->1, A[1,0]->3, A[1,1]->4
    # B1 is rows j=0,1 of B.
    
    # Strassen decomposition for 2x2x2
    # Local U indices: 0(a), 1(b), 2(c), 3(d)
    # Local V indices: 0(e), 1(f), 2(g), 3(h)
    # Local W indices: 0(C00), 1(C01), 2(C10), 3(C11)
    
    strassen_U = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1], # P3
        [0, 0, 1, -1], # P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    strassen_W = [
        [1, 0, 0, -1], # P1
        [0, 0, 1, -1], # P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, -1]  # P7
    ]
    
    # Mapping for U (A1)
    u_map = [0, 1, 3, 4]
    
    # Define blocks for B1 (k ranges)
    # Block 0: k=0,1
    # Block 1: k=2,3
    # Block 2: k=4,5
    # Block 3: k=6 (Schoolbook)
    
    for block_idx in range(3):
        k_start = block_idx * 2
        k0 = k_start
        k1 = k_start + 1
        
        # V mapping for this block
        # e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
        v_map = [
            0 * p + k0, # e
            0 * p + k1, # f
            1 * p + k0, # g
            1 * p + k1  # h
        ]
        
        # W mapping for this block
        # C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        # Indices: k*n + i
        w_map = [
            k0 * n + 0, # C[0,k0]
            k1 * n + 0, # C[0,k1]
            k0 * n + 1, # C[1,k0]
            k1 * n + 1  # C[1,k1]
        ]
        
        for t in range(7):
            for loc_u, val_u in enumerate(strassen_U[t]):
                if val_u != 0:
                    U[u_map[loc_u], col] += val_u
            for loc_v, val_v in enumerate(strassen_V[t]):
                if val_v != 0:
                    V[v_map[loc_v], col] += val_v
            for loc_w, val_w in enumerate(strassen_W[t]):
                if val_w != 0:
                    W[w_map[loc_w], col] += val_w
            col += 1
            
    # Block 3: k=6 (Schoolbook 2x2x1)
    k = 6
    # Terms:
    # A[0,0] * B[0,6] -> C[0,6]
    # A[0,1] * B[1,6] -> C[0,6]
    # A[1,0] * B[0,6] -> C[1,6]
    # A[1,1] * B[1,6] -> C[1,6]
    
    # Term 1
    U[0, col] = 1.0
    V[0 * p + k, col] = 1.0
    W[k * n + 0, col] = 1.0
    col += 1
    
    # Term 2
    U[1, col] = 1.0
    V[1 * p + k, col] = 1.0
    W[k * n + 0, col] = 1.0
    col += 1
    
    # Term 3
    U[3, col] = 1.0
    V[0 * p + k, col] = 1.0
    W[k * n + 1, col] = 1.0
    col += 1
    
    # Term 4
    U[4, col] = 1.0
    V[1 * p + k, col] = 1.0
    W[k * n + 1, col] = 1.0
    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
