"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition coefficients
    # u, v, w are [top-left, top-right, bottom-left, bottom-right]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, -1, 0, 1])
    ]
    
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Define the 2x2 block indices for U (A matrix, 2x4)
    # A_L: cols 0-1 -> indices 0,1,2,3
    # A_R: cols 2-3 -> indices 4,5,6,7
    u_blocks = {
        'L': [0, 1, 2, 3],
        'R': [4, 5, 6, 7]
    }
    
    # Define the 2x2 block indices for V (B matrix, 4x6)
    # B is 4x6. Row-major index = row * 6 + col.
    # Top half (rows 0-1):
    #   B_T1 (cols 0-1): 0,1, 6,7
    #   B_T2 (cols 2-3): 2,3, 8,9
    #   B_T3 (cols 4-5): 4,5, 10,11
    # Bottom half (rows 2-3):
    #   B_B1 (cols 0-1): 12,13, 18,19
    #   B_B2 (cols 2-3): 14,15, 20,21
    #   B_B3 (cols 4-5): 16,17, 22,23
    v_blocks = {
        'T1': [0, 1, 6, 7],
        'T2': [2, 3, 8, 9],
        'T3': [4, 5, 10, 11],
        'B1': [12, 13, 18, 19],
        'B2': [14, 15, 20, 21],
        'B3': [16, 17, 22, 23]
    }
    
    # Define the 2x2 block indices for W (C matrix, 2x6)
    # C is 2x6. Row-major index = row * 6 + col.
    # C1 (cols 0-1): 0,1, 6,7  -- WAIT. W shape is p*n = 6*2 = 12.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies W is indexed by (k, i) where k is col of C (0..p-1) and i is row of C (0..n-1)?
    # Let's check standard tensor notation for matrix mult $C = AB$.
    # Tensor $T_{i,j,k}$ where $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # Usually U corresponds to A, V to B, W to C.
    # Shape of U: n*m. Shape of V: m*p. Shape of W: p*n.
    # If W index is $k*n + i$, then $k$ is the column index of C (0..p-1) and $i$ is the row index of C (0..n-1).
    # So W is stored column-major with respect to C? Or just flattened with k as major index?
    # "W[k*n+i,r]" means row index in the W matrix is $k*n + i$.
    # So for C[0,0] (row 0, col 0): $i=0, k=0 \rightarrow$ index 0.
    # For C[0,1] (row 0, col 1): $i=0, k=1 \rightarrow$ index 1.
    # For C[1,0] (row 1, col 0): $i=1, k=0 \rightarrow$ index 2.
    # For C[1,1] (row 1, col 1): $i=1, k=1 \rightarrow$ index 3.
    # So for a 2x2 block of C starting at col $c_{start}$ and row $r_{start}$ (here rows are always 0-1):
    # Indices are:
    # C[r, c] -> index c*n + r.
    # Block C1 (cols 0-1, rows 0-1):
    #   C[0,0] -> 0*2+0=0
    #   C[0,1] -> 1*2+0=2  (Wait, k=1, i=0 -> 1*2+0=2)
    #   C[1,0] -> 0*2+1=1
    #   C[1,1] -> 1*2+1=3
    # So the 2x2 block C1 corresponds to W indices [0, 2, 1, 3] in the order [TL, TR, BL, BR]?
    # TL: i=0, k=0 -> 0
    # TR: i=0, k=1 -> 2
    # BL: i=1, k=0 -> 1
    # BR: i=1, k=1 -> 3
    # So the indices for a 2x2 C block at columns $2j, 2j+1$ are:
    # TL: (2j)*2 + 0 = 4j
    # TR: (2j+1)*2 + 0 = 4j + 2
    # BL: (2j)*2 + 1 = 4j + 1
    # BR: (2j+1)*2 + 1 = 4j + 3
    # Let's verify for C2 (cols 2-3, j=1):
    # TL: 4
    # TR: 6
    # BL: 5
    # BR: 7
    # C3 (cols 4-5, j=2):
    # TL: 8
    # TR: 10
    # BL: 9
    # BR: 11
    
    w_blocks = {
        'C1': [0, 2, 1, 3],
        'C2': [4, 6, 5, 7],
        'C3': [8, 10, 9, 11]
    }

    col = 0
    # Multiplications:
    # 1. A_L * B_T1 = C1_part1
    # 2. A_R * B_B1 = C1_part2
    # 3. A_L * B_T2 = C2_part1
    # 4. A_R * B_B2 = C2_part2
    # 5. A_L * B_T3 = C3_part1
    # 6. A_R * B_B3 = C3_part2
    
    mults = [
        ('L', 'T1', 'C1'),
        ('R', 'B1', 'C1'),
        ('L', 'T2', 'C2'),
        ('R', 'B2', 'C2'),
        ('L', 'T3', 'C3'),
        ('R', 'B3', 'C3')
    ]
    
    for u_key, v_key, w_key in mults:
        u_idx = u_blocks[u_key]
        v_idx = v_blocks[v_key]
        w_idx = w_blocks[w_key]
        
        for r in range(7):
            if time.time() > deadline:
                break
            u_coeff, v_coeff, w_coeff = strassen_terms[r]
            for i in range(4):
                if u_coeff[i] != 0:
                    U[u_idx[i]][col] += u_coeff[i]
                if v_coeff[i] != 0:
                    V[v_idx[i]][col] += v_coeff[i]
                if w_coeff[i] != 0:
                    W[w_idx[i]][col] += w_coeff[i]
            col += 1
            
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
