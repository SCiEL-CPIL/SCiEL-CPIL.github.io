import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 10  # 10s is plenty for this O(1) task
    
    n, m, p = DIMENSIONS
    # The tensor is a direct sum of 8 independent 2x2x2 matrix multiplications.
    # The rank of 2x2x2 is 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    rank = 56
    
    # Initialize factor matrices as lists of lists for the required output format
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen coefficients for one 2x2x2 block
    # Each term is (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, val) for A_ij
    # v_entries: list of (j, k_local, val) for B_jk
    # w_entries: list of (k_local, i_prime, val) for C_ki
    
    # Standard Strassen 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    terms = [
        # M1
        ([(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)],
         [(0,0,1)],
         [(0,1,1), (1,1,-1)]),
        # M3
        ([(0,0,1)],
         [(0,1,1), (1,1,-1)],
         [(1,0,1), (1,1,1)]),
        # M4
        ([(1,1,1)],
         [(1,0,1), (0,0,-1)],
         [(0,0,1), (0,1,1)]),
        # M5
        ([(0,0,1), (0,1,1)],
         [(1,1,1)],
         [(0,0,-1), (1,0,1)]),
        # M6
        ([(1,0,1), (0,0,-1)],
         [(0,0,1), (0,1,1)],
         [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)],
         [(1,0,1), (1,1,1)],
         [(0,0,1)]),
    ]

    num_blocks = p // 2  # 8 blocks
    
    for block_idx in range(num_blocks):
        if time.time() > deadline:
            break
        k_base = 2 * block_idx
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u_entries, v_entries, w_entries = terms[term_idx]

            # Fill U: index is (i, j) -> row i*m + j
            for (i, j, val) in u_entries:
                U[i * m + j][col] = val

            # Fill V: index is (j, k) -> row j*p + k, where k = k_base + k_local
            for (j, k_local, val) in v_entries:
                k = k_base + k_local
                V[j * p + k][col] = val

            # Fill W: index is (k, i') -> row k*n + i', where k = k_base + k_local
            for (k_local, ip, val) in w_entries:
                k = k_base + k_local
                W[k * n + ip][col] = val

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
