import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Block decomposition:
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1 B1 + A2 B2
    # B1 (4x6) = [B1a (4x3) | B1b (4x3)]
    # C = A1 B1a + A1 B1b + A2 B2
    
    # Rank contributions:
    # 1. A1 B1a: 4x4x3 schoolbook rank = 4*4*3 = 48
    # 2. A1 B1b: 4x4x3 schoolbook rank = 4*4*3 = 48
    # 3. A2 B2: 4x1x6 outer product rank = 4
    
    total_rank = 48 + 48 + 4
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    col = 0
    
    # Term 1: A1 B1a
    # A1: i in [0,4), j in [0,4)
    # B1a: j in [0,4), k in [0,3)
    # C1: k in [0,3), i in [0,4)
    for i in range(4):
        for j in range(4):
            for k in range(3):
                a_idx = i * 5 + j
                b_idx = j * 6 + k
                c_idx = k * 4 + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
                
    # Term 2: A1 B1b
    # A1: i in [0,4), j in [0,4)
    # B1b: j in [0,4), k in [3,6)
    # C2: k in [3,6), i in [0,4)
    for i in range(4):
        for j in range(4):
            for k in range(3, 6):
                a_idx = i * 5 + j
                b_idx = j * 6 + k
                c_idx = k * 4 + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
                
    # Term 3: A2 B2
    # A2: i in [0,4), j in [4,5) -> j=4
    # B2: j in [4,5), k in [0,6) -> j=4
    # C3: k in [0,6), i in [0,4)
    # Rank 4 (one per i)
    for i in range(4):
        a_idx = i * 5 + 4
        # V and W are 1 for all k
        for k in range(6):
            b_idx = 4 * 6 + k
            c_idx = k * 4 + i
            # We use the same column index for all k for a given i?
            # No, in the tensor sum, for a fixed r, U_a V_b W_c must be 1 for the tuple.
            # For the outer product term, we established that we need 4 rank-1 tensors.
            # For r = 96 + i:
            # U[i*5+4, r] = 1
            # V[4*6+k, r] = 1 for all k
            # W[k*4+i, r] = 1 for all k
            # This works because the sum is over r.
            # For a specific tuple (a,b,c) in the support, there is exactly one r (determined by i)
            # such that U, V, W are all 1.
            # Wait, if V and W have 1s for all k, then for a fixed r=i,
            # U[i*5+4, r] = 1.
            # V[4*6+k, r] = 1.
            # W[k*4+i, r] = 1.
            # Product is 1.
            # This covers all k for that i.
            # So we just set V and W to 1 for all k.
            pass
        
        # Set U
        U[a_idx][col] = 1
        # Set V for all k
        for k in range(6):
            b_idx = 4 * 6 + k
            V[b_idx][col] = 1
        # Set W for all k
        for k in range(6):
            c_idx = k * 4 + i
            W[c_idx][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
