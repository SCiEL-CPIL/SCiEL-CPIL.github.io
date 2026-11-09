"""Rewritten solution for matrix multiplication 4,4,4."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # 5 second safety margin

    n, m, p = 4, 4, 4
    rank = 49

    # Element-level Strassen data for 2x2 matrices
    # Indexing: [elem00, elem01, elem10, elem11]
    U_elem = [
        [1, 0, 0, 1],   # t=0: a00+a11
        [0, 0, 1, 1],   # t=1: a10+a11
        [1, 0, 0, 0],   # t=2: a00
        [0, 0, 0, 1],   # t=3: a11
        [1, 1, 0, 0],   # t=4: a00+a01
        [-1, 0, 1, 0],  # t=5: a10-a00
        [0, 1, 0, -1],  # t=6: a01-a11
    ]
    V_elem = [
        [1, 0, 0, 1],   # t=0: b00+b11
        [1, 0, 0, 0],   # t=1: b00
        [0, 1, 0, -1],  # t=2: b01-b11
        [-1, 0, 1, 0],  # t=3: b10-b00
        [0, 0, 0, 1],   # t=4: b11
        [1, 1, 0, 0],   # t=5: b00+b01
        [0, 0, 1, 1],   # t=6: b10+b11
    ]
    W_elem = [
        [[1, 0], [0, 1]],   # t=0
        [[0, 0], [1, -1]],  # t=1
        [[0, 1], [0, 1]],   # t=2
        [[1, 0], [1, 0]],   # t=3
        [[-1, 1], [0, 0]],  # t=4
        [[0, 0], [0, 1]],   # t=5
        [[1, 0], [0, 0]],   # t=6
    ]

    # Block-level Strassen data: which 2x2 blocks are used and with what coefficient
    # Format: list of (block_row, block_col, coefficient)
    A_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: A00+A11
        [(1,0,1), (1,1,1)],      # s=1: A10+A11
        [(0,0,1)],               # s=2: A00
        [(1,1,1)],               # s=3: A11
        [(0,0,1), (0,1,1)],      # s=4: A00+A01
        [(1,0,1), (0,0,-1)],     # s=5: A10-A00
        [(0,1,1), (1,1,-1)],     # s=6: A01-A11
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: B00+B11
        [(0,0,1)],               # s=1: B00
        [(0,1,1), (1,1,-1)],     # s=2: B01-B11
        [(1,0,1), (0,0,-1)],     # s=3: B10-B00
        [(1,1,1)],               # s=4: B11
        [(0,0,1), (0,1,1)],      # s=5: B00+B01
        [(1,0,1), (1,1,1)],      # s=6: B10+B11
    ]
    C_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: C00+=M1, C11+=M1
        [(1,0,1), (1,1,-1)],     # s=1: C10+=M2, C11-=M2
        [(0,1,1), (1,1,1)],      # s=2: C01+=M3, C11+=M3
        [(0,0,1), (1,0,1)],      # s=3: C00+=M4, C10+=M4
        [(0,0,-1), (0,1,1)],     # s=4: C00-=M5, C01+=M5
        [(1,1,1)],               # s=5: C11+=M6
        [(0,0,1)],               # s=6: C00+=M7
    ]

    # Build U, V, W matrices
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)

    for s in range(7):
        for t in range(7):
            col = s * 7 + t

            # Build U for this term
            for (bi, bj, c_a) in A_blocks[s]:
                for ei in range(2):
                    for ej in range(2):
                        row_a = (2 * bi + ei) * 4 + (2 * bj + ej)
                        U[row_a, col] += c_a * U_elem[t][ei * 2 + ej]

            # Build V for this term
            for (bj, bk, c_b) in B_blocks[s]:
                for ej in range(2):
                    for ek in range(2):
                        row_b = (2 * bj + ej) * 4 + (2 * bk + ek)
                        V[row_b, col] += c_b * V_elem[t][ej * 2 + ek]

            # Build W for this term
            for (bi, bk, c_c) in C_blocks[s]:
                for ei in range(2):
                    for ek in range(2):
                        # W index: k*n + i where n=4
                        row_c = (2 * bk + ek) * 4 + (2 * bi + ei)
                        W[row_c, col] += c_c * W_elem[t][ei][ek]

    # Verify correctness with a random test
    try:
        if time.time() > deadline:
            # Out of time, return what we have
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        import random
        random.seed(42)
        A_test = np.array([[random.randint(-5, 5) for _ in range(4)] for _ in range(4)], dtype=int)
        B_test = np.array([[random.randint(-5, 5) for _ in range(4)] for _ in range(4)], dtype=int)

        # Compute A*B using schoolbook
        AB = A_test @ B_test

        # Compute using decomposition
        AB_decomp = np.zeros((4, 4), dtype=int)
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        # Vectorized verification for speed and robustness
        # U: (16, rank), V: (16, rank), W: (16, rank)
        # u_vals: (rank,) = U^T A_flat
        # v_vals: (rank,) = V^T B_flat
        # scalars: (rank,) = u_vals * v_vals
        # We need sum_r W[:, r] * scalars[r] -> vector of length 16
        # Then reshape to (4,4) with index k*4+i -> C[i,k]
        
        u_vals = U.T @ A_flat
        v_vals = V.T @ B_flat
        scalars = u_vals * v_vals
        
        # W @ scalars gives vector of length 16
        # W is (16, rank). W @ scalars is (16,)
        # Index idx = k*4 + i. Value is W[idx, :] @ scalars
        W_flat_decomp = W @ scalars
        
        # Reshape W_flat_decomp to (4,4) such that W_flat_decomp[k*4+i] maps to C[i,k]
        # C[i,k] = W_flat_decomp[k*4+i]
        # We can create a matrix C where C[i,k] = W_flat_decomp[k*4+i]
        # This is equivalent to:
        # C = W_flat_decomp.reshape(4,4).T ?
        # Let's check:
        # W_flat_decomp is [W[0,0], W[0,1], W[0,2], W[0,3], W[1,0], ...]
        # Index 0: k=0, i=0 -> C[0,0]
        # Index 1: k=0, i=1 -> C[1,0]
        # Index 2: k=0, i=2 -> C[2,0]
        # Index 3: k=0, i=3 -> C[3,0]
        # Index 4: k=1, i=0 -> C[0,1]
        # ...
        # So if we reshape W_flat_decomp to (4,4) in C-order:
        # M[0,0]=W[0], M[0,1]=W[1], M[0,2]=W[2], M[0,3]=W[3]
        # M[1,0]=W[4], ...
        # We want C[i,k] = M[k,i].
        # So C = M.T
        
        M = W_flat_decomp.reshape(4, 4)
        AB_decomp = M.T

        # Check if they match
        if not np.array_equal(AB, AB_decomp):
            # If mismatch, fall back to schoolbook
            rank_fb = 64
            U_fb = np.zeros((16, rank_fb), dtype=int)
            V_fb = np.zeros((16, rank_fb), dtype=int)
            W_fb = np.zeros((16, rank_fb), dtype=int)
            col_fb = 0
            for i2 in range(4):
                for j2 in range(4):
                    for k2 in range(4):
                        U_fb[i2 * 4 + j2, col_fb] = 1
                        V_fb[j2 * 4 + k2, col_fb] = 1
                        W_fb[k2 * 4 + i2, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
