"""Matrix multiplication 4x4 by 4x6 with rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 4x4 by 4x6 matrix multiplication.
    
    Strategy: Use Strassen's block decomposition with 2x2 blocks.
    - A (4x4) partitioned into four 2x2 blocks
    - B (4x6) partitioned into four 2x3 blocks
    - Apply Strassen's 7-step algorithm to each block-pair
    
    Rank = 4 block-pairs * 7 Strassen steps * 3 columns = 84
    """
    start = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    rank = 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen steps for 2x2 blocks
    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    def get_u_row(block, i1, j1):
        if block == '00':
            i, j = i1, j1
        elif block == '01':
            i, j = i1, j1 + 2
        elif block == '10':
            i, j = i1 + 2, j1
        elif block == '11':
            i, j = i1 + 2, j1 + 2
        else:
            raise ValueError(f"Invalid block {block}")
        return i * m + j

    def get_v_row(block, j1, k1):
        if block == '00':
            j, k = j1, k1
        elif block == '01':
            j, k = j1, k1 + 3
        elif block == '10':
            j, k = j1 + 2, k1
        elif block == '11':
            j, k = j1 + 2, k1 + 3
        else:
            raise ValueError(f"Invalid block {block}")
        return j * p + k

    def get_w_row(block, i1, k1):
        if block == '00':
            i, k = i1, k1
        elif block == '01':
            i, k = i1, k1 + 3
        elif block == '10':
            i, k = i1 + 2, k1
        elif block == '11':
            i, k = i1 + 2, k1 + 3
        else:
            raise ValueError(f"Invalid block {block}")
        return k * n + i

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        for i1 in range(2):
            for j1 in range(2):
                for k1 in range(3):
                    for b, val in alpha.items():
                        U[get_u_row(b, i1, j1), col] += val
                    for b, val in beta.items():
                        V[get_v_row(b, j1, k1), col] += val
                    for b, val in gamma.items():
                        W[get_w_row(b, i1, k1), col] += val
                    col += 1

    # Verify correctness with a random test
    if time.time() - start < DEADLINE:
        rng = np.random.RandomState(42)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test

        C_recon = np.zeros((n, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            for i in range(n):
                for k in range(p):
                    s_val = 0.0
                    for j in range(m):
                        s_val += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                    C_recon[i,k] += s_val

        if np.max(np.abs(C_recon - C_expected)) > 1e-10:
            rank2 = n * m * p
            U2 = np.zeros((n * m, rank2), dtype=float)
            V2 = np.zeros((m * p, rank2), dtype=float)
            W2 = np.zeros((p * n, rank2), dtype=float)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U2[i*m+j, idx] = 1.0
                        V2[j*p+k, idx] = 1.0
                        W2[k*n+i, idx] = 1.0
                        idx += 1
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
