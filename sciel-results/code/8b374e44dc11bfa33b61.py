import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE

    n, m, p = 3, 4, 8

    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: schoolbook rank 96
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Target Rank: 87
    # Decomposition:
    # 1. M_3,3,3 (Rank 23) for A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]
    # 2. M_3,3,3 (Rank 23) for A[0:3,0:3] * B[0:3,3:6] -> C[0:3,3:6]
    # 3. M_3,3,2 (Rank 17) for A[0:3,0:3] * B[0:3,6:8] -> C[0:3,6:8]
    # 4. M_3,1,3 (Rank 9)  for A[0:3,3:4] * B[3:4,0:3] -> C[0:3,0:3]
    # 5. M_3,1,3 (Rank 9)  for A[0:3,3:4] * B[3:4,3:6] -> C[0:3,3:6]
    # 6. M_3,1,2 (Rank 6)  for A[0:3,3:4] * B[3:4,6:8] -> C[0:3,6:8]
    
    rank = 87
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Helper to add a block
    def add_block(u_base, v_base, w_base, u_dims, v_dims, w_dims, terms):
        nonlocal col
        for t in terms:
            # t is (u_indices, v_indices, w_indices, coeffs)
            # u_indices: list of (local_i, local_j)
            # v_indices: list of (local_j, local_k)
            # w_indices: list of (local_k, local_i)
            for (li, lj), (lj2, lk), (lk2, li2), coeff in zip(t[0], t[1], t[2], t[3]):
                # Verify index consistency
                if lj != lj2 or lk != lk2 or li != li2:
                    continue # Should not happen
                g_i = u_base[0] + li
                g_j = u_base[1] + lj
                g_k = w_base[0] + lk
                U[g_i * m + g_j, col] += coeff
                V[g_j * p + g_k, col] += coeff
                W[g_k * n + g_i, col] += coeff
            col += 1

    # 1. M_3,3,3 Rank 23
    # Using a known 23-term decomposition
    # This is complex to hardcode from memory. 
    # I will use the 88 solution instead to ensure correctness.
    # The 88 solution is:
    # 1. M_1,4,8 Rank 32
    # 2. M_2,4,8 Rank 56 (8 * 7)
    
    # Reverting to 88 solution for safety.
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # Part 1: i=0 (1x4x8 schoolbook)
    for j in range(m):
        for k in range(p):
            u_idx = 0 * m + j
            v_idx = j * p + k
            w_idx = k * n + 0
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1

    # Part 2: i=1,2 (2x4x8 Strassen)
    S_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    S_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])

    for j_block in range(2):
        j_global_base = j_block * 2
        for k_block in range(4):
            k_global_base = k_block * 2
            for t in range(7):
                for l in range(4):
                    coeff = S_A[t, l]
                    if coeff != 0:
                        loc_i = l // 2
                        loc_j = l % 2
                        g_i = 1 + loc_i
                        g_j = j_global_base + loc_j
                        U[g_i * m + g_j, col] += coeff

                for l in range(4):
                    coeff = S_B[t, l]
                    if coeff != 0:
                        loc_j = l // 2
                        loc_k = l % 2
                        g_j = j_global_base + loc_j
                        g_k = k_global_base + loc_k
                        V[g_j * p + g_k, col] += coeff

                for l in range(4):
                    coeff = S_C[t, l]
                    if coeff != 0:
                        loc_i = l // 2
                        loc_k = l % 2
                        g_i = 1 + loc_i
                        g_k = k_global_base + loc_k
                        W[g_k * n + g_i, col] += coeff

                col += 1

    assert col == rank, f"Rank mismatch: {col} != {rank}"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
