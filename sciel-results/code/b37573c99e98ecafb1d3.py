import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin before 150s kill

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct for matrix multiplication."""
        R = len(U[0])
        # Check shapes
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        for r in range(R):
            if len(U[0]) != R or len(V[0]) != R or len(W[0]) != R:
                return False
        # Verify: for each (i,j,k), sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should equal 1
        # Actually the tensor entry is 1 for all valid (i,j,k)
        # The decomposition satisfies: T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)}
        # where T_{i,j,k} = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    # Trivial decomposition: rank = n*m*p = 90
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1

    # Verify trivial decomposition
    if not verify_decomposition(U_trivial, V_trivial, W_trivial, n, m, p):
        # Should never happen, but fall back
        return {"U": U_trivial, "V": V_trivial, "W": W_trivial}

    # Try block decomposition for improvement
    # Split J into {0,1,2} and {3,4}, K into {0,1,2} and {3,4,5}
    # Blocks: (3,3,3), (3,3,3), (3,2,3), (3,2,3)
    # Use trivial for each: 27+27+18+18 = 90 (no improvement)
    # But let's try to find better decompositions for sub-blocks

    def trivial_decomp(n_b, m_b, p_b):
        U = [[0] * (n_b * m_b * p_b) for _ in range(n_b * m_b)]
        V = [[0] * (n_b * m_b * p_b) for _ in range(m_b * p_b)]
        W = [[0] * (n_b * m_b * p_b) for _ in range(p_b * n_b)]
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i * m_b + j][col] = 1
                    V[j * p_b + k][col] = 1
                    W[k * n_b + i][col] = 1
                    col += 1
        return U, V, W

    def combine_blocks(blocks):
        """Combine block decompositions into a full decomposition."""
        total_rank = sum(b[3] for b in blocks)
        U_full = [[0] * total_rank for _ in range(n * m)]
        V_full = [[0] * total_rank for _ in range(m * p)]
        W_full = [[0] * total_rank for _ in range(p * n)]
        offset = 0
        for (i_idx, j_idx, k_idx, U_b, V_b, W_b, rank_b) in blocks:
            for li, gi in enumerate(i_idx):
                for lj in range(len(j_idx)):
                    gj = j_idx[lj]
                    for r in range(rank_b):
                        U_full[gi * m + gj][offset + r] = U_b[li * len(j_idx) + lj][r]
            for lj, gj in enumerate(j_idx):
                for lk in range(len(k_idx)):
                    gk = k_idx[lk]
                    for r in range(rank_b):
                        V_full[gj * p + gk][offset + r] = V_b[lj * len(k_idx) + lk][r]
            for lk, gk in enumerate(k_idx):
                for li in range(len(i_idx)):
                    gi = i_idx[li]
                    for r in range(rank_b):
                        W_full[gk * n + gi][offset + r] = W_b[lk * len(i_idx) + li][r]
            offset += rank_b
        return U_full, V_full, W_full, total_rank

    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = rank_trivial

    # Try block decomposition with trivial sub-blocks
    try:
        # Block 1: i={0,1,2}, j={0,1,2}, k={0,1,2}
        U1, V1, W1 = trivial_decomp(3, 3, 3)
        # Block 2: i={0,1,2}, j={0,1,2}, k={3,4,5}
        U2, V2, W2 = trivial_decomp(3, 3, 3)
        # Block 3: i={0,1,2}, j={3,4}, k={0,1,2}
        U3, V3, W3 = trivial_decomp(3, 2, 3)
        # Block 4: i={0,1,2}, j={3,4}, k={3,4,5}
        U4, V4, W4 = trivial_decomp(3, 2, 3)

        blocks = [
            ([0, 1, 2], [0, 1, 2], [0, 1, 2], U1, V1, W1, 27),
            ([0, 1, 2], [0, 1, 2], [3, 4, 5], U2, V2, W2, 27),
            ([0, 1, 2], [3, 4], [0, 1, 2], U3, V3, W3, 18),
            ([0, 1, 2], [3, 4], [3, 4, 5], U4, V4, W4, 18),
        ]

        U_full, V_full, W_full, total_rank = combine_blocks(blocks)

        if verify_decomposition(U_full, V_full, W_full, n, m, p) and total_rank < best_rank:
            best_rank = total_rank
            best_result = {"U": U_full, "V": V_full, "W": W_full}

    except Exception:
        pass

    # Try alternative split: J into {0,1} and {2,3,4}
    try:
        # Block 1: i={0,1,2}, j={0,1}, k={0,1,2,3,4,5}
        U1, V1, W1 = trivial_decomp(3, 2, 6)
        # Block 2: i={0,1,2}, j={2,3,4}, k={0,1,2,3,4,5}
        U2, V2, W2 = trivial_decomp(3, 3, 6)

        blocks = [
            ([0, 1, 2], [0, 1], [0, 1, 2, 3, 4, 5], U1, V1, W1, 36),
            ([0, 1, 2], [2, 3, 4], [0, 1, 2, 3, 4, 5], U2, V2, W2, 54),
        ]

        U_full, V_full, W_full, total_rank = combine_blocks(blocks)

        if verify_decomposition(U_full, V_full, W_full, n, m, p) and total_rank < best_rank:
            best_rank = total_rank
            best_result = {"U": U_full, "V": V_full, "W": W_full}

    except Exception:
        pass

    return best_result

# EVOLVE-BLOCK-END
