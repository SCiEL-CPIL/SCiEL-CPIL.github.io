"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    # Verify decomposition
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(rank):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    # Generate all possible 2x2x2 Strassen blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    # Greedy search for maximum disjoint packing with random restarts
    best_num_blocks = 0
    best_packing = []

    num_trials = 2000
    rng = random.Random(42)
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break

        # Shuffle a copy of blocks
        shuffled = all_blocks[:]
        rng.shuffle(shuffled)

        used_indices = set()
        current_packing = []

        for block_params, block_indices in shuffled:
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices

        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing

            # Theoretical max: floor(105/8) = 13, but constrained by dimensions
            if best_num_blocks >= 13:
                break

    # Construct U, V, W based on the best packing found
    U_cols = []
    V_cols = []
    W_cols = []

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        # M1 = (a11+a22)*(b11+b22)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a11] = 1.0; u[a22] = 1.0
        v[b11] = 1.0; v[b22] = 1.0
        w[c11] = 1.0; w[c22] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M2 = (a21+a22)*b11
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a21] = 1.0; u[a22] = 1.0
        v[b11] = 1.0
        w[c21] = 1.0; w[c22] = -1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M3 = a11*(b12-b22)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a11] = 1.0
        v[b12] = 1.0; v[b22] = -1.0
        w[c12] = 1.0; w[c22] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M4 = a22*(b21-b11)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a22] = 1.0
        v[b21] = 1.0; v[b11] = -1.0
        w[c11] = 1.0; w[c21] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M5 = (a11+a12)*b22
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a11] = 1.0; u[a12] = 1.0
        v[b22] = 1.0
        w[c11] = -1.0; w[c12] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M6 = (a21-a11)*(b11+b12)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a21] = 1.0; u[a11] = -1.0
        v[b11] = 1.0; v[b12] = 1.0
        w[c22] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M7 = (a12-a22)*(b21+b22)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a12] = 1.0; u[a22] = -1.0
        v[b21] = 1.0; v[b22] = 1.0
        w[c11] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

    # Add Strassen blocks from the best packing
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 2.0:
                    break
                if (i, j, k) not in covered:
                    u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    U_cols.append(u); V_cols.append(v); W_cols.append(w)

    rank = len(U_cols)
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T

    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
