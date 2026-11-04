"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for c in range(rank):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
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


def construct():
    deadline = time.time() + 135

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)

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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    # Greedy + local search for maximum disjoint packing
    best_num_blocks = 0
    best_packing = []

    rng = random.Random(42)

    # Phase 1: Greedy with random restarts
    num_trials = 3000
    for t in range(num_trials):
        if time.time() > deadline - 15.0:
            break
        shuffled = all_blocks[:]
        rng.shuffle(shuffled)
        used = set()
        packing = []
        for params, idx in shuffled:
            if not (idx & used):
                packing.append(params)
                used |= idx
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]
            if best_num_blocks >= 13:
                break

    # Phase 2: Local improvement on best packing
    # Try to swap a block in for a pair of overlapping blocks, etc.
    # Simplified: try removing one block and re-greedy for remaining slots
    if time.time() < deadline - 10.0:
        for _ in range(500):
            if time.time() > deadline - 10.0:
                break
            if not best_packing:
                break
            # Remove a random block from best_packing
            rem_idx = rng.randrange(len(best_packing))
            reduced = best_packing[:]
            removed = reduced.pop(rem_idx)

            # Re-run greedy starting with the reduced set of used indices
            used = set()
            for bp in reduced:
                # Compute indices for bp
                i1, i2, j1, j2, k1, k2 = bp
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            used.add((i, j, k))

            # Now greedily add blocks
            shuffled2 = all_blocks[:]
            rng.shuffle(shuffled2)
            for params, idx in shuffled2:
                if not (idx & used):
                    reduced.append(params)
                    used |= idx

            if len(reduced) > best_num_blocks:
                best_num_blocks = len(reduced)
                best_packing = reduced[:]

    # Build U, V, W
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

    covered = set()
    for params in best_packing:
        i1, i2, j1, j2, k1, k2 = params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

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

    if _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
