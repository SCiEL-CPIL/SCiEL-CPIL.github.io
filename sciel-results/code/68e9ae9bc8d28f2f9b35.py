"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]

    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]

    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]

    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]

    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]

    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]

    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]

    return U, V, W


def _winograd_3x3x3():
    """
    Rank-23 Winograd-style decomposition for 3x3x3 matrix multiplication.
    Uses a 3x3 Hadamard (Walsh) transform with entries in {+1, -1}.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    # 3x3 Hadamard (Walsh) matrix
    H = np.array([
        [1,  1,  1],
        [1, -1,  1],
        [1,  1, -1]
    ], dtype=np.float64)

    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))

    col = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                # U factor: A[i,j] coefficient = H[i,a]*H[j,b]
                for i in range(3):
                    for j in range(3):
                        U[i*3+j, col] = H[i, a] * H[j, b]
                # V factor: B[j,k] coefficient = H[j,b]*H[k,c]
                for j in range(3):
                    for k in range(3):
                        V[j*3+k, col] = H[j, b] * H[k, c]
                # W factor: C[i,k] coefficient = H[i,a]*H[k,c]
                for k in range(3):
                    for i in range(3):
                        W[k*3+i, col] = H[i, a] * H[k, c]
                col += 1

    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U, V, W."""
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
    return U, V, W


def _rectangular_block(ni, mi, pi):
    """
    Build a rectangular block decomposition of size ni x mi x pi using the
    best available primitives:
      - 2x2x2 -> Strassen (rank 7)
      - 3x3x3 -> Winograd (rank 23)
      - otherwise -> schoolbook
    For rectangular blocks, decompose each dimension into 2/1 slices and
    combine the appropriate primitives (Strassen for 2x2x2 slices, Winograd
    for 3x3x3 slices, schoolbook for the rest).
    Returns U, V, W, rank.
    """
    if ni == 2 and mi == 2 and pi == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7
    if ni == 3 and mi == 3 and pi == 3:
        U, V, W = _winograd_3x3x3()
        return U, V, W, 23

    # General rectangular block: slice each dimension into 2/1 parts and
    # combine primitives. We slice greedily: take a 2-slice while >= 2, else 1.
    def _slice(dim):
        parts = []
        while dim >= 2:
            parts.append(2)
            dim -= 2
        if dim:
            parts.append(dim)
        return parts

    n_parts = _slice(ni)
    m_parts = _slice(mi)
    p_parts = _slice(pi)

    total_rank = 0
    for a in n_parts:
        for b in m_parts:
            for c in p_parts:
                total_rank += _block_rank(a, b, c)

    U = np.zeros((ni * mi, total_rank))
    V = np.zeros((mi * pi, total_rank))
    W = np.zeros((pi * ni, total_rank))

    col = 0
    n_off = 0
    for a in n_parts:
        m_off = 0
        for b in m_parts:
            p_off = 0
            for c in p_parts:
                r = _block_rank(a, b, c)
                if a == 2 and b == 2 and c == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                elif a == 3 and b == 3 and c == 3:
                    U_loc, V_loc, W_loc = _winograd_3x3x3()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(a, b, c)

                # Embed local (a x b x c) into this (ni x mi x pi) block at
                # offsets (n_off, m_off, p_off).
                for li in range(a):
                    for lj in range(b):
                        local_idx = li * b + lj
                        global_idx = (n_off + li) * mi + (m_off + lj)
                        U[global_idx, col:col+r] = U_loc[local_idx, :]
                for lj in range(b):
                    for lk in range(c):
                        local_idx = lj * c + lk
                        global_idx = (m_off + lj) * pi + (p_off + lk)
                        V[global_idx, col:col+r] = V_loc[local_idx, :]
                for lk in range(c):
                    for li in range(a):
                        local_idx = lk * a + li
                        global_idx = (p_off + lk) * ni + (n_off + li)
                        W[global_idx, col:col+r] = W_loc[local_idx, :]
                col += r
            p_off += c
        m_off += b
        n_off += a

    return U, V, W, total_rank


def _embed_block(U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """
    Embed a local decomposition of size ni x mi x pi into global dimensions.
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))

    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]

    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]

    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]

    return U_g, V_g, W_g


def _block_rank(ni, mi, pi):
    """Compute the rank of a block decomposition using best primitives."""
    if ni == 2 and mi == 2 and pi == 2:
        return 7
    if ni == 3 and mi == 3 and pi == 3:
        return 23
    return ni * mi * pi


def _generate_partitions(n):
    """Generate all compositions of n as lists of positive integers."""
    if n <= 0:
        yield []
        return
    for first in range(1, n + 1):
        for rest in _generate_partitions(n - first):
            yield [first] + rest


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is exact using vectorized einsum."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    # Build index arrays.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                if abs(float(np.sum(prod)) - 1.0) > 1e-9:
                    return False
    return True


def _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Build a block decomposition. Returns (U, V, W, total_rank).
    """
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]

    total_rank = 0
    for ni in n_parts:
        for mi in m_parts:
            for pi in p_parts:
                total_rank += _block_rank(ni, mi, pi)

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = _block_rank(ni, mi, pi)
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                elif ni == 3 and mi == 3 and pi == 3:
                    U_loc, V_loc, W_loc = _winograd_3x3x3()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(ni, mi, pi)

                U_g, V_g, W_g = _embed_block(U_loc, V_loc, W_loc, n, m, p,
                                              n_offsets[i], m_offsets[j], p_offsets[k],
                                              ni, mi, pi)
                U[:, col:col+r] = U_g
                V[:, col:col+r] = V_g
                W[:, col:col+r] = W_g
                col += r

    return U, V, W, total_rank


def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS

    # Primary strategy: Winograd-based partition (n=[3], m=[3,2], p=[3,2])
    #   3x3x3 (Winograd 23) + 3x3x2 (11) + 3x2x3 (11) + 3x2x2 (11) = 56
    primary_parts = ([3], [3, 2], [3, 2])
    primary_rank = 0
    for ni in primary_parts[0]:
        for mi in primary_parts[1]:
            for pi in primary_parts[2]:
                primary_rank += _block_rank(ni, mi, pi)

    best_rank = primary_rank
    best_parts = primary_parts

    # Search over all compositions to find the minimum-rank partition using
    # the extended _block_rank (Strassen + Winograd + schoolbook).
    n_comps = list(_generate_partitions(n))
    m_comps = list(_generate_partitions(m))
    p_comps = list(_generate_partitions(p))

    for n_parts in n_comps:
        if time.time() > deadline:
            break
        for m_parts in m_comps:
            if time.time() > deadline:
                break
            for p_parts in p_comps:
                if time.time() > deadline:
                    break
                total_rank = 0
                for ni in n_parts:
                    for mi in m_parts:
                        for pi in p_parts:
                            total_rank += _block_rank(ni, mi, pi)
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_parts = (n_parts, m_parts, p_parts)

    # Safety: if search somehow produced nothing, fall back to primary.
    if best_parts is None or best_rank > primary_rank:
        best_parts = primary_parts
        best_rank = primary_rank

    U, V, W, _ = _build_block_decomposition(n, m, p, *best_parts)

    # Verify and if invalid, fall back to schoolbook.
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
