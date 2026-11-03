"""Combined: Strassen rank-7 3x3x3 sub-blocks + schoolbook for remaining columns."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_3x3x3_terms():
    """Return 7 rank-1 terms (u,v,w) each of length 9 for the 3x3x3 multiplication tensor.
    Index convention: A index i*3+j, B index j*3+k, C index k*3+i."""
    def make_term(a_combos, b_combos, c_combos):
        u = np.zeros(9)
        v = np.zeros(9)
        w = np.zeros(9)
        for idx, coeff in a_combos:
            u[idx] = coeff
        for idx, coeff in b_combos:
            v[idx] = coeff
        for idx, coeff in c_combos:
            w[idx] = coeff
        return u, v, w

    terms = []
    # M1 = (A00 + A22)(B00 + B22)
    terms.append(make_term(
        [(0, 1), (8, 1)],
        [(0, 1), (8, 1)],
        [(0, 1), (3, -1), (4, -1), (8, 1)]
    ))
    # M2 = (A20 + A22)B00
    terms.append(make_term(
        [(6, 1), (8, 1)],
        [(0, 1)],
        [(2, 1), (4, 1), (5, -1), (8, -1)]
    ))
    # M3 = A00(B01 - B21)
    terms.append(make_term(
        [(0, 1)],
        [(1, 1), (7, -1)],
        [(1, 1), (3, -1), (6, 1)]
    ))
    # M4 = A22(B21 - B01)
    terms.append(make_term(
        [(8, 1)],
        [(7, 1), (1, -1)],
        [(0, 1), (4, 1), (7, -1), (8, -1)]
    ))
    # M5 = (A01 + A11)B11
    terms.append(make_term(
        [(1, 1), (4, 1)],
        [(4, 1)],
        [(0, -1), (1, 1), (3, 1), (8, -1)]
    ))
    # M6 = (A10 + A11)(B01 + B11)
    terms.append(make_term(
        [(3, 1), (4, 1)],
        [(1, 1), (4, 1)],
        [(2, 1), (5, 1), (6, -1)]
    ))
    # M7 = (A00 + A10)(B10 - B11)
    terms.append(make_term(
        [(0, 1), (3, 1)],
        [(3, 1), (4, -1)],
        [(0, 1), (3, -1), (4, 1), (6, 1), (7, -1)]
    ))
    return terms


def _build_3x3x3_block(terms, k_offset):
    """Return (u, v, w) arrays of length 9 each for a 3x3x3 block whose k values
    are k_offset, k_offset+1, k_offset+2 (within the p=7 B/C indexing).
    A index unchanged (i*3+j). B index j*7 + k. C index k*3 + i."""
    u = np.zeros(9)
    v = np.zeros(9)
    w = np.zeros(9)
    for (tu, tv, tw) in terms:
        # A part: tu is over 9 A-slots (i*3+j) -> same
        u += tu
        # B part: tv over 9 slots indexed j*3+k_local -> map to j*7 + (k_offset+k_local)
        for j in range(3):
            for k_local in range(3):
                local = j * 3 + k_local
                if tv[local] != 0:
                    v[j * 7 + (k_offset + k_local)] += tv[local]
        # C part: tw over 9 slots indexed k_local*3+i -> map to (k_offset+k_local)*3 + i
        for k_local in range(3):
            for i in range(3):
                local = k_local * 3 + i
                if tw[local] != 0:
                    w[(k_offset + k_local) * 3 + i] += tw[local]
    return u, v, w


def _schoolbook_columns(k_values, n, m, p):
    """Return list of (u,v,w) rank-1 terms (each length n*m, m*p, p*n) for the
    schoolbook terms over the given set of k values."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in k_values:
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                terms.append((u, v, w))
    return terms


def _verify(U, V, W, n, m, p, tol=1e-9):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(T_recon - T)) <= tol


def _full_schoolbook(n, m, p):
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


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Strategy: partition the p=7 columns of B into groups. Each group of 3
    # consecutive k-values can use the rank-7 Strassen 3x3x3 decomposition.
    # Remaining single columns use schoolbook (rank 9 each).
    # 7 = 3 + 3 + 1  ->  rank 7 + 7 + 9 = 23
    # We try several partitions and keep the verified one with minimal rank.

    terms3 = _strassen_3x3x3_terms()

    # Candidate partitions of range(p) into blocks of size 3 (Strassen) and 1 (schoolbook)
    def partition_to_blocks(p):
        """Yield partitions as list of (kind, k_values) where kind in {'strassen','school'}."""
        # Generate all ways to cover 0..p-1 with blocks of size 3 or 1, in order.
        results = []
        def rec(start, blocks):
            if start >= p:
                results.append(list(blocks))
                return
            if time.time() > deadline:
                return
            # try block of 3
            if start + 3 <= p:
                blocks.append(('strassen', list(range(start, start + 3))))
                rec(start + 3, blocks)
                blocks.pop()
            # try block of 1
            blocks.append(('school', [start]))
            rec(start + 1, blocks)
            blocks.pop()
        rec(0, [])
        return results

    best = None  # (rank, U, V, W)

    for blocks in partition_to_blocks(p):
        if time.time() > deadline:
            break
        U_list, V_list, W_list = [], [], []
        for kind, kvals in blocks:
            if kind == 'strassen':
                # must be exactly 3 consecutive k values
                if len(kvals) != 3:
                    continue
                u, v, w = _build_3x3x3_block(terms3, kvals[0])
                # each strassen block contributes 7 rank-1 terms; expand
                for r in range(7):
                    # We stored aggregated per-term? No: _build_3x3x3_block aggregates all 7 terms into one (u,v,w) which is WRONG.
                    pass
                # Actually we need per-term vectors. Redo: build per-term.
                for (tu, tv, tw) in terms3:
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    u += tu  # A part unchanged (n*m = 9)
                    for j in range(m):
                        for k_local in range(3):
                            local = j * 3 + k_local
                            if tv[local] != 0:
                                v[j * p + (kvals[0] + k_local)] += tv[local]
                    for k_local in range(3):
                        for i in range(n):
                            local = k_local * 3 + i
                            if tw[local] != 0:
                                w[(kvals[0] + k_local) * n + i] += tw[local]
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
            else:
                for (u, v, w) in _schoolbook_columns(kvals, n, m, p):
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
        if not U_list:
            continue
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        rank = U.shape[1]
        if _verify(U, V, W, n, m, p):
            if best is None or rank < best[0]:
                best = (rank, U, V, W)

    if best is None:
        # Fallback: full schoolbook
        U, V, W = _full_schoolbook(n, m, p)
        rank = U.shape[1]
    else:
        rank, U, V, W = best

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
