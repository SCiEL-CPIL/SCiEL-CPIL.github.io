"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 120.0

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    U_sel = U[i_idx * m + j_idx, :]
    V_sel = V[j_idx * p + k_idx, :]
    W_sel = W[k_idx * n + i_idx, :]
    T_flat = np.sum(U_sel * V_sel * W_sel, axis=1)
    return np.allclose(T_flat, np.ones(n * m * p), atol=tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_products():
    """7 Strassen rank-1 terms for 2x2x2 MM.
    Each: (U_dict, V_dict, W_dict) with local 2x2 indices."""
    return [
        # P1 = (A00+A11)*(B00+B11) -> C00, C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2 = (A10+A11)*B00 -> C10
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # P3 = A00*(B10-B11) -> C00, C01
        ({0: 1}, {2: 1, 3: -1}, {0: 1, 1: 1}),
        # P4 = A11*(B01-B00) -> C00, C11
        ({3: 1}, {1: 1, 0: -1}, {3: 1, 2: 1}),
        # P5 = (A00+A01)*B11 -> C01
        ({0: 1, 1: 1}, {3: 1}, {1: 1}),
        # P6 = (A10-A00)*(B00+B10) -> C10
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {2: 1}),
        # P7 = (A01-A11)*(B01+B11) -> C01, C11
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {1: 1, 3: 1}),
    ]


def _blocks_overlap(b1, b2):
    return not (b1[0]+1 < b2[0] or b2[0]+1 < b1[0] or
                b1[1]+1 < b2[1] or b2[1]+1 < b1[1] or
                b1[2]+1 < b2[2] or b2[2]+1 < b1[2])


def _construct_strassen(n, m, p, blocks):
    """Build decomposition: Strassen blocks + schoolbook for remainder."""
    prods = _strassen_products()
    covered = set()
    for (i0, j0, k0) in blocks:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    covered.add((i0+a, j0+b, k0+c))
    rank = 7 * len(blocks) + (n*m*p - 8*len(blocks))
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for (i0, j0, k0) in blocks:
        for (ud, vd, wd) in prods:
            for li, cf in ud.items():
                U[(i0 + li//2) * m + (j0 + li%2), col] += cf
            for li, cf in vd.items():
                V[(j0 + li//2) * p + (k0 + li%2), col] += cf
            for li, cf in wd.items():
                W[(k0 + li//2) * n + (i0 + li%2), col] += cf
            col += 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    return U, V, W


def _find_best_blocks(n, m, p):
    """Find the set of non-overlapping 2x2x2 Strassen blocks that gives lowest rank."""
    prods = _strassen_products()
    best_rank = n * m * p
    best_blocks = []
    # For 3x3: i0 in {0,1}, j0 in {0,1} — only one i0 and one j0 can be chosen
    # For k: non-overlapping pairs
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            # Find max set of non-overlapping k0 values
            # k0 in {0,...,p-2}, each covers {k0, k0+1}
            # Greedy: k0 = 0, 2, 4, 6 (for p=8)
            k_vals = list(range(0, p - 1, 2))
            # Verify non-overlap
            valid = True
            for a in range(len(k_vals)):
                for b in range(a+1, len(k_vals)):
                    if not (k_vals[a]+1 < k_vals[b] or k_vals[b]+1 < k_vals[a]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            blocks = [(i0, j0, k) for k in k_vals]
            # Check mutual non-overlap
            ok = True
            for a in range(len(blocks)):
                for b in range(a+1, len(blocks)):
                    if _blocks_overlap(blocks[a], blocks[b]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            rank = 7 * len(blocks) + (n*m*p - 8*len(blocks))
            if rank < best_rank:
                best_rank = rank
                best_blocks = blocks
    return best_blocks


def _reduce_rank(U, V, W, n, m, p, max_iter=200):
    """Try to reduce rank by finding and eliminating dependent rank-1 terms."""
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    rank = U.shape[1]
    for _ in range(max_iter):
        found = -1
        for c in range(rank):
            # Check if column c is a linear combination of others
            # Simple check: is column c zero?
            if np.allclose(U[:, c], 0, atol=1e-12) and \
               np.allclose(V[:, c], 0, atol=1e-12) and \
               np.allclose(W[:, c], 0, atol=1e-12):
                found = c
                break
        if found < 0:
            break
        U = np.delete(U, found, axis=1)
        V = np.delete(V, found, axis=1)
        W = np.delete(W, found, axis=1)
        rank -= 1
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()

    # Baseline: schoolbook
    U, V, W = _schoolbook(n, m, p)
    assert _verify(U, V, W, n, m, p), "Schoolbook failed"
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Try Strassen block configurations
    if time.time() - t0 < DEADLINE - 10:
        blocks = _find_best_blocks(n, m, p)
        if blocks:
            U2, V2, W2 = _construct_strassen(n, m, p, blocks)
            if _verify(U2, V2, W2, n, m, p):
                if U2.shape[1] < best_rank:
                    best_rank = U2.shape[1]
                    best_U, best_V, best_W = U2, V2, W2

    # Try to reduce rank further
    if time.time() - t0 < DEADLINE - 5:
        U3, V3, W3, r3 = _reduce_rank(best_U, best_V, best_W, n, m, p)
        if r3 < best_rank and _verify(U3, V3, W3, n, m, p):
            best_rank = r3
            best_U, best_V, best_W = U3, V3, W3

    # Convert to int if possible, else keep as float
    def _to_list(mat):
        if np.allclose(mat, np.round(mat), atol=1e-9):
            return np.round(mat).astype(np.int64).tolist()
        return mat.tolist()

    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}

# EVOLVE-BLOCK-END
