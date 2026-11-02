"""Program for 3x5x5 matrix multiplication tensor decomposition.

Structural approach: enumerate every valid tiling of the 3x5x5 index grid by
disjoint 2x2x2 Strassen blocks (each saves 1 rank over 8 schoolbook terms),
fill the uncovered triples with schoolbook, verify each candidate exactly via
einsum, and return the minimum-rank verified decomposition.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]


def _strassen_2x2x2_terms(i0, j0, k0, n, m, p):
    """Return the 7 (U, V, W) sparse term lists for one 2x2x2 Strassen block.

    Each term is a tuple (u_idx, v_idx, w_idx, sign).
    U index: i*m + j ; V index: j*p + k ; W index: k*n + i
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append((i0*m + j0, j0*p + k0, k0*n + i0, 1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1))
    # M2 = (A10+A11)B00
    terms.append(((i0+1)*m + j0, j0*p + k0, k0*n + (i0+1), 1))
    terms.append(((i0+1)*m + (j0+1), j0*p + k0, (k0+1)*n + (i0+1), -1))
    # M3 = A00(B01-B11)
    terms.append((i0*m + j0, j0*p + (k0+1), (k0+1)*n + i0, 1))
    terms.append((i0*m + j0, (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), -1))
    # M4 = A11(B10-B00)
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + k0, k0*n + i0, 1))
    terms.append(((i0+1)*m + (j0+1), j0*p + k0, k0*n + (i0+1), -1))
    # M5 = (A00+A01)B11
    terms.append((i0*m + j0, (j0+1)*p + (k0+1), k0*n + i0, -1))
    terms.append((i0*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + i0, 1))
    # M6 = (A10-A00)(B00+B01)
    terms.append(((i0+1)*m + j0, j0*p + k0, (k0+1)*n + (i0+1), 1))
    terms.append((i0*m + j0, j0*p + k0, (k0+1)*n + (i0+1), -1))
    terms.append(((i0+1)*m + j0, j0*p + (k0+1), (k0+1)*n + (i0+1), 1))
    terms.append((i0*m + j0, j0*p + (k0+1), (k0+1)*n + (i0+1), -1))
    # M7 = (A01-A11)(B10+B11)
    terms.append((i0*m + (j0+1), (j0+1)*p + k0, k0*n + i0, 1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + k0, k0*n + i0, -1))
    terms.append((i0*m + (j0+1), (j0+1)*p + (k0+1), k0*n + i0, 1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), k0*n + i0, -1))
    return terms


def _build_target(n, m, p):
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    return target


def _build_decomposition(blocks, n, m, p):
    """Build U,V,W (as lists) for a given list of Strassen block origins."""
    covered = np.zeros((n, m, p), dtype=bool)
    for (i0, j0, k0) in blocks:
        covered[i0:i0+2, j0:j0+2, k0:k0+2] = True

    n_blocks = len(blocks)
    n_school = int((~covered).sum())
    total_rank = n_blocks * 7 + n_school

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    for (i0, j0, k0) in blocks:
        for (ui, vi, wi, s) in _strassen_2x2x2_terms(i0, j0, k0, n, m, p):
            U[ui][col] += s
            V[vi][col] += s
            W[wi][col] += s
        col += 7

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1
    return U, V, W


def _verify(U, V, W, target):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    return bool(np.allclose(T, target, atol=1e-10))


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][c] = 1
                V[j*p + k][c] = 1
                W[k*n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _enumerate_tilings(n, m, p, deadline):
    """Enumerate tilings by 2x2x2 blocks; yield (blocks, rank) best-first."""
    # All candidate block origins that fit inside the grid.
    candidates = [(i, j, k)
                  for i in range(n - 1)
                  for j in range(m - 1)
                  for k in range(p - 1)]

    best = []  # (rank, blocks)
    state = np.zeros((n, m, p), dtype=bool)

    def dfs(start, blocks):
        if time.time() > deadline:
            return
        # Rank if we stop here: 7 per block + schoolbook for the rest.
        n_blocks = len(blocks)
        n_school = int((~state).sum())
        rank = n_blocks * 7 + n_school
        best.append((rank, list(blocks)))

        # Try extending with a block at or after `start` (to avoid duplicates).
        for idx in range(start, len(candidates)):
            if time.time() > deadline:
                return
            (i, j, k) = candidates[idx]
            if state[i, j, k] or state[i+1, j, k] or state[i, j+1, k] or \
               state[i, j, k+1] or state[i+1, j+1, k] or state[i+1, j, k+1] or \
               state[i, j+1, k+1] or state[i+1, j+1, k+1]:
                continue
            # place block
            state[i, j, k] = state[i+1, j, k] = state[i, j+1, k] = state[i, j, k+1] = True
            state[i+1, j+1, k] = state[i+1, j, k+1] = state[i, j+1, k+1] = state[i+1, j+1, k+1] = True
            blocks.append((i, j, k))
            dfs(idx + 1, blocks)
            blocks.pop()
            state[i, j, k] = state[i+1, j, k] = state[i, j+1, k] = state[i, j, k+1] = False
            state[i+1, j+1, k] = state[i+1, j, k+1] = state[i, j+1, k+1] = state[i+1, j+1, k+1] = False

    dfs(0, [])
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5
    target = _build_target(n, m, p)

    # Structural search: enumerate tilings, keep the minimum-rank verified one.
    best_rank = n * m * p
    best_decomp = None
    best_blocks = []

    try:
        tilings = _enumerate_tilings(n, m, p, deadline)
        # Sort by rank ascending; verify each until one passes.
        tilings.sort(key=lambda x: x[0])
        for rank, blocks in tilings:
            if time.time() > deadline:
                break
            if rank >= best_rank and best_decomp is not None:
                break  # can't improve
            U, V, W = _build_decomposition(blocks, n, m, p)
            if _verify(U, V, W, target):
                best_rank = rank
                best_decomp = {"U": U, "V": V, "W": W}
                best_blocks = blocks
                break  # first verified at this (minimum) rank is optimal
    except Exception:
        best_decomp = None

    if best_decomp is None or time.time() > deadline:
        return _schoolbook(n, m, p)

    return best_decomp
