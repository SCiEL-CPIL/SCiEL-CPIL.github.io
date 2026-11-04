import time
import numpy as np
import itertools

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 135.0  # safety margin from 150s kill

# Correct Strassen 2x2x2 multipliers.
# A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = A*B
STR_U = [
    {(0, 0): 1, (1, 1): 1},   # M1 (a00+a11)
    {(1, 0): 1, (1, 1): 1},   # M2 (a10+a11)
    {(0, 0): 1},              # M3 a00
    {(1, 1): 1},              # M4 a11
    {(0, 0): 1, (0, 1): 1},   # M5 (a00+a01)
    {(1, 0): 1, (0, 0): -1},  # M6 (a10-a00)
    {(0, 1): 1, (1, 1): -1},  # M7 (a01-a11)
]
STR_V = [
    {(0, 0): 1, (1, 1): 1},   # (b00+b11)
    {(0, 0): 1},              # b00
    {(0, 1): 1, (1, 1): -1},  # (b01-b11)
    {(1, 0): 1, (0, 0): -1},  # (b10-b00)
    {(1, 1): 1},              # b11
    {(0, 0): 1, (0, 1): 1},   # (b00+b01)
    {(1, 0): 1, (1, 1): 1},   # (b10+b11)
]
# W[r] = dict {(i,k): coeff} : multiplier r's contribution to C[i,k]
STR_W = [
    {(0, 0): 1, (1, 1): 1},   # M1 -> C00(+), C11(+)
    {(1, 0): 1, (1, 1): -1},  # M2 -> C10(+), C11(-)
    {(0, 1): 1, (1, 1): 1},   # M3 -> C01(+), C11(+)
    {(0, 0): 1, (1, 0): 1},   # M4 -> C00(+), C10(+)
    {(0, 0): -1, (0, 1): 1},  # M5 -> C00(-), C01(+)
    {(1, 1): 1},              # M6 -> C11(+)
    {(0, 0): 1},              # M7 -> C00(+)
]


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def time_left():
        return DEADLINE_S - (time.time() - start_time)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def validate(U, V, W, tol=1e-8):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.ones((n, m, p))
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i, j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        return np.max(np.abs(T_computed - T_expected)) < tol

    def build_rank_from_blocks(blocks):
        """Build a decomposition using the given list of 2x2x2 Strassen block
        placements. Each block is (i0, j0, k0) with i0 in {0,1}, j0 in {0,1},
        k0 in {0..6} (so the 2x2x2 box fits in the 3x3x8 tensor).
        Blocks must be non-overlapping in their (i,j,k) coverage.
        Covered (i,j,k) entries are handled by Strassen; uncovered entries
        are handled by schoolbook single terms.
        Returns (U, V, W, rank)."""
        # Determine coverage: a box covers (i0+di, j0+dj, k0+dk) for di,dj,dk in {0,1}
        covered = set()
        for (i0, j0, k0) in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0 + di, j0 + dj, k0 + dk))

        # Count schoolbook terms needed: all (i,j,k) not covered
        uncovered = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)
                     if (i, j, k) not in covered]

        rank = 7 * len(blocks) + len(uncovered)
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        # Strassen blocks
        for (i0, j0, k0) in blocks:
            for r in range(7):
                for (ii, jj), coeff in STR_U[r].items():
                    U[(i0 + ii) * m + (j0 + jj), col] += coeff
                for (jj_l, kk_l), coeff in STR_V[r].items():
                    V[(j0 + jj_l) * p + (k0 + kk_l), col] += coeff
                for (ii_l, kk_l), coeff in STR_W[r].items():
                    W[(k0 + kk_l) * n + (i0 + ii_l), col] += coeff
                col += 1
        # Schoolbook for uncovered
        for (i, j, k) in uncovered:
            U[i * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
        assert col == rank, (col, rank)
        return U, V, W, rank

    def blocks_nonoverlap(blocks):
        """Check that no two blocks share any (i,j,k) cell."""
        cells = []
        for (i0, j0, k0) in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        cells.append((i0 + di, j0 + dj, k0 + dk))
        return len(cells) == len(set(cells))

    def all_block_positions():
        """All valid (i0,j0,k0) for a 2x2x2 box inside 3x3x8."""
        pos = []
        for i0 in range(n - 1):   # 0,1
            for j0 in range(m - 1):  # 0,1
                for k0 in range(p - 1):  # 0..6
                    pos.append((i0, j0, k0))
        return pos

    def try_reduce(U0, V0, W0):
        """Try to lower rank by finding a term that is a half-integer linear
        combination of the others, then absorb it."""
        rank = U0.shape[1]
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_rank = rank

        def build_term_flat(matrices):
            r = matrices[0].shape[1]
            tt = np.zeros((r, n, m, p))
            for c in range(r):
                um = matrices[0][:, c].reshape(n, m)
                vm = matrices[1][:, c].reshape(m, p)
                wm = matrices[2][:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        if abs(um[i, j]) < 1e-15:
                            continue
                        for k in range(p):
                            tt[c, i, j, k] = um[i, j] * vm[j, k] * wm[k, i]
            return tt.reshape(r, -1)

        term_flat = build_term_flat((best_U, best_V, best_W))
        for _ in range(60):  # allow multiple reductions
            if time_left() < 4:
                break
            rank = best_U.shape[1]
            progress = False
            for c in range(rank):
                if time_left() < 4:
                    break
                others = [x for x in range(rank) if x != c]
                A = term_flat[others].T
                b = term_flat[c]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    if np.max(np.abs(A @ x - b)) > 1e-9:
                        continue
                    x_rounded = np.round(x * 2) / 2
                    if np.max(np.abs(A @ x_rounded - b)) > 1e-9:
                        continue
                    new_rank = rank - 1
                    nU = np.zeros((nm, new_rank))
                    nV = np.zeros((mp, new_rank))
                    nW = np.zeros((pn, new_rank))
                    for new_c, old_c in enumerate(others):
                        nU[:, new_c] = best_U[:, old_c] - x_rounded[new_c] * best_U[:, c]
                        nV[:, new_c] = best_V[:, old_c] - x_rounded[new_c] * best_V[:, c]
                        nW[:, new_c] = best_W[:, old_c] - x_rounded[new_c] * best_W[:, c]
                    if validate(nU, nV, nW):
                        best_U, best_V, best_W = nU, nV, nW
                        term_flat = build_term_flat((best_U, best_V, best_W))
                        progress = True
                        break  # restart c-loop with new rank
                except Exception:
                    continue
            if not progress:
                break
        return best_U, best_V, best_W, best_rank

    # Fallback
    if not validate(*build_rank_from_blocks([(0, 0, 0)])[0:3]):
        return schoolbook()

    best_U, best_V, best_W, best_rank = None, None, None, n * m * p

    def consider(U, V, W, rank):
        nonlocal best_U, best_V, best_W, best_rank
        if rank < best_rank and validate(U, V, W):
            best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    positions = all_block_positions()

    # 0 blocks (pure schoolbook)
    U0, V0, W0, r0 = build_rank_from_blocks([])
    consider(U0, V0, W0, r0)

    # 1 block: all 27 positions
    for pos in positions:
        if time_left() < 15:
            break
        U, V, W, r = build_rank_from_blocks([pos])
        consider(U, V, W, r)

    # 2 blocks: all non-overlapping pairs
    for pos_a, pos_b in itertools.combinations(positions, 2):
        if time_left() < 20:
            break
        if not blocks_nonoverlap([pos_a, pos_b]):
            continue
        U, V, W, r = build_rank_from_blocks([pos_a, pos_b])
        consider(U, V, W, r)

    # 3 blocks: all non-overlapping triples (prune by time)
    triples = 0
    for pos_a, pos_b, pos_c in itertools.combinations(positions, 3):
        if time_left() < 30:
            break
        if not blocks_nonoverlap([pos_a, pos_b, pos_c]):
            continue
        triples += 1
        U, V, W, r = build_rank_from_blocks([pos_a, pos_b, pos_c])
        consider(U, V, W, r)

    # Attempt rank reduction via linear-dependency absorption on best found.
    if best_U is not None and time_left() > 20:
        rU, rV, rW, rrank = try_reduce(best_U, best_V, best_W)
        if rrank < best_rank and validate(rU, rV, rW):
            best_U, best_V, best_W, best_rank = rU, rV, rW, rrank

    if best_U is None:
        return schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
