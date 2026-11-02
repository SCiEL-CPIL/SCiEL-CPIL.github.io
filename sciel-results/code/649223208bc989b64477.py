"""Stronger design: exact rank-7 2x2x2 blocks (Strassen + symmetries) + standard tail,
searched over all dimension partitions and block placements."""
import itertools
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen2x2x2():
    U = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 1],
    ]
    V = [
        [0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 1],
    ]
    W = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
    ]
    return U, V, W


def _permute_block(U, V, W, pi, pj, pk):
    n = 2
    U2 = [[0] * 7 for _ in range(n * n)]
    V2 = [[0] * 7 for _ in range(n * n)]
    W2 = [[0] * 7 for _ in range(n * n)]
    for i in range(n):
        for j in range(n):
            for r in range(7):
                U2[i * n + j][r] = U[pi[i] * n + pj[j]][r]
    for j in range(n):
        for k in range(n):
            for r in range(7):
                V2[j * n + k][r] = V[pj[j] * n + pk[k]][r]
    for k in range(n):
        for i in range(n):
            for r in range(7):
                W2[k * n + i][r] = W[pk[k] * n + pi[i]][r]
    return U2, V2, W2


def _block_forms():
    base = _build_strassen2x2x2()
    forms = []
    seen = set()
    for pi in itertools.permutations(range(2)):
        for pj in itertools.permutations(range(2)):
            for pk in itertools.permutations(range(2)):
                U, V, W = _permute_block(base[0], base[1], base[2], pi, pj, pk)
                key = (tuple(map(tuple, U)), tuple(map(tuple, V)), tuple(map(tuple, W)))
                if key not in seen:
                    seen.add(key)
                    forms.append((U, V, W))
    return forms


def _standard_block(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W, rank


def _verify(U, V, W, n, m, p, tol=1e-6):
    import random
    random.seed(12345)
    for _ in range(3):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C[i][k] += A[i][j] * B[j][k]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(len(U[0])):
                        s += A[i][j] * B[j][k] * U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - C[i][k]) > tol:
                    return False
    return True


def _all_partitions(n):
    res = []
    def rec(rem, acc):
        if rem == 0:
            res.append(list(acc))
            return
        for part in (1, 2):
            if part <= rem:
                acc.append(part)
                rec(rem - part, acc)
                acc.pop()
    rec(n, [])
    return res


_RESULT = {"rank": None, "U": None, "V": None, "W": None}


def _search(deadline, forms):
    n, m, p = DIMENSIONS
    best_rank = n * m * p

    part_m = _all_partitions(m)
    part_p = _all_partitions(p)
    part_n = _all_partitions(n)

    def try_blocks(pm, pp, pn):
        nonlocal best_rank
        i_segs2 = [a for a in range(len(pn)) if pn[a] == 2]
        j_segs2 = [b for b in range(len(pm)) if pm[b] == 2]
        k_segs2 = [c for c in range(len(pp)) if pp[c] == 2]
        blocks = [(ia, jb, kc) for ia in i_segs2 for jb in j_segs2 for kc in k_segs2]
        maxb = len(blocks)
        from itertools import combinations
        for size in range(maxb, -1, -1):
            if time.time() > deadline:
                return
            rank = 7 * size + (n * m * p - 4 * size)
            if rank >= best_rank:
                continue
            for S in combinations(range(maxb), size):
                if time.time() > deadline:
                    return
                chosen = [blocks[t] for t in S]
                # try form assignments
                for form_idx in itertools.product(range(len(forms)), repeat=size):
                    if time.time() > deadline:
                        return
                    covered = set()
                    block_cells = []
                    for (ia, jb, kc), fi in zip(chosen, form_idx):
                        i0 = sum(pn[:ia])
                        j0 = sum(pm[:jb])
                        k0 = sum(pp[:kc])
                        fu, fv, fw = forms[fi]
                        block_cells.append((i0, j0, k0, fu, fv, fw))
                        for di in range(2):
                            for dj in range(2):
                                for dk in range(2):
                                    covered.add((i0 + di, j0 + dj, k0 + dk))
                    rank2 = 7 * size + (n * m * p - len(covered))
                    if rank2 >= best_rank:
                        continue
                    U = [[0] * rank2 for _ in range(n * m)]
                    V = [[0] * rank2 for _ in range(m * p)]
                    W = [[0] * rank2 for _ in range(p * n)]
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if (i, j, k) in covered:
                                    continue
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
                    for (i0, j0, k0, fu, fv, fw) in block_cells:
                        for di in range(2):
                            for dj in range(2):
                                for dk in range(2):
                                    for r in range(7):
                                        u = fu[di * 2 + dj][r]
                                        v = fv[dj * 2 + dk][r]
                                        w = fw[dk * 2 + di][r]
                                        if u:
                                            U[(i0 + di) * m + (j0 + dj)][col + r] += u
                                        if v:
                                            V[(j0 + dj) * p + (k0 + dk)][col + r] += v
                                        if w:
                                            W[(k0 + dk) * n + (i0 + di)][col + r] += w
                        col += 7
                    if col != rank2:
                        continue
                    if _verify(U, V, W, n, m, p):
                        best_rank = rank2
                        _RESULT["rank"] = rank2
                        _RESULT["U"] = U
                        _RESULT["V"] = V
                        _RESULT["W"] = W
                        return

    for pn in part_n:
        for pm in part_m:
            for pp in part_p:
                if time.time() > deadline:
                    return
                try_blocks(pm, pp, pn)


def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = DIMENSIONS
    U0, V0, W0, rank0 = _standard_block(n, m, p)
    _RESULT["rank"] = rank0
    _RESULT["U"] = U0
    _RESULT["V"] = V0
    _RESULT["W"] = W0
    try:
        forms = _block_forms()
        _search(deadline, forms)
    except Exception:
        pass
    return {"U": _RESULT["U"], "V": _RESULT["V"], "W": _RESULT["W"]}

# EVOLVE-BLOCK-END
