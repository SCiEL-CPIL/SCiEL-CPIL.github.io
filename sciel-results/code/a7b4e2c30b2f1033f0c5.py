"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [4, 5, 6]


def _comps(sz, k):
    """All compositions of sz into exactly k positive parts."""
    if k <= 0:
        return []
    if k == 1:
        yield (sz,)
        return
    for first in range(1, sz - k + 2):
        for rest in _comps(sz - first, k - 1):
            yield (first,) + rest


def _gen_partitions(dim):
    """All ordered partitions of dim as lists of positive ints."""
    out = []
    for k in range(1, dim + 1):
        for c in _comps(dim, k):
            out.append(list(c))
    return out


def _strassen2():
    """Strassen 2x2x2, local row order: U/V rows (a,b) -> a*2+b, W rows (c,a) -> c*2+a."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    # M1=(A00+A11)(B00+B11): C00+=, C11+=
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # M2=(A10+A11)B10: C10+=
    U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
    # M3=A00(B01-B11): C01+=
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
    # M4=A11(B10-B00): C00+=, C10+=
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    # M5=(A00+A01)B11: C00-=, C01+=
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    # M6=(A10-A00)(B00+B01): C11+=
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # M7=(A01-A11)(B10+B11): C00+=, C11+=
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
    return U, V, W


def _strassen3():
    """Strassen 3x3x3 (rank 23), local row order: U/V rows (a,b)->a*3+b, W rows (c,a)->c*3+a."""
    U = [[0] * 23 for _ in range(9)]
    V = [[0] * 23 for _ in range(9)]
    W = [[0] * 23 for _ in range(9)]

    def a(i, j):
        return i * 3 + j

    def c(i, j):
        return i * 3 + j

    def add(A, row, col, val):
        A[row][col] += val

    # M1=(A00+A11+A22)(B00+B11+B22): C00+=, C11+=, C22+=
    for i in (0, 1, 2):
        add(U, a(i, i), 0, 1); add(V, a(i, i), 0, 1); add(W, c(i, i), 0, 1)
    # M2=(A10+A20)(B10+B20): C10+=, C20+=
    for i in (1, 2):
        add(U, a(i, 0), 1, 1); add(V, a(i, 0), 1, 1); add(W, c(i, 0), 1, 1)
    # M3=A00(B01+B12+B22): C01+=, C02+=
    add(U, a(0, 0), 2, 1); add(V, a(0, 1), 2, 1); add(V, a(1, 2), 2, 1); add(V, a(2, 2), 2, 1)
    add(W, c(0, 1), 2, 1); add(W, c(0, 2), 2, 1)
    # M4=A11(B10+B21): C10+=, C12+=
    add(U, a(1, 1), 3, 1); add(V, a(1, 0), 3, 1); add(V, a(2, 1), 3, 1)
    add(W, c(1, 0), 3, 1); add(W, c(1, 2), 3, 1)
    # M5=A22(B20+B02): C20+=, C02+=
    add(U, a(2, 2), 4, 1); add(V, a(2, 0), 4, 1); add(V, a(0, 2), 4, 1)
    add(W, c(2, 0), 4, 1); add(W, c(0, 2), 4, 1)
    # M6=(A10+A11)(B01+B11): C01+=, C11+=
    add(U, a(1, 0), 5, 1); add(U, a(1, 1), 5, 1); add(V, a(0, 1), 5, 1); add(V, a(1, 1), 5, 1)
    add(W, c(0, 1), 5, 1); add(W, c(1, 1), 5, 1)
    # M7=(A20+A21)(B01+B11): C01+=, C11+=
    add(U, a(2, 0), 6, 1); add(U, a(2, 1), 6, 1); add(V, a(0, 1), 6, 1); add(V, a(1, 1), 6, 1)
    add(W, c(0, 1), 6, 1); add(W, c(1, 1), 6, 1)
    # M8=(A01+A12)(B10+B20): C10+=, C20+=
    add(U, a(0, 1), 7, 1); add(U, a(1, 2), 7, 1); add(V, a(1, 0), 7, 1); add(V, a(2, 0), 7, 1)
    add(W, c(1, 0), 7, 1); add(W, c(2, 0), 7, 1)
    # M9=(A00+A01)(B10+B12): C10+=, C12+=
    add(U, a(0, 0), 8, 1); add(U, a(0, 1), 8, 1); add(V, a(1, 0), 8, 1); add(V, a(1, 2), 8, 1)
    add(W, c(1, 0), 8, 1); add(W, c(1, 2), 8, 1)
    # M10=(A10+A11)(B01+B02): C01+=, C02+=
    add(U, a(1, 0), 9, 1); add(U, a(1, 1), 9, 1); add(V, a(0, 1), 9, 1); add(V, a(0, 2), 9, 1)
    add(W, c(0, 1), 9, 1); add(W, c(0, 2), 9, 1)
    # M11=(A20+A21)(B01+B02): C01+=, C02+=
    add(U, a(2, 0), 10, 1); add(U, a(2, 1), 10, 1); add(V, a(0, 1), 10, 1); add(V, a(0, 2), 10, 1)
    add(W, c(0, 1), 10, 1); add(W, c(0, 2), 10, 1)
    # M12=(A01+A10)(B10+B20): C10+=, C20+=
    add(U, a(0, 1), 11, 1); add(U, a(1, 0), 11, 1); add(V, a(1, 0), 11, 1); add(V, a(2, 0), 11, 1)
    add(W, c(1, 0), 11, 1); add(W, c(2, 0), 11, 1)
    # M13=A12(B11+B20): C11+=, C20+=
    add(U, a(1, 2), 12, 1); add(V, a(1, 1), 12, 1); add(V, a(2, 0), 12, 1)
    add(W, c(1, 1), 12, 1); add(W, c(2, 0), 12, 1)
    # M14=A21(B20+B02): C20+=, C02+=
    add(U, a(2, 1), 13, 1); add(V, a(2, 0), 13, 1); add(V, a(0, 2), 13, 1)
    add(W, c(2, 0), 13, 1); add(W, c(0, 2), 13, 1)
    # M15=(A00+A11+A22)(B10+B21): C10+=, C20+=
    for i in (0, 1, 2):
        add(U, a(i, i), 14, 1); add(V, a(1, 0), 14, 1); add(V, a(2, 1), 14, 1)
        add(W, c(1, 0), 14, 1); add(W, c(2, 0), 14, 1)
    # M16=(A00+A11+A22)(B01+B20): C01+=, C20+=
    for i in (0, 1, 2):
        add(U, a(i, i), 15, 1); add(V, a(0, 1), 15, 1); add(V, a(2, 0), 15, 1)
        add(W, c(0, 1), 15, 1); add(W, c(2, 0), 15, 1)
    # M17=(A01+A10)(B20+B12): C20+=, C12+=
    add(U, a(0, 1), 16, 1); add(U, a(1, 0), 16, 1); add(V, a(2, 0), 16, 1); add(V, a(1, 2), 16, 1)
    add(W, c(2, 0), 16, 1); add(W, c(1, 2), 16, 1)
    # M18=(A12+A21)(B01+B20): C01+=, C20+=
    add(U, a(1, 2), 17, 1); add(U, a(2, 1), 17, 1); add(V, a(0, 1), 17, 1); add(V, a(2, 0), 17, 1)
    add(W, c(0, 1), 17, 1); add(W, c(2, 0), 17, 1)
    # M19=(A10+A11+A22)(B01+B11+B22): C01+=, C11+=, C22+=
    for i, v in ((1, 1), (2, 1), (0, 1)):
        pass
    for i in (1, 2, 0):
        add(U, a(i, i), 18, 1)
    add(V, a(0, 1), 18, 1); add(V, a(1, 1), 18, 1); add(V, a(2, 2), 18, 1)
    add(W, c(0, 1), 18, 1); add(W, c(1, 1), 18, 1); add(W, c(2, 2), 18, 1)
    # M20=(A12+A21)(B10+B22): C10+=, C22+=
    add(U, a(1, 2), 19, 1); add(U, a(2, 1), 19, 1); add(V, a(1, 0), 19, 1); add(V, a(2, 2), 19, 1)
    add(W, c(1, 0), 19, 1); add(W, c(2, 2), 19, 1)
    # M21=(A01+A10)(B12+B20): C20+=, C12+=
    add(U, a(0, 1), 20, 1); add(U, a(1, 0), 20, 1); add(V, a(1, 2), 20, 1); add(V, a(2, 0), 20, 1)
    add(W, c(2, 0), 20, 1); add(W, c(1, 2), 20, 1)
    # M22=(A12+A21)(B20+B11): C20+=, C11+=
    add(U, a(1, 2), 21, 1); add(U, a(2, 1), 21, 1); add(V, a(2, 0), 21, 1); add(V, a(1, 1), 21, 1)
    add(W, c(2, 0), 21, 1); add(W, c(1, 1), 21, 1)
    # M23=(A01+A12)(B20+B11): C20+=, C11+=
    add(U, a(0, 1), 22, 1); add(U, a(1, 2), 22, 1); add(V, a(2, 0), 22, 1); add(V, a(1, 1), 22, 1)
    add(W, c(2, 0), 22, 1); add(W, c(1, 1), 22, 1)
    return U, V, W


def _block_factors(nb, mb, pb):
    """Return the list of scalar factors (rank-1 tensors) for a block:
    Strassen when possible, else one factor per (i,j,k)."""
    if nb == 2 and mb == 2 and pb == 2:
        U, V, W = _strassen2()
        R = 7
        return [
            (U[r], V[r], W[r]) for r in range(R)
        ]
    if nb == 3 and mb == 3 and pb == 3:
        U, V, W = _strassen3()
        R = 23
        return [
            (U[r], V[r], W[r]) for r in range(R)
        ]
    # Schoolbook scalar factors
    factors = []
    for i in range(nb):
        for j in range(mb):
            for k in range(pb):
                u = [0.0] * (nb * mb); v = [0.0] * (mb * pb); w = [0.0] * (pb * nb)
                u[i * mb + j] = 1.0
                v[j * pb + k] = 1.0
                w[k * nb + i] = 1.0
                factors.append((u, v, w))
    return factors


def _merge_factors(factors, nb, mb, pb):
    """Try to reduce the number of factors by merging a pair of scalar factors.
    Return a new factor list of equal or smaller size."""
    best = list(factors)
    best_rank = len(best)
    n_f = len(best)
    if n_f <= 2:
        return best
    for a in range(n_f):
        for b in range(a + 1, n_f):
            fu, fv, fw = best[a]
            gu, gv, gw = best[b]
            # merged scalar factor = sum of the two rank-1 tensors
            mu = [fu[r] + gu[r] for r in range(len(fu))]
            mv = [fv[r] + gv[r] for r in range(len(fv))]
            mw = [fw[r] + gw[r] for r in range(len(fw))]
            new_list = best[:a] + [best[a + 1:b]] + [best[b + 1:]] + [(mu, mv, mw)]
            if len(new_list) < best_rank:
                best_rank = len(new_list)
                best = new_list
    return best


def _try_partition(n, m, p, np_, mp_, pp_):
    """Build factor matrices for a specific partition. Return (total_rank, U,V,W) or None."""
    # offsets
    def offs(part):
        o = []; cur = 0
        for s in part:
            o.append(cur); cur += s
        return o
    noff = offs(np_); moff = offs(mp_); poff = offs(pp_)

    blocks = []  # (ni, mi, pi, factors)
    for ni, nb in enumerate(np_):
        for mi, mb in enumerate(mp_):
            for pi, pb in enumerate(pp_):
                factors = _block_factors(nb, mb, pb)
                blocks.append((ni, mi, pi, factors))

    # Total rank (after per-block merging of scalar blocks)
    total = 0
    for ni, mi, pi, factors in blocks:
        nb = np_[ni]; mb = mp_[mi]; pb = pp_[pi]
        if nb == 2 and mb == 2 and pb == 2 or (nb == 3 and mb == 3 and pb == 3):
            total += len(factors)
        else:
            total += len(_merge_factors(factors, nb, mb, pb))

    nm = n * m; mp = m * p; pn = p * n
    U_full = [[0] * total for _ in range(nm)]
    V_full = [[0] * total for _ in range(mp)]
    W_full = [[0] * total for _ in range(pn)]

    col = 0
    for ni, mi, pi, factors in blocks:
        nb = np_[ni]; mb = mp_[mi]; pb = pp_[pi]
        if (nb == 2 and mb == 2 and pb == 2) or (nb == 3 and mb == 3 and pb == 3):
            used = factors
        else:
            used = _merge_factors(factors, nb, mb, pb)
        no = noff[ni]; mo = moff[mi]; po = poff[pi]
        for (u, v, w) in used:
            for r in range(len(u)):
                if u[r]:
                    U_full[no * m + (r % mb)] and None
                U_full[(no + r // mb) * m + (r % mb)][col] += u[r]
            for r in range(len(v)):
                V_full[(mo + r // pb) * p + (r % pb)][col] += v[r]
            for r in range(len(w)):
                W_full[(po + r // nb) * n + (r % nb)][col] += w[r]
            col += 1

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total):
                    s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return None
    return total, U_full, V_full, W_full


def _best_over_partitions(n, m, p, deadline):
    """Search over partition shapes; return (rank, U, V, W) for the best valid one."""
    best_rank = n * m * p
    best = None
    n_parts = _gen_partitions(n)
    m_parts = _gen_partitions(m)
    p_parts = _gen_partitions(p)
    # Order by expected rank (product of block products), cheaper first
    candidates = []
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                # rough lower bound: sum over blocks of min(strassen, product)
                lb = 0
                for nb in np_:
                    for mb in mp_:
                        for pb in pp_:
                            if nb == 2 and mb == 2 and pb == 2:
                                lb += 7
                            elif nb == 3 and mb == 3 and pb == 3:
                                lb += 23
                            else:
                                lb += nb * mb * pb
                candidates.append((lb, np_, mp_, pp_))
    candidates.sort(key=lambda x: x[0])
    rng = random.Random(12345)
    # shuffle equal-lb entries to diversify order
    for i in range(len(candidates) - 1, 0, -1):
        if candidates[i][0] == candidates[i - 1][0]:
            j = rng.randint(i - 1, i)
            candidates[i], candidates[i - 1] = candidates[j], candidates[i - 1]

    seen = set()
    for lb, np_, mp_, pp_ in candidates:
        if time.time() > deadline:
            break
        if lb >= best_rank:
            break
        key = (tuple(np_), tuple(mp_), tuple(pp_))
        if key in seen:
            continue
        seen.add(key)
        res = _try_partition(n, m, p, np_, mp_, pp_)
        if res is None:
            continue
        rank, U, V, W = res
        if rank < best_rank:
            best_rank = rank
            best = (U, V, W)
    return best_rank, best


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135.0  # safety margin before 150s kill
    nm = n * m; mp = m * p; pn = p * n

    # Start from a known-good fixed partition as the current best.
    start_np = [2, 2]; start_mp = [2, 2, 1]; start_pp = [2, 2, 2]
    res = _try_partition(n, m, p, start_np, start_mp, start_pp)
    if res is not None:
        best_rank, U, V, W = res
    else:
        best_rank, U, V, W = n * m * p, None, None

    # Improve over a wider search of partitions.
    if time.time() < deadline:
        r, res2 = _best_over_partitions(n, m, p, deadline)
        if res2 is not None and r < best_rank:
            best_rank, U, V, W = r, res2[0], res2[1], res2[2]

    # Fallback to pure schoolbook if nothing valid found.
    if U is None:
        rk = n * m * p
        U = [[0] * rk for _ in range(nm)]
        V = [[0] * rk for _ in range(mp)]
        W = [[0] * rk for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_rank = rk

    return {"U": U, "V": V, "W": W}
