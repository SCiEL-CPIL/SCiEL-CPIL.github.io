"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0   # safety margin before the 120 s evaluator kill

# Strassen level-1 structure (fixed, known-correct). 6x6 = 2x2 blocks of 3x3.
STRASSEN_TERMS = [
    ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
    ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}),
    ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}),
    ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
    ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),
    ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}),
]
RESULT_CONTRIBS = {
    (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
    (0, 1): [(2, 1), (4, 1)],
    (1, 0): [(1, 1), (3, 1)],
    (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)],
}


def _build_6x6(base_U, base_V, base_W):
    r = len(base_U)
    U = [[0] * r for _ in range(36)]
    V = [[0] * r for _ in range(36)]
    W = [[0] * r for _ in range(36)]
    for col in range(r):
        bu = base_U[col]
        bv = base_V[col]
        bw = base_W[col]
        t_idx = col // 27
        i = (col % 27) // 9
        a = (col % 27) // 3
        k = col % 3
        left_combo, right_combo = STRASSEN_TERMS[t_idx]
        for (br, bc), coeff_l in left_combo.items():
            U[(3 * br + i) * 6 + (3 * bc + a)][col] += coeff_l * bu[i * 3 + a]
        for (br, bc), coeff_r in right_combo.items():
            V[(3 * br + a) * 6 + (3 * bc + k)][col] += coeff_r * bv[a * 3 + k]
        for (rbr, rbc), signs in RESULT_CONTRIBS.items():
            for (si, sign) in signs:
                if si == t_idx:
                    W[(3 * rbc + k) * 6 + (3 * rbr + i)][col] += sign * bw[i * 3 + k]
    return U, V, W


def _build_6x6_schoolbook():
    bu, bv, bw = [], [], []
    for i in range(3):
        for a in range(3):
            for k in range(3):
                U = [0] * 9; V = [0] * 9; W = [0] * 9
                U[i * 3 + a] = 1
                V[a * 3 + k] = 1
                W[i * 3 + k] = 1
                bu.append(U); bv.append(V); bw.append(W)
    return _build_6x6(bu, bv, bw)


def _search_base(rank_target, max_iters, rng):
    N = 27
    school = []
    for i in range(3):
        for a in range(3):
            for k in range(3):
                U = [0] * 9; V = [0] * 9; W = [0] * 9
                U[i * 3 + a] = 1.0
                V[a * 3 + k] = 1.0
                W[i * 3 + k] = 1.0
                school.append((U, V, W))
    r = rank_target
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            return None
        dropped = int(rng.integers(0, N))
        keep = [j for j in range(N) if j != dropped]
        A = np.zeros((N, 3 * r))
        b = np.zeros(N)
        for eq, (i, a, k) in enumerate(
                [(i, a, k) for i in range(3) for a in range(3) for k in range(3)]):
            for s, j in enumerate(keep):
                Uj, Vj, Wj = school[j]
                A[eq, s] = Uj[i * 3 + a] * Wj[i * 3 + k]
                A[eq, r + s] = Vj[a * 3 + k] * Wj[i * 3 + k]
                A[eq, 2 * r + s] = Uj[i * 3 + a] * Vj[a * 3 + k]
            b[eq] = 1.0 - school[dropped][2][i * 3 + k]
        try:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
        except Exception:
            continue
        ok = True
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    tot = 0.0
                    for s, j in enumerate(keep):
                        Uj, Vj, Wj = school[j]
                        tot += sol[s] * Uj[i * 3 + a] * Vj[a * 3 + k] * Wj[i * 3 + k]
                    if abs(tot - 1.0) > 1e-7:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            bu = [0.0] * 9
            bv = [0.0] * 9
            bw = [0.0] * 9
            for s, j in enumerate(keep):
                Uj, Vj, Wj = school[j]
                cu, cv, cw = sol[s], sol[r + s], sol[2 * r + s]
                for idx in range(9):
                    bu[idx] += cu * Uj[idx]
                    bv[idx] += cv * Vj[idx]
                    bw[idx] += cw * Wj[idx]
            return (bu, bv, bw)
    return None


def _to_result(base):
    bu = [tuple(x) for x in base[0]]
    bv = [tuple(x) for x in base[1]]
    bw = [tuple(x) for x in base[2]]
    U, V, W = _build_6x6(bu, bv, bw)
    return {"U": U, "V": V, "W": W}


def construct():
    # Guaranteed-correct rank-189 result (7 Strassen x 27 schoolbook base).
    best = _build_6x6_schoolbook()

    rng = np.random.default_rng(12345)
    # Attempt a rank-26 base decomposition (=> 178 total) via randomised
    # drop-one-term linear solve.  Bounded by the deadline.
    found = None
    outer = 0
    while time.time() < _DEADLINE - 8.0:
        found = _search_base(26, max_iters=400, rng=rng)
        if found is not None:
            break
        outer += 1
        if outer >= 2000:
            break
    if found is not None:
        best = _to_result(found)

    return best
# EVOLVE-BLOCK-END
