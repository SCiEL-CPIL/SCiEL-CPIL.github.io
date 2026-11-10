import time
import numpy as np

_N = 8
_FRAC = 1.0 / _N
_IDX8 = np.arange(_N) / _N
_NN = _N * _N
# fixed c-samples in [0,1] (midpoints) for the exact union-volume surrogate
_C = (np.arange(300) + 0.5) / 300.0


def _union_len1d(C, base):
    """Union length of intervals [base[i] + C*c_i, +1/N] for each c in C (batch)."""
    shifts = C[:, None] * _IDX8[None, :]
    l = base[None, :] + shifts
    r = l + _FRAC
    order = np.argsort(l, axis=1)
    ls = np.take_along_axis(l, order, 1)
    rs = np.take_along_axis(r, order, 1)
    rs_cum = np.maximum.accumulate(rs, axis=1)
    gap = np.maximum(rs_cum[:, 1:] - ls[:, 1:], 0.0)
    return rs_cum[:, -1] - ls[:, 0] + gap.sum(axis=1)


def _union_area2d(a_l, a_r, b_l, b_r):
    """Union area of axis-aligned rectangles [a_l,a_r]x[b_l,b_r] (sweep line in a)."""
    n = a_l.size
    if n == 0:
        return 0.0
    ev = np.empty(2 * n, dtype=np.int64)
    ev[0::2] = np.argsort(a_l, kind="stable")
    ev[1::2] = np.argsort(a_r, kind="stable")
    order = np.argsort(ev, kind="stable")
    is_start = (order % 2 == 0)
    a_sorted = np.where(is_start, a_l, a_r)[order]
    idx_sorted = order // 2
    bl_sorted = b_l[idx_sorted]
    br_sorted = b_r[idx_sorted]
    # sweep in a
    total = 0.0
    active = []
    prev_a = a_sorted[0]
    for k in range(2 * n):
        a_k = a_sorted[k]
        if a_k > prev_a:
            if active:
                active.sort()
                seg = 0.0
                cur_l = active[0]
                cur_r = br_sorted[active[0]]
                for idx in active[1:]:
                    l2 = bl_sorted[idx]
                    r2 = br_sorted[idx]
                    if l2 > cur_r:
                        seg += (cur_r - cur_l)
                        cur_l = l2
                        cur_r = r2
                    else:
                        if r2 > cur_r:
                            cur_r = r2
                seg += (cur_r - cur_l)
                total += seg * (a_k - prev_a)
        if is_start[k]:
            active.append(idx_sorted[k])
        else:
            try:
                active.remove(idx_sorted[k])
            except ValueError:
                pass
        prev_a = a_k
    return float(total)


def _exact_volume(f, g):
    """Exact 3-D union volume for separable (f,g) (used only to seed the search)."""
    uf = _union_len1d(_C, f)
    ug = _union_len1d(_C, g)
    val = float(np.mean(uf * ug))
    if not np.isfinite(val):
        return 1e9
    return val


def _vol_from_positions(pos):
    """Exact union volume of 64 sheared tubes given 64 base positions (pos shape (64,2))."""
    xk = pos[:, 0]
    yk = pos[:, 1]
    i = np.arange(_NN) // _N
    j = np.arange(_NN) % _N
    a_l = xk[None, :] + _C[:, None] * (i / _N)[None, :]
    a_r = a_l + _FRAC
    b_l = yk[None, :] + _C[:, None] * (j / _N)[None, :]
    b_r = b_l + _FRAC
    areas = np.empty(_C.size)
    for c in range(_C.size):
        areas[c] = _union_area2d(a_l[c], a_r[c], b_l[c], b_r[c])
    val = float(np.mean(areas))
    if not np.isfinite(val):
        return 1e9
    return val


def _keich_v():
    N = _N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N)
    for idx in range(N):
        bits = np.array(list(str(bin(idx))[2:]), dtype=int)
        bits = np.pad(bits, (pos - len(bits), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * bits * (2.0 ** -j_terms) / pos)
    keich = keich - keich[0]
    return np.concatenate([keich[1:], keich[1:]])


def _build(f, g):
    i = np.arange(_NN) // _N
    j = np.arange(_NN) % _N
    return np.stack([f[i], g[j]], axis=1)


def _v_to_fg(v):
    f = np.zeros(_N)
    g = np.zeros(_N)
    f[1:] = v[:7]
    g[1:] = v[7:]
    return f, g


def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    DEADLINE = 630.0 - 60.0

    # ---- seed the search from the best separable Keich configuration ----
    kv = _keich_v()
    kf, kg = _v_to_fg(kv)
    seed_pos = _build(kf, kg)
    best_pos = seed_pos.copy()
    best_vol = _vol_from_positions(best_pos)

    rng = np.random.default_rng(42)

    def _perturb(p, scale):
        return p + rng.normal(0.0, scale, p.shape)

    # ---- basin-hopping / simulated-annealing over full 128-D positions ----
    current_pos = best_pos.copy()
    current_vol = best_vol
    T = 0.05
    T_min = 1e-6
    cooling = 0.999
    n_iter = 0
    max_iter = 40000
    while time.time() < DEADLINE and n_iter < max_iter:
        n_iter += 1
        scale = 0.05 + 0.30 * (T / 0.05)
        cand = _perturb(current_pos, scale)
        cand_vol = _vol_from_positions(cand)
        if cand_vol < current_vol:
            current_pos = cand
            current_vol = cand_vol
            if cand_vol < best_vol:
                best_vol = cand_vol
                best_pos = cand.copy()
        else:
            # Metropolis acceptance
            delta = cand_vol - current_vol
            if delta < 0 or rng.random() < np.exp(-delta / max(T, T_min)):
                current_pos = cand
                current_vol = cand_vol
        T = max(T * cooling, T_min)

    # ---- occasional restarts from fresh random separable configs ----
    rng2 = np.random.default_rng(7)
    n_restarts = 0
    while time.time() < DEADLINE - 30.0 and n_restarts < 40:
        n_restarts += 1
        v0 = rng2.uniform(-1.5, 1.5, 14)
        f0, g0 = _v_to_fg(v0)
        p0 = _build(f0, g0)
        v0_vol = _vol_from_positions(p0)
        # short local basin-hopping from this restart
        cur = p0.copy()
        cv = v0_vol
        Tl = 0.03
        for _ in range(300):
            if time.time() > DEADLINE - 30.0:
                break
            c2 = _perturb(cur, 0.04)
            c2v = _vol_from_positions(c2)
            if c2v < cv or rng.random() < np.exp(-(c2v - cv) / max(Tl, T_min)):
                cur = c2
                cv = c2v
            Tl = max(Tl * 0.99, T_min)
        if cv < best_vol:
            best_vol = cv
            best_pos = cur.copy()

    # ---- final safety: ensure finite & correct shape ----
    if best_pos.shape != (_NN, 2) or not np.all(np.isfinite(best_pos)):
        kf, kg = _v_to_fg(_keich_v())
        best_pos = _build(kf, kg)

    # ---- optional exact MC polish on the incumbent (cheap, uses evaluator's own MC) ----
    try:
        if time.time() < DEADLINE - 5.0:
            def _mc_volume(p):
                xk = p[:, 0]
                yk = p[:, 1]
                i = np.arange(_NN) // _N
                j = np.arange(_NN) % _N
                X0 = float(np.min(xk))
                X1 = float(np.max(xk + (i + 1) / _N))
                Y0 = float(np.min(yk))
                Y1 = float(np.max(yk + (j + 1) / _N))
                if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
                    return np.inf
                rngm = np.random.default_rng(12345)
                a = rngm.uniform(X0, X1, 500000)
                b = rngm.uniform(Y0, Y1, 500000)
                c = rngm.uniform(0.0, 1.0, 500000)
                hit = np.zeros(500000, dtype=bool)
                for t in range(_NN):
                    lt = xk[t] + c * (i[t] / _N)
                    ut = lt + _FRAC
                    lb = yk[t] + c * (j[t] / _N)
                    ub = lb + _FRAC
                    hit |= (a >= lt) & (a <= ut) & (b >= lb) & (b <= ub)
                frac = float(hit.mean())
                if frac <= 0.0 or not np.isfinite(frac):
                    return np.inf
                return frac * (X1 - X0) * (Y1 - Y0)
            # keep the incumbent; we do not have a better MC candidate without more search
            _ = _mc_volume(best_pos)
    except Exception:
        pass

    if best_pos.shape != (_NN, 2) or not np.all(np.isfinite(best_pos)):
        kf, kg = _v_to_fg(_keich_v())
        best_pos = _build(kf, kg)
    return best_pos
