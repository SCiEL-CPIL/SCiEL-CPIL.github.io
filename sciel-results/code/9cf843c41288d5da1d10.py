import numpy as np
import time
from scipy.optimize import minimize

_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

_diff_shifted = _x_flat - _y_flat + 9
_h0_vals = _x_flat
_h1_vals = _x_flat + _y_flat
_h2_vals = _x_flat + 2 * _y_flat

_N_BINS_DIFF = 19
_N_BINS_H0 = 10
_N_BINS_H1 = 19
_N_BINS_H2 = 28

_LN2 = np.log(2.0)
_EPS = 1e-9


def _entropy_from_bincounts(bin_counts, total_weight):
    if total_weight <= 1e-15:
        return 0.0
    probs = bin_counts / total_weight
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    return -np.sum(p * np.log2(p))


def _compute_ratio_fast(w):
    total = w.sum()
    if total <= 1e-15:
        return 0.0
    diff_counts = np.bincount(_diff_shifted, weights=w, minlength=_N_BINS_DIFF)
    h0_counts = np.bincount(_h0_vals, weights=w, minlength=_N_BINS_H0)
    h1_counts = np.bincount(_h1_vals, weights=w, minlength=_N_BINS_H1)
    h2_counts = np.bincount(_h2_vals, weights=w, minlength=_N_BINS_H2)

    D = _entropy_from_bincounts(diff_counts, total)
    H0 = _entropy_from_bincounts(h0_counts, total)
    H1 = _entropy_from_bincounts(h1_counts, total)
    H2 = _entropy_from_bincounts(h2_counts, total)

    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M


def _compute_ratio(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    return _compute_ratio_fast(P.flatten())


def _neg_ratio_and_grad(v):
    """Objective in softmax space: minimize -(D/M). Returns (f, grad)."""
    v = v - v.max()
    expv = np.exp(v)
    w = expv / expv.sum()
    S = 1.0  # w is normalized

    def _entropy_grad(counts, nbins):
        p = counts  # already normalized (sum 1)
        m = p > 1e-12
        E = 0.0
        if np.any(m):
            pm = p[m]
            E = -np.sum(pm * np.log2(pm))
        # dE/dp_i = -(log2 p_i + 1)/ln2 for p_i>0 else 0
        dE = np.zeros(nbins)
        if np.any(m):
            dE[m] = -(np.log2(p[m]) + 1.0) / _LN2
        return E, dE

    dc = np.bincount(_diff_shifted, weights=w, minlength=_N_BINS_DIFF)
    c0 = np.bincount(_h0_vals, weights=w, minlength=_N_BINS_H0)
    c1 = np.bincount(_h1_vals, weights=w, minlength=_N_BINS_H1)
    c2 = np.bincount(_h2_vals, weights=w, minlength=_N_BINS_H2)

    D, dD = _entropy_grad(dc, _N_BINS_DIFF)
    H0, dH0 = _entropy_grad(c0, _N_BINS_H0)
    H1, dH1 = _entropy_grad(c1, _N_BINS_H1)
    H2, dH2 = _entropy_grad(c2, _N_BINS_H2)

    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 1e6, np.zeros_like(v)

    val = D / M
    # gradient of val wrt w
    def _dw(dE, counts, nbins):
        # dE/dw_j = (1/S) * sum_i dE_i (delta_ij - p_i)  with S=1
        # = dE_j - p_j * sum_i dE_i
        return dE - counts * np.sum(dE)

    gD = _dw(dD, dc, _N_BINS_DIFF)
    gH0 = _dw(dH0, c0, _N_BINS_H0)
    gH1 = _dw(dH1, c1, _N_BINS_H1)
    gH2 = _dw(dH2, c2, _N_BINS_H2)

    # dM/dw: subgradient, use the argmax
    if H0 >= H1 and H0 >= H2:
        gM = gH0
    elif H1 >= H0 and H1 >= H2:
        gM = gH1
    else:
        gM = gH2

    # d(D/M)/dw = (gD*M - D*gM)/M^2
    gval = (gD * M - D * gM) / (M * M)

    # chain rule: w = softmax(v), dw/dv = w * (I - w)
    # grad_v = w * (gval - sum(gval * w))
    grad_v = w * (gval - np.dot(gval, w))
    return -val, grad_v


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    timeout = 600
    safety_margin = 20
    deadline = start_time + timeout - safety_margin

    def _elapsed():
        return time.time() - start_time

    # ---- Build diverse seed weight matrices ----
    def _norm(P):
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-9:
            return np.ones((n, n)) / 100.0
        return P / s

    seeds = []

    # uniform
    seeds.append(np.ones((n, n), dtype=np.float64) / 100.0)

    # diagonal (x=y)
    w = np.zeros((n, n)); np.fill_diagonal(w, 1.0); seeds.append(_norm(w))

    # anti-diagonal (x+y=9)
    w = np.zeros((n, n))
    for i in range(n):
        j = 9 - i
        if 0 <= j < n:
            w[i, j] = 1.0
    seeds.append(_norm(w))

    # 1/sqrt(count_d): uniformize diagonals
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = j - i
            cd = min(n - abs(d), n)
            w[i, j] = 1.0 / np.sqrt(cd)
    seeds.append(_norm(w))

    # central x focus
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            w[i, j] = np.exp(-0.3 * (j - 4.5) ** 2)
    seeds.append(_norm(w))

    # combined gaussian
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = j - i
            s_xy = i + j
            w[i, j] = np.exp(-0.1 * abs(d)) * np.exp(-0.05 * abs(s_xy - 9))
    seeds.append(_norm(w))

    # 1/sqrt(count_d) * 1/(x+0.5)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = j - i
            cd = min(n - abs(d), n)
            w[i, j] = (1.0 / np.sqrt(cd)) * (1.0 / (j + 0.5))
    seeds.append(_norm(w))

    # random seeds
    rng = np.random.RandomState(1234)
    for _ in range(6):
        w = rng.rand(n, n) ** rng.uniform(1.0, 4.0)
        seeds.append(_norm(w))

    # ---- Convert seeds to softmax space ----
    def _to_v(P):
        P = np.clip(P, 1e-12, None)
        return np.log(P)

    # ---- Multi-start L-BFGS-B ----
    best_P = seeds[0].copy()
    best_score = _compute_ratio_fast(best_P.flatten())

    # Pre-sort seeds by score so we spend time on the good ones first
    scored = []
    for P in seeds:
        sc = _compute_ratio_fast(P.flatten())
        scored.append((sc, P))
    scored.sort(key=lambda t: -t[0])

    for sc, P in scored:
        if _elapsed() > deadline - 30:
            break
        if sc > best_score:
            best_score = sc
            best_P = P.copy()

    # Optimize from each seed (best first), with time budget per start
    n_starts = len(scored)
    per_start = (deadline - _elapsed() - 40) / max(n_starts, 1)
    per_start = max(per_start, 8.0)

    for sc, P in scored:
        if _elapsed() > deadline - 30:
            break
        v0 = _to_v(P).flatten()
        start_opt = time.time()
        try:
            res = minimize(
                _neg_ratio_and_grad,
                v0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10,
                         'maxls': 50, 'maxfun': 50000},
            )
        except Exception:
            continue
        if time.time() - start_opt > per_start + 5:
            # still evaluate partial result
            pass
        v_opt = res.x
        w_opt = np.exp(v_opt - v_opt.max())
        w_opt = w_opt / w_opt.sum()
        P_opt = w_opt.reshape(n, n)
        sc_opt = _compute_ratio_fast(P_opt.flatten())
        if sc_opt > best_score:
            best_score = sc_opt
            best_P = P_opt.copy()

    # ---- Polish with very fine L-BFGS-B from best ----
    for _ in range(3):
        if _elapsed() > deadline - 20:
            break
        v0 = _to_v(best_P).flatten()
        start_opt = time.time()
        try:
            res = minimize(
                _neg_ratio_and_grad,
                v0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12,
                         'maxls': 50, 'maxfun': 200000},
            )
        except Exception:
            break
        v_opt = res.x
        w_opt = np.exp(v_opt - v_opt.max())
        w_opt = w_opt / w_opt.sum()
        P_opt = w_opt.reshape(n, n)
        sc_opt = _compute_ratio_fast(P_opt.flatten())
        if sc_opt > best_score:
            best_score = sc_opt
            best_P = P_opt.copy()
        else:
            break

    # ---- Final validation and normalization ----
    result = np.clip(best_P, 0, None)
    s = result.sum()
    if s <= 1e-9:
        result = np.ones((n, n)) / 100.0
    else:
        result = result / s

    return result
