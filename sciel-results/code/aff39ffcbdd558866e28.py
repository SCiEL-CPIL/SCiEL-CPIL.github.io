"""Broadband antireflection coating design: annealed layer pattern + Nelder-Mead."""
import time
import math
import numpy as np

# ---- thin-film characteristic-matrix model (vectorized over wavelengths) ----
def _reflectance(matrices, n_s, n_0):
    # matrices: (M,2,2) complex; n_s,n_0 real. Returns R in [0,1], shape (M,).
    M = matrices.shape[0]
    m11 = np.empty((M, 2, 2), dtype=complex)
    m11[:, 0, 0] = 1.0
    m11[:, 1, 1] = 1.0
    for k in range(matrices.shape[1]):
        m11 = m11 @ matrices[:, k, :, :]
    y0 = np.array([n_0, 1.0], dtype=complex)
    y_s = np.array([1.0, n_s], dtype=complex)
    v0 = m11[:, 0, 0] * y0[0] + m11[:, 0, 1] * y0[1]
    w0 = m11[:, 1, 0] * y0[0] + m11[:, 1, 1] * y0[1]
    vs = m11[:, 0, 0] * y_s[0] + m11[:, 0, 1] * y_s[1]
    ws = m11[:, 1, 0] * y_s[0] + m11[:, 1, 1] * y_s[1]
    r = (vs * w0 - ws * v0) / (vs * w0 + ws * v0)
    return (r * np.conj(r)).real


def _make_evaluator(problem):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat = problem["materials"]
    nL = mat["L"]; nH = mat["H"]
    w0, w1 = problem["band_nm"]
    wl = np.linspace(w0, w1, 401)
    k = 2.0 * math.pi / wl
    maxL = problem["max_layer_nm"]
    maxN = problem["max_layers"]
    maxT = problem["max_total_nm"]

    def evalf(x):
        N = int(min(round(x[0]), maxN))
        if N <= 0:
            N = 1
        types = x[1:1 + N].copy()
        types[types < 0.5] = 0.0
        types[types >= 0.5] = 1.0
        d = x[1 + N:1 + N + N].copy()
        d = np.clip(d, 0.0, maxL)
        if d.sum() > maxT:
            d *= maxT / d.sum()
            d = np.clip(d, 0.0, maxL)
        nref = np.where(types > 0.5, nH, nL)
        beta = k * nref
        c = np.cos(beta)
        s = np.sin(beta)
        y = nref
        mats = np.empty((len(wl), N, 2, 2), dtype=complex)
        mats[:, :, 0, 0] = c
        mats[:, :, 0, 1] = 1j * s / y
        mats[:, :, 1, 0] = 1j * y * s
        mats[:, :, 1, 1] = c
        R = _reflectance(mats, ns, n0)
        R = np.clip(R, 0.0, 1.0)
        return float(R.mean()) * 100.0

    return evalf, maxN, maxL


# ---- Nelder-Mead (self-contained) ----
def _nm(f, x0, maxfev=400):
    n = len(x0)
    simplex = [x0.astype(float).copy()]
    for i in range(n):
        p = x0.copy(); p[i] += 0.05 if x0[i] == 0 else 0.05 * x0[i]
        simplex.append(p)
    vals = [f(p) for p in simplex]
    fe = n + 1
    for _ in range(4000):
        if fe >= maxfev:
            break
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        xw, xg = simplex[-1], simplex[-2]
        xc = sum(simplex[:-1]) / n
        xr = xc + (xc - xw)
        fe += 1; fr = f(xr)
        if fr < vals[0]:
            xe = xc + 2.0 * (xc - xw); fe += 1; fe_ = f(xe)
            if fe_ < fr:
                simplex[-1] = xe; vals[-1] = fe_
            else:
                simplex[-1] = xr; vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr; vals[-1] = fr
        else:
            xc = xc + 0.5 * (xw - xc); fe += 1; fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc; vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i]); fe += 1
                    if fe >= maxfev:
                        break
        if fe >= maxfev:
            break
    order = np.argsort(vals)
    return simplex[order[0]], vals[order[0]]


def _coord(f, x, maxfev=200):
    n = len(x); best = f(x); fe = 1
    for _ in range(30):
        improved = False
        for i in range(n):
            for step in (0.5, 0.05):
                for sgn in (1.0, -1.0):
                    if fe >= maxfev:
                        break
                    x2 = x.copy(); x2[i] += sgn * step * max(abs(x[i]), 1.0)
                    fe += 1; fv = f(x2)
                    if fv < best - 1e-12:
                        x = x2; best = fv; improved = True
        if not improved:
            break
    return x, best


def _optimize_pattern(evalf, N, maxL, maxT, starts, deadline):
    nL, nH = 1.38, 2.35
    centre = 600.0
    lam4L = centre / (4.0 * nL)
    lam4H = centre / (4.0 * nH)
    best_x, best_f = None, float("inf")
    for it in range(starts):
        if time.time() > deadline:
            break
        if it == 0:
            d = np.empty(N)
            for j in range(N):
                d[j] = lam4L if (j % 2 == 0) else lam4H
        else:
            d = np.random.uniform(0.05 * maxL, 0.95 * maxL, N)
        if d.sum() > maxT:
            d *= maxT / d.sum()
        x0 = np.concatenate(([float(N)], np.zeros(N), d))
        x, fv = _nm(evalf, x0, maxfev=300)
        x, fv = _coord(evalf, x, maxfev=150)
        if fv < best_f:
            best_f, best_x = fv, x
    return best_x, best_f


def _unpack(x, maxN, maxL):
    N = int(min(round(x[0]), maxN)); N = max(1, N)
    types = x[1:1 + N].copy(); types[types < 0.5] = 0.0; types[types >= 0.5] = 1.0
    d = x[1 + N:1 + N + N].copy(); d = np.clip(d, 0.0, maxL)
    if d.sum() > 1000.0:
        d *= 1000.0 / d.sum()
    labels = ["H" if t > 0.5 else "L" for t in types]
    return labels, [float(v) for v in d]


def design(problem):
    t0 = time.time()
    deadline = t0 + 140.0          # safety margin well under the 150 s wall-clock kill
    evalf, maxN, maxL = _make_evaluator(problem)
    np.random.seed(12345)

    best_x, best_f = None, float("inf")
    # initial: quarter-wave quarter-wave stack (L H)
    d0 = np.array([600.0 / (4.0 * 1.38), 600.0 / (4.0 * 2.35)])
    x0 = np.concatenate(([2.0], [0.0, 1.0], d0))
    x, fv = _nm(evalf, x0, maxfev=400); x, fv = _coord(evalf, x, maxfev=200)
    best_x, best_f = x, fv

    # anneal the layer pattern
    N = 2
    cur_types = np.array([0.0, 1.0])
    cur_x, cur_f = best_x, best_f
    T = 0.15
    for it in range(60):
        if time.time() > deadline:
            break
        # propose a pattern mutation
        r = np.random.random()
        if r < 0.30 and N < maxN:
            N += 1; cur_types = np.append(cur_types, float(np.random.randint(0, 2)))
        elif r < 0.55 and N > 1:
            N -= 1; cur_types = cur_types[:N]
        elif N > 1:
            idx = int(np.random.randint(0, N)); cur_types[idx] = float(np.random.randint(0, 2))
        # optimize thicknesses for this pattern (few starts, cheap)
        x, fv = _optimize_pattern(evalf, N, maxL, problem["max_total_nm"],
                                  starts=2, deadline=deadline)
        # accept/reject on the continuous optimum
        if fv < cur_f or np.random.random() < math.exp(-(fv - cur_f) / max(T, 1e-6)):
            cur_x, cur_f = x, fv
            if fv < best_f:
                best_x, best_f = x, fv
        T *= 0.85

    if best_x is None:
        return ["L"], [600.0 / (4.0 * 1.38)]
    return _unpack(best_x, maxN, maxL)
