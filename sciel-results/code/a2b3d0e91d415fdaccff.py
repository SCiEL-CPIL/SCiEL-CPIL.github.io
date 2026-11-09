import time
import numpy as np
from scipy.optimize import minimize

def design(problem):
    t0 = time.time()
    DEADLINE = 110.0  # safety margin below the 120s wall-clock kill

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    lam0, lam1 = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    N = 401
    wavelen = np.linspace(lam0, lam1, N)
    coarse = np.linspace(lam0, lam1, 61)

    # Vectorized characteristic-matrix reflectance (percent)
    def reflectance(labels, thicks, wl):
        nmat = np.array([nL if x == 'L' else nH for x in labels])
        t = np.asarray(thicks, dtype=float)
        k = 2.0 * np.pi * nmat[:, None] * t[:, None] / wl[None, :]
        c = np.cos(k)
        s = np.sin(k)
        M = np.empty((len(labels), 2, 2), dtype=complex)
        M[:, 0, 0] = c
        M[:, 0, 1] = 1j * s / nmat[:, None]
        M[:, 1, 0] = 1j * nmat[:, None] * s
        M[:, 1, 1] = c
        # product of all layer matrices
        P = np.empty((2, 2), dtype=complex)
        P[0, 0] = 1.0
        P[0, 1] = 0.0
        P[1, 0] = 0.0
        P[1, 1] = 1.0
        for j in range(len(labels)):
            mj = M[j]
            P[0, 0] = mj[0, 0]*P[0, 0] + mj[0, 1]*P[1, 0]
            P[0, 1] = mj[0, 0]*P[0, 1] + mj[0, 1]*P[1, 1]
            P[1, 0] = mj[1, 0]*P[0, 0] + mj[1, 1]*P[1, 0]
            P[1, 1] = mj[1, 0]*P[0, 1] + mj[1, 1]*P[1, 1]
        B = P[0, 0]*ns + P[0, 1]
        C = P[1, 0]*ns + P[1, 1]
        r = (n0*B - C) / (n0*B + C)
        return np.mean(np.abs(r)**2) * 100.0

    def obj(labels, thicks, wl):
        return reflectance(labels, thicks, wl)

    def make_labels(n):
        L = ['L'] * ((n + 1) // 2)
        H = ['H'] * (n // 2)
        return L + H

    def quarter_thicks(labels, center):
        return [center / (4.0 * (nL if x == 'L' else nH)) for x in labels]

    def refine(labels, x0):
        def f(x):
            return obj(labels, x, wavelen)
        bnds = [(0.0, max_layer)] * len(labels)
        res = minimize(f, x0, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-7})
        x = np.clip(res.x, 0.0, max_layer)
        return float(np.sum(x)), float(obj(labels, x, wavelen)), x

    def feasible(total, val):
        return total <= max_total + 1e-6 and val is not None and val < 1e12

    best_total = None
    best_val = np.inf
    best_labels = None
    best_x = None

    center = 0.5 * (lam0 + lam1)
    nmax = min(max_layers, 14)
    for n in range(2, nmax + 1):
        if time.time() > DEADLINE:
            break
        labels = make_labels(n)
        x0 = np.array(quarter_thicks(labels, center))
        if float(np.sum(x0)) > max_total:
            continue
        # coarse pre-check
        vcoarse = obj(labels, x0, coarse)
        if not (vcoarse < best_val or best_val == np.inf):
            continue
        total, val, x = refine(labels, x0)
        if feasible(total, val) and val < best_val:
            best_val = val
            best_total = total
            best_labels = labels
            best_x = x
            if best_val < 1.0:
                break

    if best_labels is None:
        labels = make_labels(2)
        x0 = np.array(quarter_thicks(labels, center))
        total, val, x = refine(labels, x0)
        best_labels = labels
        best_x = x
        best_val = val
        best_total = total

    # final polish if time allows
    if time.time() < DEADLINE - 8.0:
        total, val, x = refine(best_labels, best_x)
        if feasible(total, val) and val < best_val:
            best_val = val
            best_total = total
            best_x = x

    return list(best_labels), [float(v) for v in best_x]
