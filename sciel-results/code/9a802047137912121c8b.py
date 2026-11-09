"""Broadband antireflection coating design via characteristic-matrix objective + Nelder-Mead."""
import time
import math
import numpy as np

_DEADLINE = 150.0          # wall-clock kill guard (safety margin vs 120 s / 150 s limits)
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _reflectance(materials, thicknesses, wavelengths, ambient_index, substrate_index):
    """Exact normal-incidence reflectance (%) via the characteristic-matrix method."""
    n_amb = ambient_index
    n_sub = substrate_index
    n_layers = [materials[m] for m in materials]
    d = np.asarray(thicknesses, dtype=float)
    k = 2.0 * np.pi / np.asarray(wavelengths, dtype=float)
    # per-layer phase (vectorized over wavelength)
    phase = np.outer(n_layers, d) * k          # (N, W)
    c = np.cos(phase)
    s = np.sin(phase)
    # characteristic matrix of the whole stack, M = prod M_j
    M = np.ones((2, 2, len(wavelengths)), dtype=complex)
    for j in range(len(n_layers)):
        B = (n_layers[j] / n_amb) * c[j]
        C = (1.0 / n_amb) * s[j]
        # M = M @ M_j  where M_j = [[B, iC],[iC, B]]
        M0 = M[:, 0, :]
        M1 = M[:, 1, :]
        M[:, 0, :] = M0 * B + 1j * M1 * C
        M[:, 1, :] = 1j * M0 * C + M1 * B
    # admittance Y = M21 / M11 ; R = |(Y - n_sub)/(Y + n_sub)|^2
    Y = M[1, 1, :] / M[0, 1, :]
    r = (Y - n_sub) / (Y + n_sub)
    return (np.abs(r) ** 2) * 100.0


def design(problem):
    """Return (materials, thicknesses) minimising mean reflectance over the band."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mat = problem["materials"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    wls = np.linspace(band[0], band[1], 401)

    n_L = mat["L"]
    n_H = mat["H"]
    n_mid = 0.5 * (n_L + n_H)

    # baseline (bare glass) for reference / fallback
    base = _reflectance([], [], wls, n_amb, n_sub).mean()

    # a small set of good layer counts to try
    counts = [6, 8, 10, 12, 14, 16]
    counts = [c for c in counts if c <= max_layers]
    if not counts:
        counts = [max_layers]

    # quarter-wave thickness at band centre (nm)
    qL = (band[0] + band[1]) / (4.0 * n_L)
    qH = (band[0] + band[1]) / (4.0 * n_H)
    qL = min(max(qL, 0.0), max_layer)
    qH = min(max(qH, 0.0), max_layer)

    def objective(x):
        if x is None:
            return base
        x = np.clip(np.asarray(x, dtype=float), 0.0, max_layer)
        if x.sum() > max_total:
            return base + 1.0
        # alternating L/H starting with L
        labels = ["L", "H"]
        materials = [labels[i % 2] for i in range(len(x))]
        return _reflectance(materials, x, wls, n_amb, n_sub).mean()

    best_mat = ["L"]
    best_th = [qL]
    best_score = objective([qL])

    rng = np.random.default_rng(20240517)

    for n in counts:
        if _time_left() < 8.0:
            break
        # initial points: alternating quarter-wave + jitter
        base_th = np.array([qL if i % 2 == 0 else qH for i in range(n)])
        base_th = np.clip(base_th, 0.0, max_layer)
        # scale to respect total budget
        s = base_th.sum()
        if s > max_total:
            base_th = base_th * (max_total / s)
        start = np.clip(base_th + 0.15 * base_th * rng.standard_normal(n), 0.0, max_layer)
        start = np.clip(start, 0.0, max_layer)
        if start.sum() > max_total:
            start = start * (max_total / start.sum())

        from scipy.optimize import minimize
        t0 = time.time()
        res = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={
                "xatol": 0.5,
                "fatol": 1e-4,
                "maxiter": 800,
                "maxfev": 1200,
            },
        )
        if res.x is not None:
            x = np.clip(res.x, 0.0, max_layer)
            if x.sum() <= max_total:
                sc = objective(x)
                if sc < best_score:
                    best_score = sc
                    labels = ["L", "H"]
                    best_mat = [labels[i % 2] for i in range(n)]
                    best_th = list(x)
        if time.time() - t0 > _time_left() - 4.0:
            break

    # final safety: ensure constraints and valid return
    best_th = [float(min(max(t, 0.0), max_layer)) for t in best_th]
    if sum(best_th) > max_total:
        s = sum(best_th)
        best_th = [t * (max_total / s) for t in best_th]
    if len(best_mat) > max_layers:
        best_mat = best_mat[:max_layers]
        best_th = best_th[:max_layers]
    if not best_mat:
        return ["L"], [qL]
    return best_mat, best_th
