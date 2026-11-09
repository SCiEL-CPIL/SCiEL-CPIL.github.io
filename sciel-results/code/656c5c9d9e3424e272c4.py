"""Broadband antireflection coating design; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _reflectance(mats, thicknesses, n_a, n_s, wls):
    """Vectorized thin-film characteristic-matrix reflectance (percent).

    mats: list of real refractive indices, ordered air->substrate.
    thicknesses: list of physical thicknesses in nm.
    wls: array of wavelengths in nm.
    """
    n_a = float(n_a)
    n_s = float(n_s)
    wls = np.asarray(wls, dtype=float)
    d = np.asarray(thicknesses, dtype=float)
    m = len(mats)
    n = np.array(mats, dtype=float)
    beta = 2.0 * np.pi * n * d / wls  # phase, shape (m, len(wls))
    c = np.cos(beta)
    s = np.sin(beta)
    # M[i] = [[c, i*s], [i*s, c]]  (real matrices)
    M = np.empty((m, 2, 2, len(wls)), dtype=float)
    M[:, 0, 0, :] = c
    M[:, 0, 1, :] = s
    M[:, 1, 0, :] = s
    M[:, 1, 1, :] = c
    # product M[0] @ M[1] @ ... @ M[m-1]
    P = M[0]
    for i in range(1, m):
        P = np.einsum('aij,jkl->ailk', P, M[i])
    a = P[:, 0, 0, :]
    b = P[:, 0, 1, :]
    cc = P[:, 1, 0, :]
    dd = P[:, 1, 1, :]
    y = n_s * (a * n_s + b) / (cc * n_s + dd)
    r = (n_a - y) / (n_a + y)
    return np.mean(100.0 * r * r)

def _objective(x, mats, n_a, n_s, wls, max_layer_nm, max_total_nm):
    n_layers = len(mats)
    x = np.asarray(x, dtype=float)
    thicknesses = np.clip(x, 0.0, max_layer_nm)
    if thicknesses.sum() > max_total_nm:
        return 1e6 + (thicknesses.sum() - max_total_nm)
    return _reflectance(mats, thicknesses, n_a, n_s, wls)

def _de_search(mats, n_a, n_s, wls, max_layer_nm, max_total_nm, deadline):
    n_layers = len(mats)
    bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
    # symmetric quarter-wave initial guess
    centre = 0.5 * (wls[0] + wls[-1])
    x0 = np.array([centre / (4.0 * n) for n in mats], dtype=float)
    x0 = np.clip(x0, 0.0, max_layer_nm)
    if x0.sum() > max_total_nm:
        x0 = x0 * (max_total_nm / x0.sum())
    res = differential_evolution(
        _objective, bounds, args=(mats, n_a, n_s, wls, max_layer_nm, max_total_nm),
        x0=x0, seed=12345, maxiter=150, popsize=15, tol=1e-5,
        polish=False, init='latinhypercube',
        callback=lambda xk, converg: (time.time() > deadline)
    )
    best_x = np.asarray(res.x, dtype=float)
    best_f = float(res.fun)
    if best_f >= 1e6:
        best_x = x0
        best_f = _objective(x0, mats, n_a, n_s, wls, max_layer_nm, max_total_nm)
    return best_x, best_f

def design(problem):
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin before 150s wall-clock kill
    n_a = problem["ambient_index"]
    n_s = problem["substrate_index"]
    L = problem["materials"]["L"]
    H = problem["materials"]["H"]
    wls = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem["max_layers"]

    def build_stack(n_layers, start):
        mats = [L if (i % 2 == 0) else H for i in range(n_layers)]
        if start == "H":
            mats = [mats[-1 - i] for i in range(n_layers)]  # reverse so air side starts with H
        return mats

    # candidate layer counts: prefer odd counts (start and end with same material)
    candidates = [7, 9, 11, 13, 15, 17, 19, 5, 3, 1]
    candidates = [c for c in candidates if c <= max_layers]
    if not candidates:
        candidates = [1]

    best_x = None
    best_f = float("inf")
    best_mats = None
    best_thick = None

    for n_layers in candidates:
        for start in ("L", "H"):
            if time.time() > deadline:
                break
            mats = build_stack(n_layers, start)
            x, f = _de_search(mats, n_a, n_s, wls, max_layer_nm, max_total_nm, deadline)
            if f < best_f:
                best_f = f
                best_x = x
                best_mats = mats
                best_thick = list(np.clip(x, 0.0, max_layer_nm))
        if time.time() > deadline:
            break

    if best_mats is None or best_f >= 1e6:
        # fallback: single quarter-wave low-index layer
        centre = 0.5 * (wls[0] + wls[-1])
        best_mats = [L]
        best_thick = [min(centre / (4.0 * L), max_layer_nm)]
        best_f = _reflectance(best_mats, best_thick, n_a, n_s, wls)

    # final validation and clamping
    best_thick = [float(min(max(t, 0.0), max_layer_nm)) for t in best_thick]
    if sum(best_thick) > max_total_nm:
        scale = max_total_nm / sum(best_thick)
        best_thick = [t * scale for t in best_thick]
    if len(best_thick) > max_layers:
        best_thick = best_thick[:max_layers]
        best_mats = best_mats[:max_layers]

    return list(best_mats), best_thick
# EVOLVE-BLOCK-END
