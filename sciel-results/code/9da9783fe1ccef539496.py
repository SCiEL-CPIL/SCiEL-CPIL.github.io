"""Broadband antireflection coating design: fast characteristic-matrix evaluator + Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """Vectorized reflectance (0..1) over all wavelengths.

    wls: 1D array of wavelengths (nm).
    n_mat: dict mapping 'L'/'H' -> refractive index.
    """
    # Precompute per-layer optical thickness factor: n*d/wl  (shape (n_layers, n_wl))
    nd_wl = np.empty((len(materials), len(wls)), dtype=float)
    for i, m in enumerate(materials):
        nd_wl[i, :] = n_mat[m] * thicknesses[i] / wls

    beta = 2.0 * math.pi * nd_wl
    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Accumulate total characteristic matrix M = M1 @ M2 @ ... @ Mk  (elementwise over wavelengths)
    M11 = np.ones(len(wls), dtype=complex)
    M12 = np.zeros(len(wls), dtype=complex)
    M21 = np.zeros(len(wls), dtype=complex)
    M22 = np.ones(len(wls), dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        A = c
        B = -1j * s / n
        C = 1j * n * s
        D = c

        # M = M @ M_layer
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance at input plane
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R


def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """Mean reflectance in percent over the band. Invalid -> large penalty."""
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
    for t in thicknesses:
        if t < -1e-9 or t > problem["max_layer_nm"] + 1e-9:
            return 1e6
    if sum(thicknesses) > problem["max_total_nm"] + 1e-6:
        return 1e6
    if len(materials) == 0:
        # bare substrate
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0
    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0


def _nelder_mead(objective, x0, bounds, deadline, max_iter=600):
    """Nelder-Mead with clipping to bounds. objective(x)->float."""
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds

    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (hi[i] - lo[i]) if (hi[i] - lo[i]) > 1e-9 else 10.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))

    f_vals = [objective(x) for x in simplex]

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if abs(f_vals[0] - f_vals[-1]) < 1e-9:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_r = np.clip(centroid + 1.0 * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
        if f_r < f_vals[0]:
            # Expansion
            x_e = np.clip(centroid + 2.0 * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
        # Contraction
        x_c = np.clip(centroid + 0.5 * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best = order[0]
    return np.clip(simplex[best], lo, hi), f_vals[best]


def design(problem):
    """Design a broadband antireflection coating (air -> ... -> glass)."""
    start = time.time()
    deadline = start + 145.0  # safety margin before the 150 s kill

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    wls = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    n_mat = problem["materials"]

    def objective(th):
        return _mean_reflectance_percent(materials, list(th), problem, wls, n_mat)

    def scale_to_fit(mats, ths):
        s = sum(ths)
        if s > max_total:
            f = max_total / s
            ths = [t * f for t in ths]
        return mats, ths

    # --- Build several strong initial designs ---
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    qL = centre / (4.0 * n_L)
    qH = centre / (4.0 * n_H)

    candidates = []

    # 1) single L quarter-wave
    candidates.append((["L"], [qL]))

    # 2) two-layer L/H
    candidates.append((["L", "H"], [qL, qH]))

    # 3) four-layer quarter-wave stack (L H L H)
    candidates.append((["L", "H", "L", "H"], [qL, qH, qL, qH]))

    # 4) six-layer quarter-wave stack (L H L H L H)
    candidates.append((["L", "H", "L", "H", "L", "H"], [qL, qH, qL, qH, qL, qH]))

    # 5) eight-layer quarter-wave stack (L H L H L H L H)
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H"],
                       [qL, qH, qL, qH, qL, qH, qL, qH]))

    # 6) 5-layer with a thick central low layer (broadband)
    candidates.append((["L", "H", "L", "H", "L"],
                       [qL, qH, 2.0 * qL, qH, qL]))

    # 7) 6-layer with slightly extended central layers
    candidates.append((["L", "H", "L", "H", "L", "H"],
                       [qL, qH, 1.5 * qL, 1.5 * qH, qL, qH]))

    # Enforce constraints on all candidates
    valid_candidates = []
    for mats, ths in candidates:
        if len(mats) > max_layers:
            continue
        mats, ths = scale_to_fit(mats, ths)
        ths = [min(max(t, 0.0), max_layer) for t in ths]
        if sum(ths) > max_total + 1e-6:
            f = max_total / sum(ths)
            ths = [t * f for t in ths]
        valid_candidates.append((mats, ths))

    # Pick the best initial design
    best_mats, best_ths, best_val = None, None, float("inf")
    for mats, ths in valid_candidates:
        val = _mean_reflectance_percent(mats, ths, problem, wls, n_mat)
        if val < best_val:
            best_val, best_mats, best_ths = val, mats, ths

    # Refine the best design with Nelder-Mead (thicknesses only)
    if best_mats is not None and time.time() < deadline:
        materials = list(best_mats)
        lo = np.zeros(len(materials))
        hi = np.full(len(materials), max_layer)
        x0 = np.array(best_ths, dtype=float)
        x_best, f_best = _nelder_mead(objective, x0, (lo, hi), deadline, max_iter=600)
        # Final validity check
        x_best = np.clip(x_best, 0.0, max_layer)
        if sum(x_best) > max_total + 1e-6:
            f = max_total / sum(x_best)
            x_best = x_best * f
        x_best = np.clip(x_best, 0.0, max_layer)
        val = _mean_reflectance_percent(materials, list(x_best), problem, wls, n_mat)
        if val < best_val:
            best_val, best_mats, best_ths = val, materials, list(x_best)

    # Final safety: ensure constraints
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]

    return list(best_mats), list(best_ths)
# EVOLVE-BLOCK-END
