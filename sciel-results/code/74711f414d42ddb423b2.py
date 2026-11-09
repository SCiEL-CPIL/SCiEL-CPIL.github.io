"""Antireflection coating design via graded quarter-wave stack + differential evolution."""
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize


def _reflectance_stack(labels, thicknesses, problem):
    """Mean normal-incidence reflectance (%) over the band via characteristic matrix."""
    materials = problem["materials"]
    n0 = float(problem["ambient_index"])
    ns = float(problem["substrate_index"])
    band = problem["band_nm"]

    wl = np.linspace(band[0], band[1], 401)
    k = 2.0 * np.pi / wl

    n_layers = [np.full_like(wl, n0)]
    for lab in labels:
        n_layers.append(np.full_like(wl, float(materials[lab])))
    n_layers.append(np.full_like(wl, ns))

    d_layers = [0.0]
    for t in thicknesses:
        d_layers.append(np.asarray(t, dtype=float) * k)
    d_layers.append(0.0)

    M = np.ones((2, 2, wl.size), dtype=complex)
    for i in range(len(n_layers) - 1):
        ni = n_layers[i]
        di = d_layers[i]
        c = np.cos(di)
        s = np.sin(di)
        B = c + 1j * s / ni
        C = ni * c + 1j * s
        L = np.empty((2, 2, wl.size), dtype=complex)
        L[0, 0, :] = B
        L[0, 1, :] = C
        L[1, 0, :] = 1.0
        L[1, 1, :] = 1.0
        M = M @ L

    m11 = M[0, 0, :]
    m12 = M[0, 1, :]
    m21 = M[1, 0, :]
    m22 = M[1, 1, :]

    eta0 = n_layers[0]
    etas = n_layers[-1]
    Y = (eta0 * m11 + m12) / (eta0 * m21 + m22)
    R = np.abs((etas - Y) / (etas + Y)) ** 2
    return float(np.mean(R) * 100.0)


def _build_from_params(x, problem):
    """Convert optimization parameters to a valid (labels, thicknesses) stack."""
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])

    n_layers = int(round(x[0]))
    n_layers = min(max(n_layers, 1), max_layers)
    base = min(max(x[1], 0.0), max_layer)
    ramp = min(max(x[2], -1.0), 1.0)

    labels = []
    thicknesses = []
    total = 0.0
    for i in range(n_layers):
        lab = "L" if (i % 2 == 0) else "H"
        f = 1.0 + ramp * (i / max(n_layers - 1, 1))
        t = base * f
        t = min(max(t, 0.0), max_layer)
        if total + t > max_total:
            t = max(0.0, max_total - total)
        labels.append(lab)
        thicknesses.append(t)
        total += t
    return labels, thicknesses


def _objective(x, problem):
    labels, thicknesses = _build_from_params(x, problem)
    if not labels:
        return 100.0
    try:
        return _reflectance_stack(labels, thicknesses, problem)
    except Exception:
        return 100.0


def _fallback(problem):
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    t = centre / (4.0 * problem["materials"]["L"])
    return ["L"], [t]


def design(problem):
    deadline = time.time() + 150.0
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])

    best_labels, best_thick = _fallback(problem)
    best_val = _reflectance_stack(best_labels, best_thick, problem)

    try:
        max_layers = problem["max_layers"]
        qwL = centre / (4.0 * nL)
        qwH = centre / (4.0 * nH)
        qw = 0.5 * (qwL + qwH)

        bounds = [
            (1, max_layers),
            (0.5 * qw, 1.5 * qw),
            (-0.5, 0.5),
        ]

        init = np.array([
            [8, qw, 0.0],
            [10, qw, 0.2],
            [12, qw, -0.2],
            [6, qw, 0.1],
            [14, qw, 0.0],
            [9, qw, 0.3],
            [11, qw, -0.3],
            [7, qw, 0.0],
        ], dtype=float)
        for j in range(len(bounds)):
            init[:, j] = np.clip(init[:, j], bounds[j][0], bounds[j][1])

        res = differential_evolution(
            _objective,
            bounds,
            args=(problem,),
            seed=42,
            maxiter=40,
            popsize=12,
            tol=1e-4,
            init=init,
            polish=False,
            updating="immediate",
            workers=1,
        )

        if time.time() < deadline - 5.0 and res is not None and res.x is not None:
            cand_labels, cand_thick = _build_from_params(res.x, problem)
            cand_val = _reflectance_stack(cand_labels, cand_thick, problem)
            if cand_val < best_val:
                best_labels, best_thick, best_val = cand_labels, cand_thick, cand_val

        if time.time() < deadline - 5.0:
            x0 = np.array([float(len(best_labels)),
                           best_thick[0] if best_thick else qw, 0.0])
            r2 = minimize(_objective, x0, args=(problem,), method="Nelder-Mead",
                          options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6})
            if r2 is not None and r2.x is not None:
                cand_labels, cand_thick = _build_from_params(r2.x, problem)
                cand_val = _reflectance_stack(cand_labels, cand_thick, problem)
                if cand_val < best_val:
                    best_labels, best_thick, best_val = cand_labels, cand_thick, cand_val

    except Exception:
        pass

    if time.time() > deadline - 3.0:
        return _fallback(problem)

    if not best_labels:
        return _fallback(problem)
    return best_labels, best_thick
