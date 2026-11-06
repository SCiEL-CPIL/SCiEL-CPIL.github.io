import numpy as np
from scipy.optimize import minimize

def _mean_reflectance_pct(params, indices, n0, ns, band_nm, max_total):
    thicknesses = np.clip(params, 0.0, 400.0)
    lam = np.linspace(band_nm[0], band_nm[1], 401)
    n_layers = len(indices)
    c = np.empty((n_layers, len(lam)))
    s = np.empty((n_layers, len(lam)))
    for i in range(n_layers):
        delta = 2.0 * np.pi * indices[i] * thicknesses[i] / lam
        c[i] = np.cos(delta)
        s[i] = np.sin(delta)
    M11 = np.ones(len(lam), dtype=complex)
    M12 = np.zeros(len(lam), dtype=complex)
    M21 = np.zeros(len(lam), dtype=complex)
    M22 = np.ones(len(lam), dtype=complex)
    for i in range(n_layers):
        ni = indices[i]
        a = 1j * s[i] / ni
        b = 1j * ni * s[i]
        nM11 = M11 * c[i] + M12 * a
        nM12 = M11 * a + M12 * c[i]
        nM21 = M21 * c[i] + M22 * b
        nM22 = M21 * b + M22 * c[i]
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22
    Yin = (M21 + M22 * ns) / (M11 + M12 * ns)
    r = (Yin - n0) / (Yin + n0)
    R = np.abs(r) ** 2
    mean_R = np.mean(R) * 100.0
    total = np.sum(thicknesses)
    penalty = 0.0
    if total > max_total:
        penalty = 100.0 * (total - max_total)
    return mean_R + penalty

def design(problem):
    materials = problem["materials"]
    nL = materials["L"]
    nH = materials["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    max_total = problem["max_total_nm"]
    lam_c = 0.5 * (band[0] + band[1])

    # Candidate patterns: list of label sequences
    patterns = [
        ["L", "H", "L"],
        ["H", "L", "H"],
        ["L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H", "L", "H"],
    ]

    best = None
    best_val = np.inf

    for pat in patterns:
        indices = np.array([materials[lbl] for lbl in pat], dtype=float)
        N = len(pat)
        # initial: quarter wave at center
        x0 = np.array([lam_c / (4.0 * n) for n in indices])
        bounds = [(0.0, 400.0)] * N
        try:
            res = minimize(
                _mean_reflectance_pct, x0, args=(indices, n0, ns, band, max_total),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8}
            )
            x = np.clip(res.x, 0.0, 400.0)
            val = _mean_reflectance_pct(x, indices, n0, ns, band, max_total)
            if val < best_val:
                best_val = val
                best = (pat, x)
        except Exception:
            continue

    if best is None:
        # fallback: single L quarter wave
        return ["L"], [lam_c / (4.0 * nL)]

    pat, x = best
    # final fine optimization on the best pattern
    indices = np.array([materials[lbl] for lbl in pat], dtype=float)
    N = len(pat)
    bounds = [(0.0, 400.0)] * N
    try:
        res = minimize(
            _mean_reflectance_pct, x, args=(indices, n0, ns, band, max_total),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 600, "ftol": 1e-14, "gtol": 1e-10}
        )
        x = np.clip(res.x, 0.0, 400.0)
    except Exception:
        pass

    # ensure total thickness constraint
    total = np.sum(x)
    if total > max_total:
        x = x * (max_total / total)

    return list(pat), [float(t) for t in x]
