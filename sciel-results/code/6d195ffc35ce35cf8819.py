import time
import numpy as np
from scipy.optimize import differential_evolution

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill

def _reflectance(materials, thicknesses, n0, ns, wl):
    """Exact characteristic-matrix reflectance (scalar, %)."""
    n = np.array([n0] + list(materials) + [ns])
    t = np.array([0.0] + list(thicknesses) + [0.0])
    k = 2.0 * np.pi / wl
    # cumulative phase per interface
    phases = k * np.cumsum(n * t)
    m = np.zeros((len(materials), 2, 2), dtype=complex)
    for i in range(len(materials)):
        dp = phases[i + 1] - phases[i]
        c = np.cos(dp)
        s = np.sin(dp)
        ni = n[i + 1]
        m[i, 0, 0] = c
        m[i, 0, 1] = 1j * s / ni
        m[i, 1, 0] = 1j * ni * s
        m[i, 1, 1] = c
    # total matrix
    M = m[0].copy()
    for i in range(1, len(materials)):
        M = M @ m[i]
    Y = (M[0, 0] + M[0, 1] * ns) / (M[1, 0] + M[1, 1] * ns)
    R = np.abs((n0 - Y) / (n0 + Y)) ** 2
    return R * 100.0

def _mean_reflectance(materials, thicknesses, problem, wl_grid):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    idx = {k: problem["materials"][k] for k in materials}
    r = _reflectance(idx, thicknesses, n0, ns, wl_grid)
    return float(np.mean(r))

def _build_stack(params, problem, n_layers):
    """Build an alternating L/H stack with a graded thickness profile."""
    base = params[0]
    grad = params[1]
    n0 = problem["ambient_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    ns = problem["substrate_index"]
    lam_c = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]

    materials = []
    thicknesses = []
    # alternate starting with L (low index) from air side
    for i in range(n_layers):
        frac = i / max(1, n_layers - 1)
        # linear gradient factor: thickness varies from base*(1-grad) to base*(1+grad)
        factor = 1.0 + grad * (2.0 * frac - 1.0)
        mat = 'L' if i % 2 == 0 else 'H'
        nmat = nL if mat == 'L' else nH
        thick = base * factor
        thick = min(max(thick, 0.0), max_layer)
        materials.append(mat)
        thicknesses.append(thick)

    # enforce total thickness constraint by scaling
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]

    return materials, thicknesses

def _objective(params, problem, wl_grid, n_layers):
    if time.time() > _DEADLINE:
        return 100.0
    materials, thicknesses = _build_stack(params, problem, n_layers)
    if sum(thicknesses) > problem["max_total_nm"] + 1e-6:
        return 100.0
    if any(t < -1e-9 or t > problem["max_layer_nm"] + 1e-6 for t in thicknesses):
        return 100.0
    return _mean_reflectance(materials, thicknesses, problem, wl_grid)

def design(problem):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    wl_grid = np.linspace(band[0], band[1], 401)
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    # Try several layer counts, best wins
    best_score = 100.0
    best_materials = None
    best_thicknesses = None

    # candidate layer counts (odd preferred for symmetric AR)
    layer_counts = [3, 5, 7, 9, 11, 13, 15]
    layer_counts = [lc for lc in layer_counts if lc <= max_layers]

    for n_layers in layer_counts:
        if time.time() > _DEADLINE - 20:
            break

        # parameter bounds: base thickness, gradient
        lam_c = 0.5 * (band[0] + band[1])
        # base around quarter-wave of effective index
        base_lo = 20.0
        base_hi = min(max_layer, 120.0)
        grad_lo = -0.5
        grad_hi = 0.5

        bounds = [(base_lo, base_hi), (grad_lo, grad_hi)]

        try:
            res = differential_evolution(
                _objective,
                bounds,
                args=(problem, wl_grid, n_layers),
                maxiter=200,
                popsize=15,
                tol=1e-6,
                seed=42,
                init='sobol',
                polish=True,
                update_interval=5,
            )
            if res.fun < best_score:
                best_score = res.fun
                materials, thicknesses = _build_stack(res.x, problem, n_layers)
                best_materials = materials
                best_thicknesses = thicknesses
        except Exception:
            pass

    # Fallback: single quarter-wave L layer
    if best_materials is None:
        centre = 0.5 * (band[0] + band[1])
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * problem["materials"]["L"])]
        best_score = _mean_reflectance(best_materials, best_thicknesses, problem, wl_grid)

    return best_materials, best_thicknesses
