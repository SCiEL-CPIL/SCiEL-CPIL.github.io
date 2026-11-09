"""Broadband antireflection coating design: 20-layer binary pattern search with
vectorized characteristic-matrix evaluation and L-BFGS-B local optimization."""
import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _mean_reflectance(labels, thicknesses, n0, ns, nL, nH, wl_min, wl_max, n_pts=401):
    """Compute mean reflectance (percent) over n_pts wavelengths using the
    characteristic-matrix method at normal incidence. Vectorized over wavelengths."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    # Build wavelength grid
    wavelengths = np.linspace(wl_min, wl_max, n_pts)

    # Precompute layer indices and refractive indices
    indices = np.array([nH if lab == 'H' else nL for lab in labels])
    thicknesses_arr = np.array(thicknesses, dtype=float)

    # For each wavelength, compute the total characteristic matrix
    # M_total = M_1 * M_2 * ... * M_NL  (from substrate side up)
    # Each layer matrix: [[cos(phase), i*sin(phase)/n], [i*n*sin(phase), cos(phase)]]

    # We'll compute for all wavelengths simultaneously
    # phase[i, w] = 2*pi * n_i * d_i / wl[w]
    phases = 2.0 * np.pi * np.outer(indices, thicknesses_arr) / wavelengths[np.newaxis, :]
    # phases shape: (n_layers, n_pts)

    c = np.cos(phases)
    s = np.sin(phases)

    # Initialize total matrix as identity for all wavelengths
    # M = [[m11, m12], [m21, m22]]
    m11 = np.ones(n_pts, dtype=complex)
    m12 = np.zeros(n_pts, dtype=complex)
    m21 = np.zeros(n_pts, dtype=complex)
    m22 = np.ones(n_pts, dtype=complex)

    # Multiply from last layer (closest to substrate) to first (closest to air)
    for i in range(n_layers - 1, -1, -1):
        ni = indices[i]
        L11 = c[i]
        L12 = 1j * s[i] / ni
        L21 = 1j * ni * s[i]
        L22 = c[i]
        # new = L * M
        n11 = L11 * m11 + L12 * m21
        n12 = L11 * m12 + L12 * m22
        n21 = L21 * m11 + L22 * m21
        n22 = L21 * m12 + L22 * m22
        m11, m12, m21, m22 = n11, n12, n21, n22

    # Effective admittance Y = (m11 * ns + m12) / (m21 * ns + m22)
    Y = (m11 * ns + m12) / (m21 * ns + m22)
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2

    return float(np.mean(R) * 100.0)


def _nm_fallback(n0, ns, nL, nH, wl_min, wl_max):
    """Simple quarter-wave stack fallback."""
    centre = 0.5 * (wl_min + wl_max)
    qL = centre / (4.0 * nL)
    qH = centre / (4.0 * nH)
    return ['L', 'H', 'L', 'H'], [qL, qH, qL, qH]


def _optimize_thicknesses(labels, thicknesses, n0, ns, nL, nH, wl_min, wl_max,
                          max_layer_nm, max_total_nm, deadline, max_iter=200):
    """Optimize thicknesses for a fixed label pattern using L-BFGS-B or coordinate descent."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        return thicknesses

    def objective(x):
        if time.time() > deadline:
            return 1e6
        t = np.maximum(0.0, np.minimum(max_layer_nm, x))
        if np.sum(t) > max_total_nm + 1e-6:
            return 1e6
        return _mean_reflectance(labels, t.tolist(), n0, ns, nL, nH, wl_min, wl_max)

    # L-BFGS-B with bounds
    if _HAVE_SCIPY:
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        x0 = np.array(thicknesses, dtype=float)
        # Ensure feasibility
        if np.sum(x0) > max_total_nm:
            scale = max_total_nm / np.sum(x0)
            x0 = x0 * scale

        try:
            result = _scipy_minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                                     options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.success or result.fun < objective(x0):
                t = np.maximum(0.0, np.minimum(max_layer_nm, result.x))
                if np.sum(t) <= max_total_nm + 1e-6:
                    return t.tolist()
        except Exception:
            pass

    # Coordinate descent fallback
    current = thicknesses[:]
    current_score = objective(current)
    step = 5.0
    for _ in range(500):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_layers):
            for delta in [step, -step]:
                new_t = current[:]
                new_t[i] = max(0.0, min(max_layer_nm, new_t[i] + delta))
                if np.sum(new_t) > max_total_nm:
                    continue
                s = objective(new_t)
                if s < current_score:
                    current = new_t
                    current_score = s
                    improved = True
        if not improved:
            step *= 0.5
            if step < 0.5:
                break
    return current


def design(problem):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    nL = materials['L']
    nH = materials['H']
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre_wl = 0.5 * (wl_min + wl_max)

    deadline = time.time() + 130.0  # safety margin before 150s wall-clock

    # Start with a reasonable initial design
    qL = centre_wl / (4.0 * nL)
    qH = centre_wl / (4.0 * nH)

    # Try several initial patterns
    initial_patterns = [
        (['L', 'H', 'L', 'H'], [qL, qH, qL, qH]),
        (['H', 'L', 'H', 'L'], [qH, qL, qH, qL]),
        (['L', 'H', 'H', 'L'], [qL, qH, qH, qL]),
        (['L', 'H', 'L', 'H', 'L', 'H'], [qL, qH, qL, qH, qL, qH]),
        (['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'],
         [qL, qH, qL, qH, qL, qH, qL, qH]),
        (['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'],
         [qL, qH, qL, qH, qL, qH, qL, qH, qL, qH]),
    ]

    best_labels = None
    best_thick = None
    best_score = float('inf')

    for labels, thick in initial_patterns:
        if len(labels) > max_layers or sum(thick) > max_total_nm:
            continue
        if any(t < 0 or t > max_layer_nm for t in thick):
            continue
        score = _mean_reflectance(labels, thick, n0, ns, nL, nH, wl_min, wl_max)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thick = thick[:]

    if best_labels is None:
        labels, thick = _nm_fallback(n0, ns, nL, nH, wl_min, wl_max)
        return labels, thick

    # Optimize thicknesses for the best initial pattern
    best_thick = _optimize_thicknesses(best_labels, best_thick, n0, ns, nL, nH,
                                       wl_min, wl_max, max_layer_nm, max_total_nm,
                                       deadline, max_iter=300)
    best_score = _mean_reflectance(best_labels, best_thick, n0, ns, nL, nH, wl_min, wl_max)

    # Try adding layers if we have budget
    if time.time() < deadline - 20.0:
        current_labels = best_labels[:]
        current_thick = best_thick[:]
        current_score = best_score

        for _ in range(10):  # Try up to 10 layer additions
            if len(current_labels) >= max_layers:
                break
            if time.time() > deadline - 10.0:
                break

            best_add_labels = None
            best_add_thick = None
            best_add_score = current_score

            # Try adding at each position with each material
            for pos in range(len(current_labels) + 1):
                for mat in ['L', 'H']:
                    new_labels = current_labels[:]
                    new_thick = current_thick[:]
                    new_labels.insert(pos, mat)
                    nm = qL if mat == 'L' else qH
                    new_thick.insert(pos, nm)

                    if sum(new_thick) > max_total_nm or len(new_labels) > max_layers:
                        continue
                    if any(t < 0 or t > max_layer_nm for t in new_thick):
                        continue

                    s = _mean_reflectance(new_labels, new_thick, n0, ns, nL, nH, wl_min, wl_max)
                    if s < best_add_score:
                        best_add_score = s
                        best_add_labels = new_labels
                        best_add_thick = new_thick

            if best_add_labels is None or best_add_score >= current_score - 0.01:
                break

            # Optimize the new design
            current_labels = best_add_labels
            current_thick = _optimize_thicknesses(current_labels, best_add_thick, n0, ns,
                                                  nL, nH, wl_min, wl_max, max_layer_nm,
                                                  max_total_nm, deadline, max_iter=200)
            current_score = _mean_reflectance(current_labels, current_thick, n0, ns,
                                              nL, nH, wl_min, wl_max)

            if current_score < best_score:
                best_score = current_score
                best_labels = current_labels[:]
                best_thick = current_thick[:]

    # Final optimization pass
    if time.time() < deadline - 5.0:
        best_thick = _optimize_thicknesses(best_labels, best_thick, n0, ns, nL, nH,
                                           wl_min, wl_max, max_layer_nm, max_total_nm,
                                           deadline, max_iter=200)

    return best_labels, best_thick
