"""Broadband antireflection coating design: multi-layer quarter-wave stack +
fast vectorized characteristic-matrix evaluation + coordinate-descent refinement,
all guarded by an explicit wall-clock deadline."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_BUDGET = 140.0          # wall-clock safety margin (evaluator kills at 150 s)


def _char_matrix_reflectance(wl, thicknesses, n_layers, n_amb, n_sub, n_mat):
    """Mean reflectance (%) over `wl` (nm) using the thin-film
    characteristic-matrix method.  All wavelengths are processed at once."""
    wl = np.asarray(wl, dtype=float)
    k = 2.0 * np.pi / wl
    # phase per layer: phi_j = k * n_j * d_j  (normal incidence, lossless)
    # build 2x2 matrices M_j = [[cos phi_j, i sin phi_j / n_j],
    #                            [i n_j sin phi_j, cos phi_j]]
    # n_j is per-layer (vector over layers), broadcast over wavelengths.
    n = n_layers
    phi = k[:, None] * n_mat * thicknesses[None, :]          # (W, n)
    c = np.cos(phi)                                           # (W, n)
    s = np.sin(phi)                                           # (W, n)

    # We need the product of n 2x2 matrices.  Do it via explicit 2x2
    # matrix multiplication accumulating M = M_1 M_2 ... M_n.
    # Represent each layer matrix by its four entries (vectors over W).
    # M_j = [[a_j, b_j],[c_j, d_j]] with
    #   a = cos, b = i sin / n, c = i n sin, d = cos
    # Use complex; i = 1j.
    a = c.copy()
    b = 1j * s / n_mat[None, :]
    cc = 1j * n_mat[None, :] * s
    d = c.copy()

    # Accumulate product.  Start identity.
    P00 = np.ones_like(a, dtype=complex)
    P01 = np.zeros_like(a, dtype=complex)
    P10 = np.zeros_like(a, dtype=complex)
    P11 = np.ones_like(a, dtype=complex)

    for j in range(n):
        a_j = a[:, j]
        b_j = b[:, j]
        c_j = cc[:, j]
        d_j = d[:, j]
        # new P = P * M_j
        n00 = P00 * a_j + P01 * c_j
        n01 = P00 * b_j + P01 * d_j
        n10 = P10 * a_j + P11 * c_j
        n11 = P10 * b_j + P11 * d_j
        P00, P01, P10, P11 = n00, n01, n10, n11

    # Admittance Y = (P10 + P11) / (P00 + P01)   (in units of n_amb)
    Y = (P10 + P11) / (P00 + P01)
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2
    return float(np.mean(R) * 100.0)


def _is_valid(thicknesses, n_layers, max_total, max_layer):
    if len(thicknesses) != n_layers:
        return False
    if np.any(thicknesses < -1e-9) or np.any(thicknesses > max_layer + 1e-9):
        return False
    if float(np.sum(thicknesses)) > max_total + 1e-9:
        return False
    return True


def _quarter_wave_stack(n_layers, band_lo, band_hi, nL, nH):
    """Alternating L/H quarter-wave stack centred in the band, L first
    (air side).  Returns a list of physical thicknesses in nm."""
    centre = 0.5 * (band_lo + band_hi)
    seq = []
    for j in range(n_layers):
        idx = nL if (j % 2 == 0) else nH
        seq.append(centre / (4.0 * idx))
    return seq


def _coordinate_descent(thicknesses, wl, n_layers, n_amb, n_sub, n_mat,
                        max_total, max_layer, max_iter, time_limit):
    """Local coordinate-descent minimization of mean reflectance."""
    d = np.array(thicknesses, dtype=float)
    best_d = d.copy()
    best_val = _char_matrix_reflectance(wl, best_d, n_layers, n_amb, n_sub, n_mat)

    n = n_layers
    for it in range(max_iter):
        if time.time() - _START > time_limit:
            break
        improved = False
        for j in range(n):
            if time.time() - _START > time_limit:
                break
            # Try to adjust layer j.  Keep others fixed.
            # A quick 1-D scan over a window around the current value.
            base = d[j]
            # candidate set: a coarse grid over [0, max_layer] plus
            # neighbours of the current point.
            grid = np.linspace(0.0, max_layer, 25)
            neigh = np.array([base + off for off in
                              (-0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5)])
            neigh = neigh[(neigh >= 0.0) & (neigh <= max_layer)]
            cands = np.unique(np.concatenate([grid, neigh]))
            cur_best = _char_matrix_reflectance(wl, d, n_layers, n_amb, n_sub, n_mat)
            best_c = base
            best_cv = cur_best
            for cv in cands:
                # respect total constraint
                if float(np.sum(d) - base + cv) > max_total + 1e-9:
                    continue
                d[j] = cv
                v = _char_matrix_reflectance(wl, d, n_layers, n_amb, n_sub, n_mat)
                if v < best_cv - 1e-12:
                    best_cv = v
                    best_c = cv
            d[j] = best_c
            if best_c != base:
                improved = True
        # accept if improved
        val = _char_matrix_reflectance(wl, d, n_layers, n_amb, n_sub, n_mat)
        if val < best_val - 1e-12:
            best_val = val
            best_d = d.copy()
        if not improved:
            # try a small random perturbation to escape local minima
            if time.time() - _START > time_limit:
                break
            step = 0.05 * max_layer
            for _ in range(3):
                d2 = d + step * (np.random.rand(n) - 0.5)
                d2 = np.clip(d2, 0.0, max_layer)
                if float(np.sum(d2)) > max_total + 1e-9:
                    continue
                v2 = _char_matrix_reflectance(wl, d2, n_layers, n_amb, n_sub, n_mat)
                if v2 < best_val - 1e-12:
                    d = d2
                    best_val = v2
                    best_d = d2
                    improved = True
                    break
            if not improved:
                break
    return best_d, best_val


def design(problem):
    band = problem["band_nm"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mats = problem["materials"]
    nL = mats["L"]
    nH = mats["H"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    wl = np.linspace(band[0], band[1], 401)
    n_mat = np.array([nL, nH], dtype=float)

    # Start with a 4-layer quarter-wave stack (L H L H) and refine.
    n_layers = min(4, max_layers)
    seq = _quarter_wave_stack(n_layers, band[0], band[1], nL, nH)
    if float(np.sum(seq)) > max_total:
        # scale down to fit
        scale = max_total / float(np.sum(seq))
        seq = [x * scale for x in seq]

    best_d = np.array(seq, dtype=float)
    best_val = _char_matrix_reflectance(wl, best_d, n_layers, n_amb, n_sub, n_mat)

    # Coordinate-descent refinement.
    time_limit = _BUDGET - 10.0
    d, val = _coordinate_descent(
        best_d, wl, n_layers, n_amb, n_sub, n_mat,
        max_total, max_layer, max_iter=400, time_limit=time_limit
    )
    if val < best_val:
        best_d, best_val = d, val

    # Try a couple of alternative layer counts if time remains.
    for n_try in (6, 2):
        if time.time() - _START > time_limit:
            break
        if n_try > max_layers:
            continue
        seq2 = _quarter_wave_stack(n_try, band[0], band[1], nL, nH)
        if float(np.sum(seq2)) > max_total:
            scale = max_total / float(np.sum(seq2))
            seq2 = [x * scale for x in seq2]
        d2 = np.array(seq2, dtype=float)
        d2, v2 = _coordinate_descent(
            d2, wl, n_try, n_amb, n_sub, n_mat,
            max_total, max_layer, max_iter=200, time_limit=time_limit
        )
        if v2 < best_val:
            best_d, best_val = d2, v2

    # Final safety: clamp and validate.
    best_d = np.clip(best_d, 0.0, max_layer)
    if float(np.sum(best_d)) > max_total:
        scale = max_total / float(np.sum(best_d))
        best_d = best_d * scale
        best_d = np.clip(best_d, 0.0, max_layer)

    labels = ["L" if (j % 2 == 0) else "H" for j in range(len(best_d))]
    thicknesses = [float(x) for x in best_d]
    return labels, thicknesses
# EVOLVE-BLOCK-END
