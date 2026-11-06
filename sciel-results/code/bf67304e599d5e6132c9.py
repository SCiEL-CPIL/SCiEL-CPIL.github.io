"""Combined program: QWOT global-scale combinatorial search + fast vectorized
reflectance + per-layer local polish. EVOLVE-BLOCK wraps the whole body."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _mean_reflectance(materials, thicknesses, problem, wls):
    """Vectorized mean reflectance (percent) over precomputed wavelengths.

    materials: list of 'L'/'H' labels
    thicknesses: list of physical thicknesses (nm)
    wls: 1-D array of wavelengths (nm)
    """
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mat = problem["materials"]
    if len(materials) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return float(r * r * 100.0)

    n_mat = np.array([mat[m] for m in materials], dtype=float)
    d = np.array(thicknesses, dtype=float)
    nL = len(n_mat)
    inv_lam = 2.0 * np.pi / wls                       # (W,)
    phase = n_mat[:, None] * d[:, None] * inv_lam[None, :]   # (L, W)
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)
    inv_n = 1.0 / n_mat                                # (L,)

    # M00, M01, M10, M11 each (W,)
    M00 = np.ones_like(wls, dtype=complex)
    M01 = np.zeros_like(wls, dtype=complex)
    M10 = np.zeros_like(wls, dtype=complex)
    M11 = np.ones_like(wls, dtype=complex)

    for i in range(nL):
        c = cos_p[i]
        s = sin_p[i]
        a = 1j * s * inv_n[i]          # 1j*sin/n
        b = 1j * s * n_mat[i]          # 1j*n*sin
        nM00 = M00 * c + M01 * a
        nM01 = M00 * a + M01 * c
        nM10 = M10 * c + M11 * b
        nM11 = M10 * b + M11 * c
        M00, M01, M10, M11 = nM00, nM01, nM10, nM11

    B = M00 + M01 / n_sub
    C = M10 + M11 / n_sub
    Y = C / B
    r = (n_amb - Y) / (n_amb + Y)
    R = r.real ** 2 + r.imag ** 2
    return float(np.mean(R) * 100.0)


def _scale_valid(seq, scale, problem, lambda_0, mat):
    """Return (ok, thicknesses) for a QWOT-scaled sequence."""
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    total = 0.0
    th = []
    for label in seq:
        d = scale * lambda_0 / (4.0 * mat[label])
        if d < 0.0 or d > max_layer_nm:
            return False, None
        th.append(d)
        total += d
    if total > max_total_nm:
        return False, None
    return True, th


def _seq_scale_range(seq, problem, lambda_0, mat):
    """Valid global-scale range for a QWOT sequence."""
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    lo = 0.05
    hi = 4.0
    for label in seq:
        n = mat[label]
        hi = min(hi, max_layer_nm * 4.0 * n / lambda_0)
    sum_inv_n = sum(1.0 / mat[label] for label in seq)
    if sum_inv_n > 0:
        hi = min(hi, max_total_nm * 4.0 / (lambda_0 * sum_inv_n))
    hi = max(hi, lo)
    return lo, hi


def _optimize_scale(seq, problem, wls, lambda_0, mat, deadline):
    """Coarse->fine global-scale search for a fixed QWOT sequence."""
    lo, hi = _seq_scale_range(seq, problem, lambda_0, mat)
    if hi <= lo:
        return 1.0, 100.0
    best_s, best = 1.0, 100.0
    for n_grid in (24, 40, 60):
        if time.time() > deadline:
            break
        grid = np.linspace(lo, hi, n_grid)
        for s in grid:
            ok, th = _scale_valid(seq, s, problem, lambda_0, mat)
            if not ok:
                continue
            sc = _mean_reflectance(seq, th, problem, wls)
            if sc < best:
                best, best_s = sc, s
        span = (hi - lo) / n_grid
        lo = max(lo, best_s - span)
        hi = min(hi, best_s + span)
        if hi <= lo:
            break
    return best_s, best


def _alternating_seqs(n_layers):
    """Two alternating L/H sequences of given length (air->substrate)."""
    out = []
    for start in ('L', 'H'):
        seq = []
        for i in range(n_layers):
            seq.append(start if i % 2 == 0 else ('H' if start == 'L' else 'L'))
        out.append(seq)
    return out


def design(problem):
    """QWOT global-scale combinatorial search + per-layer local polish."""
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before the 150 s kill

    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat = problem["materials"]
    max_layers = problem["max_layers"]
    wls = np.linspace(band[0], band[1], 401)

    # Baseline: single quarter-wave L layer
    best_seq = ["L"]
    best_th = [lambda_0 / (4.0 * mat["L"])]
    best_score = _mean_reflectance(best_seq, best_th, problem, wls)

    # 1) Combinatorial QWOT search over alternating sequences + global scale.
    lengths = []
    for n in range(1, min(max_layers, 20) + 1):
        lengths.append(n)
    # Prioritize smaller counts first (cheaper and often near-optimal for broadband AR).
    for n in lengths:
        if time.time() > deadline - 20:
            break
        for seq in _alternating_seqs(n):
            if time.time() > deadline - 20:
                break
            s, sc = _optimize_scale(seq, problem, wls, lambda_0, mat, deadline)
            if sc < best_score:
                ok, th = _scale_valid(seq, s, problem, lambda_0, mat)
                if ok:
                    best_score = sc
                    best_seq = seq[:]
                    best_th = th[:]

    # 2) Per-layer local polish (Nelder-Mead) on the best design.
    if time.time() < deadline - 12 and len(best_seq) >= 1:
        try:
            from scipy.optimize import minimize
            max_layer_nm = problem["max_layer_nm"]
            max_total_nm = problem["max_total_nm"]

            def obj(x):
                if np.any(x < 0) or np.any(x > max_layer_nm) or np.sum(x) > max_total_nm:
                    return 100.0
                return _mean_reflectance(best_seq, list(x), problem, wls)

            x0 = np.array(best_th, dtype=float)
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-8})
            if res.fun < best_score:
                x_opt = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm and np.all(x_opt >= 0):
                    v = _mean_reflectance(best_seq, list(x_opt), problem, wls)
                    if v < best_score:
                        best_score = v
                        best_th = list(x_opt)
        except Exception:
            pass

    # Final safety clamp
    max_layer_nm = problem["max_layer_nm"]
    best_th = [float(min(max(max(t, 0.0), 0.0), max_layer_nm)) for t in best_th]
    if sum(best_th) > problem["max_total_nm"]:
        scale = problem["max_total_nm"] / sum(best_th)
        best_th = [t * scale for t in best_th]

    return best_seq, best_th
# EVOLVE-BLOCK-END
