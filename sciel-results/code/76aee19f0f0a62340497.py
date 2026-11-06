"""Broadband antireflection coating design.

Combines a fully vectorized characteristic-matrix reflectance evaluator with a
time-budgeted multi-start local search (coordinate descent on thicknesses,
layer insertion/deletion, material swaps) and analytic starting points
(single-layer index-matching and quarter-wave stacks).
"""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _reflectance_spectrum(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    """Vectorized reflectance (0..1) at each wavelength via char-matrix method."""
    W = len(lambdas)
    M11 = np.ones(W, dtype=complex)
    M12 = np.zeros(W, dtype=complex)
    M21 = np.zeros(W, dtype=complex)
    M22 = np.ones(W, dtype=complex)

    for j in range(len(n_vals)):
        n = n_vals[j]
        d = d_vals[j]
        beta = 2.0 * np.pi * n * d / lambdas
        c = np.cos(beta)
        s = np.sin(beta)
        L11 = c
        L12 = 1j * s / n
        L21 = 1j * n * s
        L22 = c

        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22

    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    return np.abs(r) ** 2


def _mean_reflectance(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    return 100.0 * np.mean(_reflectance_spectrum(n_vals, d_vals, lambdas,
                                                 n_ambient, n_substrate))


def _is_valid(d_vals, max_total_nm, max_layer_nm):
    if d_vals.size == 0:
        return True
    if d_vals.sum() > max_total_nm + 1e-6:
        return False
    if d_vals.min() < -1e-6 or d_vals.max() > max_layer_nm + 1e-6:
        return False
    return True


def _n_array(materials, n_L, n_H):
    return np.array([n_L if m == 'L' else n_H for m in materials], dtype=float)


def _local_search(materials, thicknesses, lambdas, n_ambient, n_substrate,
                  n_L, n_H, max_layers, max_total_nm, max_layer_nm, time_budget):
    """Local search: coordinate descent + layer insertion/deletion + swaps."""
    start = time.time()
    mats = list(materials)
    th = np.array(thicknesses, dtype=float)
    n_vals = _n_array(mats, n_L, n_H)
    score = _mean_reflectance(n_vals, th, lambdas, n_ambient, n_substrate)
    if not np.isfinite(score):
        score = float('inf')

    def elapsed():
        return time.time() - start

    def refine(thick, base_mats, base_score):
        """Coordinate descent on thicknesses with shrinking steps."""
        t = np.array(thick, dtype=float)
        sc = base_score
        n_arr = _n_array(base_mats, n_L, n_H)
        for step in [200.0, 100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125]:
            if elapsed() > time_budget:
                break
            improved = True
            while improved and elapsed() < time_budget:
                improved = False
                for i in range(len(t)):
                    for delta in (-step, step):
                        nt = t.copy()
                        nt[i] = min(max_layer_nm, max(0.0, nt[i] + delta))
                        if not _is_valid(nt, max_total_nm, max_layer_nm):
                            continue
                        s = _mean_reflectance(n_arr, nt, lambdas, n_ambient, n_substrate)
                        if s < sc - 1e-9:
                            t, sc = nt, s
                            improved = True
        return t, sc

    # Phase A: thickness refinement
    th, score = refine(th, mats, score)

    # Phase B: structural moves (insert / delete / swap)
    best_th, best_score = th.copy(), score
    for _ in range(200):
        if elapsed() > time_budget:
            break
        n_layers = len(mats)
        moves = []
        # insert a new layer at each position with each material
        if n_layers < max_layers:
            for pos in range(n_layers + 1):
                for mat in ('L', 'H'):
                    nm = mats[:pos] + [mat] + mats[pos:]
                    nt = np.concatenate([th[:pos], [150.0], th[pos:]])
                    moves.append(('insert', nm, nt))
        # delete a layer (keep at least 1)
        if n_layers > 1:
            for pos in range(n_layers):
                nm = mats[:pos] + mats[pos + 1:]
                nt = np.delete(th, pos)
                moves.append(('delete', nm, nt))
        # swap a layer's material
        for pos in range(n_layers):
            nm = mats[:]
            nm[pos] = 'H' if mats[pos] == 'L' else 'L'
            moves.append(('swap', nm, th.copy()))

        best_move = None
        for kind, nm, nt in moves:
            if elapsed() > time_budget:
                break
            if not _is_valid(nt, max_total_nm, max_layer_nm):
                continue
            nv = _n_array(nm, n_L, n_H)
            s = _mean_reflectance(nv, nt, lambdas, n_ambient, n_substrate)
            if s < best_score - 1e-9:
                best_move = (kind, nm, nt, s)
        if best_move is None:
            break
        _, mats, th, score = best_move
        # refine thicknesses after the structural change
        th, score = refine(th, mats, score)
        if score < best_score:
            best_th, best_score = th.copy(), score

    # Final refinement on the best structure found
    th, score = refine(th, mats, score)
    if score < best_score:
        best_th, best_score = th.copy(), score

    return mats, best_th.tolist(), best_score


def design(problem):
    """Design a broadband antireflection coating stack (air -> substrate order)."""
    start = time.time()
    deadline = start + 140.0  # safety margin under the 150 s wall-clock kill

    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    lambdas = np.linspace(band[0], band[1], 401)
    center_lam = 0.5 * (band[0] + band[1])

    def time_ok():
        return time.time() < deadline - 3.0

    # --- Build analytic starting points -------------------------------------
    starts = []
    # Single-layer index-matching thickness: n* = sqrt(n_amb * n_sub)
    for mat, n in (('L', n_L), ('H', n_H)):
        starts.append(([mat], [center_lam / (4.0 * n)]))
    # Quarter-wave stacks of various lengths, both polarities
    for n_layers in range(1, 9):
        for start_mat in ('L', 'H'):
            materials, thicknesses = [], []
            for i in range(n_layers):
                mat = start_mat if i % 2 == 0 else ('H' if start_mat == 'L' else 'L')
                n = n_L if mat == 'L' else n_H
                materials.append(mat)
                thicknesses.append(center_lam / (4.0 * n))
            starts.append((materials, thicknesses))

    # --- Evaluate starting points and pick the best -------------------------
    best_materials, best_thicknesses, best_score = None, None, float('inf')
    for mats, thicks in starts:
        if not time_ok():
            break
        nv = _n_array(mats, n_L, n_H)
        th = np.array(thicks, dtype=float)
        if not _is_valid(th, max_total_nm, max_layer_nm):
            continue
        s = _mean_reflectance(nv, th, lambdas, n_ambient, n_substrate)
        if s < best_score:
            best_score = s
            best_materials, best_thicknesses = mats[:], thicks[:]

    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]
        best_score = _mean_reflectance(_n_array(best_materials, n_L, n_H),
                                       np.array(best_thicknesses, dtype=float),
                                       lambdas, n_ambient, n_substrate)

    # --- Local search from the best starting point --------------------------
    remaining = max(0.0, (deadline - 3.0) - time.time())
    if remaining > 2.0:
        mats, thicks, score = _local_search(
            best_materials, best_thicknesses, lambdas, n_ambient, n_substrate,
            n_L, n_H, max_layers, max_total_nm, max_layer_nm, remaining)
        if np.isfinite(score) and score < best_score:
            best_materials, best_thicknesses, best_score = mats, thicks, score

    # --- Safety fallback: guarantee a valid, in-bounds result ---------------
    if best_materials is None or not _is_valid(np.array(best_thicknesses, dtype=float),
                                               max_total_nm, max_layer_nm):
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]

    # Clamp any stray values defensively
    best_thicknesses = [min(max_layer_nm, max(0.0, float(t))) for t in best_thicknesses]
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
