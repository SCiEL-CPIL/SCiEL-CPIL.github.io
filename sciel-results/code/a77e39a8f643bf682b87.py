"""Broadband antireflection coating design: characteristic-matrix + adaptive search."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance(labels, thicknesses, wls, nL, nH, ns):
    """Mean reflectance (fraction) of a stack at normal incidence over wavelengths wls.
    Returns a scalar mean over wls."""
    n = len(labels)
    # Build per-wavelength phase: delta_j = 2*pi*n_j*t_j / lam
    # n_j depends on label
    nmat = np.array([nL if lb == 'L' else nH for lb in labels], dtype=float)
    t = np.asarray(thicknesses, dtype=float)
    # phase matrix shape (n, len(wls))
    phase = (2.0 * math.pi / np.asarray(wls, dtype=float)) * np.outer(nmat, t)
    # For each wavelength, build the transfer matrix product.
    # We process layers from air side to substrate.
    # Each layer matrix: [[cos d, i sin d / n_j], [i n_j sin d, cos d]]
    # Substrate: [[1, 0], [0, 1]] (we use admittance form: Y = n_s)
    # We'll compute the characteristic matrix M = product of layer matrices, then
    # B = M11 + M12 * n_s, C = M21 + M22 * n_s, R = |B - n0*C|^2 / |B + n0*C|^2 with n0=1.
    nw = len(wls)
    # Initialize M as identity for each wavelength
    M = np.empty((2, 2, nw), dtype=complex)
    M[0, 0, :] = 1.0 + 0j
    M[0, 1, :] = 0.0 + 0j
    M[1, 0, :] = 0.0 + 0j
    M[1, 1, :] = 1.0 + 0j
    for j in range(n):
        c = np.cos(phase[j])
        s = np.sin(phase[j])
        nj = nmat[j]
        # newM = M @ layer_j
        m00 = M[0, 0, :]
        m01 = M[0, 1, :]
        m10 = M[1, 0, :]
        m11 = M[1, 1, :]
        L00 = c
        L01 = 1j * s / nj
        L10 = 1j * nj * s
        L11 = c
        n00 = m00 * L00 + m01 * L10
        n01 = m00 * L01 + m01 * L11
        n10 = m10 * L00 + m11 * L10
        n11 = m10 * L01 + m11 * L11
        M[0, 0, :] = n00
        M[0, 1, :] = n01
        M[1, 0, :] = n10
        M[1, 1, :] = n11
    B = M[0, 0, :] + M[0, 1, :] * ns
    C = M[1, 0, :] + M[1, 1, :] * ns
    n0 = 1.0
    num = (B - n0 * C)
    den = (B + n0 * C)
    R = (num * np.conj(num)) / (den * np.conj(den))
    R = np.real(R)
    R = np.clip(R, 0.0, 1.0)
    return float(np.mean(R))


def _seed_stack(n, nL, nH, band0, band1):
    """Alternating L/H starting with L, quarter-wave at band centre."""
    centre = 0.5 * (band0 + band1)
    labels = []
    thicks = []
    for i in range(n):
        lb = 'L' if i % 2 == 0 else 'H'
        nmat = nL if lb == 'L' else nH
        labels.append(lb)
        thicks.append(centre / (4.0 * nmat))
    return labels, thicks


def _random_stack(n, nL, nH, max_layer_nm, rng):
    labels = []
    thicks = []
    for i in range(n):
        lb = 'L' if rng.random() < 0.5 else 'H'
        labels.append(lb)
        thicks.append(rng.uniform(0.0, max_layer_nm))
    return labels, thicks


def _validate(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
    if len(labels) > max_layers:
        return False
    if len(labels) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0.0 or t > max_layer_nm + 1e-9:
            return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    for lb in labels:
        if lb not in ('L', 'H'):
            return False
    return True


def design(problem):
    t_start = time.time()
    DEADLINE = 150.0
    SAFETY = 2.0
    band0, band1 = problem["band_nm"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    ns = problem["substrate_index"]
    n0 = problem.get("ambient_index", 1.0)
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    # 401 equally spaced wavelengths in [band0, band1]
    wls = np.linspace(band0, band1, 401)

    rng = np.random.default_rng(12345)

    best_score = float('inf')
    best_labels = None
    best_thicks = None

    def eval_stack(labels, thicknesses):
        nonlocal best_score, best_labels, best_thicks
        if not _validate(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
            return float('inf')
        r = _reflectance(labels, thicknesses, wls, nL, nH, ns)
        # reflectance in percent for reporting; we minimize fraction here
        if r < best_score:
            best_score = r
            best_labels = list(labels)
            best_thicks = list(thicknesses)
        return r

    # Phase 1: coarse global search over layer counts
    # Allocate time roughly proportional to usefulness; focus on mid-high counts.
    # We'll iterate counts and for each do a short stochastic + local refinement.
    counts = list(range(1, max_layers + 1))
    # Weight higher counts more (they can do better), but keep low counts cheap.
    weights = []
    for c in counts:
        w = 1.0 + 0.05 * c
        weights.append(w)
    tot_w = sum(weights)

    # Reserve a fixed polish budget at the end
    POLISH_FRAC = 0.25
    search_budget = (DEADLINE - t_start) * (1.0 - POLISH_FRAC)
    polish_budget = (DEADLINE - t_start) * POLISH_FRAC

    # For each count, give it a time slice proportional to weight, but at least a minimum.
    # We'll process counts in an order that balances exploration: start mid, then expand.
    order = sorted(range(len(counts)), key=lambda i: (abs(counts[i] - 12), counts[i]))

    # Local refinement: coordinate descent + small random perturbations (a simple, robust method)
    def local_refine(labels, thicknesses, budget):
        nonlocal best_score, best_labels, best_thicks
        cur_labels = list(labels)
        cur_thicks = [float(t) for t in thicknesses]
        cur = eval_stack(cur_labels, cur_thicks)
        n = len(cur_labels)
        # Adaptive step sizes per layer
        steps = [max_layer_nm * 0.25] * n
        t_local = time.time()
        iters = 0
        no_improve = 0
        while time.time() - t_local < budget and no_improve < 40:
            iters += 1
            improved = False
            # Try perturbing each layer's thickness
            for j in range(n):
                for sign in (+1.0, -1.0):
                    cand = list(cur_thicks)
                    cand[j] = cur_thicks[j] + sign * steps[j]
                    cand[j] = max(0.0, min(max_layer_nm, cand[j]))
                    if sum(cand) > max_total_nm + 1e-9:
                        # scale down proportionally if over total
                        s = max_total_nm / sum(cand)
                        cand = [c * s for c in cand]
                    r = _reflectance(cur_labels, cand, wls, nL, nH, ns)
                    if r < cur - 1e-12:
                        cur_thicks = cand
                        cur = r
                        steps[j] = max(max_layer_nm * 0.01, steps[j] * 1.05)
                        improved = True
                        eval_stack(cur_labels, cur_thicks)
                        no_improve = 0
                        break
                if improved:
                    break
            # Try flipping a layer's material (structural move)
            if not improved:
                flip_done = False
                for j in range(n):
                    new_lb = 'H' if cur_labels[j] == 'L' else 'L'
                    cand_labels = list(cur_labels)
                    cand_labels[j] = new_lb
                    r = _reflectance(cand_labels, cur_thicks, wls, nL, nH, ns)
                    if r < cur - 1e-12:
                        cur_labels = cand_labels
                        cur = r
                        improved = True
                        flip_done = True
                        eval_stack(cur_labels, cur_thicks)
                        no_improve = 0
                        break
                if flip_done:
                    continue
            if not improved:
                # shrink steps
                steps = [max(max_layer_nm * 0.005, s * 0.7) for s in steps]
                no_improve += 1
        return cur_labels, cur_thicks, cur

    # Phase 1 loop
    for idx in order:
        if time.time() + SAFETY > DEADLINE - polish_budget:
            break
        c = counts[idx]
        w = weights[idx]
        # time slice for this count
        slice_time = search_budget * (w / tot_w)
        slice_time = max(0.5, min(slice_time, 12.0))
        t_c0 = time.time()

        # Initialize with a physical seed
        labels, thicks = _seed_stack(c, nL, nH, band0, band1)
        cur = eval_stack(labels, thicks)

        # A few random restarts within the slice, then local refine the best
        n_restarts = 1
        best_in_c = (list(labels), list(thicks), cur)
        while time.time() - t_c0 < slice_time * 0.4:
            n_restarts += 1
            labels_r, thicks_r = _random_stack(c, nL, nH, max_layer_nm, rng)
            r = eval_stack(labels_r, thicks_r)
            if r < best_in_c[2]:
                best_in_c = (labels_r, thicks_r, r)

        # Local refine the best found for this count
        refine_budget = max(0.2, slice_time - (time.time() - t_c0))
        lab_r, th_r, r_r = local_refine(best_in_c[0], best_in_c[1], refine_budget)
        eval_stack(lab_r, th_r)

    # Phase 2: polish pass on the global best
    if best_labels is not None and time.time() + SAFETY < DEADLINE:
        polish_budget = max(0.5, DEADLINE - SAFETY - time.time())
        lab_p, th_p, r_p = local_refine(best_labels, best_thicks, polish_budget)
        eval_stack(lab_p, th_p)

    # Final safety: ensure a valid result exists
    if best_labels is None:
        # fallback: single quarter-wave L layer
        centre = 0.5 * (band0 + band1)
        best_labels = ['L']
        best_thicks = [centre / (4.0 * nL)]

    # Clamp for safety
    best_thicks = [max(0.0, min(max_layer_nm, float(t))) for t in best_thicks]
    if sum(best_thicks) > max_total_nm:
        s = max_total_nm / sum(best_thicks)
        best_thicks = [t * s for t in best_thicks]

    return best_labels, best_thicks

# EVOLVE-BLOCK-END
