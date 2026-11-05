"""Broadband antireflection coating via fast local search + analytic seed."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance(thicknesses, materials, n_amb, n_sub, wavelengths):
    """Mean reflectance (%) via characteristic-matrix method, vectorized over wavelengths."""
    nL = materials["L"]
    nH = materials["H"]
    n = np.array([nL if m == "L" else nH for m in materials], dtype=float)
    d = np.asarray(thicknesses, dtype=float)
    w = np.asarray(wavelengths, dtype=float)
    k = 2.0 * np.pi / w  # (M,)
    # MxM matrix M_j = [[cos(d_j*k_j), i*sin(d_j*k_j)/(n_j)], [i*n_j*sin(d_j*k_j), cos(d_j*k_j)]]
    c = np.cos(np.outer(d, k))          # (N, M)
    s = np.sin(np.outer(d, k))          # (N, M)
    a = c
    b = 1j * s / n[:, None]
    cmat = 1j * n[:, None] * s
    dmat = c
    # Product over layers (air -> substrate)
    A = a[:, 0].copy()
    B = b[:, 0].copy()
    C = cmat[:, 0].copy()
    D = dmat[:, 0].copy()
    for j in range(1, len(n)):
        Aj, Bj, Cj, Dj = a[:, j], b[:, j], cmat[:, j], dmat[:, j]
        A2 = A * Aj + B * Cj
        B2 = A * Bj + B * Dj
        C2 = C * Aj + D * Cj
        D2 = C * Bj + D * Dj
        A, B, C, D = A2, B2, C2, D2
    # Substrate transfer: [[1,0],[0,1]] (n_sub absorbed in admittance)
    Y = (C * n_sub + D) / (A * n_sub + B)
    r = (n_amb - Y) / (n_amb + Y)
    R = (r * np.conj(r)).real
    R = np.clip(R, 0.0, 1.0)
    return 100.0 * float(np.mean(R))


def design(problem):
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mats = problem["materials"]
    nL, nH = mats["L"], mats["H"]
    lo, hi = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000.0)
    max_layer = problem.get("max_layer_nm", 400.0)
    wls = np.linspace(lo, hi, 401)
    centre = 0.5 * (lo + hi)

    deadline = time.time() + 50.0  # safety margin

    # ---- Analytic seed: 5-layer quarter-wave stack at band centre ----
    qL = centre / (4.0 * nL)
    qH = centre / (4.0 * nH)
    seed_labels = ["L", "H", "L", "H", "L"]
    seed_thick = [qL, qH, qL, qH, qL]
    best_labels = seed_labels[:]
    best_thick = seed_thick[:]
    best_val = _reflectance(seed_thick, mats, n_amb, n_sub, wls)

    def valid(th):
        if len(th) > max_layers:
            return False
        if any(t < -1e-6 or t > max_layer + 1e-6 for t in th):
            return False
        if sum(th) > max_total + 1e-6:
            return False
        return True

    def local_search(labels, th0, budget):
        """Adaptive random-restart local search on thicknesses. Returns (val, th)."""
        n = len(th0)
        t = np.array(th0, dtype=float).copy()
        cur = _reflectance(t.tolist(), mats, n_amb, n_sub, wls)
        best_t = t.copy()
        best_v = cur
        step = np.full(n, 15.0)
        rng = np.random.default_rng(12345 + n * 7)
        end = time.time() + budget
        while time.time() < end:
            # propose a random perturbation on a random subset
            for _ in range(200):
                if time.time() > end:
                    break
                idx = rng.integers(0, n, size=rng.integers(1, max(2, n // 2 + 1)))
                cand = t.copy()
                cand[idx] += rng.uniform(-step[idx], step[idx], size=len(idx))
                cand = np.clip(cand, 0.0, max_layer)
                if cand.sum() > max_total:
                    scale = max_total / cand.sum()
                    cand *= scale
                v = _reflectance(cand.tolist(), mats, n_amb, n_sub, wls)
                if v < cur:
                    t = cand
                    cur = v
                    if cur < best_v:
                        best_v = cur
                        best_t = t.copy()
                    step = np.clip(step * 1.1, 0.2, max_layer)
                else:
                    step = np.clip(step * 0.8, 0.2, max_layer)
        # final polish with tiny steps
        for _ in range(300):
            idx = rng.integers(0, n)
            cand = t.copy()
            cand[idx] += rng.uniform(-1.0, 1.0)
            cand = np.clip(cand, 0.0, max_layer)
            if cand.sum() > max_total:
                cand *= max_total / cand.sum()
            v = _reflectance(cand.tolist(), mats, n_amb, n_sub, wls)
            if v < cur:
                t = cand
                cur = v
                if cur < best_v:
                    best_v = cur
                    best_t = t.copy()
        return best_v, best_t.tolist()

    # ---- Try to improve by adding layers (greedy) ----
    remaining = max_layers - len(best_labels)
    while remaining > 0 and time.time() < deadline:
        improved = False
        # try inserting a layer of each material at each position
        for pos in range(len(best_labels) + 1):
            if time.time() > deadline:
                break
            for mat in ("L", "H"):
                new_labels = best_labels[:pos] + [mat] + best_labels[pos:]
                # seed the new layer near a quarter wave
                new_thick = best_thick[:pos] + [centre / (4.0 * (nL if mat == "L" else nH))] + best_thick[pos:]
                if not valid(new_thick):
                    continue
                v, th = local_search(new_labels, new_thick, budget=1.5)
                if v < best_val - 1e-9:
                    best_val = v
                    best_labels = new_labels
                    best_thick = th
                    remaining = max_layers - len(best_labels)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # ---- Final polish on the best stack ----
    if time.time() < deadline:
        v, th = local_search(best_labels, best_thick, budget=4.0)
        if v < best_val:
            best_val = v
            best_thick = th

    # clamp for safety
    best_thick = [max(0.0, min(max_layer, t)) for t in best_thick]
    if sum(best_thick) > max_total:
        scale = max_total / sum(best_thick)
        best_thick = [t * scale for t in best_thick]

    return best_labels, best_thick

# EVOLVE-BLOCK-END
