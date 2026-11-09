"""Broadband antireflection coating design: global pattern search + Nelder-Mead polish."""
import time
import random
import numpy as np


def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    # delta_j = 2*pi*n_j*d_j/lambda  -> (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def design(problem):
    start_time = time.time()
    deadline = start_time + 135  # 150s kill, 15s safety

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]

    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)

    def make_initial_thicknesses(pattern):
        ths = []
        for m in pattern:
            ths.append(qw[m])
        return ths

    def constrain_thicknesses(ths, pattern):
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        for i in range(len(ths)):
            ths[i] = max(0.0, min(max_layer, ths[i]))
        total2 = sum(ths)
        if total2 > max_total:
            scale = max_total / total2
            ths = [t * scale for t in ths]
        return ths

    def nelder_mead(materials_pattern, init_ths, problem, deadline, max_iters=400):
        """Nelder-Mead simplex local optimization."""
        ths = list(init_ths)
        ths = constrain_thicknesses(ths, materials_pattern)
        best_score = _mean_reflectance(materials_pattern, ths, problem)
        best_ths = list(ths)
        n = len(ths)

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        step = 15.0
        simplex = [list(ths)]
        for i in range(n):
            pt = list(ths)
            pt[i] = max(0.0, min(max_layer, pt[i] + step))
            simplex.append(pt)

        scores = [_mean_reflectance(materials_pattern, simplex[i], problem) for i in range(n + 1)]

        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]

            if scores[0] < best_score:
                best_score = scores[0]
                best_ths = list(simplex[0])

            centroid = np.mean(simplex[:-1], axis=0)

            xr = centroid + alpha * (centroid - simplex[-1])
            xr = constrain_thicknesses(list(xr), materials_pattern)
            fr = _mean_reflectance(materials_pattern, xr, problem)

            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = constrain_thicknesses(list(xe), materials_pattern)
                fe = _mean_reflectance(materials_pattern, xe, problem)
                if fe < fr:
                    simplex[-1] = xe
                    scores[-1] = fe
                else:
                    simplex[-1] = xr
                    scores[-1] = fr
            elif fr < scores[-2]:
                simplex[-1] = xr
                scores[-1] = fr
            else:
                if fr < scores[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = constrain_thicknesses(list(xc), materials_pattern)
                    fc = _mean_reflectance(materials_pattern, xc, problem)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = constrain_thicknesses(simplex[i], materials_pattern)
                        scores[i] = _mean_reflectance(materials_pattern, simplex[i], problem)
                    if np.max(scores) - np.min(scores) < 1e-7:
                        break

        return materials_pattern, best_ths, best_score

    def coarse_scan(materials_pattern, problem, deadline, n_restarts=5):
        """Do a few random restarts + short NM to find a good basin."""
        best_score = float('inf')
        best_ths = None
        for _ in range(n_restarts):
            if time.time() >= deadline:
                break
            # Random thicknesses in [0.5*qw, 1.5*qw]
            ths = []
            for m in materials_pattern:
                base = qw[m]
                ths.append(random.uniform(0.3 * base, 1.8 * base))
            ths = constrain_thicknesses(ths, materials_pattern)
            score = _mean_reflectance(materials_pattern, ths, problem)
            if score < best_score:
                best_score = score
                best_ths = list(ths)
        if best_ths is None:
            best_ths = constrain_thicknesses(make_initial_thicknesses(materials_pattern), materials_pattern)
            best_score = _mean_reflectance(materials_pattern, best_ths, problem)
        # Polish with NM
        _, polished_ths, polished_score = nelder_mead(materials_pattern, best_ths, problem, deadline, max_iters=250)
        return polished_score, polished_ths

    # Generate candidate patterns
    candidates = []

    # 1-layer
    candidates.append(("L",))
    candidates.append(("H",))

    # 2-layer
    for a in ("L", "H"):
        for b in ("L", "H"):
            if a != b:
                candidates.append((a, b))

    # 3-layer
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                candidates.append((a, b, c))

    # 4-layer symmetric: A B B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            if a != b:
                candidates.append((a, b, b, a))

    # 5-layer symmetric: A B C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                candidates.append((a, b, c, b, a))

    # 6-layer symmetric: A B C C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                if a != c:
                    candidates.append((a, b, c, c, b, a))

    # 7-layer symmetric: A B C D C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                for d in ("L", "H"):
                    if a != d:
                        candidates.append((a, b, c, d, c, b, a))

    # 8-layer symmetric: A B C D D C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                for d in ("L", "H"):
                    if a != d:
                        candidates.append((a, b, c, d, d, c, b, a))

    # Random longer patterns
    random.seed(42)
    for _ in range(40):
        length = random.randint(4, min(max_layers, 14))
        pattern = tuple(random.choice("LH") for _ in range(length))
        candidates.append(pattern)

    # Remove duplicates
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    candidates = unique_candidates

    # Sort by length ascending (shorter patterns are faster to evaluate)
    candidates.sort(key=lambda p: len(p))

    best_result = None
    best_score = float('inf')

    for pattern in candidates:
        if time.time() >= deadline:
            break

        # Coarse scan to find good basin
        score, ths = coarse_scan(pattern, problem, deadline, n_restarts=3)
        if score < best_score:
            best_score = score
            best_result = (pattern, ths)

    if best_result is None:
        # Fallback: single L layer
        return ["L"], [qw["L"]]

    mats, ths = best_result
    # Final polish with more iterations
    _, final_ths, final_score = nelder_mead(mats, ths, problem, deadline, max_iters=300)
    return list(mats), list(final_ths)
