"""Broadband antireflection coating design: seeded global + local optimization."""
import time
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

    K = wavs.shape[0]
    A = np.ones(K)
    B = np.zeros(K)
    C = np.zeros(K)
    D = np.ones(K)
    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return float(100.0 * np.mean(R))

def design(problem):
    start_time = time.time()
    deadline = start_time + 132  # 150s kill, 18s safety margin

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]

    # Center quarter-wave period (full L+H quarter-wave pair reference)
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    T = wl_center / (4.0 * nL) + wl_center / (4.0 * nH)  # one full L+H quarter-wave period

    # Maier 7-layer symmetric design: (L H H L H H L)
    # Fractions of T: (0.09, 0.175, 0.21, 0.35, 0.21, 0.175, 0.09)
    maier_pattern = ("L", "H", "H", "L", "H", "H", "L")
    maier_frac = (0.09, 0.175, 0.21, 0.35, 0.21, 0.175, 0.09)

    def clamp_ths(ths):
        ths = np.clip(np.asarray(ths, dtype=float), 0.0, max_layer)
        tot = ths.sum()
        if tot > max_total:
            ths = ths * (max_total / tot)
        return ths

    def objective(ths):
        ths = clamp_ths(ths)
        return _mean_reflectance(maier_pattern, list(ths), problem)

    # Initial best: Maier design
    init_ths = np.array([f * T for f in maier_frac])
    init_ths = clamp_ths(init_ths)
    best_score = objective(init_ths)
    best_ths = list(init_ths)

    def update(ths, score):
        nonlocal best_score, best_ths
        if score < best_score:
            best_score = score
            best_ths = list(ths)

    # Global optimization with differential evolution
    try:
        from scipy.optimize import differential_evolution, minimize
        import warnings
        warnings.filterwarnings("ignore")

        n = len(maier_pattern)
        bounds = [(0.0, max_layer) for _ in range(n)]

        def de_obj(x):
            x = clamp_ths(x)
            # penalize if sum exceeds max_total (shouldn't after clamp, but safe)
            if x.sum() > max_total + 1e-6:
                return best_score + 10.0
            return _mean_reflectance(maier_pattern, list(x), problem)

        if time.time() < deadline - 20:
            de_res = differential_evolution(
                de_obj, bounds,
                seed=12345,
                maxiter=150,
                popsize=12,
                tol=1e-5,
                mutation=(0.3, 1.0),
                recombination=0.8,
                init="latinhypercube",
                polish=True,
                updating="immediate",
                disp=False,
            )
            if de_res.success or de_res.fun < best_score:
                cand = clamp_ths(de_res.x)
                s = de_obj(cand)
                update(cand, s)
    except Exception:
        pass

    # Local Nelder-Mead polish from the best found
    try:
        from scipy.optimize import minimize
        import warnings
        warnings.filterwarnings("ignore")

        if time.time() < deadline - 8:
            def lm_obj(x):
                x = clamp_ths(x)
                if x.sum() > max_total + 1e-6:
                    return best_score + 5.0
                return _mean_reflectance(maier_pattern, list(x), problem)

            res = minimize(
                lm_obj, np.asarray(best_ths),
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6, "adaptive": True},
            )
            if res.fun < best_score:
                cand = clamp_ths(res.x)
                s = lm_obj(cand)
                update(cand, s)
    except Exception:
        pass

    # Final safety: ensure constraints are met
    final_ths = clamp_ths(best_ths)
    # Re-verify sum
    if sum(final_ths) > max_total:
        final_ths = final_ths * (max_total / sum(final_ths))
    for i in range(len(final_ths)):
        final_ths[i] = max(0.0, min(max_layer, final_ths[i]))

    return list(maier_pattern), list(final_ths)
