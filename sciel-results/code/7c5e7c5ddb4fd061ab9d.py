# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses a parametric representation with a single global shear parameter per axis
    plus per-index offsets, optimized via differential evolution and local polish.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 580.0

    # --- Parametric model: x[i] = global_shear * i/N + offset[i] ---
    # This reduces the search space from 2N to 2*(N+1) = 18 params
    # The shear term captures the dominant structure (Keich-like), while offsets
    # allow fine-tuning.

    # --- Exact 2-D area of union of sheared rectangles ---
    # Use Gauss-Legendre quadrature with 64 points
    c_vals, w_vals = np.polynomial.legendre.leggauss(64)
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals
    idx_arange = np.arange(N, dtype=np.float64)
    interval_widths = (idx_arange + 1.0) / N

    def area_2d(base):
        """Compute the area of the union of N sheared rectangles in (coord, c) plane."""
        total_area = 0.0
        for c, w in zip(c_vals, w_vals):
            lefts = base + c * idx_arange / N
            rights = lefts + interval_widths

            order = np.argsort(lefts)
            lefts_sorted = lefts[order]
            rights_sorted = rights[order]

            total_len = 0.0
            cur_l = lefts_sorted[0]
            cur_r = rights_sorted[0]
            for k in range(1, N):
                if lefts_sorted[k] <= cur_r:
                    if rights_sorted[k] > cur_r:
                        cur_r = rights_sorted[k]
                else:
                    total_len += cur_r - cur_l
                    cur_l = lefts_sorted[k]
                    cur_r = rights_sorted[k]
            total_len += cur_r - cur_l

            total_area += w * total_len

        return total_area

    def base_from_params(params):
        """Convert parametric representation to base positions.
        params[0] = global shear for x, params[1:N+1] = offsets for x
        params[N+1] = global shear for y, params[N+2:] = offsets for y
        """
        shear_x = params[0]
        offsets_x = params[1:N+1]
        shear_y = params[N+1]
        offsets_y = params[N+2:]

        x_base = shear_x * idx_arange / N + offsets_x
        y_base = shear_y * idx_arange / N + offsets_y
        return x_base, y_base

    def objective(params):
        x_opt, y_opt = base_from_params(params)
        vx = area_2d(x_opt)
        vy = area_2d(y_opt)
        if vx <= 0 or vy <= 0:
            return 1e10
        return vx * vy

    # --- Initial Keich construction in parametric form ---
    # Keich: x[i] = sum_{j=1}^{log2(N)} (1-j) * eps_j(i) * 2^{-j} / log2(N)
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    # Extract shear and offset from Keich
    # keich[i] ≈ shear * i/N + offset[i]
    # Fit a line to estimate shear
    if N > 1:
        shear_x_keich = np.polyfit(idx_arange / N, keich, 1)[0]
        offsets_x_keich = keich - shear_x_keich * idx_arange / N
    else:
        shear_x_keich = 0.0
        offsets_x_keich = keich.copy()

    # --- Build initial candidates in parametric form ---
    # Candidate 1: Keich-based
    init1 = np.concatenate([[shear_x_keich], offsets_x_keich, [shear_x_keich], offsets_x_keich])

    # Candidate 2: Zero shear, uniform offsets
    init2 = np.concatenate([[0.0], np.zeros(N), [0.0], np.zeros(N)])

    # Candidate 3: Small positive shear
    init3 = np.concatenate([[0.1], np.zeros(N), [0.1], np.zeros(N)])

    # Candidate 4: Negative shear
    init4 = np.concatenate([[-0.1], np.zeros(N), [-0.1], np.zeros(N)])

    # Evaluate and pick best
    best_params = init1.copy()
    best_obj = objective(best_params)

    for candidate in [init2, init3, init4]:
        obj = objective(candidate)
        if obj < best_obj:
            best_obj = obj
            best_params = candidate.copy()

    # --- Phase 1: Differential Evolution global search ---
    if time.time() < deadline - 120.0:
        # Bounds: shear in [-0.5, 0.5], offsets in [-0.5, 1.5]
        bounds = [(-0.5, 0.5)] + [(-0.5, 1.5)] * N + [(-0.5, 0.5)] + [(-0.5, 1.5)] * N

        try:
            from scipy.optimize import minimize
            result_de = minimize(
                objective,
                best_params,
                method='differential_evolution',
                bounds=bounds,
                options={
                    'maxiter': 300,
                    'popsize': 25,
                    'tol': 1e-9,
                    'mutate': (0.5, 1.0),
                    'recombine': 0.7,
                    'seed': 42,
                    'polish': False,
                    'init': 'latinhypercube'
                }
            )
            if result_de.fun < best_obj:
                best_obj = result_de.fun
                best_params = result_de.x.copy()
        except Exception:
            pass

    # --- Phase 2: Nelder-Mead local refinement ---
    if time.time() < deadline - 60.0:
        try:
            from scipy.optimize import minimize
            result_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True,
                    'initial_simplex': 0.01 * np.ones(2 * (N + 1))
                }
            )
            if result_nm.fun < best_obj:
                best_obj = result_nm.fun
                best_params = result_nm.x.copy()
        except Exception:
            pass

    # --- Phase 3: Coordinate-descent polish ---
    if time.time() < deadline - 10.0:
        step = 1e-5
        no_improve = 0
        for iteration in range(500):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for idx in range(len(best_params)):
                old_val = best_params[idx]
                for mult in [1.0, 0.5, 0.25, 0.125]:
                    delta = step * mult
                    for sign in [1, -1]:
                        best_params[idx] = old_val + sign * delta
                        obj = objective(best_params)
                        if obj < best_obj - 1e-16:
                            best_obj = obj
                            improved = True
                            break
                        best_params[idx] = old_val
                    if improved:
                        break
                if not improved:
                    best_params[idx] = old_val
            if not improved:
                no_improve += 1
                step *= 0.5
                if step < 1e-10 or no_improve > 40:
                    break
            else:
                no_improve = 0

    # --- Extract final base positions ---
    x_final, y_final = base_from_params(best_params)

    # --- Build final positions array ---
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = x_final[i_flat]
    base_y = y_final[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    # Validate
    if not np.all(np.isfinite(positions)):
        # Fallback to Keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
