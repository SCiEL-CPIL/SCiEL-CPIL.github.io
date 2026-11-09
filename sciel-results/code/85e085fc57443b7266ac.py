# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    The union volume is exactly separable: V = Vx * Vy, where Vx (resp. Vy) is the
    2-D area of the union of N sheared parallelograms in the (x, c) (resp. (y, c))
    plane. We compute that area exactly via Gauss-Legendre quadrature and minimize
    it using a combination of structured initializations, differential-evolution
    global search, and local refinement.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    # --- Initial Keich construction ---
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    x_base = keich.copy()
    y_base = keich.copy()

    # --- Additional structured initializations ---
    # Quadratic interpolation
    t = np.arange(N, dtype=np.float64) / N
    x_quad = t ** 2
    y_quad = t ** 2

    # Height-specific overlap: spread positions to maximize overlap
    x_overlap = np.linspace(0, 1, N)
    y_overlap = np.linspace(0, 1, N)

    # --- Exact 2-D area of union of sheared rectangles in (coord, c) plane ---
    # Use 64-point Gauss-Legendre quadrature for high accuracy
    c_vals, w_vals = np.polynomial.legendre.leggauss(64)
    c_vals = 0.5 * (c_vals + 1.0)  # map from [-1,1] to [0,1]
    w_vals = 0.5 * w_vals          # scale weights
    idx_arange = np.arange(N, dtype=np.float64)
    interval_widths = (idx_arange + 1.0) / N  # (i+1)/N for i in 0..N-1

    def area_2d(base):
        """
        Compute the area of the union of N sheared rectangles (parallelograms)
        in the (coord, c) plane. Rectangle i at height c occupies
        [base[i] + c*i/N, base[i] + (i+1)/N + c*i/N].
        We integrate the 1-D union length over c using Gauss-Legendre quadrature.
        """
        total_area = 0.0
        for c, w in zip(c_vals, w_vals):
            lefts = base + c * idx_arange / N
            rights = lefts + interval_widths

            # Merge intervals to compute union length
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

    # --- Objective: minimize Vx * Vy ---
    def objective(params):
        x_opt = params[:N]
        y_opt = params[N:]
        vx = area_2d(x_opt)
        vy = area_2d(y_opt)
        if vx <= 0 or vy <= 0:
            return 1e10
        return vx * vy

    # --- Evaluate all initializations and pick the best ---
    initial_candidates = [
        np.concatenate([x_base, y_base]),
        np.concatenate([x_quad, y_quad]),
        np.concatenate([x_overlap, y_overlap]),
    ]

    best_params = initial_candidates[0].copy()
    best_obj = objective(best_params)

    for candidate in initial_candidates[1:]:
        obj = objective(candidate)
        if obj < best_obj:
            best_obj = obj
            best_params = candidate.copy()

    # --- Phase 1: Differential Evolution global search ---
    if time.time() < deadline - 120.0:
        # Define bounds for each variable
        bounds = [(-0.5, 1.5)] * (2 * N)

        try:
            result_de = minimize(
                objective,
                best_params,
                method='differential_evolution',
                bounds=bounds,
                options={
                    'maxiter': 200,
                    'popsize': 20,
                    'tol': 1e-8,
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
            result_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': 0.05 * np.ones(2 * N)
                }
            )
            if result_nm.fun < best_obj:
                best_obj = result_nm.fun
                best_params = result_nm.x.copy()
        except Exception:
            pass

    # --- Phase 3: Coordinate-descent polish with adaptive step size ---
    if time.time() < deadline - 10.0:
        step = 1e-4
        no_improve = 0
        for iteration in range(300):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for idx in range(2 * N):
                old_val = best_params[idx]
                # Try several step sizes in both directions
                for mult in [1.0, 0.5, 0.25, 0.125]:
                    delta = step * mult
                    for sign in [1, -1]:
                        best_params[idx] = old_val + sign * delta
                        obj = objective(best_params)
                        if obj < best_obj - 1e-15:
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
                step *= 0.6
                if step < 1e-8 or no_improve > 30:
                    break
            else:
                no_improve = 0

    # --- Extract final base positions ---
    x_final = best_params[:N]
    y_final = best_params[N:]

    # --- Build final positions array ---
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = x_final[i_flat]
    base_y = y_final[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    # Validate: check that positions are finite
    if not np.all(np.isfinite(positions)):
        # Fallback to Keich construction
        base_x = x_base[i_flat]
        base_y = y_base[i_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
