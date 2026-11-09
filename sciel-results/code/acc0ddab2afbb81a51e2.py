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
    it using a combination of spectral (Fourier) global search and local refinement.
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

    # --- Exact 2-D area of union of sheared rectangles in (coord, c) plane ---
    # Use 80-point Gauss-Legendre quadrature for high accuracy
    c_vals, w_vals = np.polynomial.legendre.leggauss(80)
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

    # --- Spectral (Fourier) parametric deformation ---
    # Use more Fourier modes for better expressiveness
    n_modes = 3  # use 3 Fourier modes

    def parametric_deform(fourier_params, base):
        """
        Apply a Fourier-mode deformation to the base positions.
        fourier_params: array of shape (2*n_modes,) = [a1, b1, a2, b2, ..., an, bn]
        base: array of shape (N,)
        Returns deformed positions.
        """
        t = np.arange(N, dtype=np.float64) / N
        perturbation = np.zeros(N)
        for m in range(n_modes):
            a = fourier_params[2 * m]
            b = fourier_params[2 * m + 1]
            freq = (m + 1) * 2 * np.pi
            perturbation += a * np.sin(freq * t) + b * np.cos(freq * t)
        return base + perturbation

    # --- Objective with parametric deformation (for global coherence search) ---
    def objective_parametric(params):
        # First 2*n_modes: x Fourier params, next 2*n_modes: y Fourier params
        x_f = params[:2 * n_modes]
        y_f = params[2 * n_modes:4 * n_modes]
        x_opt = parametric_deform(x_f, x_base)
        y_opt = parametric_deform(y_f, y_base)
        vx = area_2d(x_opt)
        vy = area_2d(y_opt)
        if vx <= 0 or vy <= 0:
            return 1e10
        return vx * vy

    # --- Phase 1: Spectral global search with multiple restarts ---
    best_obj = np.inf
    best_params = None
    best_x_start = x_base.copy()
    best_y_start = y_base.copy()

    # Try multiple random initial points in Fourier space
    rng = np.random.default_rng(42)
    n_restarts = 15

    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break

        if restart == 0:
            initial_param = np.zeros(4 * n_modes)
        else:
            initial_param = rng.uniform(-0.1, 0.1, size=4 * n_modes)

        try:
            result_p = minimize(
                objective_parametric,
                initial_param,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': 0.05 * np.ones(4 * n_modes)
                }
            )
            if result_p.fun < best_obj:
                best_obj = result_p.fun
                best_params = result_p.x.copy()
                best_x_start = parametric_deform(result_p.x[:2 * n_modes], x_base)
                best_y_start = parametric_deform(result_p.x[2 * n_modes:4 * n_modes], y_base)
        except Exception:
            pass

    # Also try the Keich construction directly
    keich_obj = objective(np.concatenate([x_base, y_base]))
    if keich_obj < best_obj:
        best_obj = keich_obj
        best_x_start = x_base.copy()
        best_y_start = y_base.copy()

    # --- Phase 2: Full 16-variable Nelder-Mead optimization ---
    initial_params = np.concatenate([best_x_start, best_y_start])
    best_params_full = initial_params.copy()
    best_obj_full = objective(best_params_full)

    # Try multiple initial simplex sizes for robustness
    simplex_sizes = [0.05, 0.02, 0.1]
    for ss in simplex_sizes:
        if time.time() > deadline - 30.0:
            break
        try:
            result = minimize(
                objective,
                best_params_full,
                method='Nelder-Mead',
                options={
                    'maxiter': 15000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': ss * np.ones(2 * N)
                }
            )
            if result.fun < best_obj_full:
                best_obj_full = result.fun
                best_params_full = result.x.copy()
        except Exception:
            pass

    # --- Phase 3: Coordinate-descent polish with adaptive step size ---
    if time.time() < deadline - 5.0:
        step = 1e-4
        no_improve = 0
        for iteration in range(300):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for idx in range(2 * N):
                old_val = best_params_full[idx]
                # Try several step sizes in both directions
                for mult in [1.0, 0.5, 0.25, 0.125]:
                    delta = step * mult
                    for sign in [1, -1]:
                        best_params_full[idx] = old_val + sign * delta
                        obj = objective(best_params_full)
                        if obj < best_obj_full - 1e-15:
                            best_obj_full = obj
                            improved = True
                            break
                        best_params_full[idx] = old_val
                    if improved:
                        break
                if not improved:
                    best_params_full[idx] = old_val
            if not improved:
                no_improve += 1
                step *= 0.6
                if step < 1e-8 or no_improve > 30:
                    break
            else:
                no_improve = 0

    # --- Extract final base positions ---
    x_final = best_params_full[:N]
    y_final = best_params_full[N:]

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
        base_y = y_base[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
