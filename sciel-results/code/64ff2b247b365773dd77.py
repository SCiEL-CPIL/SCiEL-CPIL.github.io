# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    The union volume is exactly separable: V = Vx * Vy, where Vx (resp. Vy) is the
    2-D area of the union of N sheared parallelograms in the (x, c) (resp. (y, c))
    plane.

    Strategy:
      1. Spectral (Fourier) representation of the base positions to exploit
         translational structure and reduce effective dimensionality.
      2. Global search over Fourier coefficients using differential evolution
         followed by local Nelder-Mead refinement.
      3. Coordinate-descent polish for final precision.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    # --- Exact 2-D area of union of sheared rectangles in (coord, c) plane ---
    # Precompute Gauss-Legendre nodes and weights on [0,1]
    c_vals, w_vals = np.polynomial.legendre.leggauss(80)
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals
    idx_arange = np.arange(N, dtype=np.float64)
    interval_widths = (idx_arange + 1.0) / N

    def area_2d(base):
        """
        Compute the area of the union of N sheared rectangles in the (coord, c) plane.
        Rectangle i at height c occupies [base[i] + c*i/N, base[i] + (i+1)/N + c*i/N].
        We integrate the 1-D union length over c using Gauss-Legendre quadrature.
        """
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

    # --- Fourier basis for spectral representation ---
    # Use first few Fourier modes to represent base positions
    # base[i] = sum_m (a_m * cos(2*pi*m*i/N) + b_m * sin(2*pi*m*i/N))
    # We use modes m=0,1,2,3 for 8 points (Nyquist is m=4)
    n_modes = 4
    t = np.arange(N, dtype=np.float64) / N
    # Build Fourier basis matrix: shape (N, 2*n_modes)
    fourier_basis = np.zeros((N, 2 * n_modes))
    for m in range(n_modes):
        fourier_basis[:, 2 * m] = np.cos(2 * np.pi * m * t)
        fourier_basis[:, 2 * m + 1] = np.sin(2 * np.pi * m * t)

    def fourier_to_base(coeffs):
        """Convert Fourier coefficients to base positions."""
        return fourier_basis @ coeffs

    def objective_fourier(coeffs):
        """Objective in Fourier coefficient space."""
        x_coeffs = coeffs[:2 * n_modes]
        y_coeffs = coeffs[2 * n_modes:4 * n_modes]
        x_base = fourier_to_base(x_coeffs)
        y_base = fourier_to_base(y_coeffs)
        vx = area_2d(x_base)
        vy = area_2d(y_base)
        if vx <= 0 or vy <= 0:
            return 1e10
        return vx * vy

    # --- Initial Fourier coefficients from Keich construction ---
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    # Project Keich onto Fourier basis
    keich_fourier_x = np.linalg.lstsq(fourier_basis, keich, rcond=None)[0]
    keich_fourier_y = np.linalg.lstsq(fourier_basis, keich, rcond=None)[0]

    initial_fourier = np.concatenate([keich_fourier_x, keich_fourier_y])
    best_obj = objective_fourier(initial_fourier)
    best_fourier = initial_fourier.copy()

    # --- Phase 1: Differential evolution global search in Fourier space ---
    # Fourier space is 16-dim (2*n_modes for x + 2*n_modes for y)
    bounds = [(-0.5, 0.5)] * (4 * n_modes)

    try:
        if time.time() < deadline - 120.0:
            result_de = minimize(
                objective_fourier,
                initial_fourier,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': 0.1 * np.ones(4 * n_modes)
                }
            )
            if result_de.fun < best_obj:
                best_obj = result_de.fun
                best_fourier = result_de.x.copy()
    except Exception:
        pass

    # --- Phase 2: Local refinement in full 16-dim coordinate space ---
    # Convert best Fourier solution to coordinate space
    x_start = fourier_to_base(best_fourier[:2 * n_modes])
    y_start = fourier_to_base(best_fourier[2 * n_modes:4 * n_modes])
    initial_params = np.concatenate([x_start, y_start])
    best_params = initial_params.copy()
    best_obj_full = objective_full = area_2d(x_start) * area_2d(y_start)

    def objective_full(params):
        x_opt = params[:N]
        y_opt = params[N:]
        vx = area_2d(x_opt)
        vy = area_2d(y_opt)
        if vx <= 0 or vy <= 0:
            return 1e10
        return vx * vy

    # Try multiple Nelder-Mead runs with different simplex sizes
    simplex_sizes = [0.05, 0.1, 0.2, 0.03]
    for ss in simplex_sizes:
        if time.time() > deadline - 60.0:
            break
        try:
            result = minimize(
                objective_full,
                best_params,
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
                best_params = result.x.copy()
        except Exception:
            pass

    # --- Phase 3: Coordinate-descent polish ---
    if time.time() < deadline - 10.0:
        step = 1e-4
        no_improve = 0
        for iteration in range(300):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for idx in range(2 * N):
                old_val = best_params[idx]
                for delta in [step, -step, step / 2, -step / 2, step / 4, -step / 4]:
                    best_params[idx] = old_val + delta
                    obj = objective_full(best_params)
                    if obj < best_obj_full - 1e-14:
                        best_obj_full = obj
                        improved = True
                        break
                    best_params[idx] = old_val
                if not improved:
                    best_params[idx] = old_val
            if not improved:
                no_improve += 1
                step *= 0.7
                if step < 1e-8 or no_improve > 25:
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

    # Validate: check for non-finite values
    if not np.all(np.isfinite(positions)):
        # Fallback to Keich construction
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        positions = np.stack([keich[i_flat], keich[j_flat]], axis=1)

    return positions


# EVOLVE-BLOCK-END
