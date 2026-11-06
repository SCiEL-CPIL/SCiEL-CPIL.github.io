"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill

    def time_ok():
        return time.time() - t_start < DEADLINE

    # Plane wave setup: 17x17 = 289 plane waves
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289

    # K-points in irreducible BZ (30 points as specified)
    pi = np.pi
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)

    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2

    # Precompute difference indices for Fourier coupling
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            idx_x = int(dGx) + half
            idx_y = int(dGy) + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)

    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs

    def compute_band_ratio(coeffs):
        """Compute gap-midgap ratio."""
        max_band4 = -np.inf
        min_band5 = np.inf

        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            H_vals = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                if eigvals[3] > max_band4:
                    max_band4 = eigvals[3]
                if eigvals[4] < min_band5:
                    min_band5 = eigvals[4]

        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def enforce_symmetry(cell):
        """Enforce 4-fold rotational + reflection symmetry."""
        cell = np.array(cell, dtype=int)
        rot90 = np.rot90(cell, 1)
        rot180 = np.rot90(cell, 2)
        rot270 = np.rot90(cell, 3)
        hflip = cell[:, ::-1].copy()
        vflip = cell[::-1, :].copy()
        trans = cell.T.copy()
        avg = (cell + rot90 + rot180 + rot270 + hflip + vflip + trans + np.rot90(trans, 1))
        result = (avg >= 4).astype(int)
        return result

    def generate_four_rods(radius, offset, shape_exp=2.0):
        """
        Generate a cell with 4 superellipses at (+/-offset, +/-offset).
        Equation: |x/a|^n + |y/a|^n <= 1
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')

        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]

        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            val = np.abs(dx / radius) ** shape_exp + np.abs(dy / radius) ** shape_exp
            cell[val <= 1.0] = 1

        return enforce_symmetry(cell)

    def generate_grating_cell(fx, fy, px, py, thresh):
        """
        Generate a cell from the product/intersection of two orthogonal sinusoidal
        gratings. The X-grating has spatial frequency fx and phase px, the Y-grating
        has spatial frequency fy and phase py. A pixel is dielectric where
        sin(2*pi*fx*x + px) * sin(2*pi*fy*y + py) >= thresh.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        val = np.sin(2 * pi * fx * X + px) * np.sin(2 * pi * fy * Y + py)
        cell[val >= thresh] = 1
        return enforce_symmetry(cell)

    # Phase 1: Global Search over design families
    best_ratio = -np.inf
    best_cell = None
    best_params = None
    best_family = None

    def evaluate_cell(cell):
        nonlocal best_ratio, best_cell, best_params, best_family
        if cell is None:
            return
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        return ratio

    # Family A: Four rods / superellipses (primary, proven approach)
    # Include the proven (radius=0.15, offset=0.25) point from the inspiration.
    n_values = [1.5, 2.0, 2.5, 3.0, 4.0]
    a_values = [0.10, 0.12, 0.14, 0.15, 0.16, 0.18]
    off_values = [0.22, 0.24, 0.25, 0.26, 0.28]

    for n in n_values:
        for a in a_values:
            for off in off_values:
                if not time_ok():
                    break
                cell = generate_four_rods(a, off, n)
                ratio = evaluate_cell(cell)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = (n, a, off)
                    best_cell = cell.copy()
                    best_family = 'four_rods'
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break

    # Family B: Sinusoidal grating interference (new direction)
    if time_ok():
        fx_values = [1, 2, 3]
        fy_values = [1, 2, 3]
        px_values = [0.0, pi / 2, pi, 3 * pi / 2]
        py_values = [0.0, pi / 2, pi, 3 * pi / 2]
        thresh_values = [-0.3, -0.1, 0.0, 0.1, 0.3]
        for fx in fx_values:
            for fy in fy_values:
                for px in px_values:
                    for py in py_values:
                        for thresh in thresh_values:
                            if not time_ok():
                                break
                            cell = generate_grating_cell(fx, fy, px, py, thresh)
                            ratio = evaluate_cell(cell)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_params = (fx, fy, px, py, thresh)
                                best_cell = cell.copy()
                                best_family = 'grating'
                            if not time_ok():
                                break
                        if not time_ok():
                            break
                    if not time_ok():
                        break
                if not time_ok():
                    break
            if not time_ok():
                break

    if best_cell is None:
        best_cell = generate_four_rods(0.15, 0.25, 2.0)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
        best_params = (2.0, 0.15, 0.25)
        best_family = 'four_rods'

    # Phase 2: Local Optimization (multi-restart random search around the best)
    current_cell = best_cell.copy()
    current_ratio = best_ratio

    search_start = time.time()
    search_time_limit = 130.0  # Leave time for pixel refinement

    if best_family == 'four_rods':
        current_n, current_a, current_off = best_params
        no_improve_count = 0
        max_no_improve = 60

        # Multi-restart: occasionally jump to a new random point to escape local minima
        n_restarts = 0
        max_restarts = 6

        while time.time() - search_start < search_time_limit and time_ok():
            if no_improve_count >= max_no_improve:
                # Restart from a random point near the best
                if n_restarts >= max_restarts:
                    break
                current_n = np.random.uniform(1.5, 4.0)
                current_a = np.random.uniform(0.10, 0.20)
                current_off = np.random.uniform(0.22, 0.30)
                n_restarts += 1
                no_improve_count = 0

            new_n = current_n * (1.0 + np.random.normal(0, 0.1))
            new_a = current_a * (1.0 + np.random.normal(0, 0.08))
            new_off = current_off + np.random.normal(0, 0.015)

            new_n = np.clip(new_n, 1.0, 20.0)
            new_a = np.clip(new_a, 0.06, 0.22)
            new_off = np.clip(new_off, 0.18, 0.35)

            new_cell = generate_four_rods(new_a, new_off, new_n)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)

            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_n, current_a, current_off = new_n, new_a, new_off
                no_improve_count = 0
            else:
                no_improve_count += 1
    else:
        # Grating optimization
        current_fx, current_fy, current_px, current_py, current_thresh = best_params
        no_improve_count = 0
        max_no_improve = 40

        while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
            new_fx = int(np.clip(np.round(current_fx + np.random.choice([-1, 0, 1])), 1, 4))
            new_fy = int(np.clip(np.round(current_fy + np.random.choice([-1, 0, 1])), 1, 4))
            new_px = current_px + np.random.normal(0, 0.3)
            new_py = current_py + np.random.normal(0, 0.3)
            new_thresh = current_thresh + np.random.normal(0, 0.1)
            new_thresh = np.clip(new_thresh, -0.5, 0.5)

            new_cell = generate_grating_cell(new_fx, new_fy, new_px, new_py, new_thresh)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)

            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_fx, current_fy = new_fx, new_fy
                current_px, current_py = new_px, new_py
                current_thresh = new_thresh
                no_improve_count = 0
            else:
                no_improve_count += 1

    # Phase 3: Pixel Refinement (multiple passes)
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 40:
        indep_pixels = [(i, j) for i in range(16) for j in range(16)]

        refine_start = time.time()
        refine_limit = 35.0

        # Multiple passes over the independent pixels
        for pass_num in range(3):
            np.random.shuffle(indep_pixels)
            no_improve_refine = 0
            for (i, j) in indep_pixels:
                if time.time() - refine_start > refine_limit or not time_ok():
                    break

                old_val = current_cell[i, j]
                new_cell = current_cell.copy()
                new_cell[i, j] = 1 - old_val
                new_cell = enforce_symmetry(new_cell)

                # Skip if no actual change
                if np.array_equal(new_cell, current_cell):
                    continue

                coeffs = get_fourier_coeffs(new_cell)
                new_ratio = compute_band_ratio(coeffs)

                if new_ratio > current_ratio + 1e-8:
                    current_cell = new_cell
                    current_ratio = new_ratio
                    no_improve_refine = 0
                else:
                    no_improve_refine += 1
                    if no_improve_refine > 80:
                        break
            if time.time() - refine_start > refine_limit or not time_ok():
                break

    # Final verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)

    if final_ratio < best_ratio - 0.01:
        current_cell = best_cell
    else:
        best_ratio = final_ratio

    return current_cell.tolist()
# EVOLVE-BLOCK-END
