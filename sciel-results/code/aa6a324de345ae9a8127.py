"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    start_time = time.time()
    deadline = 215.0  # safety margin before 230s kill

    def time_ok():
        return time.time() - start_time < deadline

    # ---- Plane wave setup: 17x17 = 289 plane waves ----
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289

    # ---- K-points in irreducible BZ (30 points) ----
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

    # Precompute difference indices for Fourier coupling (sparse)
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
    # Precompute the row-index array and the coefficient-index array (both 1D)
    vrow = valid_rows
    vcol = valid_cols
    vcoeff = diff_idx[vrow, vcol]
    n_valid = vrow.shape[0]

    # Fast screening subset of k-points (corners + midpoints + Gamma)
    k_subset = [0, 4, 9, 14, 19, 24, 29]
    k2_subset = k2_all[k_subset]

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

    def _bands_from_coeffs(coeffs):
        """Return (max_band4, min_band5) over all k-points."""
        max_band4 = -np.inf
        min_band5 = np.inf
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            # H[i,j] = (k_i+G_i)^2 * c_{G_i-G_j}
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[vrow, vcol] = k2[vrow] * coeffs[vcoeff]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            max_band4 = max(max_band4, eigvals[3])
            min_band5 = min(min_band5, eigvals[4])
        return max_band4, min_band5

    def compute_band_ratio_full(coeffs):
        max_band4, min_band5 = _bands_from_coeffs(coeffs)
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def compute_band_ratio_fast(coeffs):
        """Fast screening using a subset of k-points."""
        max_band4 = -np.inf
        min_band5 = np.inf
        for k_idx in range(len(k_subset)):
            k2 = k2_subset[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[vrow, vcol] = k2[vrow] * coeffs[vcoeff]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            max_band4 = max(max_band4, eigvals[3])
            min_band5 = min(min_band5, eigvals[4])
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def enforce_symmetry(cell):
        """Enforce full square (d4) symmetry about the cell centre."""
        cell = np.asarray(cell, dtype=int)
        rot180 = cell[::-1, ::-1]
        hflip = cell[:, ::-1]
        vflip = cell[::-1, :]
        result = (cell + rot180 + hflip + vflip)
        result = (result >= 2).astype(int)
        return result

    def make_cell_from_params(params):
        """Build a symmetric dielectric mask from Fourier-harmonic amplitudes.

        params = [a20, a02, a22, a10, a01, a30, a03, a33]
        The field is a superposition of low-order harmonics that couple to the
        4th/5th TM bands; binarized and symmetrized into a 0/1 pixel mask.
        """
        a20, a02, a22, a10, a01, a30, a03, a33 = params
        xs = (np.arange(size) + 0.5) / size - 0.5
        ys = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        C = (a20 * np.cos(2 * pi * X) +
             a02 * np.cos(2 * pi * Y) +
             a22 * np.cos(2 * pi * X) * np.cos(2 * pi * Y) +
             a10 * np.cos(pi * X) +
             a01 * np.cos(pi * Y) +
             a30 * np.cos(3 * pi * X) +
             a03 * np.cos(3 * pi * Y) +
             a33 * np.cos(3 * pi * X) * np.cos(3 * pi * Y))
        # Normalize to keep the mask in a sensible range
        Cmax = np.max(np.abs(C))
        if Cmax > 1e-12:
            C = C / Cmax
        mask = (C > 0.0).astype(int)
        return enforce_symmetry(mask)

    # ---- Parameter-space (Fourier-harmonic) optimization ----
    def optimize_fourier(params0, time_limit):
        """Hill-climbing over the low-order Fourier amplitudes."""
        p = np.array(params0, dtype=float)
        cur_cell = make_cell_from_params(p)
        cur_ratio = compute_band_ratio_full(get_fourier_coeffs(cur_cell))
        step = 0.12
        no_improve = 0
        max_no_improve = 60
        threshold = 1e-6
        t0 = time.time()
        while time.time() - t0 < time_limit and no_improve < max_no_improve and time_ok():
            improved = False
            dim = p.shape[0]
            for _ in range(dim):
                if time.time() - t0 > time_limit or not time_ok():
                    break
                d = np.random.randint(dim)
                old = p[d]
                p[d] = old + np.random.uniform(-step, step)
                new_cell = make_cell_from_params(p)
                coeffs = get_fourier_coeffs(new_cell)
                # screen first
                fast = compute_band_ratio_fast(coeffs)
                if fast > cur_ratio - 0.05:
                    new_ratio = compute_band_ratio_full(coeffs)
                else:
                    p[d] = old
                    continue
                if new_ratio > cur_ratio + threshold:
                    cur_ratio = new_ratio
                    cur_cell = new_cell
                    improved = True
                    no_improve = 0
                else:
                    p[d] = old
                    no_improve += 1
            if not improved:
                step *= 0.8
                no_improve += 1
        return cur_cell, cur_ratio

    # ---- Candidate generation (geometric) ----
    def generate_rod(radius):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_four_rods(radius, offset):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for sx in (1, -1):
                    for sy in (1, -1):
                        dx = x - sx * offset
                        dy = y - sy * offset
                        if dx * dx + dy * dy < radius * radius:
                            cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_ring(inner_r, outer_r):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x * x + y * y
                if inner_r * inner_r < r2 < outer_r * outer_r:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    candidates = []
    for r in [0.18, 0.20, 0.22, 0.25, 0.28]:
        candidates.append(generate_rod(r))
    for r in [0.08, 0.10, 0.12, 0.14]:
        for off in [0.22, 0.25, 0.28, 0.30]:
            candidates.append(generate_four_rods(r, off))
    for ir, orr in [(0.12, 0.22), (0.15, 0.25), (0.18, 0.28)]:
        candidates.append(generate_ring(ir, orr))

    # ---- Evaluate candidates (fast screen, then full) ----
    best_ratio = -np.inf
    best_cell = None
    scored = []
    for cell in candidates:
        if not time_ok():
            break
        coeffs = get_fourier_coeffs(cell)
        fast = compute_band_ratio_fast(coeffs)
        scored.append((fast, cell))
    scored.sort(key=lambda x: -x[0])
    for fast, cell in scored[:15]:
        if not time_ok():
            break
        full = compute_band_ratio_full(get_fourier_coeffs(cell))
        if full > best_ratio:
            best_ratio = full
            best_cell = cell.copy()

    # ---- Fourier-harmonic parameter optimization (multiple restarts) ----
    np.random.seed(20240517)
    fourier_starts = [
        [1.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.4, 0.4, 0.0, 0.0, 0.0],
        [0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.3, 0.3, 0.0],
        [0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5],
    ]
    per_restart = 25.0
    for s in fourier_starts:
        if not time_ok():
            break
        cell, ratio = optimize_fourier(s, per_restart)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()

    if best_cell is None:
        best_cell = generate_four_rods(0.10, 0.25)
        best_ratio = compute_band_ratio_full(get_fourier_coeffs(best_cell))

    # ---- Local pixel search (refinement machinery) ----
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    indep_pixels = [(i, j) for i in range(size // 2) for j in range(size // 2)]
    no_improve = 0
    max_no_improve = 60
    threshold = 1e-6
    search_start = time.time()
    search_time_limit = 60.0
    while time.time() - search_start < search_time_limit and no_improve < max_no_improve and time_ok():
        improved = False
        np.random.shuffle(indep_pixels)
        for (i, j) in indep_pixels:
            if not time_ok() or time.time() - search_start > search_time_limit:
                break
            old_val = current_cell[i, j]
            current_cell[i, j] = 1 - old_val
            current_cell = enforce_symmetry(current_cell)
            coeffs = get_fourier_coeffs(current_cell)
            fast_ratio = compute_band_ratio_fast(coeffs)
            if fast_ratio > current_ratio - 0.05:
                new_ratio = compute_band_ratio_full(coeffs)
            else:
                new_ratio = fast_ratio * 0.9
            if new_ratio > current_ratio + threshold:
                current_ratio = new_ratio
                improved = True
                no_improve = 0
            else:
                current_cell[i, j] = old_val
                current_cell = enforce_symmetry(current_cell)
                no_improve += 1
            if not time_ok() or time.time() - search_start > search_time_limit:
                break
        if not improved:
            no_improve += 1

    # ---- Final full evaluation ----
    final_ratio = compute_band_ratio_full(get_fourier_coeffs(current_cell))
    if final_ratio >= best_ratio:
        best_ratio = final_ratio
    else:
        current_cell = best_cell

    return current_cell.tolist()
# EVOLVE-BLOCK-END
