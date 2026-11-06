"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """Optimize 32x32 dielectric pattern for max TM gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 185.0  # Safety margin before 230s kill
    
    # Plane wave grid: 17x17 = 289 plane waves, G from -8 to 8
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289
    
    # Precompute k-points (30 points in irreducible BZ)
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
    
    # Precompute difference indices
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
        """Compute gap-midgap ratio for bands 4 and 5."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            valid = valid_mask
            rows, cols = np.where(valid)
            H[rows, cols] = k2[rows] * coeffs[diff_idx[rows, cols]]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0
    
    def make_symmetric(cell):
        """Enforce 4-fold rotational symmetry."""
        cell = np.array(cell, dtype=int)
        cell_rot = cell[::-1, ::-1].copy()
        cell_hflip = cell[:, ::-1].copy()
        cell_vflip = cell[::-1, :].copy()
        result = (cell + cell_rot + cell_hflip + cell_vflip)
        result = (result >= 2).astype(int)
        return result
    
    def initial_cell(radius):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1
        return make_symmetric(cell)
    
    # Evaluate initial guesses
    best_ratio = -np.inf
    best_cell = None
    
    for radius in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        if time.time() - start_time > time_limit * 0.2:
            break
        cell = initial_cell(radius)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
    
    if best_cell is None:
        best_cell = initial_cell(0.3)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
    
    # Local search
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    indep_pixels = [(i, j) for i in range(size // 2) for j in range(size // 2)]
    
    no_improve = 0
    max_no_improve = 30
    threshold = 1e-7
    
    while time.time() - start_time < time_limit * 0.85 and no_improve < max_no_improve:
        improved = False
        np.random.shuffle(indep_pixels)
        
        for (i, j) in indep_pixels:
            if time.time() - start_time > time_limit * 0.85:
                break
            
            old_val = current_cell[i, j]
            current_cell[i, j] = 1 - old_val
            current_cell = make_symmetric(current_cell)
            coeffs = get_fourier_coeffs(current_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + threshold:
                current_ratio = new_ratio
                improved = True
                no_improve = 0
            else:
                current_cell[i, j] = old_val
                current_cell = make_symmetric(current_cell)
                no_improve += 1
            
            if time.time() - start_time > time_limit * 0.85:
                break
        
        if not improved:
            no_improve += 1
    
    return current_cell.tolist()
# EVOLVE-BLOCK-END
