"""
Photonic crystal unit cell design for large TM band gap between bands 4 and 5.
Strategy: Optimize parametric superellipse-based shapes to enhance specific Fourier components.
"""
import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = 220.0  # Safety margin before 230s kill
    
    def time_ok():
        return time.time() - start_time < deadline

    # Plane wave setup: 17x17 = 289 plane waves
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289
    
    # K-points in irreducible BZ (30 points)
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

    def compute_band_ratio_full(coeffs):
        """Compute gap-midgap ratio using all 30 k-points and 289 PWs."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
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

    def compute_band_ratio_fast(coeffs):
        """Fast approximate evaluation using subset of k-points for search guidance."""
        # Use fewer k-points for speed - focus on high-symmetry points
        k_subset = [0, 4, 9, 14, 19, 24, 29]  # 7 representative k-points
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_subset:
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
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

    def enforce_symmetry(cell):
        """Enforce square symmetry by averaging over the group."""
        cell = np.array(cell, dtype=int)
        cell_rot = cell[::-1, ::-1].copy()
        cell_hflip = cell[:, ::-1].copy()
        cell_vflip = cell[::-1, :].copy()
        cell_rot90 = cell.T[::-1, :].copy()
        cell_rot270 = cell.T[:, ::-1].copy()
        cell_diag = cell.T.copy()
        cell_anti_diag = cell[::-1, ::-1].T.copy()
        result = (cell + cell_rot + cell_hflip + cell_vflip + 
                  cell_rot90 + cell_rot270 + cell_diag + cell_anti_diag)
        result = (result >= 4).astype(int)
        return result

    def generate_superellipse(radius, exponent, n_lobs=1):
        """
        Generate a superellipse-based pattern.
        radius: scaling factor
        exponent: shape exponent (2=circle, >2=squarer, <2=sharper)
        n_lobs: number of lobes (1=central, 4=quadrants)
        """
        cell = np.zeros((size, size), dtype=float)
        # Use pixel centers
        x = (np.arange(size) + 0.5) / size - 0.5
        y = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x, y)
        
        if n_lobs == 1:
            # Single central superellipse
            # Equation: |x/r|^n + |y/r|^n = 1
            mask = (np.abs(X / radius) ** exponent + np.abs(Y / radius) ** exponent) <= 1.0
        elif n_lobs == 4:
            # Four superellipses at (±radius/2, ±radius/2) approx
            # Actually, let's place them at (±offset, ±offset)
            # For simplicity in parametric opt, let's stick to a single central shape
            # or a 4-fold symmetric multi-lobe defined by polar modulation.
            # Polar form: r(theta) = radius * (1 + 0.3 * cos(n_lobs * theta))
            # But superellipse is cartesian. Let's stick to n_lobs=1 for superellipse.
            # For n_lobs=4, we can use a "four-petal" shape:
            # |x|^n + |y|^n <= radius^n * (1 + 0.5*cos(4*theta))? No, keep it simple.
            # Let's implement a 4-lobe shape as the intersection of 4 superellipses shifted?
            # Better: Use a function that is naturally 4-fold symmetric.
            # r(theta) = radius * (1 + 0.4 * cos(4 * theta))
            # Convert to cartesian mask:
            R = np.sqrt(X**2 + Y**2)
            Theta = np.arctan2(Y, X)
            R_max = radius * (1 + 0.4 * np.cos(4 * Theta))
            # Superellipse aspect in polar is hard. Let's just use circular petals for n_lobs=4
            # or stick to the central superellipse which proved effective.
            # The knowledge says "tunable superellipses" improved score. So focus on n_lobs=1.
            mask = R <= R_max
        else:
            mask = np.zeros_like(X, dtype=bool)
            
        cell[mask] = 1.0
        return enforce_symmetry(cell.astype(int))

    def objective(params):
        """Objective function for optimization: negative gap ratio."""
        if not time_ok():
            return 0.0
            
        radius = params[0]
        exponent = params[1]
        
        # Constraints: radius in [0.1, 0.45], exponent in [1.0, 6.0]
        if radius < 0.1 or radius > 0.45:
            return 0.0
        if exponent < 1.0 or exponent > 6.0:
            return 0.0
            
        cell = generate_superellipse(radius, exponent)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio_fast(coeffs)
        return -ratio

    # Optimization Strategy
    # 1. Coarse grid search to find promising regions
    # 2. Local refinement using Nelder-Mead

    best_ratio = -np.inf
    best_params = [0.25, 2.0] # Default circle

    # Coarse Grid Search
    radii = np.linspace(0.15, 0.40, 10)
    exponents = np.linspace(1.2, 5.0, 10)
    
    for r in radii:
        if not time_ok(): break
        for n in exponents:
            if not time_ok(): break
            cell = generate_superellipse(r, n)
            coeffs = get_fourier_coeffs(cell)
            ratio = compute_band_ratio_fast(coeffs)
            if ratio > best_ratio:
                best_ratio = ratio
                best_params = [r, n]

    # Local Refinement
    if time_ok():
        try:
            res = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'xatol': 1e-3, 'fatol': 1e-4, 'maxiter': 200}
            )
            if res.success:
                new_params = res.x
                new_ratio = -res.fun
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_params = new_params
        except Exception:
            pass

    # Generate final cell from best parameters
    final_cell = generate_superellipse(best_params[0], best_params[1])
    
    # Optional: Small local pixel flip refinement around the best parametric shape
    # This can fine-tune the boundaries
    current_cell = final_cell.copy()
    current_ratio = compute_band_ratio_fast(get_fourier_coeffs(current_cell))
    
    # Independent pixels: top-left quadrant
    indep_pixels = [(i, j) for i in range(size // 2) for j in range(size // 2)]
    np.random.shuffle(indep_pixels)
    
    # Limit pixel flips to save time, only if we have plenty of time left
    time_left = deadline - (time.time() - start_time)
    if time_left > 20:
        flips_performed = 0
        max_flips = 50 # Cap flips to ensure we finish
        
        for (i, j) in indep_pixels:
            if flips_performed >= max_flips:
                break
            if not time_ok():
                break
                
            old_val = current_cell[i, j]
            current_cell[i, j] = 1 - old_val
            current_cell = enforce_symmetry(current_cell)
            coeffs = get_fourier_coeffs(current_cell)
            new_ratio = compute_band_ratio_fast(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_ratio = new_ratio
                flips_performed += 1
            else:
                current_cell[i, j] = old_val
                current_cell = enforce_symmetry(current_cell)
                flips_performed += 1

    # Final evaluation with full k-points
    if time_ok():
        coeffs = get_fourier_coeffs(current_cell)
        final_ratio = compute_band_ratio_full(coeffs)
        # We return the cell, the evaluator scores it.
        # The score returned here is not directly used by the evaluator for the final result,
        # but the cell is what matters.
        
    return current_cell.tolist()

# EVOLVE-BLOCK-END
