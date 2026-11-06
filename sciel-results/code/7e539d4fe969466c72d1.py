import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap
    between the 4th and 5th bands using topological optimization driven by
    local field intensities of the 4th and 5th TM bands.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 108  # Leave 12s buffer
    
    # --- PWE Solver with eigenvectors (17x17 = 289 plane waves to match evaluator) ---
    
    # Precompute G vectors for 17x17 grid
    _G_CACHE = {}
    
    def get_G_vectors(n_pw):
        """
        Returns G vectors (as integer multiples of 2pi/a) sorted by magnitude.
        Caches results.
        """
        if n_pw in _G_CACHE:
            return _G_CACHE[n_pw]
        G_list = []
        half = n_pw // 2
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        G_list.sort(key=lambda x: (x[0]**2 + x[1]**2, x[0], x[1]))
        result = G_list[:n_pw * n_pw]
        _G_CACHE[n_pw] = result
        return result

    def fft_index(g, N):
        """Map integer G component to FFT array index."""
        return g % N

    def get_fourier_coeffs(cell):
        """
        Computes the Fourier coefficients of 1/epsilon(r).
        cell is a 2D numpy array of 0s and 1s.
        1 -> dielectric (eps_high), 0 -> air (eps_low).
        """
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs

    def solve_band_structure(cell, kx, ky, n_pw=17, return_eigvecs=False):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        Returns sorted omega^2 values, and optionally eigenvectors.
        Uses 17x17=289 plane waves to match the evaluator.
        """
        fft_coeffs = get_fourier_coeffs(cell)
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        # Precompute G vector arrays
        Gx_arr = np.array([g[0] for g in G_vectors], dtype=int)
        Gy_arr = np.array([g[1] for g in G_vectors], dtype=int)
        
        # Build matrix M where M[i,j] = D(G_i - G_j) using vectorized approach
        # For efficiency, compute differences and index into fft_coeffs
        Gx_diff = Gx_arr[:, None] - Gx_arr[None, :]
        Gy_diff = Gy_arr[:, None] - Gy_arr[None, :]
        idx_x = Gx_diff % size
        idx_y = Gy_diff % size
        M = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        Kx = kx + Gx_arr
        Ky = ky + Gy_arr
        K_sq = Kx**2 + Ky**2
        
        # A = diag(K_sq) - M
        A = np.diag(K_sq) - M
        
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
            idx_sort = np.argsort(eigvals)
            eigvals = eigvals[idx_sort]
            if return_eigvecs:
                eigvecs = eigvecs[:, idx_sort]
                return eigvals, eigvecs
            return eigvals
        except np.linalg.LinAlgError:
            if return_eigvecs:
                return np.zeros(n), np.zeros((n, n), dtype=complex)
            return np.zeros(n)

    def compute_field_intensities(cell, k_points, band_idx, n_pw=17):
        """
        Computes the local field intensity at each pixel for a given band
        across multiple k-points, averaged.
        Returns a 2D array of shape (size, size).
        """
        total_intensity = np.zeros((size, size))
        count = 0
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        # Precompute pixel positions (center of each pixel)
        px = (np.arange(size) + 0.5) / size
        ry, rx = np.meshgrid(px, px, indexing='ij')
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            # Get eigenvector for the specified band
            vec = eigvecs[:, band_idx]
            
            # Compute field at each pixel
            field = np.zeros((size, size), dtype=complex)
            for i in range(n):
                gx_i, gy_i = G_vectors[i]
                Kx = kx + gx_i
                Ky = ky + gy_i
                phase = 2 * np.pi * (Kx * rx + Ky * ry)
                field += vec[i] * np.exp(1j * phase)
            
            total_intensity += np.abs(field)**2
            count += 1
        
        if count > 0:
            total_intensity /= count
        return total_intensity

    def estimate_gap_ratio(cell, k_points, n_pw=17):
        """
        Estimates the gap-midgap ratio between band 4 and 5.
        Returns 2*(min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        """
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            eigvals = solve_band_structure(cell, kx, ky, n_pw)
            if len(eigvals) < 5:
                continue
            w4 = np.sqrt(max(0.0, eigvals[3]))
            w5 = np.sqrt(max(0.0, eigvals[4]))
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf') or (min_omega_5 + max_omega_4) < 1e-10:
            return -1.0
        ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return ratio

    # --- Symmetry handling ---
    
    def fill_symmetric(unique_mask):
        """
        Fills the full 32x32 cell from a 16x16 unique mask.
        The unique mask covers rows 0-15, cols 0-15.
        Symmetries: row flip (31-i), col flip (31-j), transpose (j,i).
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = unique_mask[i][j]
                coords = [
                    (i, j),
                    (size-1-i, j),
                    (i, size-1-j),
                    (size-1-i, size-1-j),
                    (j, i),
                    (size-1-j, i),
                    (j, size-1-i),
                    (size-1-j, size-1-i)
                ]
                for r, c in coords:
                    cell[r][c] = val
        return cell

    def get_unique_pixel_coords():
        """
        Returns a list of (i, j, [all symmetric coords]) for unique pixels in [0,15]x[0,15].
        """
        mapping = []
        for i in range(16):
            for j in range(16):
                coords = [
                    (i, j),
                    (size-1-i, j),
                    (i, size-1-j),
                    (size-1-i, size-1-j),
                    (j, i),
                    (size-1-j, i),
                    (j, size-1-i),
                    (size-1-j, size-1-i)
                ]
                unique_coords = list(set(coords))
                mapping.append((i, j, unique_coords))
        return mapping

    # --- Optimization ---
    
    # Define k-points for the surrogate - cover the irreducible BZ
    # Irreducible BZ: 0 <= kx <= ky <= 0.5 (for square lattice, first BZ boundary at 0.5)
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
        (0.25, 0.0),   # along Gamma-X
        (0.25, 0.25),  # along Gamma-M
        (0.5, 0.25),   # along X-M
        (0.1, 0.1),
        (0.3, 0.1),
        (0.4, 0.2),
        (0.2, 0.3),
        (0.15, 0.0),
        (0.35, 0.0),
        (0.0, 0.15),
        (0.0, 0.35),
        (0.45, 0.45),
        (0.4, 0.4),
        (0.1, 0.3),
        (0.3, 0.3),
        (0.2, 0.2),
        (0.4, 0.1),
        (0.1, 0.4),
    ]
    
    # Precompute pixel coordinate mapping
    pixel_mapping = get_unique_pixel_coords()
    
    # Initialize with a structured pattern - circular dielectric inclusion
    unique_mask = np.zeros((16, 16), dtype=int)
    center_i = 7.5
    center_j = 7.5
    radius = 5.5
    for i in range(16):
        for j in range(16):
            dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            if dist < radius:
                unique_mask[i][j] = 1  # Dielectric
            else:
                unique_mask[i][j] = 0  # Air
    
    cell = fill_symmetric(unique_mask)
    current_score = estimate_gap_ratio(cell, k_points, n_pw=17)
    best_score = current_score
    best_cell = cell.copy()
    
    # Topological optimization loop
    n_iter = 0
    patience = 0
    max_patience = 50
    
    while time.time() - start_time < time_limit:
        n_iter += 1
        
        # Compute field intensities for bands 4 and 5 (0-indexed: 3 and 4)
        intensity_4 = compute_field_intensities(cell, k_points, 3, n_pw=17)
        intensity_5 = compute_field_intensities(cell, k_points, 4, n_pw=17)
        
        # Compute pressure: where band 4 has high field and band 5 has low field,
        # adding dielectric should raise band 5 more. Where band 5 has high field and
        # band 4 has low field, removing dielectric should lower band 5 more.
        # pressure > 0: add dielectric (set to 1)
        # pressure < 0: remove dielectric (set to 0)
        pressure = intensity_4 - intensity_5
        
        # Compute average pressure for each unique pixel
        pixel_pressures = []
        for i, j, coords in pixel_mapping:
            avg_p = np.mean(pressure[coords])
            pixel_pressures.append((i, j, avg_p, coords))
        
        # Sort pixels by absolute pressure (descending)
        pixel_pressures.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Try the top N pixels with highest pressure
        found_improvement = False
        N_try = min(10, len(pixel_pressures))
        
        for idx in range(N_try):
            i, j, avg_p, coords = pixel_pressures[idx]
            
            if abs(avg_p) < 1e-8:
                continue
            
            # Decide direction based on pressure sign
            if avg_p > 0:
                new_val = 1
            else:
                new_val = 0
            
            # If pixel already has desired value, try opposite
            if unique_mask[i][j] == new_val:
                new_val = 1 - new_val
                if unique_mask[i][j] == new_val:
                    continue
            
            # Apply the change
            old_val = unique_mask[i][j]
            unique_mask[i][j] = new_val
            new_cell = fill_symmetric(unique_mask)
            new_score = estimate_gap_ratio(new_cell, k_points, n_pw=17)
            
            if new_score > current_score:
                current_score = new_score
                cell = new_cell
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell.copy()
                    patience = 0
                else:
                    patience += 1
                found_improvement = True
                break
            else:
                # Revert
                unique_mask[i][j] = old_val
        
        if not found_improvement:
            patience += 1
            # Random perturbation to escape local minima
            if patience > 10:
                n_random = min(5, patience // 10)
                for _ in range(n_random):
                    ri = random.randint(0, 15)
                    rj = random.randint(0, 15)
                    old_val = unique_mask[ri][rj]
                    unique_mask[ri][rj] = 1 - old_val
                    new_cell = fill_symmetric(unique_mask)
                    new_score = estimate_gap_ratio(new_cell, k_points, n_pw=17)
                    if new_score > current_score:
                        current_score = new_score
                        cell = new_cell
                        if new_score > best_score:
                            best_score = new_score
                            best_cell = new_cell.copy()
                            patience = 0
                    else:
                        unique_mask[ri][rj] = old_val
            
            # If stuck for too long, try a larger random restart from best
            if patience > max_patience:
                # Reset to best and try random flips
                unique_mask = np.zeros((16, 16), dtype=int)
                for i in range(16):
                    for j in range(16):
                        unique_mask[i][j] = best_cell[i][j]
                cell = fill_symmetric(unique_mask)
                current_score = best_score
                patience = 0
                
                # Do several random flips
                for _ in range(5):
                    ri = random.randint(0, 15)
                    rj = random.randint(0, 15)
                    old_val = unique_mask[ri][rj]
                    unique_mask[ri][rj] = 1 - old_val
                    new_cell = fill_symmetric(unique_mask)
                    new_score = estimate_gap_ratio(new_cell, k_points, n_pw=17)
                    if new_score > current_score:
                        current_score = new_score
                        cell = new_cell
                        if new_score > best_score:
                            best_score = new_score
                            best_cell = new_cell.copy()
                            patience = 0
                    else:
                        unique_mask[ri][rj] = old_val
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
    
    # Final check: ensure best_cell is valid and symmetric
    if best_cell is None:
        best_cell = cell
    
    # Verify symmetry
    result = best_cell.tolist()
    # Ensure it's a proper nested list of ints
    for i in range(size):
        for j in range(size):
            result[i][j] = int(result[i][j])
    
    return result
