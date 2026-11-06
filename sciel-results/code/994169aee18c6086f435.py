import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between the 4th and 5th bands using topological optimization driven by
    local field intensities of the 4th and 5th TM bands.
    Uses random binary mask initialization and smooth proxy objective.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 100  # Leave 20s buffer
    
    # --- PWE Solver ---
    
    def get_fourier_coeffs(cell):
        """
        Computes the Fourier coefficients of 1/epsilon(r).
        """
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs

    def get_G_vectors(n_pw):
        """
        Returns G vectors (as integer multiples of 2pi/a) sorted by magnitude.
        """
        G_list = []
        half = n_pw // 2
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        G_list.sort(key=lambda x: x[0]**2 + x[1]**2)
        return G_list[:n_pw * n_pw]

    def fft_index(g, N):
        """Map integer G component to FFT array index."""
        return g if g >= 0 else N + g

    def solve_band_structure(cell, kx, ky, n_pw=7, return_eigvecs=False):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        """
        fft_coeffs = get_fourier_coeffs(cell)
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        # Build matrix M where M[i,j] = D(G_i - G_j)
        M = np.zeros((n, n), dtype=complex)
        for i in range(n):
            gx_i, gy_i = G_vectors[i]
            for j in range(n):
                gx_j, gy_j = G_vectors[j]
                Gx_diff = gx_i - gx_j
                Gy_diff = gy_i - gy_j
                idx_x = fft_index(Gx_diff, size)
                idx_y = fft_index(Gy_diff, size)
                M[i, j] = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        K_sq = np.zeros(n)
        for i in range(n):
            gx_i, gy_i = G_vectors[i]
            Kx = kx + gx_i
            Ky = ky + gy_i
            K_sq[i] = Kx**2 + Ky**2
        
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
                return [0] * n, np.zeros((n, n), dtype=complex)
            return [0] * n

    def compute_field_intensities(cell, k_points, band_idx, n_pw=7):
        """
        Computes the local field intensity at each pixel for a given band
        across multiple k-points, averaged.
        """
        total_intensity = np.zeros((size, size))
        count = 0
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        # Precompute pixel positions
        px_range = np.arange(size)
        ry, rx = np.meshgrid(px_range / size, px_range / size, indexing='ij')
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            vec = eigvecs[:, band_idx]
            
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

    def compute_smooth_proxy(cell, k_points, n_pw=7):
        """
        Computes a smooth proxy for the band gap using field intensities.
        The proxy is: (avg_intensity_band4 - avg_intensity_band5) weighted by gap
        """
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            eigvals = solve_band_structure(cell, kx, ky, n_pw)
            if len(eigvals) < 5:
                continue
            w4 = np.sqrt(max(0, eigvals[3]))
            w5 = np.sqrt(max(0, eigvals[4]))
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf'):
            return 0.0, 0.0, 0.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2
        ratio = 2 * gap / (min_omega_5 + max_omega_4) if (min_omega_5 + max_omega_4) > 0 else 0.0
        
        return gap, midgap, ratio

    # --- Symmetry handling ---
    
    def fill_symmetric(unique_mask):
        """
        Fills the full 32x32 cell from a 16x16 unique mask.
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = unique_mask[i][j]
                coords = [
                    (i, j),
                    (31-i, j),
                    (i, 31-j),
                    (31-i, 31-j),
                    (j, i),
                    (31-j, i),
                    (j, 31-i),
                    (31-j, 31-i)
                ]
                for r, c in coords:
                    cell[r][c] = val
        return cell

    def get_symmetric_coords(i, j):
        """Returns all symmetric positions for pixel (i,j) in 16x16 quadrant."""
        coords = [
            (i, j),
            (31-i, j),
            (i, 31-j),
            (31-i, 31-j),
            (j, i),
            (31-j, i),
            (j, 31-i),
            (31-j, 31-i)
        ]
        return list(set(coords))

    # --- Optimization ---
    
    # Define k-points for the surrogate
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.1, 0.1),
        (0.4, 0.1),
        (0.1, 0.4),
        (0.3, 0.2),
        (0.2, 0.3),
        (0.15, 0.0),
        (0.35, 0.0),
        (0.0, 0.15),
        (0.0, 0.35),
        (0.45, 0.45),
        (0.45, 0.55),
        (0.55, 0.45),
        (0.55, 0.55),
    ]
    
    # Random initialization
    random.seed(42)
    unique_mask = np.random.randint(0, 2, (16, 16), dtype=int)
    
    cell = fill_symmetric(unique_mask)
    current_gap, current_midgap, current_ratio = compute_smooth_proxy(cell, k_points)
    best_gap = current_gap
    best_ratio = current_ratio
    best_cell = cell.copy()
    
    # Precompute symmetric coordinates for all unique pixels
    sym_coords = {}
    for i in range(16):
        for j in range(16):
            sym_coords[(i, j)] = get_symmetric_coords(i, j)
    
    # Topological optimization loop
    n_iter = 0
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() - start_time < time_limit:
        n_iter += 1
        
        if no_improve_count > max_no_improve:
            # Try a larger perturbation to escape local minimum
            n_flips = random.randint(3, 8)
            for _ in range(n_flips):
                ri = random.randint(0, 15)
                rj = random.randint(0, 15)
                unique_mask[ri][rj] = 1 - unique_mask[ri][rj]
            new_cell = fill_symmetric(unique_mask)
            new_gap, new_midgap, new_ratio = compute_smooth_proxy(new_cell, k_points)
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_cell = new_cell.copy()
                best_gap = new_gap
                current_gap = new_gap
                no_improve_count = 0
            else:
                # Revert
                unique_mask = np.random.randint(0, 2, (16, 16), dtype=int)
                cell = fill_symmetric(unique_mask)
                current_gap, current_midgap, current_ratio = compute_smooth_proxy(cell, k_points)
                no_improve_count = 0
            continue
        
        # Compute field intensities for bands 4 and 5
        intensity_4 = compute_field_intensities(cell, k_points, 3)
        intensity_5 = compute_field_intensities(cell, k_points, 4)
        
        # Compute pressure field
        pressure = intensity_4 - intensity_5
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
        
        # Find the pixel with highest absolute pressure
        best_pixel = None
        best_pressure = 0
        
        for (i, j), coords in sym_coords.items():
            avg_pressure = np.mean(pressure[coords])
            if abs(avg_pressure) > abs(best_pressure):
                best_pressure = avg_pressure
                best_pixel = (i, j)
        
        if best_pixel is None or abs(best_pressure) < 0.03:
            i = random.randint(0, 15)
            j = random.randint(0, 15)
            best_pixel = (i, j)
        
        i, j = best_pixel
        
        # Decide new value based on pressure sign
        if best_pressure > 0:
            new_val = 1
        else:
            new_val = 0
        
        if unique_mask[i][j] == new_val:
            new_val = 1 - new_val
            if unique_mask[i][j] == new_val:
                continue
        
        # Apply change
        unique_mask[i][j] = new_val
        new_cell = fill_symmetric(unique_mask)
        new_gap, new_midgap, new_ratio = compute_smooth_proxy(new_cell, k_points)
        
        # Accept if improved
        if new_ratio > current_ratio:
            current_ratio = new_ratio
            current_gap = new_gap
            cell = new_cell
            no_improve_count = 0
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_cell = new_cell.copy()
                best_gap = new_gap
        else:
            # Reject and revert
            unique_mask[i][j] = 1 - new_val
            cell = fill_symmetric(unique_mask)
            no_improve_count += 1
    
    # Final validation
    if best_cell is None:
        best_cell = cell
    
    return best_cell.tolist()
