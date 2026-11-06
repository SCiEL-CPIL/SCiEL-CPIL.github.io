import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between the 4th and 5th bands using topological optimization with random
    initialization and smooth proxy minimization based on local field intensities.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 105  # Leave 15s buffer
    
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
        Returns G vectors sorted by magnitude.
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
        
        K_sq = np.zeros(n)
        for i in range(n):
            gx_i, gy_i = G_vectors[i]
            Kx = kx + gx_i
            Ky = ky + gy_i
            K_sq[i] = Kx**2 + Ky**2
        
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

    def compute_smooth_proxy(cell, k_points, n_pw=7):
        """
        Computes a smooth proxy for band overlap using field intensities.
        Returns (proxy_value, intensity_4, intensity_5) where proxy is negative
        when there's a gap (good) and positive when bands overlap (bad).
        """
        total_proxy = 0.0
        total_intensity_4 = np.zeros((size, size))
        total_intensity_5 = np.zeros((size, size))
        count = 0
        
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        px_range = np.arange(size)
        ry, rx = np.meshgrid(px_range / size, px_range / size, indexing='ij')
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) < 5:
                continue
            
            w4_sq = eigvals[3]
            w5_sq = eigvals[4]
            
            # Smooth proxy: use tanh to smoothly penalize overlap
            # When w5_sq > w4_sq (gap exists), proxy is negative
            # When w5_sq < w4_sq (overlap), proxy is positive
            gap_metric = w5_sq - w4_sq
            # Scale for smoothness
            scale = max(0.1, (w4_sq + w5_sq) / 2)
            proxy_val = np.tanh(gap_metric / scale)
            total_proxy += proxy_val
            
            # Compute field intensities
            vec4 = eigvecs[:, 3]
            vec5 = eigvecs[:, 4]
            
            field4 = np.zeros((size, size), dtype=complex)
            field5 = np.zeros((size, size), dtype=complex)
            for i in range(n):
                gx_i, gy_i = G_vectors[i]
                Kx = kx + gx_i
                Ky = ky + gy_i
                phase = 2 * np.pi * (Kx * rx + Ky * ry)
                exp_phase = np.exp(1j * phase)
                field4 += vec4[i] * exp_phase
                field5 += vec5[i] * exp_phase
            
            total_intensity_4 += np.abs(field4)**2
            total_intensity_5 += np.abs(field5)**2
            count += 1
        
        if count > 0:
            total_proxy /= count
            total_intensity_4 /= count
            total_intensity_5 /= count
        
        return total_proxy, total_intensity_4, total_intensity_5

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

    def get_unique_pixel_indices():
        """
        Returns a mapping from unique pixel (i,j) in [0,15]x[0,15] 
        to all symmetric positions in the full cell.
        """
        mapping = {}
        for i in range(16):
            for j in range(16):
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
                unique_coords = list(set(coords))
                mapping[(i, j)] = unique_coords
        return mapping

    # --- Optimization ---
    
    # Define k-points for the surrogate
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
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
        (0.2, 0.4),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.1, 0.3),
        (0.3, 0.1),
    ]
    
    pixel_mapping = get_unique_pixel_indices()
    
    # Random initialization
    random.seed(42)
    unique_mask = np.random.randint(0, 2, (16, 16), dtype=int)
    cell = fill_symmetric(unique_mask)
    
    # Compute initial proxy
    current_proxy, _, _ = compute_smooth_proxy(cell, k_points)
    best_proxy = current_proxy
    best_cell = cell.copy()
    
    # Optimization loop with simulated annealing
    temperature = 1.0
    cooling_rate = 0.95
    
    for iteration in range(150):
        if time.time() - start_time > time_limit:
            break
        
        # Compute smooth proxy and field intensities
        current_proxy, intensity_4, intensity_5 = compute_smooth_proxy(cell, k_points)
        
        # Compute pressure: where to add/remove dielectric
        # Where band 4 has high intensity relative to band 5: add dielectric
        # Where band 5 has high intensity relative to band 4: remove dielectric
        pressure = intensity_4 - intensity_5
        
        # Normalize pressure
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
        
        # Find the pixel with highest absolute pressure
        best_pixel = None
        best_pressure = 0
        
        for (i, j), coords in pixel_mapping.items():
            avg_pressure = np.mean(pressure[coords])
            if abs(avg_pressure) > abs(best_pressure):
                best_pressure = avg_pressure
                best_pixel = (i, j)
        
        if best_pixel is None or abs(best_pressure) < 0.03:
            # Random flip
            i = random.randint(0, 15)
            j = random.randint(0, 15)
            best_pixel = (i, j)
            best_pressure = 0
        
        i, j = best_pixel
        
        # Decide direction based on pressure
        if best_pressure > 0:
            new_val = 1
        elif best_pressure < 0:
            new_val = 0
        else:
            new_val = 1 - unique_mask[i][j]
        
        # Only change if different
        if unique_mask[i][j] == new_val:
            new_val = 1 - new_val
            if unique_mask[i][j] == new_val:
                continue
        
        # Apply change
        old_val = unique_mask[i][j]
        unique_mask[i][j] = new_val
        new_cell = fill_symmetric(unique_mask)
        
        # Evaluate new proxy
        new_proxy, _, _ = compute_smooth_proxy(new_cell, k_points)
        
        # Simulated annealing acceptance
        delta = new_proxy - current_proxy
        if delta < 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
            current_proxy = new_proxy
            cell = new_cell
            if new_proxy < best_proxy:
                best_proxy = new_proxy
                best_cell = new_cell.copy()
        else:
            # Revert
            unique_mask[i][j] = old_val
            cell = fill_symmetric(unique_mask)
        
        # Cool down
        temperature *= cooling_rate
        
        # Occasionally do random flips to escape local minima
        if iteration % 15 == 14:
            ri = random.randint(0, 15)
            rj = random.randint(0, 15)
            old_rv = unique_mask[ri][rj]
            unique_mask[ri][rj] = 1 - old_rv
            new_cell = fill_symmetric(unique_mask)
            new_proxy, _, _ = compute_smooth_proxy(new_cell, k_points)
            if new_proxy < current_proxy:
                current_proxy = new_proxy
                cell = new_cell
                if new_proxy < best_proxy:
                    best_proxy = new_proxy
                    best_cell = new_cell.copy()
            else:
                unique_mask[ri][rj] = old_rv
                cell = fill_symmetric(unique_mask)
    
    # Final verification: compute actual gap ratio
    def compute_gap_ratio(cell, k_points, n_pw=7):
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
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return 0.0
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2
        if midgap == 0:
            return 0.0
        return 2 * gap / midgap
    
    # Try a few final local optimizations
    for _ in range(3):
        if time.time() - start_time > time_limit:
            break
        gap_ratio = compute_gap_ratio(best_cell, k_points)
        
        # Try flipping each unique pixel and see if it improves the gap
        improved = False
        for (i, j), coords in pixel_mapping.items():
            if time.time() - start_time > time_limit:
                break
            old_val = best_cell[i][j]
            new_val = 1 - old_val
            
            # Temporarily flip
            temp_cell = best_cell.copy()
            for r, c in coords:
                temp_cell[r][c] = new_val
            
            new_gap = compute_gap_ratio(temp_cell, k_points)
            if new_gap > gap_ratio:
                best_cell = temp_cell
                gap_ratio = new_gap
                improved = True
                break
        
        if not improved:
            break
    
    return best_cell.tolist()
