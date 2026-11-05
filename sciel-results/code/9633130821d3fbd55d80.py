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
    time_limit = 100  # Leave 20s buffer
    
    # --- PWE Solver with eigenvectors ---
    
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
        Returns sorted omega^2 values, and optionally eigenvectors.
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

    def compute_field_intensities_fast(cell, k_points, band_idx, n_pw=7):
        """
        Computes the local field intensity at each pixel for a given band
        across multiple k-points, averaged. Uses vectorized computation.
        Returns a 2D array of shape (size, size).
        """
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        # Precompute pixel positions
        px_vals = np.arange(size) / size
        py_vals = np.arange(size) / size
        RX, RY = np.meshgrid(px_vals, px_vals, indexing='ij')
        
        total_intensity = np.zeros((size, size))
        count = 0
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            vec = eigvecs[:, band_idx]
            
            # Compute phases for all pixels at once
            # phase[i, px, py] = 2*pi*(Kx_i*RX + Ky_i*RY)
            Kx_arr = kx + np.array([g[0] for g in G_vectors])
            Ky_arr = ky + np.array([g[1] for g in G_vectors])
            
            # Build phase matrix: shape (n, size, size)
            phase = 2 * np.pi * (Kx_arr[:, None, None] * RX[None, :, :] + 
                                  Ky_arr[:, None, None] * RY[None, :, :])
            
            # Compute field: shape (n, size, size)
            field = np.sum(vec[:, None, None] * np.exp(1j * phase), axis=0)
            
            total_intensity += np.abs(field)**2
            count += 1
        
        if count > 0:
            total_intensity /= count
        return total_intensity

    def estimate_gap(cell, k_points, n_pw=7):
        """
        Estimates the gap between band 4 and 5.
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
            return 0.0
        gap = min_omega_5 - max_omega_4
        return gap

    def estimate_score(cell, k_points, n_pw=7):
        """
        Estimates the gap-midgap ratio score.
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
            return 0.0
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap < 1e-10:
            return 0.0
        gap = min_omega_5 - max_omega_4
        return 2.0 * gap / midgap

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

    def get_unique_pixel_indices(cell):
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
                # Remove duplicates
                unique_coords = list(set(coords))
                mapping[(i, j)] = unique_coords
        return mapping

    # --- Optimization ---
    
    # Define k-points for the surrogate - focus on high-symmetry points
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
    ]
    
    # Try multiple initializations and pick the best
    random.seed(42)
    np.random.seed(42)
    
    best_score = -float('inf')
    best_cell = None
    best_unique_mask = None
    
    # Candidate initializations
    candidates = []
    
    # 1. Circular dielectric island (current approach)
    um1 = np.zeros((16, 16), dtype=int)
    center = 7.5
    for i in range(16):
        for j in range(16):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < 5.5:
                um1[i][j] = 1
    candidates.append(um1.copy())
    
    # 2. Circular air hole in dielectric
    um2 = np.ones((16, 16), dtype=int)
    for i in range(16):
        for j in range(16):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < 5.5:
                um2[i][j] = 0
    candidates.append(um2.copy())
    
    # 3. Cross pattern
    um3 = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in range(16):
            if i >= 6 and i <= 9 and j >= 6 and j <= 9:
                um3[i][j] = 1
    candidates.append(um3.copy())
    
    # 4. Checkerboard-like
    um4 = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in range(16):
            if (i % 4 < 2) == (j % 4 < 2):
                um4[i][j] = 1
    candidates.append(um4.copy())
    
    # 5. Random
    um5 = np.random.randint(0, 2, (16, 16), dtype=int)
    candidates.append(um5.copy())
    
    # Evaluate all candidates
    for idx, um in enumerate(candidates):
        if time.time() - start_time > time_limit * 0.1:
            break
        cell = fill_symmetric(um)
        score = estimate_score(cell, k_points)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            best_unique_mask = um.copy()
    
    if best_cell is None:
        best_cell = fill_symmetric(candidates[0])
        best_unique_mask = candidates[0].copy()
    
    cell = best_cell
    unique_mask = best_unique_mask
    current_score = best_score
    
    pixel_mapping = get_unique_pixel_indices(cell)
    
    # Topological optimization loop with batch updates
    n_batch = 4  # Number of pixels to flip per iteration
    
    for iteration in range(150):
        if time.time() - start_time > time_limit:
            break
        
        # Compute field intensities for bands 4 and 5
        intensity_4 = compute_field_intensities_fast(cell, k_points, 3)
        intensity_5 = compute_field_intensities_fast(cell, k_points, 4)
        
        # Compute pressure: positive means add dielectric, negative means remove
        pressure = intensity_4 - intensity_5
        
        # Normalize pressure
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
        
        # Find top N pixels by absolute pressure
        pixel_pressures = []
        for (i, j), coords in pixel_mapping.items():
            avg_pressure = np.mean(pressure[coords])
            pixel_pressures.append((abs(avg_pressure), avg_pressure, (i, j)))
        
        pixel_pressures.sort(reverse=True)
        
        # Try flipping top N pixels
        improved = False
        for rank in range(min(n_batch, len(pixel_pressures))):
            if time.time() - start_time > time_limit:
                break
            
            _, avg_pressure, (i, j) = pixel_pressures[rank]
            
            if abs(avg_pressure) < 0.05:
                continue
            
            # Decide direction
            if avg_pressure > 0:
                new_val = 1
            else:
                new_val = 0
            
            # Only flip if different
            if unique_mask[i][j] == new_val:
                new_val = 1 - new_val
                if unique_mask[i][j] == new_val:
                    continue
            
            # Apply the change
            old_val = unique_mask[i][j]
            unique_mask[i][j] = new_val
            new_cell = fill_symmetric(unique_mask)
            new_score = estimate_score(new_cell, k_points)
            
            if new_score > current_score:
                current_score = new_score
                cell = new_cell
                improved = True
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell.copy()
                    best_unique_mask = unique_mask.copy()
            else:
                # Revert
                unique_mask[i][j] = old_val
                cell = fill_symmetric(unique_mask)
        
        # If no improvement in a batch, do a random perturbation
        if not improved:
            for _ in range(3):
                if time.time() - start_time > time_limit:
                    break
                ri = random.randint(0, 15)
                rj = random.randint(0, 15)
                old_val = unique_mask[ri][rj]
                unique_mask[ri][rj] = 1 - old_val
                new_cell = fill_symmetric(unique_mask)
                new_score = estimate_score(new_cell, k_points)
                if new_score > current_score:
                    current_score = new_score
                    cell = new_cell
                    if new_score > best_score:
                        best_score = new_score
                        best_cell = new_cell.copy()
                        best_unique_mask = unique_mask.copy()
                else:
                    unique_mask[ri][rj] = old_val
                    cell = fill_symmetric(unique_mask)
    
    # Final check: ensure best_cell is valid
    if best_cell is None:
        best_cell = cell
    
    # Verify symmetry
    result = best_cell.tolist()
    # Ensure exact symmetry by re-symmetrizing
    sym_cell = np.zeros((size, size), dtype=int)
    for i in range(16):
        for j in range(16):
            val = best_cell[i][j]
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
                sym_cell[r][c] = val
    return sym_cell.tolist()
