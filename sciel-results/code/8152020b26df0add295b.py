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
    time_limit = 110  # Leave 10s buffer for safety

    # --- PWE Solver ---
    
    # Precompute G vectors for n_pw=7 (49 plane waves)
    # The evaluator uses 289 plane waves (17x17), but for optimization surrogate, 7x7 or 9x9 is sufficient.
    # Let's use 9x9 (81 plane waves) for better accuracy in the surrogate.
    n_pw = 9
    half = n_pw // 2
    G_list = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_list.append((gx, gy))
    G_list.sort(key=lambda x: x[0]**2 + x[1]**2)
    G_vectors = np.array(G_list[:n_pw * n_pw])
    n_pw_actual = len(G_vectors)
    
    # Precompute G differences for matrix construction
    # M[i,j] depends on G_i - G_j
    G_diffs = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    Gx_diffs = G_diffs[:, :, 0]
    Gy_diffs = G_diffs[:, :, 1]
    
    # Map G components to FFT indices
    # FFT index for value g is g if g>=0 else size+g
    idx_x = Gx_diffs % size
    idx_y = Gy_diffs % size

    def get_fourier_coeffs(cell):
        """
        Computes the Fourier coefficients of 1/epsilon(r).
        """
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs

    def solve_band_structure(cell, kx, ky, return_eigvecs=False):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        """
        fft_coeffs = get_fourier_coeffs(cell)
        
        # Extract coefficients for all G_i - G_j pairs
        # M[i,j] = fft_coeffs[idx_x[i,j], idx_y[i,j]]
        M = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        Kx = kx + G_vectors[:, 0]
        Ky = ky + G_vectors[:, 1]
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
                return np.zeros(n_pw_actual), np.zeros((n_pw_actual, n_pw_actual), dtype=complex)
            return np.zeros(n_pw_actual)

    def compute_field_intensities(cell, k_points, band_indices, n_pw=9):
        """
        Computes the local field intensity at each pixel for given bands
        across multiple k-points, averaged.
        Returns a list of 2D arrays, one for each band.
        """
        # Positions of pixels
        rx = np.arange(size) / size
        ry = np.arange(size) / size
        RX, RY = np.meshgrid(rx, ry, indexing='ij')
        
        # Phases for all G vectors: exp(i * 2pi * (Kx*Rx + Ky*Ry))
        # We need to compute this for each k-point and each band.
        
        intensities = [np.zeros((size, size)) for _ in band_indices]
        count = 0
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) < max(band_indices) + 1:
                continue
            
            Kx = kx + G_vectors[:, 0]
            Ky = ky + G_vectors[:, 1]
            
            # Compute field at all pixels for all bands at once?
            # Field(r) = sum_i c_i * exp(i 2pi (Kx_i Rx + Ky_i Ry))
            # Let's compute the matrix of phases: shape (n_pw, size, size)
            # Phase[i, px, py] = 2pi * (Kx[i]*RX[px,py] + Ky[i]*RY[px,py])
            
            # To save memory and time, we can do it in chunks or just vectorize carefully.
            # n_pw is ~81. size is 32. 81*32*32 = 82944 complex numbers. Manageable.
            
            Phases = 2 * np.pi * (Kx[:, np.newaxis, np.newaxis] * RX[np.newaxis, :, :] + 
                                  Ky[:, np.newaxis, np.newaxis] * RY[np.newaxis, :, :])
            
            # exp(i * Phases)
            ExpPhases = np.exp(1j * Phases)
            
            for idx, band_idx in enumerate(band_indices):
                vec = eigvecs[:, band_idx]
                # Field = sum_i vec[i] * ExpPhases[i, :, :]
                # This is a matrix-vector multiplication: (n_pw, size, size) x (n_pw,) -> (size, size)
                Field = np.tensordot(ExpPhases, vec, axes=([0], [0]))
                intensity = np.abs(Field)**2
                intensities[idx] += intensity
                count += 1
        
        if count > 0:
            for i in range(len(intensities)):
                intensities[i] /= count
        
        return intensities

    def estimate_gap(cell, k_points, n_pw=9):
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
    
    # Define k-points for the surrogate
    # Include high symmetry points and some interior points
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
        (0.45, 0.45),
        (0.05, 0.05),
    ]
    
    # Initialize with a structured pattern (circular hole)
    unique_mask = np.zeros((16, 16), dtype=int)
    center = 7.5
    radius = 5
    for i in range(16):
        for j in range(16):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < radius:
                unique_mask[i][j] = 1  # Dielectric
            else:
                unique_mask[i][j] = 0  # Air
    
    cell = fill_symmetric(unique_mask)
    current_gap = estimate_gap(cell, k_points)
    best_gap = current_gap
    best_cell = cell.copy()
    
    pixel_mapping = get_unique_pixel_indices(cell)
    
    # Optimization loop
    # We will calculate the "pressure" for all unique pixels and update the best ones.
    for iteration in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Compute field intensities for bands 4 and 5
        # Band 4 is index 3, Band 5 is index 4
        intensities = compute_field_intensities(cell, k_points, [3, 4], n_pw=n_pw)
        intensity_4 = intensities[0]
        intensity_5 = intensities[1]
        
        # Pressure: 
        # We want to set pixel to 1 if intensity_4 > intensity_5
        # We want to set pixel to 0 if intensity_5 > intensity_4
        # So "score" for setting to 1 is intensity_4 - intensity_5.
        # If score > 0, prefer 1. If score < 0, prefer 0.
        
        pressure = intensity_4 - intensity_5
        
        # Normalize pressure
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
        
        # Evaluate candidate flips
        # We will try flipping the top N pixels with highest |pressure|
        # that are currently in the "wrong" state or have high potential for improvement.
        
        candidates = []
        for (i, j), coords in pixel_mapping.items():
            avg_pressure = np.mean(pressure[coords])
            current_val = unique_mask[i][j]
            
            # If pressure > 0, we prefer 1. If current is 0, flipping to 1 is good.
            # If pressure < 0, we prefer 0. If current is 1, flipping to 0 is good.
            
            # Score for flipping:
            # If we flip, the new value is 1-current_val.
            # The "gain" is roughly how much the pressure disagrees with the current value.
            
            if current_val == 0:
                # Currently 0. If pressure > 0, we should be 1. Gain = pressure.
                # If pressure < 0, we should be 0. Gain = -pressure (if we flipped to 1, it would be bad).
                # Actually, let's just score the flip action.
                # Flip 0->1: Good if pressure > 0. Score = max(0, pressure)
                # Flip 1->0: Good if pressure < 0. Score = max(0, -pressure)
                
                if avg_pressure > 0:
                    score = avg_pressure
                else:
                    score = 0.1 * abs(avg_pressure) # Small penalty for flipping against gradient? Or just 0?
                    # Let's encourage exploration by allowing flips even if against gradient, but with lower score.
                    score = abs(avg_pressure) * 0.5
            
            else: # current_val == 1
                if avg_pressure < 0:
                    score = -avg_pressure
                else:
                    score = abs(avg_pressure) * 0.5

            candidates.append((score, i, j, avg_pressure))
        
        # Sort candidates by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Try the top few candidates
        updated = False
        for score, i, j, avg_pressure in candidates[:10]:
            if time.time() - start_time > time_limit:
                break
                
            current_val = unique_mask[i][j]
            
            # Determine new value based on pressure
            if avg_pressure > 0:
                new_val = 1
            else:
                new_val = 0
                
            if new_val == current_val:
                # No change
                continue
            
            # Apply change
            unique_mask[i][j] = new_val
            new_cell = fill_symmetric(unique_mask)
            new_gap = estimate_gap(new_cell, k_points)
            
            # Accept if better
            if new_gap > current_gap:
                current_gap = new_gap
                cell = new_cell
                if new_gap > best_gap:
                    best_gap = new_gap
                    best_cell = new_cell.copy()
                updated = True
                break # Restart evaluation with new cell? Or continue?
                # Restarting is safer to avoid using stale pressure info.
            else:
                # Revert
                unique_mask[i][j] = current_val
                cell = fill_symmetric(unique_mask)
        
        if not updated:
            # If no single pixel flip improved the gap, try a random flip to escape local minima
            if iteration % 5 == 0:
                ri = random.randint(0, 15)
                rj = random.randint(0, 15)
                unique_mask[ri][rj] = 1 - unique_mask[ri][rj]
                new_cell = fill_symmetric(unique_mask)
                new_gap = estimate_gap(new_cell, k_points)
                if new_gap > current_gap:
                    current_gap = new_gap
                    cell = new_cell
                    if new_gap > best_gap:
                        best_gap = new_gap
                        best_cell = new_cell.copy()
                else:
                    unique_mask[ri][rj] = 1 - unique_mask[ri][rj]
                    cell = fill_symmetric(unique_mask)

    # Final check
    if best_cell is None:
        best_cell = cell
    
    return best_cell.tolist()
