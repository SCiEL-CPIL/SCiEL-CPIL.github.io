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
    time_limit = 110  # Leave 10s buffer for final checks/output
    
    # --- PWE Solver ---
    
    # Pre-compute G vectors for a specific number of plane waves
    # 7x7 = 49 PWs is a good balance. 5x5 = 25 is faster.
    # Let's use 5x5 for the surrogate to allow more iterations, 
    # but verify with 7x7 occasionally? 
    # Actually, 49 is small enough. Let's stick to 49 for better accuracy in surrogate.
    N_PW = 7 
    half = N_PW // 2
    G_list = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_list.append((gx, gy))
    G_list.sort(key=lambda x: x[0]**2 + x[1]**2)
    G_vectors = np.array(G_list[:N_PW*N_PW]) # Shape (N, 2)
    N_G = len(G_vectors)
    
    # Pre-compute pixel positions
    # Pixels are centers of the grid. 
    # For 32x32, pixel (i,j) is at (i+0.5)/32, (j+0.5)/32
    x_coords = (np.arange(size) + 0.5) / size
    y_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    R_flat = np.column_stack((X.flatten(), Y.flatten())) # Shape (1024, 2)
    
    def get_fourier_coeffs(cell):
        """
        Computes the Fourier coefficients of 1/epsilon(r).
        cell is a 2D numpy array of 0s and 1s.
        """
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs

    def fft_index(g, N):
        """Map integer G component to FFT array index."""
        return g if g >= 0 else N + g

    def solve_band_structure(cell, kx, ky, return_eigvecs=False):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        Returns sorted omega^2 values, and optionally eigenvectors.
        """
        fft_coeffs = get_fourier_coeffs(cell)
        
        # Build matrix M where M[i,j] = D(G_i - G_j)
        # G_vectors is shape (N, 2)
        # We need D(G_i - G_j) for all i, j
        # G_diff[i, j, 0] = G_vectors[i, 0] - G_vectors[j, 0]
        G_diff_x = G_vectors[:, 0][:, np.newaxis] - G_vectors[:, 0][np.newaxis, :]
        G_diff_y = G_vectors[:, 1][:, np.newaxis] - G_vectors[:, 1][np.newaxis, :]
        
        # Map to FFT indices
        idx_x = np.where(G_diff_x >= 0, G_diff_x, size + G_diff_x)
        idx_y = np.where(G_diff_y >= 0, G_diff_y, size + G_diff_y)
        
        M = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        # K = k + G
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
                return [0] * N_G, np.zeros((N_G, N_G), dtype=complex)
            return [0] * N_G

    def compute_field_intensities(cell, k_points, band_idx):
        """
        Computes the local field intensity at each pixel for a given band
        across multiple k-points, averaged.
        Returns a 2D array of shape (size, size).
        """
        total_intensity = np.zeros((size, size))
        count = 0
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            # Get eigenvector for the specified band
            vec = eigvecs[:, band_idx]
            
            # Compute field at all pixels
            # E(r) = sum_i vec_i * exp(i (k+G_i) . r)
            # K = k + G_vectors
            Kx = kx + G_vectors[:, 0]
            Ky = ky + G_vectors[:, 1]
            
            # Phase matrix: Shape (N_pixels, N_G)
            # Phase[p, i] = 2 * pi * (Kx[i] * R_flat[p, 0] + Ky[i] * R_flat[p, 1])
            # We can compute this as:
            # R_flat @ K_vec? No.
            # Dot product: R_flat (N_pix, 2) dot K (N_G, 2).T -> (N_pix, N_G)
            K_col = np.column_stack((Kx, Ky)) # Shape (N_G, 2)
            dot_products = R_flat @ K_col.T # Shape (N_pix, N_G)
            phases = 2 * np.pi * dot_products
            
            # Field = exp(i * phases) @ vec
            # exp(i * phases) is (N_pix, N_G)
            # vec is (N_G,)
            # Result is (N_pix,)
            fields = np.exp(1j * phases) @ vec
            
            total_intensity += np.abs(fields)**2
            count += 1
        
        if count > 0:
            total_intensity /= count
        return total_intensity.reshape((size, size))

    def estimate_gap(cell, k_points):
        """
        Estimates the gap between band 4 and 5.
        """
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            eigvals = solve_band_structure(cell, kx, ky)
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
        # Vectorized fill
        # unique_mask is 16x16
        # Symmetries:
        # 1. (i, j)
        # 2. (31-i, j)
        # 3. (i, 31-j)
        # 4. (31-i, 31-j)
        # 5. (j, i)
        # 6. (31-j, i)
        # 7. (j, 31-i)
        # 8. (31-j, 31-i)
        
        # We can just loop 16x16 = 256 times, it's fast enough.
        for i in range(16):
            for j in range(16):
                val = unique_mask[i, j]
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
                    cell[r, c] = val
        return cell

    def get_unique_pixel_indices():
        """
        Returns a list of (i, j) for the 16x16 unique block.
        """
        return [(i, j) for i in range(16) for j in range(16)]

    # --- Optimization ---
    
    # Define k-points for the surrogate
    # Cover Gamma, X, M, and mid-points
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
        (0.25, 0.0),   # Mid-GX
        (0.25, 0.25),  # Mid-GM
        (0.0, 0.25),   # Mid-GY
        (0.5, 0.25),   # Mid-XY
        (0.25, 0.5),   # Mid-XM
        (0.1, 0.1),
        (0.4, 0.1),
        (0.1, 0.4),
        (0.3, 0.2),
        (0.2, 0.3),
    ]
    
    # Initialize with a structured pattern (circular hole) or random
    unique_mask = np.zeros((16, 16), dtype=int)
    center = 7.5
    radius = 5
    for i in range(16):
        for j in range(16):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < radius:
                unique_mask[i, j] = 1  # Dielectric
            else:
                unique_mask[i, j] = 0  # Air
    
    cell = fill_symmetric(unique_mask)
    current_gap = estimate_gap(cell, k_points)
    best_gap = current_gap
    best_cell = cell.copy()
    
    unique_indices = get_unique_pixel_indices()
    
    # Topological optimization loop
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Compute field intensities for bands 4 and 5
        # Band 4 is index 3, Band 5 is index 4
        intensity_4 = compute_field_intensities(cell, k_points, 3)
        intensity_5 = compute_field_intensities(cell, k_points, 4)
        
        # Pressure: 
        # We want to lower omega_4 (add dielectric where E4 is high)
        # We want to raise omega_5 (remove dielectric where E5 is high)
        # So, if we set pixel to 1 (Dielectric):
        #   Gain from lowering w4 ~ E4
        #   Loss from lowering w5 ~ E5 (since we want w5 to go UP, lowering it is bad)
        #   Net Score for Setting to 1: E4 - E5
        # If Score > 0, Setting to 1 is better than Setting to 0 (which would be -E4 + E5)
        # Diff = (E4 - E5) - (-E4 + E5) = 2(E4 - E5)
        # So we prefer 1 if E4 > E5.
        
        pressure = intensity_4 - intensity_5
        
        # Find the unique pixel with the highest absolute pressure
        # We need to map the full cell pressure back to the unique mask
        # Since the cell is symmetric, the pressure is symmetric.
        # We just pick the unique pixel (i,j) in 16x16 with max |pressure[i,j]|
        
        # pressure is 32x32. unique_mask is 16x16.
        # pressure[i,j] corresponds to unique_mask[i,j] for i,j in 0..15
        # We just look at the top-left 16x16 quadrant of pressure.
        p_unique = pressure[:16, :16]
        
        # Flatten to find max abs
        flat_p = p_unique.flatten()
        idx_max = np.argmax(np.abs(flat_p))
        i, j = np.unravel_index(idx_max, (16, 16))
        
        # Get the pressure value
        p_val = flat_p[idx_max]
        
        # Determine desired state
        if p_val > 0:
            desired_state = 1
        else:
            desired_state = 0
        
        # If the pixel is already in the desired state, we might want to flip it anyway 
        # or skip. If we skip, we might get stuck. 
        # Let's flip the pixel with the second highest pressure if the top one is already correct?
        # Or just flip the top one to the *opposite* state if it's already correct? 
        # No, that would be bad. 
        # Strategy: If the best candidate is already in the desired state, 
        # pick the next best candidate (by abs pressure) that is NOT in its desired state.
        
        # Create a list of candidates sorted by abs pressure
        candidates = np.argsort(np.abs(flat_p))[::-1]
        
        updated = False
        for idx in candidates:
            ci, cj = np.unravel_index(idx, (16, 16))
            cp_val = flat_p[idx]
            if cp_val > 0:
                c_desired = 1
            else:
                c_desired = 0
            
            if unique_mask[ci, cj] != c_desired:
                # This is a promising move
                i, j = ci, cj
                new_val = c_desired
                updated = True
                break
        
        if not updated:
            # All pixels are in their "locally optimal" state based on current fields
            # Try a random flip to escape local minimum
            i = random.randint(0, 15)
            j = random.randint(0, 15)
            new_val = 1 - unique_mask[i, j]
        
        # Apply change
        unique_mask[i, j] = new_val
        new_cell = fill_symmetric(unique_mask)
        new_gap = estimate_gap(new_cell, k_points)
        
        # Accept if better
        if new_gap > current_gap:
            current_gap = new_gap
            cell = new_cell
            if new_gap > best_gap:
                best_gap = new_gap
                best_cell = new_cell.copy()
        else:
            # Revert
            unique_mask[i, j] = 1 - new_val
            cell = fill_symmetric(unique_mask)
            
        # Every 50 iterations, do a random restart or mutation?
        # Maybe just ensure we don't timeout.
        
    # Final check: ensure best_cell is valid
    if best_cell is None:
        best_cell = cell
    
    return best_cell.tolist()
