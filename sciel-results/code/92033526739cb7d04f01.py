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
    time_limit = 105  # Leave 15s buffer
    
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
        return fft_coeffs, inv_eps_map

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
        fft_coeffs, _ = get_fourier_coeffs(cell)
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        # Vectorized Matrix Construction
        # G_vectors is a list of tuples. Convert to array.
        G_arr = np.array(G_vectors) # shape (n, 2)
        
        # Compute G_i - G_j for all i, j
        # G_diff[i, j, 0] = G_i[0] - G_j[0]
        # G_diff[i, j, 1] = G_i[1] - G_j[1]
        G_diff_x = G_arr[:, 0:1] - G_arr[0:1, :] # shape (n, n)
        G_diff_y = G_arr[:, 1:2] - G_arr[1:2, :] # shape (n, n)
        
        # Map to FFT indices
        idx_x = G_diff_x
        idx_x[idx_x < 0] += size
        idx_y = G_diff_y
        idx_y[idx_y < 0] += size
        
        # M[i, j] = fft_coeffs[idx_x[i, j], idx_y[i, j]]
        M = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        Kx = kx + G_arr[:, 0]
        Ky = ky + G_arr[:, 1]
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
                return [0] * n, np.zeros((n, n), dtype=complex)
            return [0] * n

    def compute_field_intensities(cell, k_points, band_idx, n_pw=7):
        """
        Computes the local field intensity at each pixel for a given band
        across multiple k-points, averaged.
        Returns a 2D array of shape (size, size).
        """
        total_intensity = np.zeros((size, size))
        count = 0
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        G_arr = np.array(G_vectors)
        
        # Precompute pixel positions
        px, py = np.meshgrid(np.arange(size), np.arange(size))
        rx = px / size
        ry = py / size
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            vec = eigvecs[:, band_idx]
            
            # Compute phase matrix: shape (size, size, n)
            # Phase = 2 * pi * (Kx * rx + Ky * ry)
            Kx = kx + G_arr[:, 0]
            Ky = ky + G_arr[:, 1]
            
            # rx, ry are (size, size)
            # Kx, Ky are (n,)
            # We want phase[px, py, i] = 2*pi*(Kx[i]*rx[px,py] + Ky[i]*ry[px,py])
            # Reshape rx, ry to (size, size, 1) and Kx, Ky to (1, 1, n)
            rx_b = rx[:, :, np.newaxis]
            ry_b = ry[:, :, np.newaxis]
            Kx_b = Kx[np.newaxis, np.newaxis, :]
            Ky_b = Ky[np.newaxis, np.newaxis, :]
            
            phases = 2 * np.pi * (Kx_b * rx_b + Ky_b * ry_b)
            exp_phases = np.exp(1j * phases)
            
            # Field = sum_i vec[i] * exp_phases[..., i]
            # vec is (n,)
            field = np.tensordot(exp_phases, vec, axes=([2], [0]))
            
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

    # --- Symmetry handling ---
    
    def fill_symmetric(unique_mask):
        """
        Fills the full 32x32 cell from a 16x16 unique mask.
        """
        cell = np.zeros((size, size), dtype=int)
        # Vectorized fill
        # unique_mask is 16x16
        # We need to map (i,j) to 4 corners and transpose them.
        # Let's just do the loop, it's only 256 ops.
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
                unique_coords = list(set(coords))
                mapping[(i, j)] = unique_coords
        return mapping

    # --- Optimization ---
    
    # Define k-points for the surrogate
    # Use a denser grid in the irreducible BZ
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
        (0.35, 0.15),
        (0.15, 0.35),
        (0.45, 0.25),
        (0.25, 0.45),
        (0.15, 0.15),
        (0.35, 0.35),
    ]
    
    # Initialize with a random symmetric pattern
    unique_mask = np.random.randint(0, 2, (16, 16), dtype=int)
    
    cell = fill_symmetric(unique_mask)
    current_gap = estimate_gap(cell, k_points)
    best_gap = current_gap
    best_cell = cell.copy()
    
    pixel_mapping = get_unique_pixel_indices(cell)
    unique_pixels = list(pixel_mapping.keys())
    
    # Topological optimization loop
    for iteration in range(300):
        if time.time() - start_time > time_limit:
            break
        
        # Compute field intensities for bands 4 and 5
        intensity_4 = compute_field_intensities(cell, k_points, 3)
        intensity_5 = compute_field_intensities(cell, k_points, 4)
        
        pressure = intensity_4 - intensity_5
        
        # Find the pixel that gives the best improvement
        # We will test flipping a subset of pixels to save time
        # Sort pixels by absolute pressure to prioritize high-signal pixels
        pixel_pressures = []
        for (i, j), coords in pixel_mapping.items():
            avg_pressure = np.mean(pressure[coords])
            pixel_pressures.append((abs(avg_pressure), i, j, avg_pressure))
        
        pixel_pressures.sort(reverse=True)
        
        # Try top 10 candidates
        best_new_gap = current_gap
        best_new_mask = None
        best_new_cell = None
        
        for _, i, j, p in pixel_pressures[:10]:
            # Determine direction based on pressure
            # If p > 0 (I4 > I5), we want to increase contrast. 
            # If current is 0, set to 1. If 1, set to 0?
            # Actually, adding dielectric (0->1) increases epsilon.
            # High I4 means band 4 energy is there. Increasing epsilon there might push it up?
            # Or does it push band 5 up more?
            # Generally, for a gap between 4 and 5:
            # We want to raise 5 and lower 4.
            # If I5 is high at a pixel, adding dielectric (increasing eps) will raise the local energy, pushing band 5 up.
            # If I4 is high at a pixel, adding dielectric will raise band 4.
            # So we want to add dielectric where I5 is high and I4 is low.
            # i.e., where I5 - I4 is large.
            # Our pressure is I4 - I5.
            # So we want to add dielectric where pressure is negative (I5 > I4).
            # And remove dielectric where pressure is positive (I4 > I5).
            
            if p < 0:
                # I5 > I4: Add dielectric
                new_val = 1
            else:
                # I4 > I5: Remove dielectric
                new_val = 0
            
            if unique_mask[i][j] == new_val:
                # No change, or try the other way?
                # If we want to add but it's already 1, maybe we should remove?
                # No, that would be wrong. Just skip or try random.
                continue
                
            unique_mask[i][j] = new_val
            new_cell = fill_symmetric(unique_mask)
            new_gap = estimate_gap(new_cell, k_points)
            
            if new_gap > best_new_gap:
                best_new_gap = new_gap
                best_new_mask = unique_mask.copy()
                best_new_cell = new_cell
            else:
                # Revert
                unique_mask[i][j] = 1 - new_val
        
        if best_new_mask is not None:
            unique_mask = best_new_mask
            cell = best_new_cell
            current_gap = best_new_gap
            if current_gap > best_gap:
                best_gap = current_gap
                best_cell = cell.copy()
        else:
            # No improvement found in top candidates, try a random flip
            ri, rj = random.choice(unique_pixels)
            unique_mask[ri][rj] = 1 - unique_mask[ri][rj]
            new_cell = fill_symmetric(unique_mask)
            new_gap = estimate_gap(new_cell, k_points)
            if new_gap > current_gap:
                current_gap = new_gap
                cell = new_cell
                if current_gap > best_gap:
                    best_gap = current_gap
                    best_cell = cell.copy()
            else:
                unique_mask[ri][rj] = 1 - unique_mask[ri][rj]
                cell = fill_symmetric(unique_mask)

    if best_cell is None:
        best_cell = cell
    
    return best_cell.tolist()
