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
        Returns G vectors (as integer multiples of 2pi/a) for an n_pw x n_pw grid.
        For odd n_pw, this gives a symmetric set of G vectors.
        """
        G_list = []
        half = n_pw // 2
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        return G_list

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
        Computes local field intensity at each pixel for a given band
        across multiple k-points, using FFT-based evaluation.
        """
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        total_intensity = np.zeros((size, size))
        count = 0
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            vec = eigvecs[:, band_idx]
            
            # Build field on a grid using direct summation but vectorized
            # r = (px/size, py/size) for px, py in range(size)
            pxs = np.arange(size) / size
            pys = np.arange(size) / size
            PX, PY = np.meshgrid(pxs, pys, indexing='ij')
            
            field = np.zeros((size, size), dtype=complex)
            for i in range(n):
                gx_i, gy_i = G_vectors[i]
                Kx = kx + gx_i
                Ky = ky + gy_i
                phase = 2 * np.pi * (Kx * PX + Ky * PY)
                field += vec[i] * np.exp(1j * phase)
            
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
    
    # Define k-points for the surrogate - cover the irreducible BZ triangle
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= 0.5
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
        (0.25, 0.0),   # X/2
        (0.25, 0.25),  # M/2
        (0.125, 0.0),
        (0.375, 0.0),
        (0.125, 0.125),
        (0.375, 0.125),
        (0.125, 0.25),
        (0.375, 0.25),
        (0.25, 0.125),
        (0.25, 0.375),
        (0.4, 0.4),
        (0.1, 0.1),
    ]
    
    # Initialize with a structured pattern - cross/diamond shape for better gap
    unique_mask = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in range(16):
            # Diamond/cross pattern centered at (7.5, 7.5)
            dist = abs(i - 7.5) + abs(j - 7.5)
            if dist < 6.0:
                unique_mask[i][j] = 1  # Dielectric
            else:
                unique_mask[i][j] = 0  # Air
    
    cell = fill_symmetric(unique_mask)
    current_gap = estimate_gap(cell, k_points)
    best_gap = current_gap
    best_cell = cell.copy()
    
    pixel_mapping = get_unique_pixel_indices(cell)
    
    # Multi-pixel topological optimization with simulated annealing
    temperature = 0.3
    cooling_rate = 0.995
    
    for iteration in range(300):
        if time.time() - start_time > time_limit:
            break
        
        # Compute field intensities for bands 4 and 5 using fast method
        intensity_4 = compute_field_intensities_fast(cell, k_points, 3)
        intensity_5 = compute_field_intensities_fast(cell, k_points, 4)
        
        # Compute pressure field: positive means add dielectric, negative means remove
        pressure = intensity_4 - intensity_5
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
        
        # Select multiple pixels to flip in this iteration
        num_flips = 3 if iteration < 50 else 2
        
        # Get all unique pixels sorted by absolute pressure
        pixel_pressures = []
        for (i, j), coords in pixel_mapping.items():
            avg_pressure = np.mean(pressure[coords])
            pixel_pressures.append((abs(avg_pressure), avg_pressure, i, j))
        
        pixel_pressures.sort(reverse=True)
        
        # Try flipping the top pixels
        for _, avg_pressure, i, j in pixel_pressures[:num_flips * 2]:
            if time.time() - start_time > time_limit:
                break
            
            # Decide direction based on pressure
            if avg_pressure > 0:
                new_val = 1
            else:
                new_val = 0
            
            # Skip if already at desired value
            if unique_mask[i][j] == new_val:
                continue
            
            # Apply the change
            unique_mask[i][j] = new_val
            new_cell = fill_symmetric(unique_mask)
            new_gap = estimate_gap(new_cell, k_points)
            
            # Simulated annealing acceptance
            delta = new_gap - current_gap
            if delta > 0:
                accept = True
            else:
                accept = random.random() < np.exp(delta / temperature)
            
            if accept:
                current_gap = new_gap
                cell = new_cell
                if new_gap > best_gap:
                    best_gap = new_gap
                    best_cell = new_cell.copy()
            else:
                # Revert
                unique_mask[i][j] = 1 - new_val
                cell = fill_symmetric(unique_mask)
        
        # Cool down
        temperature *= cooling_rate
        
        # Random perturbation to escape local minima
        if iteration % 30 == 29:
            for _ in range(3):
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
                # Revert the random changes
                cell = fill_symmetric(unique_mask)
    
    # Final check: ensure best_cell is valid and symmetric
    if best_cell is None:
        best_cell = cell
    
    # Verify symmetry
    result = best_cell.tolist()
    # Ensure it's exactly 32x32
    assert len(result) == 32 and all(len(row) == 32 for row in result)
    
    return result
