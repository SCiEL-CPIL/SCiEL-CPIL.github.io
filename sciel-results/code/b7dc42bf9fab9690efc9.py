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
    
    # --- PWE Solver ---
    
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

    def compute_field_intensities(cell, k_points, band_indices, n_pw=7):
        """
        Computes the local field intensity at each pixel for given bands
        across multiple k-points, averaged.
        Returns a list of 2D arrays, one per band index.
        """
        n_bands = len(band_indices)
        total_intensities = [np.zeros((size, size)) for _ in range(n_bands)]
        count = 0
        
        # Precompute positions
        positions = np.array([(px / size, py / size) for px in range(size) for py in range(size)])
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) <= max(band_indices):
                continue
            
            G_vectors = get_G_vectors(n_pw)
            n = len(G_vectors)
            
            # Precompute phases for all pixels and all G vectors
            # phases[px, py, i] = 2*pi*(Kx_i*rx + Ky_i*ry)
            phases = np.zeros((size, size, n))
            for i in range(n):
                gx_i, gy_i = G_vectors[i]
                Kx = kx + gx_i
                Ky = ky + gy_i
                phases[:, :, i] = 2 * np.pi * (Kx * positions[:, 0] + Ky * positions[:, 1])
            
            for bi, band_idx in enumerate(band_indices):
                if band_idx >= len(eigvals):
                    continue
                vec = eigvecs[:, band_idx]
                
                # Compute field at each pixel: E(r) = sum_i vec_i * exp(i * phase_i)
                field = np.zeros((size, size), dtype=complex)
                for i in range(n):
                    field += vec[i] * np.exp(1j * phases[:, :, i])
                
                total_intensities[bi] += np.abs(field)**2
                count += 1
        
        if count > 0:
            for bi in range(n_bands):
                total_intensities[bi] /= count
        
        return total_intensities

    def estimate_gap(cell, k_points, n_pw=7):
        """
        Estimates the gap between band 4 and 5.
        Returns the gap-midgap ratio.
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
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return 0.0
        
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        
        gap = min_omega_5 - max_omega_4
        return gap / midgap

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
    
    # Define k-points for the surrogate (irreducible BZ of square lattice)
    # k in units of pi/a, so BZ boundary is at k=0.5
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
        (0.4, 0.4),
        (0.35, 0.15),
        (0.15, 0.35),
    ]
    
    # Random initialization
    unique_mask = np.random.randint(0, 2, (16, 16))
    cell = fill_symmetric(unique_mask)
    current_score = estimate_gap(cell, k_points)
    best_score = current_score
    best_cell = cell.copy()
    
    pixel_mapping = get_unique_pixel_indices()
    unique_pixel_keys = list(pixel_mapping.keys())
    
    # Simulated annealing parameters
    T_initial = 0.5
    T_final = 0.01
    n_iterations = 300
    cooling_rate = (T_final / T_initial) ** (1.0 / max(1, n_iterations - 1))
    
    # Topological optimization loop
    for iteration in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
        
        T = T_initial * (cooling_rate ** iteration)
        
        # Compute field intensities for bands 4 and 5 (0-indexed: 3 and 4)
        intensities = compute_field_intensities(cell, k_points, [3, 4])
        intensity_4 = intensities[0]
        intensity_5 = intensities[1]
        
        # Strategy: 
        # - Where band 4 has high intensity and band 5 has low intensity: add dielectric (set to 1)
        #   This raises band 5 more than band 4, potentially opening the gap
        # - Where band 5 has high intensity and band 4 has low intensity: remove dielectric (set to 0)
        #   This lowers band 5 more than band 4, potentially opening the gap
        # - Where both have high intensity: neutral, skip
        # - Where both have low intensity: neutral, skip
        
        # Compute a "pressure" field: positive means add dielectric, negative means remove
        pressure = intensity_4 - intensity_5
        
        # Normalize pressure
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
        
        # For each unique pixel, compute average pressure and decide direction
        candidates = []
        for (i, j), coords in pixel_mapping.items():
            avg_pressure = np.mean(pressure[coords])
            # Desired value: 1 if pressure > 0, 0 if pressure < 0
            if avg_pressure > 0.1:
                desired = 1
            elif avg_pressure < -0.1:
                desired = 0
            else:
                desired = None  # Neutral
            
            if desired is not None and unique_mask[i][j] != desired:
                # This pixel should be flipped
                candidates.append(((i, j), abs(avg_pressure), desired))
        
        # Sort by absolute pressure (strongest signal first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Try top candidates
        accepted = False
        for (i, j), _, desired in candidates[:5]:
            if time.time() - start_time > time_limit:
                break
            
            new_val = desired
            if unique_mask[i][j] == new_val:
                continue
            
            # Apply the change
            old_val = unique_mask[i][j]
            unique_mask[i][j] = new_val
            new_cell = fill_symmetric(unique_mask)
            new_score = estimate_gap(new_cell, k_points)
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if new_score > current_score:
                current_score = new_score
                cell = new_cell
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell.copy()
                accepted = True
                break
            else:
                delta = current_score - new_score
                if delta < 0:
                    delta = 0
                prob = np.exp(-delta / max(T, 1e-10))
                if random.random() < prob:
                    current_score = new_score
                    cell = new_cell
                    if new_score > best_score:
                        best_score = new_score
                        best_cell = new_cell.copy()
                    accepted = True
                    break
                else:
                    # Reject and revert
                    unique_mask[i][j] = old_val
                    cell = fill_symmetric(unique_mask)
        
        # If no candidate was accepted, do a random flip
        if not accepted:
            ri = random.randint(0, 15)
            rj = random.randint(0, 15)
            old_val = unique_mask[ri][rj]
            unique_mask[ri][rj] = 1 - old_val
            new_cell = fill_symmetric(unique_mask)
            new_score = estimate_gap(new_cell, k_points)
            
            if new_score > current_score:
                current_score = new_score
                cell = new_cell
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell.copy()
            else:
                delta = current_score - new_score
                if delta < 0:
                    delta = 0
                prob = np.exp(-delta / max(T, 1e-10))
                if random.random() < prob:
                    current_score = new_score
                    cell = new_cell
                    if new_score > best_score:
                        best_score = new_score
                        best_cell = new_cell.copy()
                else:
                    unique_mask[ri][rj] = old_val
                    cell = fill_symmetric(unique_mask)
        
        # Occasionally restart from best to escape local minima
        if iteration % 50 == 49 and iteration > 0:
            cell = best_cell.copy()
            current_score = best_score
            unique_mask = np.zeros((16, 16), dtype=int)
            for i in range(16):
                for j in range(16):
                    # Recover unique mask from best_cell
                    unique_mask[i][j] = best_cell[i][j]
    
    # Final check: ensure best_cell is valid
    if best_cell is None:
        best_cell = cell
    
    # Verify symmetry
    result = best_cell.tolist()
    # Ensure perfect symmetry
    for i in range(size):
        for j in range(size):
            result[i][j] = int(best_cell[i][j])
            result[size-1-i][j] = int(best_cell[i][j])
            result[i][size-1-j] = int(best_cell[i][j])
            result[size-1-i][size-1-j] = int(best_cell[i][j])
            result[j][i] = int(best_cell[i][j])
            result[size-1-j][i] = int(best_cell[i][j])
            result[j][size-1-i] = int(best_cell[i][j])
            result[size-1-j][size-1-i] = int(best_cell[i][j])
    
    return result
