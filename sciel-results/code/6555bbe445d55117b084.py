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
    time_limit = 110  # Leave 10s buffer
    
    # --- PWE Solver ---
    
    # Precompute G vectors for n_pw=7 (49 plane waves)
    n_pw = 7
    half = n_pw // 2
    G_list = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_list.append((gx, gy))
    G_list.sort(key=lambda x: x[0]**2 + x[1]**2)
    G_vectors = G_list[:n_pw * n_pw]
    n_G = len(G_vectors)
    
    # Precompute G difference indices for FFT lookup
    # fft_index(g, N) maps integer g to [0, N)
    def fft_idx(g, N):
        return g if g >= 0 else N + g
    
    # Precompute the G difference matrix indices
    G_diff_idx = np.zeros((n_G, n_G), dtype=int)
    for i in range(n_G):
        gx_i, gy_i = G_vectors[i]
        for j in range(n_G):
            gx_j, gy_j = G_vectors[j]
            G_diff_idx[i, j] = (fft_idx(gx_i - gx_j, size), fft_idx(gy_i - gy_j, size))
    
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
    
    def solve_band_structure(cell, kx, ky, return_eigvecs=False):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        Returns sorted omega^2 values, and optionally eigenvectors.
        """
        fft_coeffs = get_fourier_coeffs(cell)
        
        # Build matrix M where M[i,j] = D(G_i - G_j)
        M = np.zeros((n_G, n_G), dtype=complex)
        for i in range(n_G):
            for j in range(n_G):
                ix, iy = G_diff_idx[i, j]
                M[i, j] = fft_coeffs[ix, iy]
        
        # K^2 values
        K_sq = np.zeros(n_G)
        for i in range(n_G):
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
                return [0] * n_G, np.zeros((n_G, n_G), dtype=complex)
            return [0] * n_G
    
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
            
            # Compute field at each pixel using matrix multiplication
            # E(r) = sum_i vec_i * exp(i * 2*pi * (k+G_i) . r)
            # For pixel (px, py): r = (px/size, py/size)
            
            # Build phase matrix: shape (n_G, size, size)
            # phase[i, px, py] = 2*pi*(Kx_i*px/size + Ky_i*py/size)
            phases = np.zeros((n_G, size, size))
            for i in range(n_G):
                gx_i, gy_i = G_vectors[i]
                Kx = kx + gx_i
                Ky = ky + gy_i
                # Vectorized phase computation
                px_arr = np.arange(size) / size
                py_arr = np.arange(size) / size
                phases[i] = 2 * np.pi * (Kx[:, None] * px_arr[None, :] + Ky[:, None] * py_arr[None, :])
            
            # field[px, py] = sum_i vec[i] * exp(i * phase[i, px, py])
            field = np.sum(vec[:, None, None] * np.exp(1j * phases), axis=0)
            intensity = np.abs(field)**2
            
            total_intensity += intensity
            count += 1
        
        if count > 0:
            total_intensity /= count
        return total_intensity
    
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
        (0.35, 0.35),
        (0.15, 0.35),
    ]
    
    pixel_mapping = get_unique_pixel_indices()
    
    # Helper: create initial patterns
    def make_initial_pattern(pattern_type, seed_val):
        """Create an initial 16x16 unique mask."""
        rng = random.Random(seed_val)
        mask = np.zeros((16, 16), dtype=int)
        
        if pattern_type == 0:
            # Square hole pattern
            for i in range(16):
                for j in range(16):
                    dist = max(abs(i - 7.5), abs(j - 7.5))
                    if dist < 4.5:
                        mask[i, j] = 1
        elif pattern_type == 1:
            # Circular hole
            for i in range(16):
                for j in range(16):
                    dist = np.sqrt((i - 7.5)**2 + (j - 7.5)**2)
                    if dist < 4.5:
                        mask[i, j] = 1
        elif pattern_type == 2:
            # Diamond hole
            for i in range(16):
                for j in range(16):
                    dist = abs(i - 7.5) + abs(j - 7.5)
                    if dist < 6:
                        mask[i, j] = 1
        elif pattern_type == 3:
            # Random with some structure
            # Start with dielectric, punch random holes
            mask[:] = 1
            for _ in range(30):
                ri = rng.randint(0, 15)
                rj = rng.randint(0, 15)
                mask[ri, rj] = 0
                # Also set symmetric positions in the 16x16 quadrant
                if ri != 15-ri:
                    mask[15-ri, rj] = 0
                    mask[15-ri, 15-rj] = 0
                if rj != 15-rj:
                    mask[ri, 15-rj] = 0
        elif pattern_type == 4:
            # Cross pattern
            for i in range(16):
                for j in range(16):
                    if abs(i - 7.5) < 2 or abs(j - 7.5) < 2:
                        mask[i, j] = 1
        elif pattern_type == 5:
            # Ring pattern
            for i in range(16):
                for j in range(16):
                    dist = np.sqrt((i - 7.5)**2 + (j - 7.5)**2)
                    if 3 < dist < 5.5:
                        mask[i, j] = 1
        elif pattern_type == 6:
            # Checkerboard-like
            for i in range(16):
                for j in range(16):
                    if (i // 4 + j // 4) % 2 == 0:
                        mask[i, j] = 1
        elif pattern_type == 7:
            # Random binary
            mask = (rng.random((16, 16)) > 0.5).astype(int)
        else:
            # Default: square hole
            for i in range(16):
                for j in range(16):
                    dist = max(abs(i - 7.5), abs(j - 7.5))
                    if dist < 4.5:
                        mask[i, j] = 1
        
        return mask
    
    def optimize_from_start(unique_mask, k_points, pixel_mapping, time_budget):
        """
        Run topological optimization from a given starting mask.
        Returns the best cell found.
        """
        cell = fill_symmetric(unique_mask)
        current_gap = estimate_gap(cell, k_points)
        best_gap = current_gap
        best_cell = cell.copy()
        best_mask = unique_mask.copy()
        
        n_iter = 0
        stall_count = 0
        max_stall = 40
        
        while time.time() - start_time < time_budget:
            n_iter += 1
            
            # Compute field intensities for bands 4 and 5
            intensity_4 = compute_field_intensities(cell, k_points, 3)
            intensity_5 = compute_field_intensities(cell, k_points, 4)
            
            # Pressure: positive means band 4 dominates -> add dielectric
            # Negative means band 5 dominates -> remove dielectric
            pressure = intensity_4 - intensity_5
            
            p_max = np.max(np.abs(pressure))
            if p_max > 1e-10:
                pressure /= p_max
            
            # Score each unique pixel: how much would flipping help?
            # For pixel (i,j) with current value v:
            # If v=0 (air), flipping to 1 (dielectric) helps if pressure > 0
            # If v=1 (dielectric), flipping to 0 (air) helps if pressure < 0
            # Score = |pressure| * (1 if flip direction matches pressure sign, else -1)
            
            best_score = -float('inf')
            best_pixel = None
            best_new_val = None
            
            for (i, j), coords in pixel_mapping.items():
                avg_pressure = np.mean(pressure[coords])
                current_val = unique_mask[i, j]
                
                if current_val == 0:
                    # Flipping to 1: good if pressure > 0
                    if avg_pressure > 0:
                        score = avg_pressure
                    else:
                        score = -avg_pressure * 0.3  # Penalty for wrong direction
                else:
                    # Flipping to 0: good if pressure < 0
                    if avg_pressure < 0:
                        score = -avg_pressure
                    else:
                        score = -avg_pressure * 0.3  # Penalty for wrong direction
                
                if score > best_score:
                    best_score = score
                    best_pixel = (i, j)
                    best_new_val = 1 - current_val
            
            if best_pixel is None:
                break
            
            i, j = best_pixel
            
            # If score is too low, try random
            if best_score < 0.05:
                i = random.randint(0, 15)
                j = random.randint(0, 15)
                best_new_val = 1 - unique_mask[i, j]
            
            # Apply the change
            unique_mask[i, j] = best_new_val
            new_cell = fill_symmetric(unique_mask)
            new_gap = estimate_gap(new_cell, k_points)
            
            if new_gap > current_gap:
                current_gap = new_gap
                cell = new_cell
                if new_gap > best_gap:
                    best_gap = new_gap
                    best_cell = new_cell.copy()
                    best_mask = unique_mask.copy()
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                # Reject and revert
                unique_mask[i, j] = 1 - best_new_val
                cell = fill_symmetric(unique_mask)
                stall_count += 1
            
            # If stalled, do a random perturbation
            if stall_count >= max_stall:
                for _ in range(5):
                    ri = random.randint(0, 15)
                    rj = random.randint(0, 15)
                    unique_mask[ri, rj] = 1 - unique_mask[ri, rj]
                new_cell = fill_symmetric(unique_mask)
                new_gap = estimate_gap(new_cell, k_points)
                if new_gap > current_gap:
                    current_gap = new_gap
                    cell = new_cell
                    if new_gap > best_gap:
                        best_gap = new_gap
                        best_cell = new_cell.copy()
                        best_mask = unique_mask.copy()
                else:
                    # Revert the random perturbations
                    for _ in range(5):
                        ri = random.randint(0, 15)
                        rj = random.randint(0, 15)
                        unique_mask[ri, rj] = 1 - unique_mask[ri, rj]
                    cell = fill_symmetric(unique_mask)
                stall_count = 0
        
        return best_cell, best_gap
    
    # Multi-start optimization
    best_overall_gap = -float('inf')
    best_overall_cell = None
    
    # Try several initial patterns
    initial_patterns = [0, 1, 2, 3, 4, 5, 6, 7]
    random.shuffle(initial_patterns)
    
    # Allocate time per start
    n_starts = len(initial_patterns)
    time_per_start = (time_limit - 5) / n_starts  # Leave 5s for finalization
    
    for pattern_idx, pat_type in enumerate(initial_patterns):
        if time.time() - start_time > time_limit - 5:
            break
        
        remaining_time = time_limit - 5 - (time.time() - start_time)
        if remaining_time < 2:
            break
        
        time_budget = start_time + min(time_per_start, remaining_time)
        
        seed_val = 42 + pattern_idx * 100
        unique_mask = make_initial_pattern(pat_type, seed_val)
        
        cell, gap = optimize_from_start(unique_mask, k_points, pixel_mapping, time_budget)
        
        if gap > best_overall_gap:
            best_overall_gap = gap
            best_overall_cell = cell
    
    # If we didn't find any valid cell, use a default
    if best_overall_cell is None:
        unique_mask = make_initial_pattern(0, 42)
        best_overall_cell = fill_symmetric(unique_mask)
    
    return best_overall_cell.tolist()
