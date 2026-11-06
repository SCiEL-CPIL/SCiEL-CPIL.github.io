"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    t_start = time.time()
    DEADLINE = 215.0  # Safety margin before 230s kill
    
    def time_ok():
        return time.time() - t_start < DEADLINE
        
    # Plane wave setup: 17x17 = 289 plane waves
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289
    
    # K-points in irreducible BZ (30 points as specified)
    pi = np.pi
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)
    
    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2
        
    # Precompute difference indices for Fourier coupling
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            idx_x = int(dGx) + half
            idx_y = int(dGy) + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)
    
    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs
        
    def compute_band_ratio(coeffs):
        """Compute gap-midgap ratio."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            # Build H matrix efficiently
            H_vals = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                if eigvals[3] > max_band4:
                    max_band4 = eigvals[3]
                if eigvals[4] < min_band5:
                    min_band5 = eigvals[4]
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def enforce_symmetry(cell):
        """Enforce 4-fold rotational + reflection symmetry."""
        cell = np.array(cell, dtype=int)
        # 4-fold rotation: 90, 180, 270 degrees
        rot90 = np.rot90(cell, 1)
        rot180 = np.rot90(cell, 2)
        rot270 = np.rot90(cell, 3)
        # Reflections
        hflip = cell[:, ::-1].copy()
        vflip = cell[::-1, :].copy()
        # Transpose
        trans = cell.T.copy()
        # All 8 symmetries
        avg = (cell + rot90 + rot180 + rot270 + hflip + vflip + trans + np.rot90(trans, 1))
        result = (avg >= 4).astype(int)
        return result

    def generate_disk_in_quadrant(cx, cy, radius):
        """
        Generate a cell by placing a disk in the top-left quadrant (0-15, 0-15)
        and enforcing full D4 symmetry.
        cx, cy are in pixel coordinates [0, 15].
        radius is in pixels.
        """
        cell = np.zeros((size, size), dtype=int)
        # Coordinates for the top-left quadrant
        # We generate on the full 32x32 grid but only fill the TL quadrant initially
        # Actually, it's easier to generate on full grid and mask, or just generate TL and replicate
        # Let's generate on full grid, but only set pixels in TL quadrant based on distance
        
        coords_y = np.arange(size)
        coords_x = np.arange(size)
        Y, X = np.meshgrid(coords_y, coords_x, indexing='ij')
        
        # Mask for top-left quadrant: rows 0-15, cols 0-15
        # Note: The center of the cell is at 15.5. 
        # The symmetry axes are at x=15.5 and y=15.5.
        # So the fundamental domain is roughly x in [0, 15.5], y in [0, 15.5].
        # We will define the disk center in this domain.
        
        dist_sq = (X - cx)**2 + (Y - cy)**2
        # Only fill if in the top-left quadrant (X < 16, Y < 16)
        # Using 16 as the split point for 32 pixels
        mask = (dist_sq <= radius**2) & (X < 16) & (Y < 16)
        cell[mask] = 1
        
        return enforce_symmetry(cell)

    # Phase 1: Global Search over Disk parameters
    # We search over center (cx, cy) and radius r
    best_ratio = -np.inf
    best_params = None
    best_cell = None
    
    # Coarse grid
    # Centers in the top-left quadrant (0-15)
    cx_values = [2, 4, 6, 8, 10, 12, 14]
    cy_values = [2, 4, 6, 8, 10, 12, 14]
    r_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    for cx in cx_values:
        for cy in cy_values:
            for r in r_values:
                if not time_ok():
                    break
                cell = generate_disk_in_quadrant(cx, cy, r)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = (cx, cy, r)
                    best_cell = cell.copy()
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
            
    if best_cell is None:
        # Fallback
        best_cell = generate_disk_in_quadrant(8, 8, 3.0)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
        best_params = (8, 8, 3.0)

    # Phase 2: Local Optimization (Hill Climbing on parameters)
    current_cx, current_cy, current_r = best_params
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    search_start = time.time()
    search_time_limit = 150.0  # Leave ~50s for safety
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
        improved = False
        
        # Generate random perturbations
        new_cx = current_cx + np.random.normal(0, 1.0)
        new_cy = current_cy + np.random.normal(0, 1.0)
        new_r = current_r * (1.0 + np.random.normal(0, 0.1))
        
        # Clamp values
        # Keep center within the quadrant [1, 15] to avoid edge artifacts in symmetry enforcement
        new_cx = np.clip(new_cx, 1.0, 15.0)
        new_cy = np.clip(new_cy, 1.0, 15.0)
        new_r = np.clip(new_r, 1.0, 8.0)
        
        new_cell = generate_disk_in_quadrant(new_cx, new_cy, new_r)
        coeffs = get_fourier_coeffs(new_cell)
        new_ratio = compute_band_ratio(coeffs)
        
        if new_ratio > current_ratio + 1e-6:
            current_cell = new_cell
            current_ratio = new_ratio
            current_cx, current_cy, current_r = new_cx, new_cy, new_r
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            
    # Phase 3: Pixel Refinement
    # Sometimes small pixel adjustments at the boundary can help
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 30:
        # Independent pixels in the fundamental domain (top-left 16x16)
        # We pick pixels from the 16x16 block
        indep_pixels = [(i, j) for i in range(16) for j in range(16)]
        np.random.shuffle(indep_pixels)
        
        refine_start = time.time()
        refine_limit = 25.0
        
        no_improve_refine = 0
        for (i, j) in indep_pixels:
            if time.time() - refine_start > refine_limit or not time_ok():
                break
                
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                no_improve_refine = 0
            else:
                no_improve_refine += 1
                if no_improve_refine > 50:
                    break

    # Final verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)
    
    if final_ratio < best_ratio - 0.01:
        current_cell = best_cell
    else:
        best_ratio = final_ratio
        
    return current_cell.tolist()
# EVOLVE-BLOCK-END
