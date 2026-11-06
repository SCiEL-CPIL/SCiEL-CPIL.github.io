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
    DEADLINE = 215.0  # safety margin before 230s kill
    
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

    def generate_superlattice(radii):
        """
        Generate a cell from a 4x4 superlattice of 8x8 sub-cells.
        radii: array of 3 independent radii [r_corner, r_edge, r_inner]
        The 4x4 grid of sub-cells is determined by symmetry.
        Sub-cell centers are at (i+0.5)*8, (j+0.5)*8 in pixel coordinates.
        """
        cell = np.zeros((size, size), dtype=int)
        
        # Define the 3 independent sub-cell types in the 2x2 fundamental domain (top-left quadrant of the 4x4 grid)
        # Indices in 4x4 grid: (0,0) is corner, (0,1) is edge, (1,1) is inner
        # Due to symmetry, we define these and replicate.
        # (0,0) -> r_corner
        # (0,1) -> r_edge
        # (1,1) -> r_inner
        # (1,0) -> r_edge (symmetric to 0,1)
        
        # Map 4x4 indices to radii
        r_map = np.zeros((4, 4))
        r_map[0, 0] = radii[0]
        r_map[0, 1] = radii[1]
        r_map[0, 2] = radii[1]
        r_map[0, 3] = radii[0]
        
        r_map[1, 0] = radii[1]
        r_map[1, 1] = radii[2]
        r_map[1, 2] = radii[2]
        r_map[1, 3] = radii[1]
        
        r_map[2, 0] = radii[1]
        r_map[2, 1] = radii[2]
        r_map[2, 2] = radii[2]
        r_map[2, 3] = radii[1]
        
        r_map[3, 0] = radii[0]
        r_map[3, 1] = radii[1]
        r_map[3, 2] = radii[1]
        r_map[3, 3] = radii[0]
        
        # Rasterize
        # Pixel centers: (i+0.5)/32 - 0.5
        # Sub-cell size is 8x8 pixels.
        # Center of sub-cell (si, sj) is at pixel (si*8 + 4, sj*8 + 4) -> coordinate (si*8 + 3.5)/32 - 0.5?
        # Let's use relative coordinates within the sub-cell for distance calculation.
        
        for si in range(4):
            for sj in range(4):
                r = r_map[si, sj]
                if r <= 0:
                    continue
                
                # Pixel indices for this sub-cell
                i_start = si * 8
                j_start = sj * 8
                i_end = i_start + 8
                j_end = j_start + 8
                
                # Coordinates of pixels in this sub-cell
                # i, j are global pixel indices
                # x, y are global coordinates in [-0.5, 0.5]
                # Center of sub-cell in global coords:
                # cx = (i_start + 4) / size - 0.5  (center of the 8 pixels)
                # cy = (j_start + 4) / size - 0.5
                
                cx = (i_start + 4) / size - 0.5
                cy = (j_start + 4) / size - 0.5
                
                # Extract sub-grid
                sub_pixels_i = np.arange(i_start, i_end)
                sub_pixels_j = np.arange(j_start, j_end)
                
                # Pixel centers
                px = (sub_pixels_i + 0.5) / size - 0.5
                py = (sub_pixels_j + 0.5) / size - 0.5
                
                X, Y = np.meshgrid(px, py, indexing='ij')
                
                # Distance from sub-cell center
                dist_sq = (X - cx)**2 + (Y - cy)**2
                
                # Disk condition
                mask = dist_sq <= (r * size / 32.0)**2  # r is in units of sub-cell size? 
                # Let's define r as fraction of sub-cell width (8 pixels).
                # So radius in pixels is r * 8.
                # Radius in global coords is (r * 8) / size.
                mask = dist_sq <= (r * 8.0 / size)**2
                
                cell[i_start:i_end, j_start:j_end] = mask.astype(int)
                
        return enforce_symmetry(cell)

    # Phase 1: Global Search over Superlattice Parameters
    # Radii range: 0.0 to 1.0 (fraction of sub-cell width)
    best_ratio = -np.inf
    best_params = None
    best_cell = None
    
    # Coarse grid
    r_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    for r0 in r_vals:
        for r1 in r_vals:
            for r2 in r_vals:
                if not time_ok():
                    break
                radii = np.array([r0, r1, r2])
                cell = generate_superlattice(radii)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = radii.copy()
                    best_cell = cell.copy()
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
            
    if best_cell is None:
        # Fallback
        best_params = np.array([0.5, 0.5, 0.5])
        best_cell = generate_superlattice(best_params)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))

    # Phase 2: Local Optimization (Hill Climbing on radii)
    current_params = best_params.copy()
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    search_start = time.time()
    search_time_limit = 100.0  # Leave ~80s for pixel refinement and safety
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
        improved = False
        
        # Generate random perturbations
        # Perturb radii
        new_params = current_params.copy()
        for i in range(3):
            new_params[i] += np.random.normal(0, 0.1)
            new_params[i] = np.clip(new_params[i], 0.0, 1.2) # Allow slightly larger to fill gaps
        
        new_cell = generate_superlattice(new_params)
        coeffs = get_fourier_coeffs(new_cell)
        new_ratio = compute_band_ratio(coeffs)
        
        if new_ratio > current_ratio + 1e-6:
            current_cell = new_cell
            current_ratio = new_ratio
            current_params = new_params
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            
    # Phase 3: Pixel Refinement
    # Refine the boundaries of the disks by flipping pixels in the independent quadrant
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 40:
        indep_pixels = [(i, j) for i in range(16) for j in range(16)]
        np.random.shuffle(indep_pixels)
        
        refine_start = time.time()
        refine_limit = 30.0
        
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
