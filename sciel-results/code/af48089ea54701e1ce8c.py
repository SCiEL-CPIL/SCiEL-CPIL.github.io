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
    DEADLINE = 220.0  # safety margin before 230s kill
    
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
        """Enforce 4-fold rotational + reflection symmetry (D4 group)."""
        cell = np.array(cell, dtype=int)
        # All 8 symmetries of the square
        symmetries = [
            cell,
            np.rot90(cell, 1),
            np.rot90(cell, 2),
            np.rot90(cell, 3),
            cell[:, ::-1].copy(),
            cell[::-1, :].copy(),
            cell.T.copy(),
            np.rot90(cell.T.copy(), 1)
        ]
        avg = np.sum(symmetries, axis=0)
        result = (avg >= 4).astype(int)
        return result

    def generate_grating_cell(freq_x, freq_y, duty_x, duty_y, phase_x=0.0, phase_y=0.0):
        """
        Generate a cell from product of two 1D square-wave gratings.
        cell[i,j] = 1 if (square_wave(i*freq_x + phase_x) AND square_wave(j*freq_y + phase_y))
        
        This creates a checkerboard-like microstructure with tunable spatial frequencies
        and duty cycles that can enhance specific Fourier components.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        
        # 1D square waves
        # square_wave(x) = 1 if |x mod 1 - 0.5| < duty/2, else 0
        def square_wave(x, duty):
            # x is in [0,1) effectively, duty is fraction
            x_mod = np.abs(x - 0.5)
            return (x_mod < duty / 2.0).astype(float)
        
        # Generate 1D patterns
        pattern_x = square_wave(coords * freq_x + phase_x, duty_x)
        pattern_y = square_wave(coords * freq_y + phase_y, duty_y)
        
        # Product creates 2D checkerboard
        cell_2d = np.outer(pattern_y, pattern_x).astype(int)
        
        return enforce_symmetry(cell)

    def generate_staggered_grating(freq_x, freq_y, duty_x, duty_y, stagger=0.0):
        """
        Generate a staggered grating pattern.
        Odd rows are shifted by 'stagger' relative to even rows.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        
        def square_wave(x, duty):
            x_mod = np.abs(x - 0.5)
            return (x_mod < duty / 2.0).astype(float)
        
        for i in range(size):
            shift = stagger if i % 2 == 1 else 0.0
            pattern = square_wave(coords * freq_x + shift, duty_x)
            cell[i, :] = pattern.astype(int)
            
        return enforce_symmetry(cell)

    def generate_dual_scale_cell(freq1, freq2, duty1, duty2, blend=0.5):
        """
        Generate a cell that is a blend of two grating patterns at different scales.
        cell = 1 if (pattern1 AND pattern2) with some blending.
        """
        cell1 = generate_grating_cell(freq1, freq1, duty1, duty1)
        cell2 = generate_grating_cell(freq2, freq2, duty2, duty2)
        
        # Blend: use XOR-like combination or weighted sum
        combined = (cell1.astype(float) * (1 - blend) + cell2.astype(float) * blend)
        cell = (combined >= 0.5).astype(int)
        
        return enforce_symmetry(cell)

    def generate_hollow_square(half_size, wall_thickness, corner_cut=0.0):
        """
        Generate a hollow square ring with optional corner cutting.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        # Square ring: |x| < half_size AND |y| < half_size, but hollow inside
        outer = np.abs(X) < half_size
        inner = np.abs(X) < (half_size - wall_thickness)
        
        # Ring = outer AND NOT inner
        ring = outer & ~inner
        
        # Corner cutting: remove corners where both |x| and |y| are close to half_size
        if corner_cut > 0:
            corner_mask = (np.abs(X) > half_size - corner_cut) & (np.abs(Y) > half_size - corner_cut)
            ring = ring & ~corner_mask
            
        cell[ring] = 1
        return enforce_symmetry(cell)

    def generate_cross_pattern(arm_width, arm_length):
        """
        Generate a cross (+) pattern.
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        # Horizontal arm
        h_arm = (np.abs(Y) < arm_width / 2.0) & (np.abs(X) < arm_length / 2.0)
        # Vertical arm
        v_arm = (np.abs(X) < arm_width / 2.0) & (np.abs(Y) < arm_length / 2.0)
        
        cell[h_arm | v_arm] = 1
        return enforce_symmetry(cell)

    # Phase 1: Global Search over diverse design families
    best_ratio = -np.inf
    best_cell = None
    best_params = None
    best_family = None
    
    # Family 1: Grating cells (NEW - product of 1D square waves)
    freq_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    duty_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    phase_values = [0.0, 0.1, 0.2]
    
    for fx in freq_values:
        for fy in freq_values:
            if fx > fy:
                continue  # Avoid duplicates due to symmetry
            for dx in duty_values:
                for dy in duty_values:
                    if not time_ok():
                        break
                    for px in phase_values:
                        for py in phase_values:
                            if not time_ok():
                                break
                            cell = generate_grating_cell(fx, fy, dx, dy, px, py)
                            coeffs = get_fourier_coeffs(cell)
                            ratio = compute_band_ratio(coeffs)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_params = (fx, fy, dx, dy, px, py)
                                best_cell = cell.copy()
                                best_family = 'grating'
                            if not time_ok():
                                break
                        if not time_ok():
                            break
                    if not time_ok():
                        break
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Family 2: Staggered gratings
    stagger_values = [0.0, 0.1, 0.2, 0.25]
    for fx in [2.0, 3.0, 4.0, 5.0]:
        for dx in [0.3, 0.4, 0.5, 0.6]:
            for st in stagger_values:
                if not time_ok():
                    break
                cell = generate_staggered_grating(fx, fx, dx, dx, st)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = (fx, fx, dx, dx, st)
                    best_cell = cell.copy()
                    best_family = 'staggered'
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Family 3: Dual-scale cells
    for f1 in [2.0, 3.0, 4.0]:
        for f2 in [5.0, 6.0, 8.0]:
            for d1 in [0.4, 0.5, 0.6]:
                for d2 in [0.3, 0.4, 0.5]:
                    for blend in [0.3, 0.5, 0.7]:
                        if not time_ok():
                            break
                        cell = generate_dual_scale_cell(f1, f2, d1, d2, blend)
                        coeffs = get_fourier_coeffs(cell)
                        ratio = compute_band_ratio(coeffs)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_params = (f1, f2, d1, d2, blend)
                            best_cell = cell.copy()
                            best_family = 'dual_scale'
                        if not time_ok():
                            break
                    if not time_ok():
                        break
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Family 4: Hollow squares
    for hs in [0.25, 0.30, 0.35, 0.40]:
        for wt in [0.05, 0.08, 0.10, 0.12]:
            for cc in [0.0, 0.05, 0.10]:
                if not time_ok():
                    break
                cell = generate_hollow_square(hs, wt, cc)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = (hs, wt, cc)
                    best_cell = cell.copy()
                    best_family = 'hollow_square'
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Family 5: Cross patterns
    for aw in [0.10, 0.15, 0.20, 0.25]:
        for al in [0.40, 0.50, 0.60, 0.70]:
            if not time_ok():
                break
            cell = generate_cross_pattern(aw, al)
            coeffs = get_fourier_coeffs(cell)
            ratio = compute_band_ratio(coeffs)
            if ratio > best_ratio:
                best_ratio = ratio
                best_params = (aw, al)
                best_cell = cell.copy()
                best_family = 'cross'
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Family 6: Superellipses (keep from original as fallback)
    n_values = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    a_values = [0.08, 0.10, 0.12, 0.14]
    off_values = [0.20, 0.22, 0.25, 0.28]
    
    for n in n_values:
        for a in a_values:
            for off in off_values:
                if not time_ok():
                    break
                # Generate superellipse cell
                cell = np.zeros((size, size), dtype=int)
                coords = (np.arange(size) + 0.5) / size - 0.5
                X, Y = np.meshgrid(coords, coords, indexing='xy')
                centers = [(off, off), (-off, off), (off, -off), (-off, -off)]
                for cx, cy in centers:
                    dx = X - cx
                    dy = Y - cy
                    val = np.abs(dx / a) ** n + np.abs(dy / a) ** n
                    cell[val <= 1.0] = 1
                cell = enforce_symmetry(cell)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = (n, a, off)
                    best_cell = cell.copy()
                    best_family = 'superellipse'
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    if best_cell is None:
        # Fallback: simple grating
        best_cell = generate_grating_cell(3.0, 3.0, 0.5, 0.5)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
        best_params = (3.0, 3.0, 0.5, 0.5, 0.0, 0.0)
        best_family = 'grating'

    # Phase 2: Local Optimization with pixel mutations
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    search_start = time.time()
    search_time_limit = 150.0  # Leave time for pixel refinement
    
    # Use pixel-level mutations for fine-tuning
    np.random.seed(42)
    indep_pixels = [(i, j) for i in range(16) for j in range(16)]
    
    # Multiple rounds of pixel mutation
    for round_num in range(3):
        if not time_ok() or time.time() - search_start > search_time_limit:
            break
        
        np.random.shuffle(indep_pixels)
        no_improve = 0
        max_no_improve = 30
        
        for (i, j) in indep_pixels:
            if time.time() - search_start > search_time_limit or not time_ok():
                break
            
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            
            # Quick check: if no change, skip
            if np.array_equal(new_cell, current_cell):
                continue
                
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
    
    # Phase 3: Second round of pixel refinement with different seed
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 30:
        np.random.seed(123)
        indep_pixels2 = [(i, j) for i in range(16) for j in range(16)]
        np.random.shuffle(indep_pixels2)
        
        refine_start = time.time()
        refine_limit = 20.0
        
        no_improve_refine = 0
        for (i, j) in indep_pixels2:
            if time.time() - refine_start > refine_limit or not time_ok():
                break
                
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            
            if np.array_equal(new_cell, current_cell):
                continue
                
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                no_improve_refine = 0
            else:
                no_improve_refine += 1
                if no_improve_refine > 40:
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
