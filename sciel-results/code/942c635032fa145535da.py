import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Use a fast internal PWE solver with a smaller basis for quick evaluation.
    2. Use local field intensities of bands 4 and 5 to guide pixel flips (topological optimization).
    3. Enforce exact D4 symmetry by constructing from a single quadrant and mirroring.
    4. Perform iterative local search: at each step, compute field intensities, 
       then flip pixels to increase dielectric where band 4 has high intensity and 
       remove dielectric where band 5 has high intensity.
    5. Maintain strict time budget with early termination.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s limit, keep 10s margin for overhead
    start_time = time.time()
    deadline = start_time + 220.0
    
    def time_left():
        return deadline - time.time()
    
    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=13, n_kpts=8):
        """
        Compute TM band structure using plane-wave expansion.
        Returns omega (n_kpts, n_bands) and optionally eigenvectors for the last band.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
        
        def get_coeff(m, n):
            idx_m = m % size
            idx_n = n % size
            return fft_coeffs[idx_m, idx_n]
        
        M = n_pw // 2
        g_indices = []
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                if m*m + n*n <= (M + 1)*(M + 1):
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        
        S = np.zeros((n_total, n_total), dtype=complex)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                S[i, j] = get_coeff(gm, gn)
        
        k_points = []
        n_seg = max(3, n_kpts // 3)
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((t, 0))
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((0.5, t))
        for t in np.linspace(0.5, 0, n_seg):
            k_points.append((t, t))
        
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        G_vecs = np.array([(2.0 * np.pi * m, 2.0 * np.pi * n) for m, n in g_indices])
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            Gk_x = G_vecs[:, 0] + 2.0 * np.pi * kx
            Gk_y = G_vecs[:, 1] + 2.0 * np.pi * ky
            k_mag_sq = Gk_x**2 + Gk_y**2
            
            Mmat = S * k_mag_sq[np.newaxis, :]
            
            try:
                eigvals = np.linalg.eigvalsh(Mmat)
                eigvals = np.sort(eigvals)
                eigvals = np.real(eigvals)
                eigvals = np.maximum(eigvals, 0)
                omega[ki, :] = np.sqrt(eigvals)
            except Exception:
                omega[ki, :] = np.inf
        
        return omega

    def compute_tm_bands_with_fields(cell, n_pw=13, n_kpts=8):
        """
        Compute TM bands and return field intensities for bands 4 and 5 at each k-point.
        Returns:
        - omega: (n_kpts, n_bands)
        - field4: (n_kpts, size, size) - |E_4|^2 intensity
        - field5: (n_kpts, size, size) - |E_5|^2 intensity
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
        
        def get_coeff(m, n):
            idx_m = m % size
            idx_n = n % size
            return fft_coeffs[idx_m, idx_n]
        
        M = n_pw // 2
        g_indices = []
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                if m*m + n*n <= (M + 1)*(M + 1):
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        
        S = np.zeros((n_total, n_total), dtype=complex)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                S[i, j] = get_coeff(gm, gn)
        
        k_points = []
        n_seg = max(3, n_kpts // 3)
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((t, 0))
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((0.5, t))
        for t in np.linspace(0.5, 0, n_seg):
            k_points.append((t, t))
        
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        field4 = np.zeros((n_kpts_actual, size, size))
        field5 = np.zeros((n_kpts_actual, size, size))
        G_vecs = np.array([(2.0 * np.pi * m, 2.0 * np.pi * n) for m, n in g_indices])
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            Gk_x = G_vecs[:, 0] + 2.0 * np.pi * kx
            Gk_y = G_vecs[:, 1] + 2.0 * np.pi * ky
            k_mag_sq = Gk_x**2 + Gk_y**2
            
            Mmat = S * k_mag_sq[np.newaxis, :]
            
            try:
                eigvals, eigvecs = np.linalg.eigh(Mmat)
                eigvals = np.sort(eigvals)
                # Reorder eigenvectors to match sorted eigenvalues
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                eigvals = np.real(eigvals)
                eigvals = np.maximum(eigvals, 0)
                omega[ki, :] = np.sqrt(eigvals)
                
                # Get eigenvectors for bands 4 and 5 (indices 3 and 4)
                if n_total >= 5:
                    vec4 = eigvecs[:, 3]
                    vec5 = eigvecs[:, 4]
                    
                    # Compute field intensity: sum over G of |c_G|^2 * exp(2i*G*r)
                    # For each pixel position (x, y), compute the field
                    coords = (np.arange(size) + 0.5) / size
                    X, Y = np.meshgrid(coords, coords, indexing='ij')
                    
                    # Field at each pixel: sum_G c_G exp(i G . r)
                    # G = 2*pi*(m, n), r = (x, y)
                    phase4 = np.zeros((size, size), dtype=complex)
                    phase5 = np.zeros((size, size), dtype=complex)
                    
                    for gi, (mi, ni) in enumerate(g_indices):
                        Gx = 2.0 * np.pi * mi
                        Gy = 2.0 * np.pi * ni
                        phase4 += vec4[gi] * np.exp(1j * (Gx * X + Gy * Y))
                        phase5 += vec5[gi] * np.exp(1j * (Gx * X + Gy * Y))
                    
                    field4[ki] = np.abs(phase4)**2
                    field5[ki] = np.abs(phase5)**2
            except Exception:
                omega[ki, :] = np.inf
        
        return omega, field4, field5

    # Enforce exact D4 symmetry
    def enforce_d4_symmetry(cell):
        cell = np.array(cell, dtype=int)
        half = size // 2
        tl = cell[:half, :half].copy()
        cell_new = np.zeros((size, size), dtype=int)
        cell_new[:half, :half] = tl
        cell_new[:half, half:] = tl[:, ::-1]
        cell_new[half:, :half] = tl[::-1, :]
        cell_new[half:, half:] = tl[::-1, ::-1]
        return cell_new

    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        for cx, cy in centers:
            DX = X - cx
            DY = Y - cy
            if rod_radius > 0:
                val = np.abs(DX/rod_radius)**exponent + np.abs(DY/rod_radius)**exponent
            else:
                val = np.zeros_like(X)
            cell[val <= 1.0 + 1e-9] = 1
        return enforce_d4_symmetry(cell).tolist()

    def score_cell(cell, n_pw=13, n_kpts=8):
        try:
            bands = compute_tm_bands(cell, n_pw=n_pw, n_kpts=n_kpts)
            omega_4 = bands[:, 3]
            omega_5 = bands[:, 4]
            
            max_omega_4 = np.max(omega_4)
            min_omega_5 = np.min(omega_5)
            
            if min_omega_5 > max_omega_4:
                gap = min_omega_5 - max_omega_4
                ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
            else:
                overlap = max_omega_4 - min_omega_5
                midgap = 0.5 * (min_omega_5 + max_omega_4)
                ratio = -2.0 * overlap / midgap if midgap > 0 else -1.0
            return ratio
        except Exception:
            return -1.0

    # Phase 1: Quick grid search to find a good starting point
    best_score = -np.inf
    best_cell = None
    
    radii = np.linspace(0.08, 0.18, 10)
    exponents = [1.5, 2.0, 3.0, 4.0]
    
    for r in radii:
        if time_left() < 15:
            break
        for exp in exponents:
            if time_left() < 5:
                break
            cell = make_four_quadrant_cell(r, exp)
            score = score_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Phase 2: Topological optimization using field intensities
    if best_cell is not None and time_left() > 30:
        best_arr = np.array(best_cell, dtype=int)
        half = size // 2
        
        # Iterative field-guided optimization
        for iteration in range(20):
            if time_left() < 15:
                break
            
            # Compute bands and field intensities
            try:
                omega, field4, field5 = compute_tm_bands_with_fields(best_arr, n_pw=13, n_kpts=8)
            except Exception:
                break
            
            # Aggregate field intensities over all k-points
            # field4 and field5 have shape (n_kpts, size, size)
            avg_field4 = np.mean(field4, axis=0)
            avg_field5 = np.mean(field5, axis=0)
            
            # Normalize fields
            if np.max(avg_field4) > 0:
                avg_field4 = avg_field4 / np.max(avg_field4)
            if np.max(avg_field5) > 0:
                avg_field5 = avg_field5 / np.max(avg_field5)
            
            # Compute guidance: where to add/remove dielectric
            # Add dielectric where band 4 has high intensity (to push band 4 down)
            # Remove dielectric where band 5 has high intensity (to push band 5 up)
            guidance = avg_field4 - avg_field5
            
            # Get the top-left quadrant of the guidance
            tl_guidance = guidance[:half, :half]
            
            # Find the pixel in the top-left quadrant with the most positive guidance
            # (where we should add dielectric) or most negative (where we should remove)
            tl_best = best_arr[:half, :half].copy()
            
            # Try flipping the pixel with the highest guidance score
            # For positive guidance: flip 0->1 (add dielectric)
            # For negative guidance: flip 1->0 (remove dielectric)
            
            # Get all pixels in the top-left quadrant
            pixels = [(i, j) for i in range(half) for j in range(half)]
            
            # Score each pixel based on guidance and current value
            best_pixel = None
            best_improvement = 0
            
            for (i, j) in pixels:
                g = tl_guidance[i, j]
                current_val = tl_best[i, j]
                
                # If guidance is positive and pixel is 0, flipping to 1 should help
                # If guidance is negative and pixel is 1, flipping to 0 should help
                if g > 0 and current_val == 0:
                    improvement = g
                elif g < 0 and current_val == 1:
                    improvement = -g
                else:
                    improvement = 0
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_pixel = (i, j)
            
            # If no good flip found, try the best pixel anyway or break
            if best_pixel is None or best_improvement < 0.05:
                # Try a random flip instead
                np.random.seed(iteration)
                best_pixel = pixels[np.random.randint(len(pixels))]
                best_improvement = 0.01
            
            i, j = best_pixel
            
            # Flip the pixel
            tl_new = tl_best.copy()
            tl_new[i, j] = 1 - tl_new[i, j]
            
            # Reconstruct full cell
            cell_new = np.zeros((size, size), dtype=int)
            cell_new[:half, :half] = tl_new
            cell_new[:half, half:] = tl_new[:, ::-1]
            cell_new[half:, :half] = tl_new[::-1, :]
            cell_new[half:, half:] = tl_new[::-1, ::-1]
            
            # Score the new cell
            new_score = score_cell(cell_new.tolist())
            
            if new_score > best_score:
                best_score = new_score
                best_cell = cell_new.tolist()
                best_arr = cell_new
                tl_best = tl_new
            else:
                # If the flip didn't help, try a few other candidates
                # Get the next few best pixels
                pixel_scores = []
                for (pi, pj) in pixels:
                    g = tl_guidance[pi, pj]
                    current_val = tl_best[pi, pj]
                    if g > 0 and current_val == 0:
                        s = g
                    elif g < 0 and current_val == 1:
                        s = -g
                    else:
                        s = 0
                    pixel_scores.append((s, pi, pj))
                
                pixel_scores.sort(reverse=True)
                
                tried = 0
                for s, pi, pj in pixel_scores:
                    if tried >= 5 or time_left() < 5:
                        break
                    if (pi, pj) == best_pixel:
                        continue
                    
                    tl_try = tl_best.copy()
                    tl_try[pi, pj] = 1 - tl_try[pi, pj]
                    
                    cell_try = np.zeros((size, size), dtype=int)
                    cell_try[:half, :half] = tl_try
                    cell_try[:half, half:] = tl_try[:, ::-1]
                    cell_try[half:, :half] = tl_try[::-1, :]
                    cell_try[half:, half:] = tl_try[::-1, ::-1]
                    
                    try_score = score_cell(cell_try.tolist())
                    if try_score > best_score:
                        best_score = try_score
                        best_cell = cell_try.tolist()
                        best_arr = cell_try
                        tl_best = tl_try
                        break
                    tried += 1
    
    # Phase 3: Final local refinement
    if best_cell is not None and time_left() > 10:
        best_arr = np.array(best_cell, dtype=int)
        half = size // 2
        tl_best = best_arr[:half, :half].copy()
        
        # Try flipping boundary pixels
        candidates = []
        for i in range(half):
            for j in range(half):
                neighbors = []
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < half and 0 <= nj < half:
                        neighbors.append(tl_best[ni, nj])
                if neighbors and tl_best[i, j] != np.mean(neighbors):
                    candidates.append((i, j))
        
        np.random.seed(123)
        np.random.shuffle(candidates)
        
        for (i, j) in candidates[:50]:
            if time_left() < 3:
                break
            tl_new = tl_best.copy()
            tl_new[i, j] = 1 - tl_new[i, j]
            
            cell_new = np.zeros((size, size), dtype=int)
            cell_new[:half, :half] = tl_new
            cell_new[:half, half:] = tl_new[:, ::-1]
            cell_new[half:, :half] = tl_new[::-1, :]
            cell_new[half:, half:] = tl_new[::-1, ::-1]
            
            score = score_cell(cell_new.tolist())
            if score > best_score:
                best_score = score
                best_cell = cell_new.tolist()
                tl_best = tl_new
    
    if best_cell is None:
        best_cell = make_four_quadrant_cell(0.12, 2.0)
    
    best_arr = np.array(best_cell, dtype=int)
    best_arr = enforce_d4_symmetry(best_arr)
    
    return best_arr.tolist()

# EVOLVE-BLOCK-END
