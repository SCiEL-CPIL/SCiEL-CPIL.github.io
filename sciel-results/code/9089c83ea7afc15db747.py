"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s limit

    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry: row/col reversal and transpose invariance."""
        final_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                vals = [
                    cell[i][j],
                    cell[size-1-i][j],
                    cell[i][size-1-j],
                    cell[size-1-i][size-1-j],
                    cell[j][i],
                    cell[size-1-j][size-1-i],
                    cell[j][size-1-i],
                    cell[size-1-j][i],
                ]
                avg = sum(vals) / len(vals)
                final_cell[i][j] = 1 if avg >= 0.5 else 0
        return final_cell

    def compute_fourier_coefficients(cell, max_order):
        """Compute Fourier coefficients of 1/epsilon using FFT for speed."""
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = {}
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                m_idx = m % size
                n_idx = n % size
                coeffs[(m, n)] = fft_2d[m_idx, n_idx] / (size * size)
        return coeffs

    def gap_ratio(cell, g_range=4, n_kpoints=5):
        """
        Approximate the gap ratio between bands 4 and 5 using plane-wave expansion.
        """
        max_order = g_range
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        if n_kpoints == 3:
            k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
        elif n_kpoints == 5:
            k_points = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.25, 0.25), (0.5, 0.5)]
        else:
            k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            H = np.zeros((N, N), dtype=complex)
            for i, Gp in enumerate(g_vectors):
                kx_total = kx + Gp[0]
                ky_total = ky + Gp[1]
                k_sq = kx_total**2 + ky_total**2
                for j, G in enumerate(g_vectors):
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    if (m_diff, n_diff) in coeffs:
                        eps_inv_G = coeffs[(m_diff, n_diff)]
                    else:
                        eps_inv_G = 0.0
                    H[i, j] = k_sq * eps_inv_G
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
        
        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    def fast_gap_ratio(cell, g_range=3, n_kpoints=3):
        """Very fast approximate gap ratio for local search."""
        return gap_ratio(cell, g_range=g_range, n_kpoints=n_kpoints)

    def get_fundamental_domain_pixels(size):
        """Get pixels in the fundamental domain of D4 symmetry."""
        pixels = []
        half = size // 2
        for i in range(half):
            for j in range(i, half):
                pixels.append((i, j))
        return pixels

    def generate_gabor_mask(freq_x, freq_y, sigma, phase, threshold):
        """
        Generate a D4-symmetric binary mask from a Gabor filter response.
        The Gabor filter is a 2D sinusoid modulated by a Gaussian envelope.
        """
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                
                # Gabor response: sinusoid modulated by Gaussian
                gabor = math.cos(2 * math.pi * (freq_x * x + freq_y * y) + phase) * \
                        math.exp(-(x**2 + y**2) / (2 * sigma**2))
                
                # For D4 symmetry, we need to average over all 8 symmetry operations
                # But since we're generating from a function, we just check if the
                # response is above threshold at this pixel
                if gabor >= threshold:
                    cell[i][j] = 1
        return enforce_d4_symmetry(cell)

    def generate_multi_scale_pattern(scales, freqs, sigmas, thresholds):
        """
        Generate a pattern by combining multiple Gabor filters at different scales.
        """
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        
        total_response = np.zeros((size, size))
        
        for freq, sigma, threshold in zip(freqs, sigmas, thresholds):
            for i in range(size):
                for j in range(size):
                    x = (j + 0.5) - cx
                    y = (i + 0.5) - cy
                    gabor = math.cos(2 * math.pi * freq * (x + y)) * \
                            math.exp(-(x**2 + y**2) / (2 * sigma**2))
                    total_response[i, j] += gabor
        
        # Threshold the combined response
        for i in range(size):
            for j in range(size):
                if total_response[i, j] > 0:
                    cell[i][j] = 1
        
        return enforce_d4_symmetry(cell)

    def generate_d4_symmetric_pattern_from_seed(seed):
        """
        Generate a D4-symmetric pattern using a deterministic seed-based approach.
        This explores a wider design space than parametric shapes.
        """
        np.random.seed(seed)
        
        # Generate a random pattern in the fundamental domain
        half = size // 2
        fd_pattern = np.zeros((half, half), dtype=int)
        
        # Fill the fundamental domain with a mix of structured and random elements
        for i in range(half):
            for j in range(i, half):
                # Use a combination of distance-based and random criteria
                dist_from_center = math.sqrt((i - half/2)**2 + (j - half/2)**2)
                norm_dist = dist_from_center / (half / 2)
                
                # Create a pattern that has some structure but is also random
                if norm_dist < 0.3:
                    fd_pattern[i, j] = 1 if np.random.random() > 0.4 else 0
                elif norm_dist < 0.6:
                    fd_pattern[i, j] = 1 if np.random.random() > 0.6 else 0
                else:
                    fd_pattern[i, j] = 1 if np.random.random() > 0.7 else 0
        
        # Expand to full cell with D4 symmetry
        cell = [[0] * size for _ in range(size)]
        for i in range(half):
            for j in range(i, half):
                val = fd_pattern[i, j]
                # Apply D4 symmetry
                cell[i][j] = val
                cell[size-1-i][j] = val
                cell[i][size-1-j] = val
                cell[size-1-i][size-1-j] = val
                cell[j][i] = val
                cell[size-1-j][size-1-i] = val
                cell[j][size-1-i] = val
                cell[size-1-j][i] = val
        
        return cell

    def generate_2x2_supercell(rod_radius_frac):
        """2x2 supercell arrangement of rods within the 32x32 unit cell."""
        cell = [[0] * size for _ in range(size)]
        r_px = rod_radius_frac * size
        offset = size / 4.0
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell

    def generate_2x2_supercell_ring(inner_frac, outer_frac):
        """2x2 supercell arrangement of rings within the 32x32 unit cell."""
        cell = [[0] * size for _ in range(size)]
        r_in = inner_frac * size
        r_out = outer_frac * size
        offset = size / 4.0
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center - offset),
            (cx_center + offset, cy_center - offset),
            (cx_center - offset, cy_center + offset),
            (cx_center + offset, cy_center + offset),
        ]
        for i in range(size):
            for j in range(size):
                x = (j + 0.5)
                y = (i + 0.5)
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    r2 = dx * dx + dy * dy
                    if r_in * r_in <= r2 < r_out * r_out:
                        cell[i][j] = 1
                        break
        return cell

    def generate_single_rod(radius_frac):
        """Central circular rod."""
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_px * r_px:
                    cell[i][j] = 1
        return cell

    def generate_ring(inner_frac, outer_frac):
        """Circular ring (dielectric annulus)."""
        cell = [[0] * size for _ in range(size)]
        r_in = inner_frac * size
        r_out = outer_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                r2 = x * x + y * y
                if r_in * r_in <= r2 < r_out * r_out:
                    cell[i][j] = 1
        return cell

    # Generate candidate designs
    candidates = []
    
    # 2x2 supercell arrangements (key for higher-order gaps)
    for r in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]:
        candidates.append(generate_2x2_supercell(r))
    
    # 2x2 supercell with rings
    for inner, outer in [(0.05, 0.15), (0.08, 0.18), (0.10, 0.20), (0.06, 0.16), 
                         (0.04, 0.12), (0.07, 0.14), (0.09, 0.17), (0.11, 0.19)]:
        candidates.append(generate_2x2_supercell_ring(inner, outer))
    
    # Single rods
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        candidates.append(generate_single_rod(r))
    
    # Rings
    for inner, outer in [(0.15, 0.30), (0.20, 0.35), (0.25, 0.40)]:
        candidates.append(generate_ring(inner, outer))
    
    # Gabor-based patterns with different frequencies and scales
    gabor_configs = [
        (0.5, 0.5, 4.0, 0.0, 0.0),
        (0.5, 0.5, 6.0, 0.0, 0.0),
        (0.75, 0.75, 4.0, 0.0, 0.0),
        (0.75, 0.75, 6.0, 0.0, 0.0),
        (1.0, 1.0, 4.0, 0.0, 0.0),
        (1.0, 1.0, 6.0, 0.0, 0.0),
        (0.5, 0.0, 5.0, 0.0, 0.0),
        (0.0, 0.5, 5.0, 0.0, 0.0),
    ]
    for freq_x, freq_y, sigma, phase, threshold in gabor_configs:
        candidates.append(generate_gabor_mask(freq_x, freq_y, sigma, phase, threshold))
    
    # Multi-scale patterns
    multi_scale_configs = [
        ([0.5, 1.0], [4.0, 6.0], [0.0, 0.0]),
        ([0.5, 0.75], [5.0, 7.0], [0.0, 0.0]),
        ([0.75, 1.25], [4.0, 5.0], [0.0, 0.0]),
    ]
    for freqs, sigmas, thresholds in multi_scale_configs:
        candidates.append(generate_multi_scale_pattern(None, freqs, sigmas, thresholds))
    
    # Random D4-symmetric patterns with various seeds
    for seed in range(20):
        candidates.append(generate_d4_symmetric_pattern_from_seed(seed))
    
    # Evaluate all candidates
    scored_candidates = []
    for cell in candidates:
        if time.time() - start_time > deadline - 30:
            break
        cell_sym = enforce_d4_symmetry(cell)
        score = gap_ratio(cell_sym, g_range=4, n_kpoints=5)
        scored_candidates.append((score, cell_sym))
    
    # Sort by score, take top candidates
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score = -1.0
    best_cell = candidates[0]
    
    # Evaluate top 15 with more accurate method
    top_n = min(15, len(scored_candidates))
    for score, cell_sym in scored_candidates[:top_n]:
        if time.time() - start_time > deadline - 20:
            break
        score_acc = gap_ratio(cell_sym, g_range=5, n_kpoints=5)
        if score_acc > best_score:
            best_score = score_acc
            best_cell = cell_sym
    
    # Local search: perturb pixels around best design
    if time.time() - start_time < deadline - 15:
        base = best_cell
        improved = True
        max_iterations = 30
        iteration = 0
        
        fd_pixels = get_fundamental_domain_pixels(size)
        
        while improved and time.time() - start_time < deadline - 10 and iteration < max_iterations:
            iteration += 1
            improved = False
            best_local = best_score
            best_local_cell = base
            
            # Try flipping individual pixels in fundamental domain only
            for fi, fj in fd_pixels:
                if time.time() - start_time > deadline - 5:
                    break
                
                cell_trial = [row[:] for row in base]
                cell_trial[fi][fj] = 1 - base[fi][fj]
                cell_sym = enforce_d4_symmetry(cell_trial)
                if cell_sym != base:
                    score = fast_gap_ratio(cell_sym, g_range=3, n_kpoints=3)
                    if score > best_local + 1e-6:
                        score_acc = gap_ratio(cell_sym, g_range=4, n_kpoints=5)
                        if score_acc > best_local + 1e-6:
                            best_local = score_acc
                            best_local_cell = cell_sym
                            improved = True
            
            if improved:
                best_score = best_local
                best_cell = best_local_cell
                base = best_cell
    
    # Final polish
    if time.time() - start_time < deadline - 5:
        base = best_cell
        fd_pixels = get_fundamental_domain_pixels(size)
        
        for fi, fj in fd_pixels:
            if time.time() - start_time > deadline - 2:
                break
            cell_trial = [row[:] for row in base]
            cell_trial[fi][fj] = 1 - base[fi][fj]
            cell_sym = enforce_d4_symmetry(cell_trial)
            if cell_sym != base:
                score = gap_ratio(cell_sym, g_range=5, n_kpoints=5)
                if score > best_score + 1e-6:
                    best_score = score
                    best_cell = cell_sym
    
    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell

# EVOLVE-BLOCK-END
