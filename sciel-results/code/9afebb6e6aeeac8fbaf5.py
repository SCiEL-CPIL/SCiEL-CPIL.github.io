"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5 using spectral parameterization.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 200s limit

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

    def approximate_gap_ratio(cell, max_order=6, g_range=4, beta=20.0):
        """
        Approximate the gap ratio between bands 4 and 5 using plane-wave expansion.
        Uses soft-min/soft-max surrogate for smoother optimization landscape.
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        # G vectors in the basis
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        # More k-points in irreducible BZ for better accuracy
        k_points = [
            (0.0, 0.0),
            (0.125, 0.0),
            (0.25, 0.0),
            (0.375, 0.0),
            (0.5, 0.0),
            (0.125, 0.125),
            (0.25, 0.125),
            (0.375, 0.125),
            (0.25, 0.25),
            (0.375, 0.25),
            (0.5, 0.25),
            (0.5, 0.5),
        ]
        
        omega_4_vals = []
        omega_5_vals = []
        
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
                omega_4_vals.append(eigvals[3])
                omega_5_vals.append(eigvals[4])
        
        if not omega_4_vals or not omega_5_vals:
            return -1.0
        
        omega_4_arr = np.array(omega_4_vals)
        omega_5_arr = np.array(omega_5_vals)
        
        # Use soft-max for max(omega_4) and soft-min for min(omega_5)
        # soft_max(x, beta) = (1/beta) * log(sum(exp(beta*x)))
        # soft_min(x, beta) = -(1/beta) * log(sum(exp(-beta*x)))
        max_omega_4 = np.max(omega_4_arr)
        min_omega_5 = np.min(omega_5_arr)
        
        # Also compute soft versions for smoother gradient
        beta_val = beta
        exp_4 = np.exp(beta_val * (omega_4_arr - max_omega_4))
        soft_max_4 = max_omega_4 + (1.0/beta_val) * np.log(np.sum(exp_4))
        
        exp_5 = np.exp(-beta_val * (omega_5_arr - min_omega_5))
        soft_min_5 = min_omega_5 - (1.0/beta_val) * np.log(np.sum(exp_5))
        
        gap = soft_min_5 - soft_max_4
        midgap = (soft_min_5 + soft_max_4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    def get_d4_symmetric_modes(max_order):
        """
        Get the set of (m, n) mode indices that are D4-symmetric.
        """
        modes = set()
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                transforms = [
                    (m, n), (-n, m), (-m, -n), (n, -m),
                    (n, m), (-m, n), (m, -n), (-n, -m)
                ]
                canon = min(transforms)
                if (m, n) == canon:
                    modes.add((m, n))
        return sorted(modes)

    def spectral_to_cell_spectral(modes, amplitudes, phase=0, size=32):
        """
        Convert spectral parameters (amplitudes for D4-symmetric modes) to a 0/1 cell.
        """
        inv_eps_fft = np.zeros((size, size), dtype=complex)
        
        for idx, (m, n) in enumerate(modes):
            amp = amplitudes[idx]
            transforms = [
                (m, n), (-n, m), (-m, -n), (n, -m),
                (n, m), (-m, n), (m, -n), (-n, -m)
            ]
            unique_transforms = list(set(transforms))
            for (mm, nn) in unique_transforms:
                m_idx = mm % size
                n_idx = nn % size
                inv_eps_fft[m_idx, n_idx] += amp
        
        inv_eps_spatial = np.fft.ifft2(inv_eps_fft).real
        
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if inv_eps_spatial[i, j] < 0.5:
                    cell[i][j] = 1
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

    def generate_2x2_supercell(rod_radius_frac):
        """
        Generate a 2x2 supercell arrangement of rods within the 32x32 unit cell.
        """
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

    def generate_four_rods(radius_frac, offset_frac):
        """Four rods at symmetric positions."""
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        offset = offset_frac * size
        cx_center = size / 2.0
        cy_center = size / 2.0
        centers = [
            (cx_center - offset, cy_center),
            (cx_center + offset, cy_center),
            (cx_center, cy_center - offset),
            (cx_center, cy_center + offset),
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

    def generate_cross(arm_width_frac, length_frac):
        """Cross shape with D4 symmetry."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        half_w = arm_width_frac * size / 2.0
        half_l = length_frac * size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                # Horizontal arm: |y| < half_w and |x| < half_l
                # Vertical arm: |x| < half_w and |y| < half_l
                if (abs(y) < half_w and abs(x) < half_l) or (abs(x) < half_w and abs(y) < half_l):
                    cell[i][j] = 1
        return cell

    def generate_2x2_supercell_ring(inner_frac, outer_frac):
        """2x2 supercell of rings."""
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

    # Get D4-symmetric modes up to order 4
    max_order = 4
    d4_modes = get_d4_symmetric_modes(max_order)
    num_modes = len(d4_modes)
    
    # Initial candidates from geometric generators
    candidates = []
    for r in [0.15, 0.20, 0.25, 0.30]:
        candidates.append(generate_single_rod(r))
    for r in [0.08, 0.12, 0.16, 0.20]:
        candidates.append(generate_2x2_supercell(r))
    for inner, outer in [(0.15, 0.30), (0.20, 0.40)]:
        candidates.append(generate_ring(inner, outer))
    for r in [0.06, 0.08, 0.10, 0.12]:
        for off in [0.15, 0.20, 0.25]:
            candidates.append(generate_four_rods(r, off))
    for w in [0.10, 0.15, 0.20]:
        for l in [0.25, 0.35, 0.45]:
            candidates.append(generate_cross(w, l))
    for inner, outer in [(0.05, 0.15), (0.08, 0.18)]:
        candidates.append(generate_2x2_supercell_ring(inner, outer))
    
    # Evaluate initial candidates
    best_score = -1.0
    best_cell = candidates[0]
    
    for cell in candidates:
        if time.time() - start_time > deadline:
            break
        cell_sym = enforce_d4_symmetry(cell)
        score = approximate_gap_ratio(cell_sym)
        if score > best_score:
            best_score = score
            best_cell = cell_sym
    
    # Spectral optimization: start from the best cell, extract its D4-symmetric Fourier coefficients,
    # then optimize amplitudes
    if time.time() - start_time < deadline - 20:
        # Extract spectral coefficients from best_cell
        eps_arr = np.array([[eps_high if best_cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps_arr = 1.0 / eps_arr
        fft_coeffs = np.fft.fft2(inv_eps_arr) / (size * size)
        
        # Get amplitudes for D4-symmetric modes
        initial_amplitudes = []
        for (m, n) in d4_modes:
            m_idx = m % size
            n_idx = n % size
            amp = fft_coeffs[m_idx, n_idx].real
            initial_amplitudes.append(amp)
        
        # Optimize amplitudes using coordinate descent
        amplitudes = initial_amplitudes[:]
        best_amp_score = best_score
        best_amp_cell = best_cell
        
        step_sizes = [0.05, 0.02, 0.01, 0.005]
        for step in step_sizes:
            if time.time() - start_time > deadline - 5:
                break
            
            improved = True
            while improved and time.time() - start_time < deadline - 5:
                improved = False
                for idx in range(num_modes):
                    # Try increasing and decreasing amplitude
                    for delta in [step, -step]:
                        trial_amplitudes = amplitudes[:]
                        trial_amplitudes[idx] += delta
                        
                        cell_trial = spectral_to_cell_spectral(d4_modes, trial_amplitudes, size=size)
                        cell_sym = enforce_d4_symmetry(cell_trial)
                        
                        if cell_sym != best_amp_cell:
                            score = approximate_gap_ratio(cell_sym)
                            if score > best_amp_score + 1e-6:
                                best_amp_score = score
                                best_amp_cell = cell_sym
                                amplitudes = trial_amplitudes
                                improved = True
        
        if best_amp_score > best_score:
            best_score = best_amp_score
            best_cell = best_amp_cell
    
    # Local pixel perturbation around best design
    if time.time() - start_time < deadline - 10:
        base = best_cell
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            best_local = best_score
            best_local_cell = base
            
            # Try flipping individual pixels with D4 symmetry
            for flip_i in range(0, size, 2):
                for flip_j in range(0, size, 2):
                    if time.time() - start_time > deadline - 3:
                        break
                    
                    cell_trial = [row[:] for row in base]
                    cell_trial[flip_i][flip_j] = 1 - base[flip_i][flip_j]
                    cell_sym = enforce_d4_symmetry(cell_trial)
                    if cell_sym != base:
                        score = approximate_gap_ratio(cell_sym)
                        if score > best_local + 1e-6:
                            best_local = score
                            best_local_cell = cell_sym
                            improved = True
                
                if time.time() - start_time > deadline - 3:
                    break
            
            if improved:
                best_score = best_local
                best_cell = best_local_cell
                base = best_cell
    
    # Final high-precision polish pass: try all single-pixel flips with tighter tolerance
    if time.time() - start_time < deadline - 3:
        base = best_cell
        for flip_i in range(size):
            for flip_j in range(size):
                if time.time() - start_time > deadline - 1:
                    break
                
                cell_trial = [row[:] for row in base]
                cell_trial[flip_i][flip_j] = 1 - base[flip_i][flip_j]
                cell_sym = enforce_d4_symmetry(cell_trial)
                if cell_sym != base:
                    score = approximate_gap_ratio(cell_sym)
                    if score > best_score + 1e-8:
                        best_score = score
                        base = cell_sym
        
        best_cell = base
    
    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell

# EVOLVE-BLOCK-END
