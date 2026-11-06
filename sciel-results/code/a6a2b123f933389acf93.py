"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = 210.0  # safety margin before 230s kill
    
    def time_ok():
        return time.time() - start_time < deadline
    
    # Plane wave setup: 17x17 = 289 plane waves
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289
    
    # K-points in irreducible BZ (30 points)
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
    n_kpts = len(k_points)
    
    # Precompute (k+G)^2
    k2_all = np.zeros((n_kpts, n_pw_total))
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
    valid_diff = diff_idx[valid_rows, valid_cols]
    
    # Fast evaluation with configurable k-point subset
    def compute_band_ratio(coeffs, k_indices=None):
        """Compute gap-midgap ratio. k_indices: list of k-point indices to use."""
        if k_indices is None:
            k_indices = range(n_kpts)
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_indices:
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[valid_diff]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0
    
    # Coarse k-point subsets for fast search
    k_coarse = [0, 4, 9, 14, 19, 24, 29]  # 7 points
    k_medium = [0, 2, 4, 6, 8, 9, 12, 14, 16, 18, 19, 22, 24, 26, 28, 29]  # 16 points
    
    def cell_to_coeffs(cell):
        """Compute Fourier coefficients of 1/eps from a 0/1 cell."""
        cell = np.array(cell, dtype=float)
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs
    
    def coeffs_to_cell(coeffs):
        """Convert Fourier coefficients back to a 0/1 cell via inverse FFT + threshold."""
        fft_full = np.zeros((size, size), dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_full[gx % size, gy % size] = coeffs[idx] * (size * size)
        inv_eps_recon = np.real(np.fft.ifft2(fft_full))
        midpoint = 0.5 * (1.0/eps_high + 1.0/eps_low)
        cell = (inv_eps_recon < midpoint).astype(int)
        return cell
    
    def enforce_symmetry(cell):
        """Enforce square symmetry by averaging over the group."""
        cell = np.array(cell, dtype=int)
        cell_rot = cell[::-1, ::-1].copy()
        cell_hflip = cell[:, ::-1].copy()
        cell_vflip = cell[::-1, :].copy()
        result = (cell + cell_rot + cell_hflip + cell_vflip)
        result = (result >= 2).astype(int)
        return result
    
    def generate_moire_pattern(freq1, freq2, angle1, angle2, amp1, amp2, threshold):
        """
        Generate a moiré pattern from two counter-rotating square-wave gratings.
        The interference creates complex high-symmetry microstructures.
        """
        x = np.linspace(-0.5, 0.5, size, endpoint=False)
        y = np.linspace(-0.5, 0.5, size, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Two counter-rotating square waves
        wave1 = np.sign(np.sin(2 * np.pi * freq1 * (X * np.cos(angle1) + Y * np.sin(angle1))))
        wave2 = np.sign(np.sin(2 * np.pi * freq2 * (X * np.cos(angle2) + Y * np.sin(angle2))))
        
        # Combine with amplitudes
        combined = amp1 * wave1 + amp2 * wave2
        
        # Threshold to binary
        cell = (combined > threshold).astype(int)
        return enforce_symmetry(cell)
    
    def generate_fractal_disk(r_base, a4, a8, phase4, phase8):
        """Generate a fractal disk with radial harmonics m=4 and m=8."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x * x + y * y)
                if r < 1e-10:
                    theta = 0.0
                else:
                    theta = np.arctan2(y, x)
                r_boundary = r_base + a4 * np.cos(4 * theta + phase4) + a8 * np.cos(8 * theta + phase8)
                if r < r_boundary:
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    def generate_rod(radius):
        """Single centered rod."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    def generate_four_rods(radius, offset):
        """Four rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        dx = x - sx * offset
                        dy = y - sy * offset
                        if dx * dx + dy * dy < radius * radius:
                            cell[i, j] = 1
        return enforce_symmetry(cell)
    
    def generate_ring(inner_r, outer_r):
        """Dielectric ring."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x * x + y * y
                if inner_r * inner_r < r2 < outer_r * outer_r:
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    # Phase 1: Generate diverse initial candidates including moiré patterns
    candidates = []
    
    # Moiré patterns with various frequencies and angles
    for freq1 in [2, 3, 4, 5]:
        for freq2 in [2, 3, 4, 5, 6]:
            if freq1 == freq2:
                continue
            for angle1 in [0, np.pi/4, np.pi/2]:
                for angle2 in [np.pi/4, np.pi/2, 3*np.pi/4]:
                    if abs(angle1 - angle2) < 0.1:
                        continue
                    for amp1 in [0.5, 0.7, 1.0]:
                        for amp2 in [0.3, 0.5, 0.7]:
                            for threshold in [-0.2, 0.0, 0.2]:
                                candidates.append(generate_moire_pattern(
                                    freq1, freq2, angle1, angle2, amp1, amp2, threshold
                                ))
    
    # Fractal disks
    for r_base in [0.20, 0.25, 0.30]:
        for a4 in [0.0, 0.02, 0.04, 0.06]:
            for a8 in [0.0, 0.01, 0.02, 0.03]:
                for phase4 in [0.0, np.pi/4, np.pi/2]:
                    for phase8 in [0.0, np.pi/4, np.pi/2]:
                        candidates.append(generate_fractal_disk(r_base, a4, a8, phase4, phase8))
    
    # Simple primitives
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        candidates.append(generate_rod(r))
    for r in [0.06, 0.08, 0.10, 0.12, 0.14]:
        for off in [0.20, 0.22, 0.25, 0.28, 0.30]:
            candidates.append(generate_four_rods(r, off))
    for ir, orr in [(0.08, 0.18), (0.10, 0.20), (0.12, 0.22), (0.14, 0.25)]:
        candidates.append(generate_ring(ir, orr))
    
    # Evaluate all candidates with coarse evaluator
    best_cell = None
    best_score = -1
    
    for cell in candidates:
        if not time_ok():
            break
        fill = np.sum(cell) / (size * size)
        if fill < 0.1 or fill > 0.7:
            continue
        coeffs = cell_to_coeffs(cell)
        score = compute_band_ratio(coeffs, k_coarse)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Phase 2: Fourier-domain optimization
    if best_cell is not None and time_ok():
        best_cell = np.array(best_cell, dtype=int)
        best_coeffs = cell_to_coeffs(best_cell)
        
        n_opt_steps = 0
        max_opt_steps = 400
        step_scale = 0.005
        
        perturbable_indices = list(range(1, n_pw_total))
        
        while n_opt_steps < max_opt_steps and time_ok():
            n_opt_steps += 1
            
            n_perturb = random.choice([2, 3, 5, 8])
            idxs = random.sample(perturbable_indices, min(n_perturb, len(perturbable_indices)))
            
            new_coeffs = best_coeffs.copy()
            for idx in idxs:
                delta_mag = step_scale * random.uniform(0.5, 2.0)
                delta_angle = random.uniform(0, 2 * np.pi)
                new_coeffs[idx] += delta_mag * np.exp(1j * delta_angle)
            
            new_cell = coeffs_to_cell(new_coeffs)
            new_cell = enforce_symmetry(new_cell)
            
            fill = np.sum(new_cell) / (size * size)
            if fill < 0.05 or fill > 0.8:
                continue
            
            new_coeffs = cell_to_coeffs(new_cell)
            score = compute_band_ratio(new_coeffs, k_coarse)
            
            if score > best_score:
                best_score = score
                best_cell = new_cell
                best_coeffs = new_coeffs
                step_scale = max(0.001, step_scale * 0.85)
            else:
                if random.random() < 0.1:
                    step_scale = min(0.02, step_scale * 1.3)
        
        # Phase 2b: Refine with medium k-point set
        if time_ok():
            refine_steps = 0
            max_refine_steps = 200
            refine_scale = 0.003
            
            while refine_steps < max_refine_steps and time_ok():
                refine_steps += 1
                n_perturb = random.choice([1, 2, 3])
                idxs = random.sample(perturbable_indices, min(n_perturb, len(perturbable_indices)))
                
                new_coeffs = best_coeffs.copy()
                for idx in idxs:
                    delta_mag = refine_scale * random.uniform(0.5, 1.5)
                    delta_angle = random.uniform(0, 2 * np.pi)
                    new_coeffs[idx] += delta_mag * np.exp(1j * delta_angle)
                
                new_cell = coeffs_to_cell(new_coeffs)
                new_cell = enforce_symmetry(new_cell)
                
                fill = np.sum(new_cell) / (size * size)
                if fill < 0.05 or fill > 0.8:
                    continue
                
                new_coeffs = cell_to_coeffs(new_cell)
                score = compute_band_ratio(new_coeffs, k_medium)
                
                if score > best_score:
                    best_score = score
                    best_cell = new_cell
                    best_coeffs = new_coeffs
                    refine_scale = max(0.0005, refine_scale * 0.85)
                else:
                    if random.random() < 0.15:
                        refine_scale = min(0.01, refine_scale * 1.2)
    
    # Phase 3: Final evaluation with full 30 k-points
    if best_cell is not None and time_ok():
        final_coeffs = cell_to_coeffs(best_cell)
        final_score = compute_band_ratio(final_coeffs, None)
    
    # Convert to nested list
    result = best_cell.tolist()
    return result
# EVOLVE-BLOCK-END
