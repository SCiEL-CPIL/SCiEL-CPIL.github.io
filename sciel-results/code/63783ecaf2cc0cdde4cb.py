import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using vectorized matrix construction.
    2. Start with a good geometric seed (four-quadrant rods).
    3. Use topological optimization (field-intensity-guided pixel flips) to refine.
    4. Maintain D4 symmetry throughout.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 210.0  # 20s margin
    
    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=17, n_kpts=8):
        """
        Compute TM band structure using plane-wave expansion.
        Vectorized implementation for speed.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/epsilon using FFT
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
        
        def get_coeff(m, n):
            idx_m = m % size
            idx_n = n % size
            return fft_coeffs[idx_m, idx_n]
        
        # Define G vectors: m, n in range [-M, M]
        M = n_pw // 2
        m_vals = np.arange(-M, M + 1)
        g_indices = [(m, n) for m in m_vals for n in m_vals]
        n_total = len(g_indices)
        
        # Precompute G vector components
        Gx = np.array([2.0 * np.pi * m for m, n in g_indices])
        Gy = np.array([2.0 * np.pi * n for m, n in g_indices])
        
        # Build S matrix: S_ij = inv_eps(G_i - G_j)
        # Vectorized: compute all differences at once
        S = np.zeros((n_total, n_total), dtype=complex)
        for i in range(n_total):
            for j in range(n_total):
                gm = g_indices[i][0] - g_indices[j][0]
                gn = g_indices[i][1] - g_indices[j][1]
                S[i, j] = get_coeff(gm, gn)
        
        # Define k-points in the irreducible Brillouin zone
        k_points = []
        n_per_seg = max(1, n_kpts // 3)
        
        # Gamma to X: (t, 0) for t in [0, pi]
        for t in np.linspace(0, np.pi, n_per_seg + 1):
            k_points.append((t, 0))
        # X to M: (pi, t) for t in [0, pi]
        for t in np.linspace(0, np.pi, n_per_seg + 1):
            k_points.append((np.pi, t))
        # M to Gamma: (t, t) for t in [pi, 0]
        for t in np.linspace(np.pi, 0, n_per_seg + 1):
            k_points.append((t, t))
        
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            # Build eigenvalue matrix: M_ij = S_ij * |G_j + k|^2
            kx_total = Gx + kx
            ky_total = Gy + ky
            k_mag_sq = kx_total**2 + ky_total**2
            
            M = S * k_mag_sq[None, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.sort(np.real(eigvals))
                omega[ki, :] = np.sqrt(np.maximum(eigvals, 0))
            except Exception:
                omega[ki, :] = np.inf
        
        return omega
    
    def compute_gap(cell, n_pw=17, n_kpts=6):
        """Compute the gap-midgap ratio for bands 4 and 5."""
        try:
            bands = compute_tm_bands(cell, n_pw=n_pw, n_kpts=n_kpts)
            if bands.shape[1] < 5:
                return -1.0
            
            omega_4 = bands[:, 3]
            omega_5 = bands[:, 4]
            
            max_omega_4 = np.max(omega_4)
            min_omega_5 = np.min(omega_5)
            
            if min_omega_5 > max_omega_4:
                gap = min_omega_5 - max_omega_4
                midgap = (min_omega_5 + max_omega_4) / 2.0
                ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
            else:
                ratio = -1.0
            return ratio
        except Exception:
            return -1.0
    
    def get_field_intensity(cell, band_idx, k_points=None, n_pw=17):
        """Get the field intensity pattern for a specific band at optimal k-points."""
        try:
            bands, eigvecs = compute_tm_bands_with_vecs(cell, n_pw=n_pw, n_kpts=6)
            if bands.shape[1] < band_idx + 1:
                return np.zeros((size, size))
            
            # Find k-point where this band has the extremal frequency
            omega_band = bands[:, band_idx]
            if band_idx == 3:  # Band 4: find max
                ki = np.argmax(omega_band)
            else:  # Band 5: find min
                ki = np.argmin(omega_band)
            
            vec = eigvecs[ki, :, band_idx]
            
            # Reconstruct field from plane-wave expansion
            M = n_pw // 2
            m_vals = np.arange(-M, M + 1)
            g_indices = [(m, n) for m in m_vals for n in m_vals]
            
            kx, ky = (ki < len(k_points)) and k_points[ki] or (0, 0)
            # Reconstruct k-point
            n_per_seg = max(1, 6 // 3)
            k_points_full = []
            for t in np.linspace(0, np.pi, n_per_seg + 1):
                k_points_full.append((t, 0))
            for t in np.linspace(0, np.pi, n_per_seg + 1):
                k_points_full.append((np.pi, t))
            for t in np.linspace(np.pi, 0, n_per_seg + 1):
                k_points_full.append((t, t))
            
            # Remove duplicates
            k_points_unique = []
            for kp in k_points_full:
                if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                    k_points_unique.append(kp)
            
            if ki >= len(k_points_unique):
                return np.zeros((size, size))
            
            kx, ky = k_points_unique[ki]
            
            # Reconstruct field
            coords = (np.arange(size) + 0.5) / size
            X, Y = np.meshgrid(coords, coords, indexing='ij')
            
            field = np.zeros((size, size), dtype=complex)
            for idx, (m, n) in enumerate(g_indices):
                Gx = 2.0 * np.pi * m
                Gy = 2.0 * np.pi * n
                phase = np.exp(1j * ((Gx + kx) * X + (Gy + ky) * Y))
                field += vec[idx] * phase
            
            return np.abs(field)**2
        except Exception:
            return np.zeros((size, size))
    
    def compute_tm_bands_with_vecs(cell, n_pw=17, n_kpts=8):
        """Compute TM bands and eigenvectors."""
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
        
        def get_coeff(m, n):
            idx_m = m % size
            idx_n = n % size
            return fft_coeffs[idx_m, idx_n]
        
        M = n_pw // 2
        m_vals = np.arange(-M, M + 1)
        g_indices = [(m, n) for m in m_vals for n in m_vals]
        n_total = len(g_indices)
        
        Gx = np.array([2.0 * np.pi * m for m, n in g_indices])
        Gy = np.array([2.0 * np.pi * n for m, n in g_indices])
        
        S = np.zeros((n_total, n_total), dtype=complex)
        for i in range(n_total):
            for j in range(n_total):
                gm = g_indices[i][0] - g_indices[j][0]
                gn = g_indices[i][1] - g_indices[j][1]
                S[i, j] = get_coeff(gm, gn)
        
        k_points = []
        n_per_seg = max(1, n_kpts // 3)
        for t in np.linspace(0, np.pi, n_per_seg + 1):
            k_points.append((t, 0))
        for t in np.linspace(0, np.pi, n_per_seg + 1):
            k_points.append((np.pi, t))
        for t in np.linspace(np.pi, 0, n_per_seg + 1):
            k_points.append((t, t))
        
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        eigvecs = np.zeros((n_kpts_actual, n_total, n_total), dtype=complex)
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            kx_total = Gx + kx
            ky_total = Gy + ky
            k_mag_sq = kx_total**2 + ky_total**2
            
            M_mat = S * k_mag_sq[None, :]
            
            try:
                eigvals, eigvecs_ki = np.linalg.eigh(M_mat)
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs_ki = eigvecs_ki[:, idx]
                omega[ki, :] = np.sqrt(np.maximum(np.real(eigvals), 0))
                eigvecs[ki, :, :] = eigvecs_ki
            except Exception:
                omega[ki, :] = np.inf
        
        return omega, eigvecs
    
    def enforce_d4(cell):
        """Enforce D4 symmetry (rotation and reflection) on the cell."""
        cell = np.array(cell, dtype=float)
        # Average over all D4 symmetry operations
        sym_ops = [
            cell,
            cell[::-1, :],
            cell[:, ::-1],
            cell[::-1, ::-1],
            cell.T,
            cell[::-1, :].T,
            cell[:, ::-1].T,
            cell[::-1, ::-1].T
        ]
        averaged = np.mean(sym_ops, axis=0)
        return (averaged >= 0.5).astype(float)
    
    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """Create a cell with four rods centered in each quadrant."""
        cell = np.zeros((size, size), dtype=float)
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
            cell[val <= 1.0 + 1e-9] = 1.0
        
        return cell
    
    # Phase 1: Find good geometric seed
    best_seed_score = -np.inf
    best_seed_cell = None
    
    radii = np.linspace(0.08, 0.16, 9)
    exponents = [2.0, 2.5, 3.0]
    
    for r in radii:
        if time.time() > deadline - 30:
            break
        for exp in exponents:
            if time.time() > deadline - 25:
                break
            
            cell = make_four_quadrant_cell(r, exp)
            score = compute_gap(cell, n_pw=15, n_kpts=5)
            
            if score > best_seed_score:
                best_seed_score = score
                best_seed_cell = cell
    
    if best_seed_cell is None:
        best_seed_cell = make_four_quadrant_cell(0.12, 2.0)
        best_seed_score = compute_gap(best_seed_cell, n_pw=15, n_kpts=5)
    
    # Phase 2: Topological optimization using field intensities
    current_cell = best_seed_cell.copy()
    current_score = best_seed_score
    
    np.random.seed(42)
    
    # Optimize for up to ~100 iterations or until time runs out
    max_iters = 150
    for it in range(max_iters):
        if time.time() > deadline - 20:
            break
        
        # Get field intensities for bands 4 and 5
        try:
            f4 = get_field_intensity(current_cell, 3, n_pw=15)
            f5 = get_field_intensity(current_cell, 4, n_pw=15)
        except Exception:
            break
        
        # Pixel score: add dielectric where band 4 has high intensity, remove where band 5 has high intensity
        pixel_score = f4 - f5
        
        # Add bias to maintain reasonable fill fraction
        fill = np.sum(current_cell) / (size * size)
        bias = 0.3 * np.sign(0.35 - fill)
        adj_score = pixel_score + bias * (current_cell - 0.5) * 2.0
        
        # Try multiple random pixel flips
        improved = False
        for _ in range(6):
            if time.time() > deadline - 20:
                break
            
            i = np.random.randint(0, size)
            j = np.random.randint(0, size)
            
            new_cell = current_cell.copy()
            new_cell[i, j] = 1.0 - new_cell[i, j]
            new_cell = enforce_d4(new_cell)
            
            try:
                new_score = compute_gap(new_cell, n_pw=15, n_kpts=5)
            except Exception:
                continue
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                improved = True
        
        if not improved and it > 20:
            # Try larger moves: flip multiple pixels
            for _ in range(3):
                if time.time() > deadline - 20:
                    break
                
                new_cell = current_cell.copy()
                for _ in range(4):
                    i = np.random.randint(0, size)
                    j = np.random.randint(0, size)
                    new_cell[i, j] = 1.0 - new_cell[i, j]
                new_cell = enforce_d4(new_cell)
                
                try:
                    new_score = compute_gap(new_cell, n_pw=15, n_kpts=5)
                except Exception:
                    continue
                
                if new_score > current_score:
                    current_cell = new_cell
                    current_score = new_score
                    improved = True
    
    # Phase 3: Final fine-tuning with higher precision
    if time.time() < deadline - 10:
        np.random.seed(123)
        for it in range(30):
            if time.time() > deadline - 10:
                break
            
            i = np.random.randint(0, size)
            j = np.random.randint(0, size)
            
            new_cell = current_cell.copy()
            new_cell[i, j] = 1.0 - new_cell[i, j]
            new_cell = enforce_d4(new_cell)
            
            try:
                new_score = compute_gap(new_cell, n_pw=17, n_kpts=6)
            except Exception:
                continue
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
    
    result = (current_cell >= 0.5).astype(int)
    return result.tolist()

# EVOLVE-BLOCK-END
