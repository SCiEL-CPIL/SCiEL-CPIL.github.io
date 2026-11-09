import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell with a 4-rod superlattice pattern optimized
    to maximize the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    def make_cell(radius, offset, size=32):
        """Create a symmetric 4-rod cell with given radius and offset from quarter-points."""
        cell = np.zeros((size, size), dtype=int)
        centers = [
            (size * 0.25 + offset, size * 0.25 + offset),
            (size * 0.75 - offset, size * 0.25 + offset),
            (size * 0.25 + offset, size * 0.75 - offset),
            (size * 0.75 - offset, size * 0.75 - offset),
        ]
        
        # Vectorized pixel generation
        y, x = np.mgrid[0:size, 0:size]
        y_pix = y + 0.5
        x_pix = x + 0.5
        
        for cx, cy in centers:
            dist_sq = (x_pix - cx)**2 + (y_pix - cy)**2
            cell[dist_sq < radius * radius] = 1
            
        return cell

    def get_fourier_coeffs(cell, n_max):
        """Compute Fourier coefficients of 1/epsilon."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        return fft_2d

    def solve_bands_vectorized(cell, n_pw=12, k_points=None):
        """
        Fast vectorized band solver.
        Returns the gap ratio between bands 4 and 5.
        """
        if k_points is None:
            k_points = [
                (0.0, 0.0), (0.5, 0.0), (0.5, 0.5),
                (0.25, 0.0), (0.25, 0.25), (0.5, 0.25)
            ]
        
        fft_2d = get_fourier_coeffs(cell, n_pw)
        
        # Generate G vectors within circular cutoff
        g_vals = np.arange(-n_pw, n_pw + 1)
        Gx, Gy = np.meshgrid(g_vals, g_vals, indexing='ij')
        mask = Gx**2 + Gy**2 <= n_pw**2
        Gx = Gx[mask]
        Gy = Gy[mask]
        n_pw_actual = len(Gx)
        
        # Precompute |k-G|^2 for all k-points and G-vectors
        # k_points is a list of (kx, ky)
        # We want a matrix K2 of shape (n_k, n_pw_actual)
        kx_arr = np.array([k[0] for k in k_points])
        ky_arr = np.array([k[1] for k in k_points])
        
        # Gx, Gy are shape (n_pw_actual,)
        # kx_arr is shape (n_k,)
        # We need |k - G|^2 = (kx - Gx)^2 + (ky - Gy)^2
        # Shape (n_k, 1) - (1, n_pw_actual) -> (n_k, n_pw_actual)
        Kx = kx_arr[:, np.newaxis] - Gx[np.newaxis, :]
        Ky = ky_arr[:, np.newaxis] - Gy[np.newaxis, :]
        K2 = Kx**2 + Ky**2
        
        max_omega4 = -1e10
        min_omega5 = 1e10
        
        for idx_k in range(len(k_points)):
            # H[i, j] = eps_inv[G_i - G_j] * |k - G_j|^2
            # G_i - G_j = (Gx_i - Gx_j, Gy_i - Gy_j)
            # We need to extract eps_inv for all pairs.
            
            # DiffGx[i, j] = Gx[i] - Gx[j]
            DiffGx = Gx[:, np.newaxis] - Gx[np.newaxis, :]
            DiffGy = Gy[:, np.newaxis] - Gy[np.newaxis, :]
            
            # Map to FFT indices
            # FFT index is (m % size, n % size)
            idx_x = DiffGx % size
            idx_y = DiffGy % size
            
            # eps_inv matrix
            EpsInv = fft_2d[idx_x, idx_y]
            
            # H = EpsInv * K2[idx_k, :] (broadcast K2 along rows)
            H = EpsInv * K2[idx_k, :][np.newaxis, :]
            
            # Solve eigenvalue problem
            # H is non-Hermitian. Use eigvals.
            try:
                eigvals = np.linalg.eigvals(H)
                # Filter real eigenvalues
                real_eigs = np.sort(np.real(eigvals))
                # We want the lowest 5 bands.
                # Note: eigvals returns all. We take the smallest positive ones?
                # Usually the lowest bands are the smallest positive eigenvalues.
                # There might be negative eigenvalues due to numerical noise or basis issues, but for TM with positive eps, they should be positive.
                # Let's take the first 5 smallest positive values.
                pos_eigs = real_eigs[real_eigs > 1e-6]
                if len(pos_eigs) >= 5:
                    omega4 = pos_eigs[3]
                    omega5 = pos_eigs[4]
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except np.linalg.LinAlgError:
                pass
        
        if max_omega4 <= 0 or min_omega5 <= 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # Initial guess
    best_score = -1e10
    best_radius = 5.5
    best_offset = 0.0
    
    # Coarse search
    radii = np.arange(4.0, 8.1, 0.5)
    offsets = np.arange(-2.0, 2.1, 0.5)
    
    for r in radii:
        for off in offsets:
            if time.time() > deadline:
                break
            cell = make_cell(r, off)
            score = solve_bands_vectorized(cell, n_pw=10)
            if score > best_score:
                best_score = score
                best_radius = r
                best_offset = off
        if time.time() > deadline:
            break
    
    # Fine search
    if time.time() < deadline - 20:
        # Refine around best
        for _ in range(2):
            step = 0.25
            improved = False
            for dr in [-step, -step/2, 0, step/2, step]:
                for do in [-step, -step/2, 0, step/2, step]:
                    if time.time() > deadline:
                        break
                    r = best_radius + dr
                    off = best_offset + do
                    if r < 3 or r > 10:
                        continue
                    cell = make_cell(r, off)
                    score = solve_bands_vectorized(cell, n_pw=12)
                    if score > best_score:
                        best_score = score
                        best_radius = r
                        best_offset = off
                        improved = True
                if time.time() > deadline:
                    break
            if not improved:
                break

    # Create final cell
    final_cell = make_cell(best_radius, best_offset)
    
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_cell[i, j]))
        result.append(row)
    
    return result
