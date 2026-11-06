"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _make_superellipse_cell(size, a, b, n, four_rod=False, offset=0.25):
    """Generate a 32x32 symmetric binary cell from a superellipse.
    
    The superellipse is defined by |x/a|^n + |y/b|^n <= 1.
    If four_rod=True, creates a 4-rod lattice by checking distance to 4 centers.
    
    a, b: semi-axes (in lattice units)
    n: shape exponent (n=2 is circle, n>2 is more square-like, n<2 is more diamond-like)
    four_rod: if True, use 4 rods at (±offset, ±offset)
    offset: offset for four-rod configuration
    """
    cell = np.zeros((size, size), dtype=int)
    
    # Pre-compute coordinates
    coords = (np.arange(size) + 0.5) / size - 0.5
    
    for i in range(size):
        for j in range(size):
            x = coords[i]
            y = coords[j]
            
            if four_rod:
                # Check distance to nearest of 4 centers
                best_val = np.inf
                for cx in [-offset, offset]:
                    for cy in [-offset, offset]:
                        dx = x - cx
                        dy = y - cy
                        val = abs(dx/a)**n + abs(dy/b)**n
                        if val < best_val:
                            best_val = val
                if best_val <= 1.0:
                    cell[i, j] = 1
            else:
                # Single centered superellipse
                val = abs(x/a)**n + abs(y/b)**n
                if val <= 1.0:
                    cell[i, j] = 1
    
    # Enforce exact symmetry: cell == cell[::-1] row-wise, col-wise, and transpose
    cell = np.minimum(cell, cell[::-1, :])
    cell = np.minimum(cell, cell[:, ::-1])
    cell = np.minimum(cell, cell.T)
    
    return cell.tolist()

def _compute_fourier_coeffs(cell, size=32, max_g=8):
    """Compute Fourier coefficients of the dielectric mask up to max_g.
    
    Returns dict mapping (m, n) to coefficient c_{mn}
    where mask(x,y) = sum_{m,n} c_{mn} exp(i*2*pi*(m*x + n*y))
    """
    eps = np.array(cell, dtype=float)
    coeffs = {}
    
    # Use FFT for efficiency
    fft_mask = np.fft.fft2(eps)
    fft_mask = np.fft.fftshift(fft_mask)
    
    # Normalize: c_{mn} = FFT_{mn} / size^2
    # With fftshift, index (size//2 + m, size//2 + n) corresponds to frequency (m, n)
    center = size // 2
    
    for m in range(-max_g, max_g + 1):
        for n in range(-max_g, max_g + 1):
            idx_m = center + m
            idx_n = center + n
            # Handle wrap-around for negative frequencies
            idx_m = idx_m % size
            idx_n = idx_n % size
            coeffs[(m, n)] = fft_mask[idx_m, idx_n] / (size * size)
    
    return coeffs

def _quick_band_gap(cell, size=32, n_pw=11, n_kpts=10):
    """Fast surrogate evaluation of the 4-5 TM band gap using a reduced plane-wave basis.
    
    Uses the generalized eigenvalue formulation:
    D a = lambda M a
    where D is diagonal with (k+G)^2 and M has entries 1/eps(G-G')
    
    Returns: gap_ratio = 2*(min_omega5 - max_omega4)/(min_omega5 + max_omega4)
    """
    eps_low = 1.0
    eps_high = 12.0
    
    # Compute 1/epsilon mask
    mask = np.array(cell, dtype=float)
    inv_eps_mask = np.zeros_like(mask)
    inv_eps_mask[mask == 1] = 1.0 / eps_high
    inv_eps_mask[mask == 0] = 1.0 / eps_low
    
    # FFT of 1/epsilon
    fft_inv_eps = np.fft.fft2(inv_eps_mask)
    fft_inv_eps = np.fft.fftshift(fft_inv_eps)
    center = size // 2
    
    def get_coeff(m, n):
        """Get Fourier coefficient c_{mn} of 1/epsilon"""
        idx_m = (center + m) % size
        idx_n = (center + n) % size
        return fft_inv_eps[idx_m, idx_n] / (size * size)
    
    # Plane wave G vectors
    half_pw = n_pw // 2
    G_vectors = []
    for i in range(-half_pw, half_pw + 1):
        for j in range(-half_pw, half_pw + 1):
            G_vectors.append((i, j))
    N_G = len(G_vectors)
    
    # K-points in the irreducible BZ (in units of 2*pi/a)
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # X
        (0.5, 0.5),      # M
        (0.25, 0.0),
        (0.25, 0.25),
        (0.1, 0.3),
        (0.3, 0.1),
        (0.15, 0.35),
        (0.35, 0.15),
        (0.4, 0.4),
    ]
    
    all_eigenvalues = []
    
    for kx, ky in k_points:
        # Build M matrix: M_{G,G'} = c_{G-G'}
        M = np.zeros((N_G, N_G), dtype=complex)
        for idx1, G1 in enumerate(G_vectors):
            for idx2, G2 in enumerate(G_vectors):
                dG = (G1[0] - G2[0], G1[1] - G2[1])
                M[idx1, idx2] = get_coeff(dG[0], dG[1])
        
        # Build D matrix: diagonal with (k+G)^2
        D = np.zeros((N_G, N_G), dtype=complex)
        for idx, G in enumerate(G_vectors):
            kgx = kx + G[0]
            kgy = ky + G[1]
            D[idx, idx] = kgx*kgx + kgy*kgy
        
        # Solve generalized eigenvalue problem: D a = lambda M a
        # Convert to standard eigenvalue: M^{-1} D a = lambda a
        # Or use scipy's eig for generalized problem
        
        try:
            # For numerical stability, solve M^{-1} D a = lambda a
            # But M might be singular. Use scipy.linalg.eig
            from scipy.linalg import eig
            eigenvalues, eigenvectors = eig(D, M)
            
            # Filter for real eigenvalues (physical modes)
            real_eigs = [ev.real for ev in eigenvalues if abs(ev.imag) < 1e-6 and ev.real > 0]
            real_eigs.sort()
            
            if len(real_eigs) >= 5:
                all_eigenvalues.append(real_eigs[:5])
        except:
            continue
    
    if not all_eigenvalues:
        return 0.0
    
    # Find max of band 4 (index 3) and min of band 5 (index 4)
    max_band4 = max(eigs[3] for eigs in all_eigenvalues)
    min_band5 = min(eigs[4] for eigs in all_eigenvalues)
    
    gap = min_band5 - max_band4
    midgap = (min_band5 + max_band4) / 2.0
    
    if midgap <= 0:
        return 0.0
    
    return 2.0 * gap / midgap

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill
    
    best_cell = None
    best_score = -np.inf
    
    # Phase 1: Coarse grid search over superellipse parameters
    # Single centered superellipse
    a_values = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    b_values = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    n_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    for a in a_values:
        if time.time() > deadline - 60.0:
            break
        for b in b_values:
            if time.time() > deadline - 30.0:
                break
            for n in n_values:
                if time.time() > deadline - 10.0:
                    break
                cell = _make_superellipse_cell(size, a, b, n, four_rod=False)
                score = _quick_band_gap(cell, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Phase 2: Four-rod lattice search
    offset_values = [0.2, 0.25, 0.3]
    a_rod_values = [0.08, 0.1, 0.12, 0.15, 0.18]
    n_rod_values = [1.5, 2.0, 2.5, 3.0]
    
    for offset in offset_values:
        if time.time() > deadline - 40.0:
            break
        for a_rod in a_rod_values:
            if time.time() > deadline - 20.0:
                break
            for n_rod in n_rod_values:
                if time.time() > deadline - 5.0:
                    break
                cell = _make_superellipse_cell(size, a_rod, a_rod, n_rod, four_rod=True, offset=offset)
                score = _quick_band_gap(cell, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Phase 3: Fine-tune around best region
    if time.time() < deadline - 30.0:
        # Determine which family was best and refine
        # For single superellipse, refine around common good regions
        refine_a = [0.2, 0.25, 0.3, 0.35]
        refine_b = [0.2, 0.25, 0.3, 0.35]
        refine_n = [1.5, 2.0, 2.5, 3.0, 3.5]
        
        for a in refine_a:
            if time.time() > deadline - 15.0:
                break
            for b in refine_b:
                if time.time() > deadline - 10.0:
                    break
                for n in refine_n:
                    if time.time() > deadline - 5.0:
                        break
                    cell = _make_superellipse_cell(size, a, b, n, four_rod=False)
                    score = _quick_band_gap(cell, size)
                    if score > best_score:
                        best_score = score
                        best_cell = cell
        
        # Refine four-rod
        refine_offset = [0.22, 0.25, 0.28]
        refine_a_rod = [0.1, 0.12, 0.14, 0.16]
        refine_n_rod = [2.0, 2.5, 3.0]
        
        for offset in refine_offset:
            if time.time() > deadline - 10.0:
                break
            for a_rod in refine_a_rod:
                if time.time() > deadline - 5.0:
                    break
                for n_rod in refine_n_rod:
                    cell = _make_superellipse_cell(size, a_rod, a_rod, n_rod, four_rod=True, offset=offset)
                    score = _quick_band_gap(cell, size)
                    if score > best_score:
                        best_score = score
                        best_cell = cell
    
    # Ensure we always return a valid cell
    if best_cell is None:
        # Fallback: simple centered disk
        best_cell = _make_superellipse_cell(size, 0.2, 0.2, 2.0, four_rod=False)
    
    # Final symmetry enforcement
    best_cell_arr = np.array(best_cell)
    best_cell_arr = np.minimum(best_cell_arr, best_cell_arr[::-1, :])
    best_cell_arr = np.minimum(best_cell_arr, best_cell_arr[:, ::-1])
    best_cell_arr = np.minimum(best_cell_arr, best_cell_arr.T)
    
    return best_cell_arr.tolist()
# EVOLVE-BLOCK-END
