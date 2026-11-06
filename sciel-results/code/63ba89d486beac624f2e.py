"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _make_superellipse_cell(size, a, b, n, offset=0.0):
    """Generate a 32x32 symmetric binary cell from a superellipse.
    
    The superellipse is defined by |x/a|^n + |y/b|^n <= 1, centered at (0,0)
    or offset for four-rod configuration.
    
    a, b: semi-axes
    n: shape exponent (n=2 is circle, n>2 is more square-like, n<2 is more diamond-like)
    offset: if > 0, creates four-rod lattice by checking distance to 4 centers
    """
    cell = np.zeros((size, size), dtype=int)
    
    # Create coordinate grid centered at 0
    coords = (np.arange(size) + 0.5) / size - 0.5
    
    if offset > 0:
        # Four-rod lattice: check distance to nearest of 4 centers
        # Centers at (±offset, ±offset)
        for i in range(size):
            for j in range(size):
                x = coords[i]
                y = coords[j]
                
                # Find minimum superellipse value over 4 centers
                min_val = np.inf
                for cx in [-offset, offset]:
                    for cy in [-offset, offset]:
                        dx = x - cx
                        dy = y - cy
                        val = abs(dx/a)**n + abs(dy/b)**n
                        if val < min_val:
                            min_val = val
                
                if min_val <= 1.0:
                    cell[i, j] = 1
    else:
        # Single centered superellipse
        for i in range(size):
            for j in range(size):
                x = coords[i]
                y = coords[j]
                val = abs(x/a)**n + abs(y/b)**n
                if val <= 1.0:
                    cell[i, j] = 1
    
    # Enforce exact symmetry
    cell = np.minimum(cell, cell[::-1, :])
    cell = np.minimum(cell, cell[:, ::-1])
    cell = np.minimum(cell, cell.T)
    
    return cell.tolist()

def _compute_fourier_coeffs(cell, size, max_order=4):
    """Compute Fourier coefficients of the dielectric function up to max_order.
    
    Returns dict mapping (m,n) to eps_{mn}
    """
    eps = np.array(cell, dtype=float)
    eps_high = 12.0
    eps_low = 1.0
    
    # Epsilon map: 12 where cell=1, 1 where cell=0
    eps_map = np.where(eps > 0.5, eps_high, eps_low)
    
    # Compute Fourier coefficients
    coeffs = {}
    for m in range(-max_order, max_order+1):
        for n in range(-max_order, max_order+1):
            # eps_{mn} = (1/A) ∫∫ eps(x,y) exp(-i(m*2πx + n*2πy)) dxdy
            # For pixelated: (1/N^2) sum_{i,j} eps(i/N, j/N) exp(-i(m*2π*i/N + n*2π*j/N))
            
            phase_x = np.exp(-1j * 2.0 * np.pi * m * np.arange(size) / size)
            phase_y = np.exp(-1j * 2.0 * np.pi * n * np.arange(size) / size)
            phase_2d = phase_x[:, None] * phase_y[None, :]
            
            coeffs[(m, n)] = np.sum(eps_map * phase_2d) / (size * size)
    
    return coeffs

def _surrogate_gap(cell, size=32, max_order=4):
    """Fast surrogate evaluation using Fourier component magnitudes.
    
    The 4-5 TM band gap is sensitive to higher-order Fourier components.
    We use the magnitude of key Fourier components as a proxy.
    """
    coeffs = _compute_fourier_coeffs(cell, size, max_order)
    
    # Key Fourier components that affect the 4-5 gap
    # For square lattice, components at (1,0), (0,1), (1,1), (2,0), (0,2), (2,2) are important
    key_components = [(1,0), (0,1), (1,1), (2,0), (0,2), (2,2), (-1,0), (0,-1), (-1,-1)]
    
    # Weighted sum of magnitudes, emphasizing higher-order components
    weights = {
        (1,0): 1.0, (0,1): 1.0, (-1,0): 1.0, (0,-1): 1.0,
        (1,1): 1.5, (-1,-1): 1.5,
        (2,0): 2.0, (0,2): 2.0, (2,2): 2.5
    }
    
    score = 0.0
    for comp in key_components:
        if comp in coeffs:
            mag = abs(coeffs[comp])
            w = weights.get(comp, 1.0)
            score += w * mag
    
    # Also penalize too high or too low fill fraction
    fill = np.sum(np.array(cell)) / (size * size)
    # Optimal fill fraction is around 0.3-0.5
    fill_penalty = 1.0 - abs(fill - 0.4)
    
    return score * fill_penalty

def _quick_band_gap(cell, size=32, n_pw=9, n_kpts=4):
    """Quick approximate band gap calculation using reduced plane-wave basis.
    
    This is faster than the full 289-plane-wave method but gives a reasonable proxy.
    """
    eps = np.array(cell, dtype=float)
    eps_high = 12.0
    eps_low = 1.0
    eps_map = np.where(eps > 0.5, eps_high, eps_low)
    
    # Plane wave indices
    if n_pw % 2 == 1:
        half = n_pw // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n_pw // 2
        pw_idx = np.arange(-half, half)
    
    # Compute Fourier coefficients
    coeffs = np.zeros((len(pw_idx), len(pw_idx)), dtype=complex)
    for im, m in enumerate(pw_idx):
        for in_, n in enumerate(pw_idx):
            phase_x = np.exp(-1j * 2.0 * np.pi * m * np.arange(size) / size)
            phase_y = np.exp(-1j * 2.0 * np.pi * n * np.arange(size) / size)
            phase_2d = phase_x[:, None] * phase_y[None, :]
            coeffs[im, in_] = np.sum(eps_map * phase_2d) / (size * size)
    
    # K-points in irreducible BZ: (0,0), (0.5,0), (0.5,0.5)
    k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    for kx, ky in k_points:
        # Build the TM eigenvalue matrix
        # For TM: [∇_⊥^2 + k0^2 eps(r)] E_z = 0
        # In plane waves: sum_{G'} [(k+G)^2 delta_{G,G'} - k0^2 eps_{G-G'}] a_{G'} = 0
        # This is a generalized eigenvalue problem: A a = lambda B a
        # where A_{G,G'} = (k+G)^2 delta_{G,G'} and B_{G,G'} = eps_{G-G'}
        # Or equivalently: [I + k0^2 * (1/(k+G)^2) * eps_{G-G'}] a = 0
        
        N = len(pw_idx)
        A = np.zeros((N*N, N*N), dtype=complex)
        B = np.zeros((N*N, N*N), dtype=complex)
        
        for i, m in enumerate(pw_idx):
            for j, n in enumerate(pw_idx):
                idx = i*N + j
                Gx = 2.0 * np.pi * m
                Gy = 2.0 * np.pi * n
                kx_G = kx + m
                ky_G = ky + n
                kG_sq = (2.0*np.pi*kx_G)**2 + (2.0*np.pi*ky_G)**2
                
                # Avoid division by zero
                if abs(kG_sq) < 1e-10:
                    kG_sq = 1e-10
                
                A[idx, idx] = kG_sq
                
                for i2, m2 in enumerate(pw_idx):
                    for j2, n2 in enumerate(pw_idx):
                        idx2 = i2*N + j2
                        dm = m - m2
                        dn = n - n2
                        # Find index of (dm, dn) in coeffs
                        im_dm = list(pw_idx).index(dm) if dm in pw_idx else None
                        in_dn = list(pw_idx).index(dn) if dn in pw_idx else None
                        if im_dm is not None and in_dn is not None:
                            B[idx, idx2] = coeffs[im_dm, in_dn]
        
        # Solve generalized eigenvalue problem A a = lambda B a
        # Convert to standard: B^{-1} A a = lambda a
        try:
            B_inv = np.linalg.inv(B)
            M = B_inv @ A
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(np.real(eigvals))
            
            # Take the smallest few eigenvalues (corresponding to lowest bands)
            # We need bands 4 and 5 (0-indexed: 3 and 4)
            if len(eigvals) >= 5:
                omega4 = np.sqrt(eigvals[3])
                omega5 = np.sqrt(eigvals[4])
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            pass
    
    if min_omega5 > max_omega4 and (min_omega5 + max_omega4) > 1e-10:
        gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    else:
        gap_ratio = -1.0
    
    return gap_ratio

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill
    
    best_cell = None
    best_score = -np.inf
    
    # Phase 1: Coarse grid search over superellipse parameters
    # Parameters: a (semi-axis x), b (semi-axis y), n (shape exponent), offset (four-rod)
    
    # Try single centered superellipse first
    a_values = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    b_values = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    n_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    for a in a_values:
        if time.time() > deadline - 60.0:
            break
        for b in b_values:
            if time.time() > deadline - 30.0:
                break
            for n in n_values:
                if time.time() > deadline - 10.0:
                    break
                cell = _make_superellipse_cell(size, a, b, n, offset=0.0)
                score = _surrogate_gap(cell, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Try four-rod lattice
    offset_values = [0.15, 0.2, 0.25]
    a_values_four = [0.1, 0.12, 0.15, 0.18]
    n_values_four = [2.0, 2.5, 3.0, 4.0]
    
    for offset in offset_values:
        if time.time() > deadline - 40.0:
            break
        for a in a_values_four:
            if time.time() > deadline - 20.0:
                break
                cell = _make_superellipse_cell(size, a, a, n, offset=offset)
                score = _surrogate_gap(cell, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Phase 2: Local refinement around best parameters using quick band gap
    if time.time() < deadline - 30.0 and best_cell is not None:
        # Use quick band gap to refine
        # We need to track the best parameters - for now, do a finer grid
        fine_a = [0.2, 0.22, 0.25, 0.28, 0.3, 0.32, 0.35]
        fine_b = [0.2, 0.22, 0.25, 0.28, 0.3, 0.32, 0.35]
        fine_n = [2.0, 2.5, 3.0, 3.5, 4.0]
        
        quick_best_score = -np.inf
        quick_best_cell = None
        
        for a in fine_a:
            if time.time() > deadline - 20.0:
                break
            for b in fine_b:
                if time.time() > deadline - 10.0:
                    break
                for n in fine_n:
                    if time.time() > deadline - 5.0:
                        break
                    cell = _make_superellipse_cell(size, a, b, n, offset=0.0)
                    # Use quick band gap for more accurate evaluation
                    qscore = _quick_band_gap(cell, size, n_pw=7, n_kpts=3)
                    if qscore > quick_best_score:
                        quick_best_score = qscore
                        quick_best_cell = cell
        
        if quick_best_cell is not None and quick_best_score > best_score:
            best_cell = quick_best_cell
            best_score = quick_best_score
    
    # Phase 3: Final refinement with more k-points if time allows
    if time.time() < deadline - 15.0 and best_cell is not None:
        # Try a few more variations around the best
        perturbations = [
            (0.22, 0.22, 2.5, 0.0),
            (0.25, 0.25, 3.0, 0.0),
            (0.2, 0.25, 2.5, 0.0),
            (0.25, 0.2, 2.5, 0.0),
            (0.15, 0.15, 3.0, 0.2),
            (0.18, 0.18, 2.5, 0.2),
        ]
        
        final_best_score = best_score
        final_best_cell = best_cell
        
        for a, b, n, offset in perturbations:
            if time.time() > deadline - 5.0:
                break
            cell = _make_superellipse_cell(size, a, b, n, offset=offset)
            qscore = _quick_band_gap(cell, size, n_pw=9, n_kpts=4)
            if qscore > final_best_score:
                final_best_score = qscore
                final_best_cell = cell
        
        if final_best_cell is not None:
            best_cell = final_best_cell
    
    return best_cell
# EVOLVE-BLOCK-END
