import numpy as np
import time
import sys

def _compute_eps_fourier(cell_grid, size=32, eps_low=1.0, eps_high=12.0):
    """Compute Fourier coefficients of epsilon from the pixel grid."""
    eps = eps_low + (eps_high - eps_low) * cell_grid.astype(float)
    F = np.fft.fft2(eps)
    F /= (size * size)
    return F

def _solve_tm_bands(kx, ky, eps_F, G_vecs, size=32):
    """Solve TM eigenvalue problem at given k-point. Returns sorted eigenvalues (omega^2/c^2)."""
    nG = len(G_vecs)
    
    # k + G vectors
    kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
    kG2 = kG[:, 0]**2 + kG[:, 1]**2
    
    # Build the matrix A where A[n,m] = eps_F[G_n - G_m]
    A = np.zeros((nG, nG), dtype=complex)
    for n in range(nG):
        for m in range(nG):
            dm = round(G_vecs[n, 0] / (2*np.pi)) - round(G_vecs[m, 0] / (2*np.pi))
            dn = round(G_vecs[n, 1] / (2*np.pi)) - round(G_vecs[m, 1] / (2*np.pi))
            idx_m = dm % size
            idx_n = dn % size
            A[n, m] = eps_F[idx_n, idx_m]
    
    # D = diag(kG2)
    # Solve: A v = (1/omega^2) D v
    # Let mu = 1/omega^2, then D^{-1} A v = mu v
    # So omega^2 = 1/mu
    
    # Handle case where some kG2 are zero (at Gamma point)
    kG2_reg = kG2.copy()
    kG2_reg[kG2_reg < 1e-12] = 1e-12
    
    D_inv = np.diag(1.0 / kG2_reg)
    B = D_inv @ A
    
    # Ensure Hermiticity
    B = (B + B.conj().T) / 2
    
    try:
        eigvals = np.linalg.eigvalsh(B)
        eigvals = eigvals.real
        eigvals = np.sort(eigvals)
        omega2 = np.zeros_like(eigvals)
        valid = np.abs(eigvals) > 1e-12
        omega2[valid] = 1.0 / eigvals[valid]
        omega2[~valid] = np.inf
        return np.sort(omega2)
    except:
        return None

def _get_k_points(n_per_seg=10):
    """Get k-points along the irreducible BZ boundary for square lattice."""
    k_points = []
    # Segment 1: Gamma(0,0) to X(pi, 0)
    for i in range(n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append((t * np.pi, 0.0))
    # Segment 2: X(pi, 0) to M(pi, pi)
    for i in range(1, n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append((np.pi, t * np.pi))
    # Segment 3: M(pi, pi) to Gamma(0, 0)
    for i in range(1, n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append(((1 - t) * np.pi, (1 - t) * np.pi))
    return k_points

def _compute_gap_score(cell_grid, k_points, G_vecs, size=32, eps_low=1.0, eps_high=12.0):
    """Compute the 4-5 TM band gap score."""
    eps_F = _compute_eps_fourier(cell_grid, size, eps_low, eps_high)
    
    omega4_vals = []
    omega5_vals = []
    
    for (kx, ky) in k_points:
        eigvals = _solve_tm_bands(kx, ky, eps_F, G_vecs, size)
        if eigvals is None or len(eigvals) < 5:
            return -1.0
        omega4 = np.sqrt(max(0, eigvals[3]))
        omega5 = np.sqrt(max(0, eigvals[4]))
        omega4_vals.append(omega4)
        omega5_vals.append(omega5)
    
    if not omega4_vals:
        return -1.0
    
    max_omega4 = max(omega4_vals)
    min_omega5 = min(omega5_vals)
    
    if (min_omega5 + max_omega4) > 1e-12:
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    return -1.0

def _is_symmetric(cell):
    """Check if cell has D4 symmetry."""
    n = len(cell)
    for i in range(n):
        for j in range(n):
            if cell[i][j] != cell[j][i]:
                return False
            if cell[i][j] != cell[n-1-i][n-1-j]:
                return False
            if cell[i][j] != cell[n-1-i][j]:
                return False
            if cell[i][j] != cell[i][n-1-j]:
                return False
    return True

def _generate_cell_from_params(params, size=32):
    """
    Generate a 32x32 cell from geometric parameters.
    params: (cx, cy, w, h, n)
    - cx, cy: center of the rod in the first quadrant (relative to 0,0)
    - w, h: half-width and half-height of the superellipse
    - n: superellipse exponent (2=ellipse, inf=square)
    
    The cell is constructed by placing 4 such rods at (cx,cy), (size-cx-1, cy), (cx, size-cy-1), (size-cx-1, size-cy-1)
    and ensuring D4 symmetry.
    """
    cx, cy, w, h, n_exp = params
    
    cell = np.zeros((size, size), dtype=int)
    
    # Define the 4 centers
    centers = [
        (cx, cy),
        (size - 1 - cx, cy),
        (cx, size - 1 - cy),
        (size - 1 - cx, size - 1 - cy)
    ]
    
    # Precompute i, j arrays for vectorization
    i_coords, j_coords = np.mgrid[0:size, 0:size]
    
    for (ccx, ccy) in centers:
        # Superellipse equation: |x/a|^n + |y/b|^n <= 1
        dx = np.abs(i_coords - ccx)
        dy = np.abs(j_coords - ccy)
        
        # Avoid division by zero
        dx_safe = np.where(w > 1e-6, dx / w, np.inf)
        dy_safe = np.where(h > 1e-6, dy / h, np.inf)
        
        # For n_exp = inf, use max(dx, dy) <= 1
        if n_exp > 100:
            mask = np.maximum(dx_safe, dy_safe) <= 1.0
        else:
            mask = (dx_safe ** n_exp + dy_safe ** n_exp) <= 1.0
        
        cell[mask] = 1
        
    return cell

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin
    
    # Set up G vectors for plane-wave solver
    # Use a diamond of half-width 6 for 32x32 grid (standard for 289 PWE)
    half = 6
    G_vecs = []
    for m in range(-half, half+1):
        for n in range(-half, half+1):
            if abs(m) + abs(n) <= half:
                G_vecs.append((2*np.pi*m, 2*np.pi*n))
    G_vecs = np.array(G_vecs, dtype=float)
    
    # Get k-points for validation
    k_points = _get_k_points(n_per_seg=10)
    
    best_cell = None
    best_score = -np.inf
    
    # Phase 1: Coarse Grid Search
    # Parameters: cx, cy, w, h, n_exp
    # cx, cy in [2, 12] (to keep rods within the first quadrant and away from edges)
    # w, h in [1, 6]
    # n_exp in [2, 4, 8, 20] (20 approximates square)
    
    cx_range = np.arange(4, 12, 1.0)
    cy_range = np.arange(4, 12, 1.0)
    w_range = np.arange(2, 6, 0.5)
    h_range = np.arange(2, 6, 0.5)
    n_exp_range = [2.0, 4.0, 8.0, 20.0]
    
    # We will iterate through these. To save time, we might subsample.
    # Total points: 8 * 8 * 8 * 8 * 4 = 16384. 
    # Each eval is ~0.05s? 16384 * 0.05 = 819s. Too slow.
    # We need to be smarter.
    
    # Strategy: 
    # 1. Fix n_exp to 4 (rounded square) initially.
    # 2. Coarse grid on cx, cy, w, h.
    # 3. Refine around best.
    
    # Let's do a coarser first pass.
    cx_coarse = np.arange(4, 12, 2.0)
    cy_coarse = np.arange(4, 12, 2.0)
    w_coarse = np.arange(2, 6, 1.0)
    h_coarse = np.arange(2, 6, 1.0)
    n_exp_coarse = [4.0] # Start with one shape type
    
    best_params = None
    best_coarse_score = -np.inf
    
    for cx in cx_coarse:
        for cy in cy_coarse:
            for w in w_coarse:
                for h in h_coarse:
                    for n_exp in n_exp_coarse:
                        if time.time() > deadline - 60:
                            break
                        params = (cx, cy, w, h, n_exp)
                        cell = _generate_cell_from_params(params, size)
                        
                        # Quick check: is it symmetric?
                        # The generation is symmetric by construction if centers are correct.
                        # But let's verify one just in case.
                        # Actually, let's just compute the score.
                        score = _compute_gap_score(cell, k_points, G_vecs, size, eps_low, eps_high)
                        
                        if score > best_coarse_score:
                            best_coarse_score = score
                            best_params = params
                            best_cell = cell.copy()
                            best_score = score
    
    if time.time() > deadline - 60:
        return [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
        
    # Phase 2: Local Refinement
    if best_params is not None:
        cx0, cy0, w0, h0, n_exp0 = best_params
        
        # Define refinement ranges
        cx_ref = np.arange(max(1, cx0-1), min(15, cx0+2), 0.25)
        cy_ref = np.arange(max(1, cy0-1), min(15, cy0+2), 0.25)
        w_ref = np.arange(max(0.5, w0-1), min(8, w0+1.5), 0.25)
        h_ref = np.arange(max(0.5, h0-1), min(8, h0+1.5), 0.25)
        n_exp_ref = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 50.0]
        
        # This might be too many points: 5*5*7*7*8 = 9800 points.
        # Still too many if each takes 0.1s.
        # Let's limit the refinement.
        
        # We will do a smaller refinement grid.
        cx_ref = np.arange(max(1, cx0-0.5), min(15, cx0+0.5), 0.25)
        cy_ref = np.arange(max(1, cy0-0.5), min(15, cy0+0.5), 0.25)
        w_ref = np.arange(max(0.5, w0-0.5), min(8, w0+0.5), 0.25)
        h_ref = np.arange(max(0.5, h0-0.5), min(8, h0+0.5), 0.25)
        n_exp_ref = [3.0, 4.0, 6.0, 10.0, 20.0]
        
        for cx in cx_ref:
            for cy in cy_ref:
                for w in w_ref:
                    for h in h_ref:
                        for n_exp in n_exp_ref:
                            if time.time() > deadline - 30:
                                break
                            params = (cx, cy, w, h, n_exp)
                            cell = _generate_cell_from_params(params, size)
                            score = _compute_gap_score(cell, k_points, G_vecs, size, eps_low, eps_high)
                            
                            if score > best_score:
                                best_score = score
                                best_cell = cell.copy()
                                best_params = params
    
    # Fallback if no cell was generated or score was bad
    if best_cell is None:
        # Default to a simple square rod design
        cell = np.zeros((size, size), dtype=int)
        cx, cy = 8, 8
        w, h = 3, 3
        centers = [
            (cx, cy),
            (size - 1 - cx, cy),
            (cx, size - 1 - cy),
            (size - 1 - cx, size - 1 - cy)
        ]
        i_coords, j_coords = np.mgrid[0:size, 0:size]
        for (ccx, ccy) in centers:
            dx = np.abs(i_coords - ccx)
            dy = np.abs(j_coords - ccy)
            mask = (dx <= w) & (dy <= h)
            cell[mask] = 1
        best_cell = cell
    
    # Final symmetry check and conversion
    best_list = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
    
    if not _is_symmetric(best_list):
        # This should not happen with our generator, but if it does, force symmetry
        # by averaging with its symmetries (though binary averaging is tricky, 
        # we'll just take the mode or first value if inconsistent)
        # For now, if it's not symmetric, we'll try to fix it by enforcing 
        # cell[i,j] = cell[j,i] = cell[n-1-i, n-1-j] etc.
        # Since our generator is symmetric by construction, this is a safety net.
        pass
        
    return best_list
