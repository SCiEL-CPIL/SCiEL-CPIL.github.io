"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fast_tm_bands(cell, n_pw=17, k_points=None):
    """Fast TM band calculation using plane-wave method.
    
    cell: 32x32 array of 0/1 (1=dielectric eps=12)
    n_pw: number of plane waves (odd, e.g., 17 means -8..8)
    k_points: list of (kx, ky) in units of 2*pi/a
    Returns: array of band frequencies (omega in units of 2*pi/a)
    """
    size = len(cell)
    eps = 1.0 + 11.0 * cell  # eps = 1 + (12-1)*cell
    eps_inv = 1.0 / eps  # For TM: 1/eps appears in the eigenvalue problem
    
    # Fourier coefficients of 1/eps
    # G indices: -n_pw//2 to n_pw//2
    half = n_pw // 2
    G_indices = np.arange(-half, half + 1)
    Gx = 2 * np.pi * G_indices
    Gy = 2 * np.pi * G_indices
    
    # Compute Fourier coefficients of 1/eps
    # Using direct sum: c_G = (1/N^2) * sum_{x,y} eps_inv(x,y) * exp(-i*G*(x,y))
    # For efficiency, use FFT
    eps_inv_arr = eps_inv.astype(np.float64)
    # Pad to at least n_pw x n_pw for FFT
    N = max(size, n_pw)
    eps_inv_padded = np.zeros((N, N))
    eps_inv_padded[:size, :size] = eps_inv_arr
    eps_inv_fft = np.fft.fft2(eps_inv_padded) / (N * N)
    
    # Extract Fourier coefficients for needed G vectors
    def get_coeff(gx, gy):
        idx_x = ((gx / (2 * np.pi)) % N)
        idx_y = ((gy / (2 * np.pi)) % N)
        return eps_inv_fft[int(idx_x) % N, int(idx_y) % N]
    
    # Build the matrix for a given k-point
    def solve_k(kx, ky):
        # Plane wave indices
        pw_indices = []
        for gi in G_indices:
            for gj in G_indices:
                pw_indices.append((gi, gj))
        n_pw_actual = len(pw_indices)
        
        # Build eigenvalue matrix: A = diag(1/eps) * (k+G)^2
        # For TM: eigenvalue = (k+G)^2, with coupling from 1/eps
        # The eigenvalue problem: sum_G' (1/eps)_{G-G'} |k+G'|^2 c_{G'} = omega c_G
        
        A = np.zeros((n_pw_actual, n_pw_actual), dtype=np.complex128)
        for i in range(n_pw_actual):
            gi, gj = pw_indices[i]
            kx_i = kx + Gx[gi]
            ky_i = ky + Gy[gj]
            for j in range(n_pw_actual):
                gjj, gjj2 = pw_indices[j]
                # Fourier coefficient of 1/eps for G_i - G_j
                dgi = gi - gjj
                dgj = gj - gjj2
                c = get_coeff(dgi * 2 * np.pi, dgj * 2 * np.pi)
                A[i, j] = c * (kx_i * kx_i + ky_i * ky_i)
        
        # Solve eigenvalue problem
        # A is Hermitian if 1/eps is real and symmetric
        A = (A + A.conj().T) / 2  # Ensure Hermitian
        eigvals = np.linalg.eigvalsh(A)
        # omega = sqrt(eigenvalue)
        omega = np.sqrt(np.maximum(eigvals, 0))
        return omega
    
    if k_points is None:
        # Use a few representative k-points in irreducible BZ
        # Irreducible BZ of square: 0 <= kx <= ky, kx + ky <= 1 (in units 2*pi/a)
        # Key points: Gamma(0,0), M(0.5,0), X(0.5,0.5), and some interior
        k_points = [
            (0.0, 0.0),      # Gamma
            (0.5, 0.0),      # X
            (0.5, 0.5),      # M
            (0.25, 0.0),
            (0.25, 0.25),
            (0.5, 0.25),
            (0.125, 0.0),
            (0.125, 0.125),
        ]
    
    all_bands = []
    for kx, ky in k_points:
        bands = solve_k(kx, ky)
        all_bands.append(bands)
    
    return all_bands

def _gap_ratio(cell, n_pw=17):
    """Compute the gap-midgap ratio for bands 4 and 5."""
    try:
        all_bands = _fast_tm_bands(cell, n_pw=n_pw)
        
        # Find max of band 4 (index 3) and min of band 5 (index 4)
        band4_vals = [b[3] for b in all_bands if len(b) > 3]
        band5_vals = [b[4] for b in all_bands if len(b) > 4]
        
        if not band4_vals or not band5_vals:
            return -1.0
        
        max_band4 = max(band4_vals)
        min_band5 = min(band5_vals)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap
    except Exception:
        return -1.0

def _make_cell_from_params(params):
    """Create a D4-symmetric 32x32 cell from parameters.
    
    params: array of shape (n_params,)
    - params[0]: inner radius of ring
    - params[1]: outer radius of ring  
    - params[2]: cross arm width
    - params[3]: corner blob radius
    - params[4:]: additional polynomial coefficients for fine tuning
    """
    size = 32
    # Use polynomial-based approach for smooth D4-symmetric shapes
    x = np.linspace(-0.5, 0.5, size, endpoint=False)
    y = np.linspace(-0.5, 0.5, size, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    r2 = X*X + Y*Y
    r = np.sqrt(r2)
    
    # Ring: dielectric where inner_r < r < outer_r
    inner_r = params[0]
    outer_r = params[1]
    cross_w = params[2]
    corner_r = params[3]
    
    cell = np.zeros((size, size), dtype=np.float64)
    
    # Add ring
    ring_mask = (r > inner_r) & (r < outer_r)
    cell[ring_mask] = 1.0
    
    # Add cross (horizontal and vertical bars)
    cross_mask = ((np.abs(X) < cross_w) | (np.abs(Y) < cross_w)) & (r < 0.45)
    cell[cross_mask] = 1.0
    
    # Add corner blobs
    corner_mask = (np.abs(X) > 0.3) & (np.abs(Y) > 0.3) & (r < 0.45)
    # Use distance from corners
    for cx, cy in [(0.35, 0.35), (-0.35, 0.35), (0.35, -0.35), (-0.35, -0.35)]:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        cell[dist < corner_r] = 1.0
    
    # Optional: use polynomial coefficients to modulate
    if len(params) > 4:
        # Add a low-order polynomial modulation
        c4 = params[4] if len(params) > 4 else 0
        c5 = params[5] if len(params) > 5 else 0
        poly = c4 * (X**2 - Y**2) + c5 * (X**4 + Y**4)
        # Modulate the cell: where poly > 0, add dielectric; where poly < 0, remove
        mod_mask = poly > 0.05
        cell[mod_mask] = 1.0
        mod_mask2 = poly < -0.05
        cell[mod_mask2] = 0.0
    
    # Ensure D4 symmetry by averaging with all 8 symmetries
    # Actually, let's just enforce it properly
    cell_sym = np.zeros_like(cell)
    for i in range(size):
        for j in range(size):
            # Average over all 8 D4 symmetries
            vals = [
                cell[i, j],
                cell[size-1-i, j],
                cell[i, size-1-j],
                cell[size-1-i, size-1-j],
                cell[j, i],
                cell[j, size-1-i],
                cell[size-1-j, i],
                cell[size-1-j, size-1-i],
            ]
            cell_sym[i, j] = 1.0 if np.mean(vals) > 0.5 else 0.0
    
    return cell_sym.astype(int)

def _make_cell_ring_cross(inner_r, outer_r, cross_w, corner_r):
    """Simpler D4-symmetric cell: ring + cross + corner blobs."""
    size = 32
    x = np.linspace(-0.5, 0.5, size, endpoint=False)
    y = np.linspace(-0.5, 0.5, size, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    r2 = X*X + Y*Y
    r = np.sqrt(r2)
    
    cell = np.zeros((size, size), dtype=np.float64)
    
    # Ring
    ring_mask = (r > inner_r) & (r < outer_r)
    cell[ring_mask] = 1.0
    
    # Cross
    cross_mask = ((np.abs(X) < cross_w) | (np.abs(Y) < cross_w)) & (r < 0.48)
    cell[cross_mask] = 1.0
    
    # Corner blobs
    for cx, cy in [(0.35, 0.35), (-0.35, 0.35), (0.35, -0.35), (-0.35, -0.35)]:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        cell[dist < corner_r] = 1.0
    
    # Enforce D4 symmetry
    cell_sym = np.zeros_like(cell)
    for i in range(size):
        for j in range(size):
            vals = [
                cell[i, j],
                cell[size-1-i, j],
                cell[i, size-1-j],
                cell[size-1-i, size-1-j],
                cell[j, i],
                cell[j, size-1-i],
                cell[size-1-j, i],
                cell[size-1-j, size-1-i],
            ]
            cell_sym[i, j] = 1.0 if np.mean(vals) > 0.5 else 0.0
    
    return cell_sym.astype(int)

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing TM band 4-5 gap."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    size = problem.get("pixels", 32)
    
    # Start with a good initial guess: ring + cross pattern
    best_cell = None
    best_score = -1.0
    
    # Initial parameter grid
    # params: [inner_r, outer_r, cross_w, corner_r]
    initial_params = [0.15, 0.30, 0.08, 0.10]
    
    # Evaluate initial design
    cell = _make_cell_ring_cross(*initial_params)
    score = _gap_ratio(cell, n_pw=13)  # Use smaller basis for speed
    if score > best_score:
        best_score = score
        best_cell = cell
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_cell.tolist()
    
    # Local optimization using Nelder-Mead
    def objective(params):
        if time.time() > deadline - 5:
            return -1.0
        cell = _make_cell_ring_cross(*params)
        return -_gap_ratio(cell, n_pw=13)
    
    try:
        result = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 50, 'fatol': 1e-3, 'xatol': 1e-3}
        )
        if result.fun < -best_score:
            best_params = result.x
            cell = _make_cell_ring_cross(*best_params)
            score = _gap_ratio(cell, n_pw=13)
            if score > best_score:
                best_score = score
                best_cell = cell
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_cell.tolist()
    
    # Try some alternative geometries
    alt_designs = [
        [0.10, 0.28, 0.10, 0.08],  # Thinner ring, wider cross
        [0.18, 0.32, 0.06, 0.12],  # Thicker ring, thinner cross
        [0.12, 0.35, 0.12, 0.06],  # Large ring
        [0.20, 0.38, 0.05, 0.15],  # Large ring, small cross
        [0.08, 0.25, 0.15, 0.10],  # Small ring, large cross
    ]
    
    for params in alt_designs:
        if time.time() > deadline - 5:
            break
        cell = _make_cell_ring_cross(*params)
        score = _gap_ratio(cell, n_pw=13)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Final refinement with larger basis if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 30 and best_cell is not None:
        try:
            score_large = _gap_ratio(best_cell, n_pw=17)
            if score_large > best_score:
                best_score = score_large
        except Exception:
            pass
    
    if best_cell is None:
        # Fallback: simple rod
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    cell[i, j] = 1
        best_cell = cell
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
