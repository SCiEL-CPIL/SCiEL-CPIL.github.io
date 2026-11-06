"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _build_pwe_matrix(inv_eps_2d, G_vectors, kx, ky, N):
    """
    Build the TM PWE matrix.
    inv_eps_2d: N x N array of 1/epsilon
    G_vectors: array of shape (n_g, 2) with (m, n) integers
    kx, ky: wave vector components
    Returns: (n_g, n_g) Hermitian matrix
    """
    n_g = len(G_vectors)
    # Precompute FFT of inv_eps
    # f_G = (1/N^2) sum_{x,y} inv_eps(x,y) exp(-2pi i (m x/N + n y/N))
    # This is exactly np.fft.fft2(inv_eps) / N^2
    F_inv = np.fft.fft2(inv_eps_2d) / (N * N)
    
    # Precompute k+G vectors
    KG = G_vectors + np.array([kx, ky])
    
    M = np.zeros((n_g, n_g), dtype=complex)
    
    # For efficiency, we can precompute the dot products
    # dot[i, j] = KG[i] . KG[j]
    dot = KG @ KG.T
    
    for i in range(n_g):
        mi, ni = G_vectors[i]
        for j in range(n_g):
            mj, nj = G_vectors[j]
            dm = int(mi - mj)
            dn = int(ni - nj)
            
            # Look up f_{dm, dn}
            im = dm % N
            in_ = dn % N
            f_diff = F_inv[im, in_]
            
            M[i, j] = dot[i, j] * f_diff
            
    # The eigenvalue equation is sum_j M_{ij} E_j = lambda E_i
    # where lambda = (omega/c)^2 * a^2 / (2pi)^2 ? 
    # Let's stick to the standard form:
    # (1/eps)_mn (k+G_m) . (k+G_n)
    # The eigenvalue lambda corresponds to (omega a / c)^2 / (2pi)^2 ?
    # Actually, if k is in units of 2pi/a, then (k+G) is dimensionless integer-like.
    # Standard PWE: 
    # sum_G' (k+G) . (k+G') (1/eps)_{G-G'} E_{G'} = (omega a / c)^2 E_G
    # So lambda = (omega a / c)^2.
    # omega = sqrt(lambda) * c / a.
    # Since we only care about ratios, we can just use sqrt(lambda) as proportional to omega.
    
    return M

def _fast_pwe_gap(cell, eps_low=1.0, eps_high=12.0, n_pw=17, n_kpoints=10):
    """
    Fast PWE surrogate.
    cell: 32x32 array of 0/1
    """
    N = len(cell)
    eps = np.array(cell, dtype=float)
    eps = eps * (eps_high - eps_low) + eps_low
    inv_eps = 1.0 / eps
    
    m_max = (n_pw - 1) // 2
    G_vectors = []
    for m in range(-m_max, m_max + 1):
        for n in range(-m_max, m_max + 1):
            G_vectors.append((m, n))
    G_vectors = np.array(G_vectors)
    n_g = len(G_vectors)
    
    # K-points in IBZ
    # IBZ: 0 <= ky <= kx <= pi (in units of 2pi/a? No, standard k is rad/length)
    # If G = 2pi/a * (m,n), then k is usually in [0, pi/a].
    # Let's assume a=1, so G = 2pi * (m,n).
    # IBZ of square lattice with a=1:
    # BZ is [-pi, pi] x [-pi, pi].
    # IBZ (1/8 BZ) is 0 <= ky <= kx <= pi.
    
    k_points = []
    # Key points
    k_points.append((0.0, 0.0))       # Gamma
    k_points.append((np.pi, 0.0))     # X
    k_points.append((np.pi, np.pi))   # M
    k_points.append((np.pi/2, 0.0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi/4, 0.0))
    k_points.append((3*np.pi/4, 0.0))
    k_points.append((np.pi/4, np.pi/4))
    k_points.append((3*np.pi/4, np.pi/4))
    k_points.append((np.pi/2, np.pi/4))
    k_points.append((np.pi/2, 3*np.pi/4))
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in k_points:
        M = _build_pwe_matrix(inv_eps, G_vectors, kx, ky, N)
        # M is Hermitian
        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.sort(eigvals.real)
        
        if len(eigvals) > 4:
            # lambda = (omega a / c)^2. So omega ~ sqrt(lambda).
            w4 = np.sqrt(eigvals[3])
            w5 = np.sqrt(eigvals[4])
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
    if min_omega_5 == np.inf:
        return 0.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap == 0:
        return 0.0
        
    return gap / midgap

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell maximizing TM band-4/5 gap.
    Uses a continuous density field optimized via a differentiable surrogate.
    """
    start_time = time.time()
    deadline = start_time + 210  # 20s safety margin for 230s limit
    
    size = problem["pixels"]
    half = size // 2
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # 1. Define the continuous density field on the top-left quadrant
    # We parameterize the density rho(x,y) in [0,1] using a low-frequency Fourier series
    # to ensure smoothness and differentiability.
    # rho(x,y) = 0.5 + sum_{m,n} A_{mn} cos(2pi m x / N) cos(2pi n y / N) ...
    # Actually, let's just use a set of control points or a simple parametric form.
    # Given the time budget, a full adjoint might be heavy.
    # Let's try a "parametric shape" approach with a few parameters, but use the 
    # fast PWE solver to guide the search.
    
    # Parameters:
    # We will use a super-ellipse / rounded square shape defined by:
    # (|x-cx|/a)^p + (|y-cy|/b)^p < 1
    # Parameters: cx, cy, a, b, p
    # But to allow more complex shapes, let's use a radial function r(theta)
    # or a low-order harmonic expansion.
    
    # Let's use a basis of 5x5 low-frequency cosines for the density in the quadrant.
    # rho[i,j] = base + sum_{m=0}^{4} sum_{n=0}^{4} c_{mn} cos(2pi m i / 32) cos(2pi n j / 32)
    # This ensures smoothness.
    
    n_basis = 5
    # Indices for the basis functions
    basis_m = np.arange(n_basis)
    basis_n = np.arange(n_basis)
    
    # Precompute basis functions on the 16x16 quadrant grid
    i_coords = np.arange(half)
    j_coords = np.arange(half)
    I, J = np.meshgrid(i_coords, j_coords, indexing='ij')
    
    # Basis functions evaluated at the pixel centers
    # We use the full 32x32 grid coordinates for the basis to maintain symmetry naturally?
    # No, we define the density on the quadrant and mirror it.
    # But the Fourier basis should be defined on the full domain to be orthogonal?
    # Let's just use a simple polynomial or trigonometric basis on the quadrant.
    
    # Let's try a different approach: 
    # Start with a good shape (e.g., the best from the GA in the current program)
    # and perform a local search in a continuous parameter space that morphs the shape.
    
    # Actually, the "Inspiration" suggests using a differentiable adjoint. 
    # Implementing a full adjoint for PWE is complex. 
    # Instead, we can use the fast PWE solver as a "black box" objective but use 
    # a smarter search than random pixel flips.
    
    # Let's use a "Shape Gradient" like approach by perturbing the boundary.
    # Or, simpler: use the fast PWE solver to evaluate a population of 
    # "parametric shapes" defined by a few continuous parameters.
    
    # Parametric shape:
    # rho(x,y) = sigmoid( (R(x,y) - d) / t )
    # where R(x,y) is a distance function to a shape.
    
    # Let's define a shape as a union of 4 super-ellipses in the quadrants?
    # No, the cell is 32x32. The "unit cell" contains 4 "rods" effectively if we look at the symmetry.
    # Wait, the problem says "symmetric 32x32-pixel unit cell".
    # The symmetry is C4v.
    # The "rods" are centered at (8,8), (24,8), (8,24), (24,24) roughly?
    # Or is it a single rod in the center?
    # The seed program uses a single central rod (cx=7.5, cy=7.5 in the quadrant, which maps to center of the 16x16 quadrant).
    # If we mirror the quadrant, the center of the quadrant (7.5, 7.5) maps to the center of the 32x32 cell?
    # Quadrant is 16x16. Indices 0..15.
    # Mirroring: 
    # cell[0:16, 0:16] = Q
    # cell[0:16, 16:32] = Q[:, ::-1]
    # cell[16:32, 0:16] = Q[::-1, :]
    # cell[16:32, 16:32] = Q[::-1, ::-1]
    
    # If Q has a rod at (7.5, 7.5), then:
    # Q[7,7] is near center of Q.
    # In full cell:
    # (7,7) -> (7,7)
    # (7, 15) -> (7, 16) ? No, Q[:, ::-1] means Q[7, 15] maps to cell[7, 16].
    # Wait, if Q is 16x16, indices 0..15.
    # Q[7,7] is "center-ish".
    # Mirrored to (7, 15) in the right half?
    # cell[7, 16+7] = cell[7, 23] = Q[7, 15-7] = Q[7, 8].
    # So the rod is duplicated at (7,7), (7,23), (23,7), (23,23).
    # These are 4 separate rods in the 32x32 cell.
    
    # So the design is 4 rods.
    
    # Let's optimize the shape of these 4 rods.
    # Parameters:
    # cx, cy: center of rod in quadrant (0..15)
    # rx, ry: radii
    # p: super-ellipse exponent
    
    def evaluate_params(params):
        cx, cy, rx, ry, p = params
        
        # Build quadrant
        Q = np.zeros((half, half), dtype=float)
        for i in range(half):
            for j in range(half):
                x = i + 0.5
                y = j + 0.5
                dx = abs(x - cx) / rx
                dy = abs(y - cy) / ry
                val = dx**p + dy**p
                if val < 1.0:
                    Q[i, j] = 1.0
                else:
                    Q[i, j] = 0.0
                    
        # Build full cell
        cell = np.zeros((size, size), dtype=int)
        cell[:half, :half] = Q
        cell[:half, half:] = Q[:, ::-1]
        cell[half:, :half] = Q[::-1, :]
        cell[half:, half:] = Q[::-1, ::-1]
        
        return _fast_pwe_gap(cell)
    
    # Optimization: Nelder-Mead or Simple Random Search with restarts.
    # Since the landscape might be rough, let's use a coarse grid search first,
    # then refine with local search.
    
    best_score = -np.inf
    best_params = (7.5, 7.5, 5.0, 5.0, 4.0)
    
    # Coarse Grid Search
    # cx, cy: 5.0 to 10.0 step 0.5
    # rx, ry: 3.0 to 7.0 step 0.5
    # p: 1.0 to 6.0 step 1.0
    
    cx_vals = np.arange(5.0, 10.1, 0.5)
    cy_vals = cx_vals # assume symmetry cx=cy for now? 
    # The symmetry of the cell forces the 4 rods to be identical.
    # But can cx != cy?
    # If Q[i,j] != Q[j,i], then the cell is not symmetric under transpose?
    # cell[i,j] = Q[i,j]. cell[j,i] = Q[j,i].
    # For cell to be symmetric, Q must be symmetric.
    # So cx must equal cy.
    
    rx_vals = np.arange(3.0, 7.1, 0.5)
    p_vals = [1.0, 2.0, 3.0, 4.0, 6.0, 10.0]
    
    for cx in cx_vals:
        if time.time() > deadline - 30: break
        for rx in rx_vals:
            if time.time() > deadline - 30: break
            for p in p_vals:
                if time.time() > deadline - 30: break
                params = (cx, cx, rx, rx, p)
                score = evaluate_params(params)
                if score > best_score:
                    best_score = score
                    best_params = params
                    # print(f"New best: {score} at {params}")
    
    # Refine around best with local search
    # Use a simple hill climbing with adaptive step size
    
    current_params = list(best_params)
    step_sizes = [0.1, 0.1, 0.1, 0.1, 0.5]
    
    for step in range(100):
        if time.time() > deadline - 10: break
        
        improved = False
        for i in range(len(current_params)):
            for sign in [-1, 1]:
                new_params = current_params[:]
                new_params[i] += sign * step_sizes[i]
                
                # Bounds
                if i == 0 or i == 1:
                    new_params[i] = np.clip(new_params[i], 5.0, 10.0)
                elif i == 2 or i == 3:
                    new_params[i] = np.clip(new_params[i], 3.0, 7.0)
                else:
                    new_params[i] = np.clip(new_params[i], 1.0, 10.0)
                
                # Enforce cx=cy
                new_params[1] = new_params[0]
                # Enforce rx=ry
                new_params[3] = new_params[2]
                
                score = evaluate_params(new_params)
                if score > best_score:
                    best_score = score
                    best_params = tuple(new_params)
                    current_params = new_params
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Reduce step sizes
            step_sizes = [s * 0.5 for s in step_sizes]
            if step_sizes[0] < 0.01:
                break
        
    # Final refinement: try to break symmetry?
    # The problem requires C4v symmetry. 
    # "invariant under the square symmetry group about the cell centre"
    # This implies C4v. 
    # So cx=cy and rx=ry are enforced by the symmetry if we assume the shape is axis-aligned super-ellipse.
    # But what if the shape is rotated?
    # A rotated super-ellipse in the quadrant, when mirrored, might not be C4v symmetric unless it's aligned.
    # Let's stick to aligned shapes.
    
    # Construct final cell
    cx, cy, rx, ry, p = best_params
    Q = np.zeros((half, half), dtype=int)
    for i in range(half):
        for j in range(half):
            x = i + 0.5
            y = j + 0.5
            dx = abs(x - cx) / rx
            dy = abs(y - cy) / ry
            if dx**p + dy**p < 1.0:
                Q[i, j] = 1
                
    cell = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(half):
        for j in range(half):
            val = Q[i][j]
            cell[i][j] = val
            cell[i][size - 1 - j] = val
            cell[size - 1 - i][j] = val
            cell[size - 1 - i][size - 1 - j] = val
            
    return cell

# EVOLVE-BLOCK-END
