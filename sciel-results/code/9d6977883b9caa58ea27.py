import numpy as np
import time

def _get_d4_orbits(n=32):
    """Get all D4 orbits of pixels in an n x n grid."""
    orbits = []
    seen = set()
    for i in range(n):
        for j in range(n):
            if (i, j) in seen:
                continue
            orbit = set()
            # Identity
            orbit.add((i, j))
            # 90 deg rotation
            orbit.add((j, n-1-i))
            # 180 deg rotation
            orbit.add((n-1-i, n-1-j))
            # 270 deg rotation
            orbit.add((n-1-j, i))
            # Mirror x (horizontal flip)
            orbit.add((i, n-1-j))
            # Mirror y (vertical flip)
            orbit.add((n-1-i, j))
            # Diagonal mirror
            orbit.add((j, i))
            # Anti-diagonal mirror
            orbit.add((n-1-j, n-1-i))
            orbit = list(orbit)
            for p in orbit:
                seen.add(p)
            orbits.append(orbit)
    return orbits

def _apply_orbits_vectorized(orbit_values, orbits, size=32):
    """Apply orbit values to a grid efficiently using numpy indexing."""
    result = np.zeros((size, size), dtype=np.float64)
    for val, orbit in zip(orbit_values, orbits):
        if val != 0:
            idxs = np.array(orbit)
            result[idxs[:, 0], idxs[:, 1]] = 1.0
    return result

def _build_pw_solver(size=32, half=6, eps_low=1.0, eps_high=12.0):
    """
    Build a pre-allocated plane-wave solver object.
    Uses a diamond of G-vectors with max |m|+|n| <= half.
    """
    G_vecs = []
    for m in range(-half, half+1):
        for n in range(-half, half+1):
            if abs(m) + abs(n) <= half:
                G_vecs.append((m, n))
    G_vecs = np.array(G_vecs, dtype=float)
    nG = len(G_vecs)
    
    # Precompute G vector components
    Gx = G_vecs[:, 0] * 2.0 * np.pi
    Gy = G_vecs[:, 1] * 2.0 * np.pi
    
    # Precompute G differences for matrix construction
    # diff_G[i, j] = G_i - G_j
    # We need epsilon_F at these differences.
    # epsilon_F is computed from the 32x32 grid.
    # The FFT of a 32x32 grid has indices 0..31.
    # G difference (dm, dn) corresponds to FFT index (dm % 32, dn % 32).
    
    dm_all = Gx[:, None] - Gx[None, :]
    dn_all = Gy[:, None] - Gy[None, :]
    
    # Normalize to integer indices
    dm_int = np.round(dm_all / (2.0 * np.pi)).astype(int)
    dn_int = np.round(dn_all / (2.0 * np.pi)).astype(int)
    
    # Map to 0..31
    idx_m = dm_int % size
    idx_n = dn_int % size
    
    # Store these indices for fast lookup
    solver = {
        'Gx': Gx,
        'Gy': Gy,
        'nG': nG,
        'idx_m': idx_m,
        'idx_n': idx_n,
        'size': size,
        'eps_low': eps_low,
        'eps_high': eps_high
    }
    return solver

def _compute_eps_fourier(cell_grid, size=32, eps_low=1.0, eps_high=12.0):
    """Compute Fourier coefficients of epsilon from the pixel grid."""
    eps = eps_low + (eps_high - eps_low) * cell_grid
    # FFT expects input in standard order.
    # The plane wave expansion uses exp(i (k+G) . r).
    # The coefficient eps_F[G] is the integral of eps(r) exp(-i G . r).
    # For a discrete grid, this is essentially the FFT.
    # Note: np.fft.fft2 computes sum_{x,y} eps[x,y] exp(-2pi i (k x/N + l y/N)).
    # This matches the definition if G = (2pi k/N, 2pi l/N).
    # Our G vectors are integers m, n in units of 2pi.
    # So G = (2pi m, 2pi n).
    # The FFT index (m, n) corresponds to frequency m/N, n/N.
    # Wait, the lattice constant is 1. So G = (2pi m, 2pi n).
    # The FFT of a function on [0,1]x[0,1] sampled at N points gives coefficients for frequencies 2pi k/N.
    # So if we want coefficient for G = (2pi m, 2pi n), we look at FFT index (m, n).
    # However, the FFT output is ordered 0, 1, ..., N-1.
    # Negative indices wrap around.
    F = np.fft.fft2(eps) / (size * size)
    return F

def _solve_tm_bands(kx, ky, eps_F, solver):
    """Solve TM eigenvalue problem at given k-point. Returns sorted omega^2."""
    Gx = solver['Gx']
    Gy = solver['Gy']
    nG = solver['nG']
    idx_m = solver['idx_m']
    idx_n = solver['idx_n']
    
    # k + G vectors
    kGx = kx + Gx
    kGy = ky + Gy
    kG2 = kGx**2 + kGy**2
    
    # Build A matrix: A[n, m] = eps_F[G_n - G_m]
    # Using precomputed indices
    A = eps_F[idx_n, idx_m]
    
    # Handle Gamma point singularity (kG2 = 0)
    # If kG2 is 0, the term is 0. In the equation A E = mu D E, if D has 0, 
    # it's a singular system. 
    # Standard approach: if kG2 == 0, set D_inv to 0 for that component? 
    # Actually, at Gamma, the lowest band is 0.
    # For TM mode, the equation is:
    # sum_m eps_F[Gn-Gm] E_m / |k+Gn|^2 = (1/omega^2) E_n
    # If |k+Gn|^2 = 0, the LHS is undefined unless E_n = 0 or eps_F... 
    # Actually, for TM, the field is parallel to the surface. 
    # The eigenvalue problem is:
    # (1/omega^2) E = D^{-1} A E
    # If D has zeros, we can't invert. 
    # However, at Gamma, the first band is 0 (omega=0). 
    # We can regularize D by adding a small epsilon to zeros.
    
    kG2_safe = kG2.copy()
    mask = kG2_safe < 1e-12
    kG2_safe[mask] = 1e-12
    
    D_inv = 1.0 / kG2_safe
    B = D_inv[:, None] * A  # Element-wise multiplication for diagonal scaling
    
    # Ensure Hermiticity
    B = (B + B.conj().T) / 2.0
    
    try:
        eigvals = np.linalg.eigvalsh(B)
        eigvals = eigvals.real
        # mu = 1/omega^2
        # omega^2 = 1/mu
        # Filter out small/negative mu which correspond to high omega or numerical noise
        valid = np.abs(eigvals) > 1e-12
        omega2 = np.zeros_like(eigvals)
        omega2[valid] = 1.0 / eigvals[valid]
        # For invalid, set to inf (high band)
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

def _compute_gap_score(cell_grid, k_points, solver, eps_low, eps_high):
    """Compute the 4-5 TM band gap score."""
    eps_F = _compute_eps_fourier(cell_grid, solver['size'], eps_low, eps_high)
    
    omega4_vals = []
    omega5_vals = []
    
    for (kx, ky) in k_points:
        eigvals = _solve_tm_bands(kx, ky, eps_F, solver)
        if eigvals is None or len(eigvals) < 5:
            return -1.0
        # eigvals are omega^2, sorted ascending
        # Band 4 is index 3, Band 5 is index 4
        # We need omega = sqrt(omega^2)
        omega4 = np.sqrt(max(0, eigvals[3]))
        omega5 = np.sqrt(max(0, eigvals[4]))
        omega4_vals.append(omega4)
        omega5_vals.append(omega5)
    
    if not omega4_vals:
        return -1.0
    
    max_omega4 = max(omega4_vals)
    min_omega5 = min(omega5_vals)
    
    if min_omega5 + max_omega4 < 1e-12:
        return -1.0
        
    return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

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

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s budget
    
    # Get D4 orbits
    orbits = _get_d4_orbits(size)
    n_orbits = len(orbits)
    
    # Set up solver
    solver = _build_pw_solver(size, half=6, eps_low=eps_low, eps_high=eps_high)
    
    # Get k-points
    k_points = _get_k_points(n_per_seg=10)
    
    # Initialize with a four-rod design
    # Rods at (8,8), (23,8), (8,23), (23,23) with radius ~4
    cell_init = np.zeros((size, size), dtype=np.float64)
    centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
    radius_sq = 16.0  # radius = 4
    
    for i in range(size):
        for j in range(size):
            for (cx, cy) in centers:
                dx = i - cx
                dy = j - cy
                if dx*dx + dy*dy < radius_sq:
                    cell_init[i, j] = 1.0
                    break
    
    # Convert to orbit-based representation
    orbit_values = np.zeros(n_orbits, dtype=int)
    for oi, orbit in enumerate(orbits):
        i, j = orbit[0]
        orbit_values[oi] = int(cell_init[i, j])
    
    current_cell = _apply_orbits_vectorized(orbit_values, orbits, size)
    
    # Compute initial score
    current_score = _compute_gap_score(current_cell, k_points, solver, eps_low, eps_high)
    best_cell = current_cell.copy()
    best_score = current_score
    best_orbit_values = orbit_values.copy()
    
    # Multi-start Local Search (Hill Climbing)
    # We will run multiple short hill-climbing sessions from different starting points
    # or restarts.
    
    rng = np.random.RandomState(42)
    
    # Pre-allocate a buffer for new cells to avoid repeated allocation if possible?
    # _apply_orbits_vectorized allocates. It's fast enough (32x32).
    
    n_restarts = 10
    max_iter_per_restart = 200
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # If not the first restart, perturb the current best or randomize
        if restart > 0:
            # Start from a random configuration or a perturbation of the best
            if restart % 2 == 0:
                # Random start
                orbit_values = rng.randint(0, 2, n_orbits).astype(int)
            else:
                # Perturbation of best
                orbit_values = best_orbit_values.copy()
                # Flip 10% of orbits
                flip_mask = rng.rand(n_orbits) < 0.1
                orbit_values[flip_mask] = 1 - orbit_values[flip_mask]
            
            current_cell = _apply_orbits_vectorized(orbit_values, orbits, size)
            current_score = _compute_gap_score(current_cell, k_points, solver, eps_low, eps_high)
        
        # Hill climbing
        for it in range(max_iter_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Pick a random orbit to flip
            oi = rng.randint(n_orbits)
            
            # Flip
            old_val = orbit_values[oi]
            orbit_values[oi] = 1 - old_val
            new_cell = _apply_orbits_vectorized(orbit_values, orbits, size)
            
            # Compute new score
            new_score = _compute_gap_score(new_cell, k_points, solver, eps_low, eps_high)
            
            if new_score > current_score:
                current_score = new_score
                current_cell = new_cell
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell.copy()
                    best_orbit_values = orbit_values.copy()
            else:
                # Revert
                orbit_values[oi] = old_val

    # Convert best_cell to nested list
    best_list = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
    
    # Verify symmetry
    if not _is_symmetric(best_list):
        # Fallback: reconstruct from orbits
        best_list = [[int(_apply_orbits_vectorized(best_orbit_values, orbits, size)[i, j]) for j in range(size)] for i in range(size)]
        
    return best_list
