"""Active local-search strategy: fast symmetry-reduced PW solver + flip hill-climb with deadline guard."""
import time
import random
import numpy as np

# ---------------------------------------------------------------------------
# Fast band solver: 17x17 plane waves (289 total), symmetry-orbit reduced
# ---------------------------------------------------------------------------
PW_HALF = 8  # -8..8 -> 17x17 = 289 plane waves

def _get_orbits(size):
    """Return list of orbits under C4v symmetry about cell centre."""
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            # Generate orbit under: identity, rot90, rot180, rot270, reflect_x, reflect_y, diag, anti-diag
            # Cell centre is at (size/2 - 0.5, size/2 - 0.5) in pixel coords
            # Rotation 90 CW about centre: (i, j) -> (j, size-1-i)
            # Rotation 180: (i, j) -> (size-1-i, size-1-j)
            # Rotation 270: (i, j) -> (size-1-j, i)
            # Reflect about vertical midline: (i, j) -> (i, size-1-j)
            # Reflect about horizontal midline: (i, j) -> (size-1-i, j)
            # Reflect about main diagonal: (i, j) -> (j, i)
            # Reflect about anti-diagonal: (i, j) -> (size-1-j, size-1-i)
            orbit = set()
            orbit.add((i, j))
            orbit.add((j, size - 1 - i))
            orbit.add((size - 1 - i, size - 1 - j))
            orbit.add((size - 1 - j, i))
            orbit.add((i, size - 1 - j))
            orbit.add((size - 1 - i, j))
            orbit.add((j, i))
            orbit.add((size - 1 - j, size - 1 - i))
            orbit = sorted(orbit)
            orbits.append(orbit)
            for pos in orbit:
                seen.add(pos)
    return orbits

def _orbit_to_cell(orbit_values, orbits, size):
    """Expand orbit values to full cell. orbit_values is list of 0/1 per orbit."""
    cell = np.zeros((size, size), dtype=np.int8)
    for val, orbit in zip(orbit_values, orbits):
        for (i, j) in orbit:
            cell[i, j] = val
    return cell

def _compute_fourier_eps_inv(cell, eps_low, eps_high):
    """
    Compute inverse epsilon Fourier coefficients on 17x17 grid.
    eps_inv = 1/eps where eps = eps_low + (eps_high - eps_low) * cell
    We need <1/eps>_{G,G'} for G, G' in the 17x17 grid.
    """
    size = cell.shape[0]
    eps = eps_low + (eps_high - eps_low) * cell.astype(np.float64)
    eps_inv = 1.0 / eps  # shape (size, size)

    # Plane wave indices: -PW_HALF .. PW_HALF
    g_indices = np.arange(-PW_HALF, PW_HALF + 1)  # 17 values
    # G vector in units of 2π/a: (gx, gy) where gx, gy in g_indices
    # Fourier coeff: c_{G} = (1/N^2) sum_{i,j} eps_inv(i,j) exp(-i G·r_{i,j})
    # r_{i,j} = ((i+0.5)/size, (j+0.5)/size)
    i_idx, j_idx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    x = (i_idx + 0.5) / size
    y = (j_idx + 0.5) / size

    # We need C_{G1,G2} = (1/N^2) sum eps_inv(i,j) exp(-i (G1 - G2) · r)
    # Let D = G1 - G2, range from -2*PW_HALF to 2*PW_HALF
    # But we only need C for G1, G2 in [-PW_HALF, PW_HALF], so D in [-16, 16]
    # However, for the matrix element we need C_{G1-G2} where both are in the grid.
    # Actually the PW method for TM uses:
    # (|k+G|^2) S_{G,G'} where S_{G,G'} = <eps^{-1}>_{G-G'}
    # So we need <eps^{-1}>_{D} for D = G - G' where G, G' in our grid.
    # D ranges from -2*PW_HALF to 2*PW_HALF in each component.

    # Compute all needed Fourier coefficients of eps_inv
    # D_x, D_y in range [-2*PW_HALF, 2*PW_HALF]
    d_range = np.arange(-2 * PW_HALF, 2 * PW_HALF + 1)  # -16..16, 33 values
    # But we only need D such that G and G' are both in [-PW_HALF, PW_HALF]
    # i.e., D = G - G' where G, G' in [-8, 8], so D in [-16, 16]
    # We'll compute the full 33x33 set and index into it.

    # For efficiency, compute using the fact that we need:
    # f(D) = (1/N^2) sum_{i,j} eps_inv(i,j) exp(-i 2π (Dx*x_ij + Dy*y_ij))
    # = (1/N^2) sum_i eps_inv_col(i) exp(-i 2π Dx * x_i) * sum_j ... 
    # Actually it's a 2D sum, not separable in general.
    # With N=32 and 33x33=1089 coefficients, direct sum is 1089*1024 ≈ 1.1M ops, fine.

    # Precompute exp factors
    # exp(-i 2π Dx * x_i) for Dx in d_range, i in 0..31
    exp_x = np.exp(-1j * 2 * np.pi * np.outer(d_range, x.ravel()))  # (33, 32)
    exp_y = np.exp(-1j * 2 * np.pi * np.outer(d_range, y.ravel()))  # (33, 32)

    # eps_inv_flat = eps_inv.ravel() shape (1024,)
    # For each (Dx, Dy): f = mean(eps_inv * exp_x[Dx_idx] * exp_y[Dy_idx])
    # Vectorize: create full grid
    # exp_x_grid: (33, 32, 1) broadcast to (33, 32, 32)
    # exp_y_grid: (33, 1, 32) broadcast to (33, 32, 32)
    exp_x_grid = exp_x[:, :, np.newaxis]  # (33, 32, 1)
    exp_y_grid = exp_y[:, np.newaxis, :]  # (33, 1, 32)
    # product: (33, 32, 33, 32) -> too big? 33*32*33*32 = 1.1M, ok
    # Actually let's do it differently: for each Dx, compute the 1D transform along x, then along y.
    # f(Dx, Dy) = (1/N^2) sum_i sum_j eps_inv(i,j) exp(-i2π Dx x_i) exp(-i2π Dy y_j)
    # = (1/N^2) [exp_x(Dx)]^T @ eps_inv @ exp_y(Dy)
    # So: for each Dx, compute A[Dx] = exp_x(Dx)^T @ eps_inv  -> shape (33, 32)
    # Then f(Dx, Dy) = (1/N^2) A[Dx] @ exp_y(Dy)  -> (33, 33)

    A = exp_x @ eps_inv  # (33, 32) : A[Dx_idx, j] = sum_i exp_x[Dx_idx, i] * eps_inv[i, j]
    F = (A @ exp_y.T) / (size * size)  # (33, 33) : F[Dx_idx, Dy_idx] = f(Dx, Dy)

    return F, d_range

def _compute_bands(cell, eps_low, eps_high, kx, ky, n_bands=6):
    """
    Compute TM band energies (omega^2 in units of 2π/a) for given k.
    Returns array of n_bands eigenvalues.
    """
    size = cell.shape[0]
    F, d_range = _compute_fourier_eps_inv(cell, eps_low, eps_high)
    
    # Plane wave grid: G in [-PW_HALF, PW_HALF]^2
    g_vals = np.arange(-PW_HALF, PW_HALF + 1)
    Gx, Gy = np.meshgrid(g_vals, g_vals, indexing='ij')  # (17, 17)
    G_flat = np.column_stack([Gx.ravel(), Gy.ravel()])  # (289, 2)
    N_pw = G_flat.shape[0]

    # k is in units of 2π/a, so (kx, ky) are dimensionless fractions
    k = np.array([kx, ky])

    # |k + G|^2
    KG = k + G_flat  # (289, 2)
    kg2 = np.sum(KG ** 2, axis=1)  # (289,)

    # Build S matrix: S_{mn} = <eps^{-1}>_{G_m - G_n}
    # G_m - G_n has components in [-16, 16]
    # d_range is [-16, 16], index offset = 16
    d_offset = 2 * PW_HALF  # 16
    S = np.zeros((N_pw, N_pw), dtype=np.complex128)
    for m in range(N_pw):
        for n in range(N_pw):
            d = G_flat[m] - G_flat[n]
            idx_x = int(d[0]) + d_offset
            idx_y = int(d[1]) + d_offset
            S[m, n] = F[idx_x, idx_y]

    # Generalized eigenvalue problem: (|k+G|^2) S v = omega^2 v
    # i.e., D @ S @ v = omega^2 v where D = diag(kg2)
    # Convert to standard: S^{-1} D S v = omega^2 v  -- no, that's not right.
    # The equation is: sum_G' |k+G|^2 <eps^{-1}>_{G-G'} c_{G'} = omega^2 c_G
    # So: (D @ S) c = omega^2 c  where D = diag(kg2)
    # This is a standard eigenvalue problem if we form the matrix M = D @ S
    # But M is not Hermitian in general? Actually for real eps_inv, S is Hermitian:
    # S_{mn} = <eps^{-1}>_{G_m - G_n} and S_{nm} = <eps^{-1}>_{G_n - G_m} = conj(S_{mn})
    # So S is Hermitian. D is real diagonal. D @ S is not Hermitian.
    # We need to solve: D S c = omega^2 c
    # Multiply by D^{-1}: S c = omega^2 D^{-1} c  -- not helpful.
    # Better: form the matrix and use eigh on a transformed system.
    # Actually, the standard approach: the matrix D @ S is similar to a Hermitian matrix.
    # Let c' = D^{1/2} c. Then D S c = omega^2 c becomes D S D^{-1/2} c' = omega^2 D^{-1/2} c'
    # Multiply by D^{1/2}: D^{1/2} D S D^{-1/2} c' = omega^2 c'
    # D^{1/2} D S D^{-1/2} = D^{3/2} S D^{-1/2} -- not Hermitian either.
    
    # Simpler: just use numpy.linalg.eig on the non-Hermitian matrix D @ S,
    # or better, use the fact that we can form the Hermitian matrix:
    # H = S^{-1/2} D S^{-1/2}  -- but S may not be positive definite in general.
    
    # For photonic crystals with eps > 1 everywhere, S should be positive definite.
    # Let's just use scipy.linalg.eigh on the generalized problem if possible,
    # or fall back to numpy.linalg.eig.
    
    M = kg2[:, np.newaxis] * S  # D @ S, (289, 289)
    
    # Use eig for non-Hermitian (safer)
    eigvals, eigvecs = np.linalg.eig(M)
    # Filter real eigenvalues and sort
    eigvals_real = eigvals[np.abs(eigvals.imag) < 1e-8 * np.abs(eigvals.real)]
    eigvals_real = eigvals_real.real
    eigvals_real = np.sort(eigvals_real)
    
    # Take the first n_bands
    if len(eigvals_real) >= n_bands:
        return eigvals_real[:n_bands]
    else:
        # Pad with large values
        result = np.zeros(n_bands)
        result[:len(eigvals_real)] = eigvals_real
        result[len(eigvals_real):] = 1e10
        return result

def _evaluate_gap(cell, eps_low, eps_high):
    """Compute the gap-midgap ratio for bands 4 and 5 (0-indexed: 3 and 4)."""
    # Irreducible BZ for square lattice: kx in [0, 0.5], ky in [0, kx] (triangle)
    # Use a set of wave vectors covering the irreducible BZ
    # Standard points: Gamma(0,0), X(0.5,0), M(0.5,0.5), and points along X-M and Gamma-M
    k_points = []
    # Along Gamma-X: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, 0.0))
    # Along X-M: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((0.5, t))
    # Along Gamma-M: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, t))
    
    max_omega4 = 0.0  # max of band 4 (index 3)
    min_omega5 = np.inf  # min of band 5 (index 4)
    
    for (kx, ky) in k_points:
        bands = _compute_bands(cell, eps_low, eps_high, kx, ky, n_bands=6)
        if len(bands) >= 5:
            max_omega4 = max(max_omega4, bands[3])
            min_omega5 = min(min_omega5, bands[4])
    
    if min_omega5 == np.inf or max_omega4 == 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / midgap

# ---------------------------------------------------------------------------
# Initial candidate generators
# ---------------------------------------------------------------------------
def _make_ring_cell(size, inner_r, outer_r):
    """Ring: dielectric where inner_r < r < outer_r."""
    cell = np.zeros((size, size), dtype=np.int8)
    cx = cy = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
            if inner_r < r < outer_r:
                cell[i, j] = 1
    return cell

def _make_cross_cell(size, width):
    """Cross: dielectric in horizontal and vertical strips of given width."""
    cell = np.zeros((size, size), dtype=np.int8)
    cx = cy = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            if abs(i - cx) < width / 2.0 or abs(j - cy) < width / 2.0:
                cell[i, j] = 1
    return cell

def _make_rod_cell(size, radius):
    """Solid rod."""
    cell = np.zeros((size, size), dtype=np.int8)
    cx = cy = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
            if r < radius:
                cell[i, j] = 1
    return cell

# ---------------------------------------------------------------------------
# Main design function
# ---------------------------------------------------------------------------
def design(problem):
    """Active local search with deadline guard."""
    start_time = time.time()
    deadline = start_time + 200.0  # 200s safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    random.seed(42)
    
    # Get symmetry orbits
    orbits = _get_orbits(size)
    n_orbits = len(orbits)
    
    def cell_to_orbits(cell):
        """Extract orbit values from a cell (first element of each orbit)."""
        vals = []
        for orbit in orbits:
            i, j = orbit[0]
            vals.append(int(cell[i, j]))
        return vals
    
    def orbits_to_cell(vals):
        return _orbit_to_cell(vals, orbits, size)
    
    # Generate initial candidates
    candidates = []
    
    # Ring with various parameters
    for inner_r, outer_r in [(3, 10), (4, 11), (5, 12), (3, 9), (4, 10)]:
        cell = _make_ring_cell(size, inner_r, outer_r)
        candidates.append(cell)
    
    # Cross with various widths
    for width in [4, 5, 6, 7, 8]:
        cell = _make_cross_cell(size, width)
        candidates.append(cell)
    
    # Rod
    for radius in [5, 6, 7, 8]:
        cell = _make_rod_cell(size, radius)
        candidates.append(cell)
    
    # Evaluate all candidates, find best
    best_score = -np.inf
    best_cell = candidates[0]
    
    for cell in candidates:
        score = _evaluate_gap(cell, eps_low, eps_high)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Local search: flip one orbit at a time
    current_orbits = cell_to_orbits(best_cell)
    current_score = best_score
    best_orbits = current_orbits[:]
    best_cell_full = best_cell.copy()
    
    # Simulated annealing style: accept worse moves with probability exp(-delta/T)
    T = 0.1  # initial temperature
    T_min = 0.01
    cooling = 0.995
    
    step_count = 0
    while time.time() < deadline - 5.0:  # 5s safety margin
        step_count += 1
        
        # Pick a random orbit to flip
        orbit_idx = random.randint(0, n_orbits - 1)
        new_orbits = current_orbits[:]
        new_orbits[orbit_idx] = 1 - new_orbits[orbit_idx]
        
        new_cell = orbits_to_cell(new_orbits)
        new_score = _evaluate_gap(new_cell, eps_low, eps_high)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(-delta / T):
            current_orbits = new_orbits
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_orbits = current_orbits[:]
                best_cell_full = new_cell.copy()
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Check time more frequently
        if step_count % 10 == 0 and time.time() > deadline - 10.0:
            break
    
    # Convert best orbit values to nested list
    final_cell = orbits_to_cell(best_orbits, orbits, size)
    
    # Ensure symmetry (should be guaranteed by orbit construction)
    # Convert to nested list of ints
    result = [[int(final_cell[i][j]) for j in range(size)] for i in range(size)]
    
    return result
