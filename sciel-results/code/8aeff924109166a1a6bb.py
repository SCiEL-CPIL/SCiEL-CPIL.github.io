"""Photonic-crystal band-gap optimizer for TM 4-5 gap on 32x32 square lattice."""
import time
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 230.0  # wall-clock limit; we'll use ~190s for safety


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _make_symmetric(cell):
    """Enforce square symmetry: reverse rows, reverse cols, transpose."""
    size = len(cell)
    arr = np.array(cell, dtype=np.int32)
    # Reverse rows
    arr = arr[::-1, :]
    # Reverse columns
    arr = arr[:, ::-1]
    # Transpose
    arr = arr.T
    # Average with original to get symmetric (but we need 0/1)
    # Actually, let's just take the 1/4 of the cell and mirror
    # Better approach: work with the fundamental domain and mirror
    return arr


def _symmetrize(cell):
    """Given a 32x32 cell, enforce full D4 symmetry about center."""
    size = len(cell)
    arr = np.array(cell, dtype=np.float64)
    # The cell is invariant under: reverse rows, reverse cols, transpose
    # So we can define it by the top-left 16x16 quadrant and mirror
    # Actually for a 32x32 cell centered at (16,16):
    # cell[i][j] = cell[31-i][j] = cell[i][31-j] = cell[j][i]
    # Let's just enforce by averaging over all 4 symmetries + transpose
    
    rev_r = arr[::-1, :]
    rev_c = arr[:, ::-1]
    rev_rc = arr[::-1, ::-1]
    trans = arr.T
    
    # The symmetry group of the square has 8 elements, but for a pixel grid
    # centered at the middle, we have: identity, rev_rows, rev_cols, rev_both,
    # transpose, transpose+rev_rows, transpose+rev_cols, transpose+rev_both
    # But the task says: cell == cell[::-1] row-wise, cell == cell[::-1] col-wise, cell == cell.T
    # So: arr[i][j] == arr[31-i][j] == arr[i][31-j] == arr[j][i]
    
    # Average over all 8 symmetries to get a symmetric result
    symmetries = [
        arr,
        arr[::-1, :],
        arr[:, ::-1],
        arr[::-1, ::-1],
        arr.T,
        arr.T[::-1, :],
        arr.T[:, ::-1],
        arr.T[::-1, ::-1],
    ]
    avg = np.mean(symmetries, axis=0)
    return (avg >= 0.5).astype(np.int32)


def _fourier_coefficients(cell, N):
    """Compute exact Fourier coefficients of the dielectric function.
    
    cell: 32x32 array of 0/1 (1 = dielectric with eps=12, 0 = air with eps=1)
    N: truncation number (N=17 for 289 plane waves, i.e., -17..17)
    Returns: dict mapping (nx, ny) -> complex coefficient of 1/eps(x,y)
    """
    size = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    
    # For TM polarization, we need 1/eps(x,y)
    # 1/eps = 1 for air (cell=0), 1/12 for dielectric (cell=1)
    inv_eps = np.where(np.array(cell) == 1, 1.0/eps_high, 1.0/eps_low)
    
    # Fourier coefficients: G_{nx,ny} = (1/A) * integral exp(-i G.r) /eps(x,y) dx dy
    # For a pixel grid, this is a discrete sum
    # G_{nx,ny} = (1/N_pixels) * sum_{i,j} inv_eps[i,j] * exp(-i (nx*2*pi*(i+0.5)/size + ny*2*pi*(j+0.5)/size))
    
    # Use FFT for efficiency
    # The DFT gives us: F[kx, ky] = sum_{i,j} f[i,j] * exp(-i 2*pi (kx*i + ky*j)/size)
    # We need: sum_{i,j} f[i,j] * exp(-i 2*pi (nx*(i+0.5)/size + ny*(j+0.5)/size))
    # = exp(-i pi nx/size) * exp(-i pi ny/size) * F[nx mod size, ny mod size]
    
    fft = np.fft.fft2(inv_eps)
    fft = fft / (size * size)  # normalize
    
    coeffs = {}
    for nx in range(-N, N+1):
        for ny in range(-N, N+1):
            # Map to FFT index
            kx = nx % size
            ky = ny % size
            phase = np.exp(-1j * np.pi * (nx + ny) / size)
            coeffs[(nx, ny)] = fft[kx, ky] * phase
    
    return coeffs


def _band_structure(cell, N=17, kpoints=None):
    """Compute TM band structure using plane-wave method.
    
    Returns: list of (kx, ky, omega_4, omega_5) for each k-point
    """
    size = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    
    # Get Fourier coefficients of 1/eps
    inv_eps = np.where(np.array(cell) == 1, 1.0/eps_high, 1.0/eps_low)
    fft = np.fft.fft2(inv_eps) / (size * size)
    
    def get_coeff(nx, ny):
        kx = nx % size
        ky = ny % size
        phase = np.exp(-1j * np.pi * (nx + ny) / size)
        return fft[kx, ky] * phase
    
    # Plane wave indices: -N..N for both x and y
    G_indices = list(range(-N, N+1))
    num_pw = len(G_indices)
    
    # Precompute G vectors
    G_vectors = np.array([(nx, ny) for nx in G_indices for ny in G_indices], dtype=np.float64)
    G_magnitudes_sq = G_vectors[:, 0]**2 + G_vectors[:, 1]**2
    
    # k-points: irreducible BZ for square lattice
    # The irreducible BZ is the triangle with vertices (0,0), (1,0), (0.5,0.5)
    # in units of 2*pi/a. Let's use 30 k-points as specified.
    if kpoints is None:
        kpoints = []
        # Along Gamma-X: (0,0) to (1,0)
        for t in np.linspace(0, 1, 10):
            kpoints.append((t, 0))
        # Along X-M: (1,0) to (0.5,0.5)
        for t in np.linspace(0, 1, 10):
            kpoints.append((1 - 0.5*t, 0.5*t))
        # Along M-Gamma: (0.5,0.5) to (0,0)
        for t in np.linspace(0, 1, 10):
            kpoints.append((0.5*(1-t), 0.5*(1-t)))
    
    results = []
    for kx, ky in kpoints:
        # (k+G)^2 for each plane wave
        kg_sq = (G_vectors[:, 0] + kx)**2 + (G_vectors[:, 1] + ky)**2
        
        # Build the eigenvalue matrix for TM:
        # omega^2 * c^2 = (k+G)^2 * sum_{G'} (1/eps)_{G-G'} * c_{G'}
        # In matrix form: M_{G,G'} = (1/eps)_{G-G'} * (k+G')^2
        # Then omega^2 * c^2 = sum_{G'} M_{G,G'} c_{G'}
        # Actually, the standard formulation is:
        # sum_{G'} (k+G)^2 delta_{G,G'} c_{G'} = omega^2/c^2 * sum_{G'} (1/eps)_{G-G'} c_{G'}
        # So: [delta_{G,G'} (k+G)^2 - omega^2/c^2 * (1/eps)_{G-G'}] c_{G'} = 0
        # Or equivalently: (1/eps)_{G-G'}^{-1} * delta_{G,G'} (k+G)^2 c_{G'} = omega^2/c^2 c_G
        # Let's use: A_{G,G'} = (1/eps)_{G-G'} and solve A^{-1} D c = omega^2 c
        # where D is diagonal with (k+G)^2
        
        # More standard: solve the generalized eigenvalue problem
        # D c = omega^2 A c, where D_{G,G'} = (k+G)^2 delta_{G,G'}
        # A_{G,G'} = (1/eps)_{G-G'}
        
        # Build A matrix
        A = np.zeros((num_pw, num_pw), dtype=np.complex128)
        for i in range(num_pw):
            for j in range(num_pw):
                dnx = G_vectors[i, 0] - G_vectors[j, 0]
                dny = G_vectors[i, 1] - G_vectors[j, 1]
                A[i, j] = get_coeff(int(dnx), int(dny))
        
        # D is diagonal
        D = kg_sq
        
        # Solve D c = omega^2 A c
        # => A^{-1} D c = omega^2 c
        # Since A is Hermitian positive definite, we can solve this as a standard
        # eigenvalue problem by computing A^{-1} D
        
        # Use scipy for eigenvalue computation
        try:
            from scipy.linalg import eigh
            # Compute A^{-1} D
            # A is Hermitian, so we can use Cholesky or direct inverse
            # A_inv_D = np.linalg.solve(A, D[:, np.newaxis])  # This doesn't work directly
            # Better: solve A x = D c for x, then x = A^{-1} D c
            # Eigenvalue problem: A^{-1} D c = omega^2 c
            # Equivalent to: D c = omega^2 A c (generalized eigenvalue)
            
            # Use scipy.linalg.eigh for generalized symmetric eigenvalue problem
            # eigh(D, A) where D and A are Hermitian
            # But D is real diagonal and A is complex Hermitian
            # Let's use the standard approach:
            
            # Actually, let's just compute A^{-1} D explicitly
            # A is 289x289, which is manageable
            A_inv = np.linalg.inv(A)
            A_inv_D = A_inv @ np.diag(D)
            
            # A_inv_D should be Hermitian (since A and D are Hermitian)
            # Take the Hermitian part
            A_inv_D = (A_inv_D + A_inv_D.conj().T) / 2
            
            # Compute eigenvalues
            eigvals = np.linalg.eigvalsh(A_inv_D)
            eigvals = np.sort(eigvals.real)
            
            # omega = sqrt(eigenvalue) / c, but we work in units where c=1 and k in units of 2pi/a
            # Actually, the eigenvalues are omega^2/c^2 in our units
            # omega_4 = eigvals[3], omega_5 = eigvals[4]
            omega_4 = math.sqrt(max(eigvals[3], 0))
            omega_5 = math.sqrt(max(eigvals[4], 0))
            
            results.append((kx, ky, omega_4, omega_5))
        except Exception:
            continue
    
    return results


def _score_cell(cell):
    """Compute the gap-midgap ratio for a given cell."""
    try:
        results = _band_structure(cell)
        if len(results) < 2:
            return -1.0
        
        max_omega_4 = max(r[2] for r in results)
        min_omega_5 = min(r[3] for r in results)
        
        numerator = min_omega_5 - max_omega_4
        denominator = min_omega_5 + max_omega_4
        
        if denominator <= 0:
            return -1.0
        
        return 2 * numerator / denominator
    except Exception:
        return -1.0


def _initial_cell():
    """Create an initial dielectric background with air holes."""
    size = 32
    cell = np.ones((size, size), dtype=np.int32)
    
    # Place air holes (0) at the center and at the midpoints of edges
    # This creates a merged-hole geometry
    center = size // 2
    
    # Central hole
    for i in range(size):
        for j in range(size):
            dx = (i - center + 0.5) / size
            dy = (j - center + 0.5) / size
            # Central hole of radius ~0.15
            if dx*dx + dy*dy < 0.15**2:
                cell[i, j] = 0
            # Holes at edge midpoints (distance 0.5 from center)
            # At (0.5, 0), (0, 0.5), etc.
            # These are at distance 0.5 from center
            # Let's add small holes there too
            # Actually, for a square lattice, the periodic images of the central hole
            # at distance 0.5 in x and y directions will merge if the hole is large enough
    
    # Enforce symmetry
    cell = _symmetrize(cell.tolist())
    return cell


def _mutate(cell, mutation_rate=0.1):
    """Randomly flip some pixels while maintaining symmetry."""
    size = len(cell)
    arr = np.array(cell, dtype=np.int32)
    
    # Work with the fundamental domain (top-left quadrant, 16x16)
    # For a 32x32 cell with D4 symmetry, the fundamental domain is 8x8
    # Actually, let's just flip random pixels and re-symmetrize
    
    for _ in range(int(mutation_rate * size * size / 4)):
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        arr[i, j] = 1 - arr[i, j]
    
    return _symmetrize(arr.tolist())


def _hill_climb(cell, max_iters=500):
    """Simple hill climbing to improve the gap."""
    best_cell = [row[:] for row in cell]
    best_score = _score_cell(cell)
    
    for it in range(max_iters):
        if _time_left() < 30:
            break
        
        # Try a few mutations
        best_trial = None
        best_trial_score = best_score
        
        for _ in range(5):
            trial = _mutate(best_cell, mutation_rate=0.05)
            trial_score = _score_cell(trial)
            if trial_score > best_trial_score:
                best_trial_score = trial_score
                best_trial = trial
        
        if best_trial is not None:
            best_cell = best_trial
            best_score = best_trial_score
        else:
            # Stuck, try a bigger mutation
            trial = _mutate(best_cell, mutation_rate=0.2)
            trial_score = _score_cell(trial)
            if trial_score > best_score:
                best_cell = trial
                best_score = trial_score
    
    return best_cell, best_score


def design(problem):
    """Design a 32x32 photonic crystal unit cell for maximum TM 4-5 band gap."""
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Start with dielectric background and air holes
    cell = _initial_cell()
    
    # Verify symmetry
    cell = _symmetrize(cell)
    
    # Initial score
    best_score = _score_cell(cell)
    best_cell = [row[:] for row in cell]
    
    # Hill climbing
    if _time_left() > 60:
        best_cell, best_score = _hill_climb(best_cell, max_iters=200)
    
    # Try a few different initial configurations
    if _time_left() > 30:
        # Try different hole sizes
        for radius in [0.1, 0.12, 0.15, 0.18, 0.2]:
            if _time_left() < 30:
                break
            
            test_cell = np.ones((size, size), dtype=np.int32)
            center = size // 2
            for i in range(size):
                for j in range(size):
                    dx = (i - center + 0.5) / size
                    dy = (j - center + 0.5) / size
                    if dx*dx + dy*dy < radius**2:
                        test_cell[i, j] = 0
            
            test_cell = _symmetrize(test_cell.tolist())
            test_score = _score_cell(test_cell)
            
            if test_score > best_score:
                best_score = test_score
                best_cell = test_cell
        
        # Hill climb the best found
        if _time_left() > 60:
            best_cell, best_score = _hill_climb(best_cell, max_iters=100)
    
    # Final symmetry check
    best_cell = _symmetrize(best_cell)
    
    return best_cell
