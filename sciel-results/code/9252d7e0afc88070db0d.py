"""Design a photonic crystal unit cell maximizing TM band gap between bands 4 and 5."""
import numpy as np
from scipy.optimize import minimize
import time

def _pw_solver(cell, n_pw=17, n_kpts=12):
    """Fast plane-wave solver for TM bands. Returns (max_omega_4, min_omega_5)."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    
    # Get irreducible BZ k-points for square lattice
    # Irreducible BZ: 0 <= kx <= ky, kx^2 + ky^2 <= (pi/a)^2
    # We'll use a set of k-points along the high-symmetry path
    # For a square lattice, the irreducible BZ is a triangle
    # k-points: (0,0), (pi,0), (pi/2, pi/2), (pi, pi)
    # But we need more points for accurate gap calculation
    
    # Generate k-points in the irreducible BZ
    # The irreducible BZ of a square lattice is bounded by kx=0, ky=kx, and kx+ky=pi
    # Actually, for a square lattice with lattice constant a, the BZ is [-pi/a, pi/a]^2
    # The irreducible BZ is the region 0 <= kx <= ky, kx^2+ky^2 <= (pi/a)^2
    
    # Let's use a simple set of k-points
    kx_vals = np.linspace(0, np.pi, 7)
    ky_vals = np.linspace(0, np.pi, 7)
    
    # Filter to irreducible BZ: 0 <= kx <= ky, kx^2 + ky^2 <= pi^2
    k_points = []
    for kx in kx_vals:
        for ky in ky_vals:
            if kx <= ky and kx**2 + ky**2 <= np.pi**2 + 1e-10:
                k_points.append((kx, ky))
    
    if not k_points:
        k_points = [(0, 0), (np.pi/2, np.pi/2), (np.pi, 0)]
    
    # Plane-wave expansion
    # For a square lattice with lattice constant a=1, reciprocal lattice vectors are (2pi, 0), (0, 2pi)
    # We use plane waves with indices in a square of side n_pw
    
    half = n_pw // 2
    g_indices = np.array([[i, j] for i in range(-half, half+1) for j in range(-half, half+1)])
    n_g = len(g_indices)
    
    # Compute Fourier coefficients of 1/eps for TM mode
    # eps is defined on a pixel grid, so we need to compute its Fourier coefficients
    # Using the pixel values directly
    
    # For TM polarization, the eigenvalue equation is:
    # sum_G' (1/eps_{G-G'}) (k+G)·(k+G') A_{G'} = omega^2 A_G
    # where 1/eps_{G} are the Fourier coefficients of 1/eps(r)
    
    # Compute 1/eps on the pixel grid
    inv_eps = 1.0 / eps
    
    # Compute Fourier coefficients using DFT
    # The pixel grid is size x size, so we can use FFT
    inv_eps_2d = inv_eps  # already 2D
    
    # Compute Fourier coefficients
    # G = (2pi*m, 2pi*n) for integers m, n
    # eps_G = (1/size^2) * sum_{i,j} eps_{i,j} * exp(-i G·r_{i,j})
    # where r_{i,j} = ((i+0.5)/size, (j+0.5)/size)
    
    # Use FFT to compute this efficiently
    # Shift the grid so that the center is at the origin for proper phase
    # Actually, let's just compute directly for the needed G vectors
    
    # For efficiency, compute all needed Fourier coefficients at once
    # The G vectors we need are those with |G| <= some cutoff
    
    # Let's compute the Fourier coefficients using a direct sum for the needed G vectors
    # This is O(n_g * size^2) which is manageable for size=32, n_g~300
    
    # Actually, let's use FFT which is O(size^2 log size)
    # The FFT gives us Fourier coefficients at discrete frequencies
    
    # For a size x size grid, the FFT gives coefficients at G = (2pi*m/size, 2pi*n/size) for m,n in [-size/2, size/2)
    # But we need G = (2pi*m, 2pi*n) for integer m, n, which corresponds to the lattice periodicity
    
    # The key insight: if we define the dielectric function on a size x size grid representing one unit cell,
    # then the Fourier coefficients at G = (2pi*m, 2pi*n) are:
    # eps_G = (1/size^2) * sum_{i=0}^{size-1} sum_{j=0}^{size-1} eps_{i,j} * exp(-i 2pi (m*i/size + n*j/size))
    
    # This is exactly the 2D DFT! So we can use FFT.
    
    # Compute 2D FFT of inv_eps
    inv_eps_fft = np.fft.fft2(inv_eps_2d)
    inv_eps_fft /= size * size
    
    # The FFT gives coefficients at frequencies m = 0, 1, ..., size-1
    # We need to map these to the correct G vectors
    # m=0 -> G_x=0, m=1 -> G_x=2pi/size, ..., m=size/2 -> G_x=pi, m=size/2+1 -> G_x=-pi, etc.
    
    # Actually, for our purposes, we need G vectors that are integer multiples of 2pi (since the lattice constant is 1)
    # The FFT of a function defined on [0,1] x [0,1] with size x size points gives:
    # f_G for G = (2pi*m/size, 2pi*n/size)
    # But we want G = (2pi*m, 2pi*n) for integer m, n
    
    # The issue is that the pixel grid represents the unit cell [0,1] x [0,1], so the natural Fourier modes are exp(i 2pi (m x + n y))
    # which have G = (2pi*m, 2pi*n). This is exactly what we want!
    
    # So the FFT coefficients at index (m, n) correspond to G = (2pi*m, 2pi*n) for m, n in the appropriate range.
    
    # For a size x size grid, the FFT indices go from 0 to size-1
    # Index m corresponds to frequency m if m <= size/2, and m-size if m > size/2
    # So G_x = 2pi * (m if m <= size/2 else m-size)
    
    # Let's create a mapping from FFT index to G vector
    fft_indices = np.arange(size)
    g_x_fft = np.where(fft_indices <= size // 2, fft_indices, fft_indices - size)
    g_y_fft = np.where(fft_indices <= size // 2, fft_indices, fft_indices - size)
    
    # Create 2D arrays of G_x and G_y
    G_x_2d, G_y_2d = np.meshgrid(g_x_fft, g_y_fft, indexing='ij')
    
    # Now we can extract the Fourier coefficient for any G = (2pi*m, 2pi*n)
    # by looking up the FFT at the appropriate index
    
    def get_fourier_coeff(m, n):
        """Get Fourier coefficient of inv_eps at G = (2pi*m, 2pi*n)."""
        # Map m, n to FFT indices
        idx_m = m % size
        idx_n = n % size
        return inv_eps_fft[idx_m, idx_n]
    
    # Now set up the eigenvalue problem for each k-point
    # For TM: sum_{G'} (1/eps)_{G-G'} (k+G)·(k+G') A_{G'} = omega^2 A_G
    
    # We'll use the plane waves with G vectors from our g_indices list
    # (k+G)·(k+G') = (kx+Gx)(kx+Gx') + (ky+Gy)(ky+Gy')
    
    max_omega_4 = 0.0
    min_omega_5 = np.inf
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        
        # Build the matrix for this k-point
        # Matrix element: M_{G,G'} = (1/eps)_{G-G'} * (k+G)·(k+G')
        
        # Precompute (k+G) for each G
        kg = k + g_indices * 2 * np.pi  # G vectors are 2pi * g_indices
        
        # (k+G)·(k+G') for all pairs
        # kg[i] · kg[j] = dot product
        
        # Build the matrix
        M = np.zeros((n_g, n_g), dtype=complex)
        
        for i in range(n_g):
            Gi = g_indices[i]
            for j in range(n_g):
                Gj = g_indices[j]
                # G - G'
                dG = Gi - Gj
                # Get Fourier coefficient of 1/eps at dG
                coeff = get_fourier_coeff(dG[0], dG[1])
                # (k+G)·(k+G')
                dot = np.dot(kg[i], kg[j])
                M[i, j] = coeff * dot
        
        # Solve the generalized eigenvalue problem: M A = omega^2 A
        # This is a standard eigenvalue problem since M is already the operator
        
        # M should be Hermitian for real eps, so we can use eigh
        # But let's use eigvals for safety
        eigenvalues = np.linalg.eigvalsh(M)
        eigenvalues = np.sort(np.real(eigenvalues))
        
        # omega = sqrt(eigenvalue)
        omegas = np.sqrt(np.maximum(eigenvalues, 0))
        
        # We need bands 4 and 5 (0-indexed: 3 and 4)
        if len(omegas) >= 5:
            omega_4 = omegas[3]  # 4th band (0-indexed: 3)
            omega_5 = omegas[4]  # 5th band (0-indexed: 4)
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
    
    return max_omega_4, min_omega_5


def _gap_ratio(cell):
    """Compute the gap-midgap ratio for a given cell."""
    max_omega_4, min_omega_5 = _pw_solver(cell)
    if min_omega_5 <= 0 or max_omega_4 <= 0:
        return -1.0
    return 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)


def design(problem):
    """Design a photonic crystal unit cell maximizing the TM band gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Parameterize the design using the top-left quadrant (due to 180° rotational symmetry)
    # cell[i][j] = cell[size-1-i][size-1-j]
    # So we only need to specify i < size/2 or (i == size/2 and j < size/2)
    # For size=32, that's 16*16 = 256 parameters
    
    half = size // 2
    
    def params_to_cell(params):
        """Convert 256 parameters to a 32x32 cell with 180° rotational symmetry."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(half):
            for j in range(half):
                val = 1 if params[i * half + j] > 0.5 else 0
                cell[i, j] = val
                cell[size - 1 - i, size - 1 - j] = val
        return cell.tolist()
    
    def cell_to_params(cell):
        """Convert a 32x32 cell to 256 parameters."""
        params = np.zeros(half * half)
        for i in range(half):
            for j in range(half):
                params[i * half + j] = cell[i][j]
        return params
    
    # Start with a simple rod-like design
    # A centered circle of radius r
    def initial_params(radius_frac=0.3):
        params = np.zeros(half * half)
        for i in range(half):
            for j in range(half):
                # Center of the cell is at (size/2 - 0.5, size/2 - 0.5) in pixel coordinates
                # For the top-left quadrant, the center is at (half - 0.5, half - 0.5) relative to the quadrant
                # Actually, let's think in terms of the full cell coordinates
                # Pixel (i, j) in the full cell has center at ((i+0.5)/size, (j+0.5)/size)
                # Distance from center (0.5, 0.5)
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius_frac * radius_frac:
                    params[i * half + j] = 1.0
        return params
    
    # Try a few initial designs and pick the best
    initial_designs = [
        initial_params(0.2),
        initial_params(0.25),
        initial_params(0.3),
        initial_params(0.35),
    ]
    
    # Also try a cross pattern
    def cross_params(arm_width_frac=0.3):
        params = np.zeros(half * half)
        for i in range(half):
            for j in range(half):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Cross: |x| < w or |y| < w
                w = arm_width_frac / 2
                if abs(x) < w or abs(y) < w:
                    params[i * half + j] = 1.0
        return params
    
    initial_designs.append(cross_params(0.3))
    initial_designs.append(cross_params(0.4))
    
    # Evaluate initial designs
    best_score = -np.inf
    best_params = initial_designs[0]
    
    for params in initial_designs:
        cell = params_to_cell(params)
        score = _gap_ratio(cell)
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # Local search: flip random pixels in the top-left quadrant
    # Use a simple hill-climbing approach
    
    current_params = best_params.copy()
    current_score = best_score
    
    # Try random flips
    np.random.seed(42)
    
    max_iterations = 500
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline:
            break
        
        # Try flipping a random pixel
        idx = np.random.randint(0, half * half)
        new_params = current_params.copy()
        new_params[idx] = 1.0 - new_params[idx]
        
        cell = params_to_cell(new_params)
        score = _gap_ratio(cell)
        
        if score > current_score:
            current_params = new_params
            current_score = score
        else:
            # Try a few more flips if we're stuck
            pass
        
        iteration += 1
    
    # Convert to final cell
    final_cell = params_to_cell(current_params)
    
    # Ensure it's a list of lists of ints
    return [[int(v) for v in row] for row in final_cell]
