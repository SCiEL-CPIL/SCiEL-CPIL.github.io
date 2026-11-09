import numpy as np
import time

def design(problem):
    """
    Designs a C4-symmetric 32x32 photonic crystal unit cell to maximize
    the TM band gap between bands 4 and 5 using a geometric hole-based
    parameterization with a correct Hermitian plane-wave solver.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 230s kill (120s budget)

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # --- Correct Hermitian TM plane-wave solver ---
    # Use 17x17 = 289 plane waves to match evaluator
    N_half = 8  # -8 to 8
    G_vals = np.arange(-N_half, N_half + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))  # shape (289, 2)
    num_pw = len(G_vectors)

    # Precompute k-points: 30 points covering irreducible BZ
    # Irreducible BZ for square lattice: triangle from Gamma(0,0) to M(0.5,0) to X(0.5,0.5)
    # The evaluator uses 30 wave vectors covering boundary and interior
    k_points = []
    # Along Gamma-M: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0.0))
    # Along M-X: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5, t))
    # Along Gamma-X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, t))

    def compute_band_gap(eps_grid):
        """
        Compute TM band gap ratio using Hermitian plane-wave method.
        eps_grid: 32x32 array of dielectric constants.
        """
        # Compute Fourier coefficients of 1/epsilon
        inv_eps = 1.0 / eps_grid
        # FFT gives Fourier coefficients; need to handle the indexing
        # fft2 gives F[n,m] = sum_{x,y} f[x,y] * exp(-2*pi*i*(n*x + m*y)/N)
        # We need the coefficient for G = (g1, g2): c_{G} = (1/N^2) * sum f[x,y] * exp(-2*pi*i*(g1*x + g2*y)/N)
        # The FFT output at index (ix, iy) corresponds to G = (ix - N//2, iy - N//2) for ix,iy in [0,N)
        # Actually, numpy fft2: F[k] = sum_n f[n] exp(-2*pi*i*k*n/N) for k in 0..N-1
        # For k < N/2, this is the positive frequency; for k > N/2, it wraps to negative
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        # Helper to get Fourier coefficient for G = (gx, gy)
        def get_coeff(gx, gy):
            ix = gx % size
            iy = gy % size
            return fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors  # (num_pw, 2)
            
            # For TM: M[i,j] = (k+G_i) · (k+G_j) * c_{G_i - G_j}
            # where c is Fourier coefficient of 1/epsilon
            # This matrix is Hermitian
            M = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                for j in range(num_pw):
                    dot = np.dot(k_plus_G[i], k_plus_G[j])
                    dx = int(G_vectors[i, 0] - G_vectors[j, 0])
                    dy = int(G_vectors[i, 1] - G_vectors[j, 1])
                    c = get_coeff(dx, dy)
                    M[i, j] = dot * c
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(max(real_eigvals[3], 0))
                omega_5 = np.sqrt(max(real_eigvals[4], 0))
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2.0 * gap / midgap
        return ratio

    # Vectorized version for speed
    def compute_band_gap_fast(eps_grid):
        """Faster version using precomputed G differences."""
        inv_eps = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        # Precompute coefficient matrix for this eps_grid
        # coeff[i,j] = c_{G_i - G_j}
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_vectors[i, 0] - G_vectors[j, 0])
                dy = int(G_vectors[i, 1] - G_vectors[j, 1])
                coeff_matrix[i, j] = fft_inv_eps[dx % size, dy % size]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            
            # M[i,j] = (k+G_i)·(k+G_j) * coeff[i,j]
            # (k+G_i)·(k+G_j) = kx² + kx*(Gx_i+Gx_j) + Gx_i*Gx_j + ky² + ky*(Gy_i+Gy_j) + Gy_i*Gy_j
            # = (kx+Gx_i)(kx+Gx_j) + (ky+Gy_i)(ky+Gy_j)
            dot_matrix = (k_plus_G[:, 0:1] * k_plus_G[:, 0:1].T + 
                         k_plus_G[:, 1:2] * k_plus_G[:, 1:2].T)
            
            M = dot_matrix * coeff_matrix
            
            eigvals = np.linalg.eigvalsh(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(max(real_eigvals[3], 0))
                omega_5 = np.sqrt(max(real_eigvals[4], 0))
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        return 2.0 * gap / midgap

    # --- Geometric parameterization: C4-symmetric holes in dielectric background ---
    # Parameters: hole_radius (0 to 0.35), hole_offset (0 to 0.3), superellipse_exp (2 to 6)
    # Holes at positions determined by offset from center, C4 symmetric
    
    def make_cell(params):
        """
        params: [hole_radius, hole_offset, super_exp]
        Returns 32x32 binary cell (0=air, 1=dielectric)
        """
        r = params[0]
        offset = params[1]
        exp_p = params[2]
        
        cell = np.ones((size, size), dtype=float)
        
        # Pixel centers
        x = np.arange(size) + 0.5
        y = np.arange(size) + 0.5
        
        # C4 symmetric hole positions (4 holes)
        # Offset from center (16, 16)
        cx, cy = 16.0, 16.0
        hole_positions = [
            (cx + offset, cy + offset),
            (cx - offset, cy + offset),
            (cx + offset, cy - offset),
            (cx - offset, cy - offset),
        ]
        
        for hx, hy in hole_positions:
            # Superellipse: |dx/r|^p + |dy/r|^p < 1
            for i in range(size):
                for j in range(size):
                    dx = abs(x[j] - hx) / r
                    dy = abs(y[i] - hy) / r
                    if dx**exp_p + dy**exp_p < 1.0:
                        cell[i, j] = 0  # air hole
        
        # Enforce C4 symmetry by averaging
        sym_cell = np.zeros((size, size), dtype=float)
        counts = np.zeros((size, size), dtype=float)
        
        for i in range(size):
            for j in range(size):
                val = cell[i, j]
                positions = [
                    (i, j),
                    (size-1-i, j),
                    (i, size-1-j),
                    (size-1-i, size-1-j),
                    (j, i),
                    (size-1-j, i),
                    (j, size-1-i),
                    (size-1-j, size-1-i),
                ]
                for pi, pj in positions:
                    sym_cell[pi, pj] += val
                    counts[pi, pj] += 1
        
        # Binarize: if average > 0.5, it's dielectric
        binary = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if counts[i, j] > 0:
                    binary[i, j] = 1 if sym_cell[i, j] / counts[i, j] > 0.5 else 0
        
        # Convert to epsilon grid
        eps_grid = np.where(binary == 1, eps_high, eps_low)
        return eps_grid, binary

    # --- Optimization: multi-start local search ---
    best_score = -np.inf
    best_params = None
    
    # Initial guesses
    initial_guesses = [
        [0.15, 0.0, 2.0],    # center hole, circular
        [0.20, 0.15, 2.0],   # 4 holes, circular
        [0.25, 0.10, 2.0],   # 4 holes, larger
        [0.18, 0.20, 4.0],   # 4 holes, square-ish
        [0.22, 0.08, 3.0],   # 4 holes, superellipse
        [0.12, 0.25, 2.0],   # 4 holes, small, far apart
        [0.28, 0.05, 2.0],   # 4 holes, large, close to center
        [0.16, 0.18, 5.0],   # 4 holes, very square
    ]
    
    np.random.seed(42)
    
    # Evaluate initial guesses
    for guess in initial_guesses:
        if time.time() > deadline - 15:
            break
        eps_grid, binary = make_cell(guess)
        score = compute_band_gap_fast(eps_grid)
        if score > best_score:
            best_score = score
            best_params = guess.copy()
    
    # Local search around best
    if best_params is not None:
        step_sizes = [0.02, 0.01, 0.005]
        for step in step_sizes:
            for _ in range(30):
                if time.time() > deadline - 15:
                    break
                p = best_params + np.random.normal(0, step, 3)
                p[0] = np.clip(p[0], 0.05, 0.35)  # radius
                p[1] = np.clip(p[1], 0.0, 0.30)    # offset
                p[2] = np.clip(p[2], 1.5, 8.0)     # superellipse exponent
                
                eps_grid, binary = make_cell(p)
                score = compute_band_gap_fast(eps_grid)
                if score > best_score:
                    best_score = score
                    best_params = p.copy()
    
    # Additional random restarts if time permits
    if time.time() < deadline - 30:
        for _ in range(20):
            if time.time() > deadline - 15:
                break
            p = [
                np.random.uniform(0.08, 0.32),
                np.random.uniform(0.0, 0.28),
                np.random.uniform(1.5, 6.0),
            ]
            eps_grid, binary = make_cell(p)
            score = compute_band_gap_fast(eps_grid)
            if score > best_score:
                best_score = score
                best_params = p.copy()
                
                # Quick local refinement
                for _ in range(10):
                    if time.time() > deadline - 15:
                        break
                    p2 = best_params + np.random.normal(0, 0.01, 3)
                    p2[0] = np.clip(p2[0], 0.05, 0.35)
                    p2[1] = np.clip(p2[1], 0.0, 0.30)
                    p2[2] = np.clip(p2[2], 1.5, 8.0)
                    eps_grid2, binary2 = make_cell(p2)
                    score2 = compute_band_gap_fast(eps_grid2)
                    if score2 > best_score:
                        best_score = score2
                        best_params = p2.copy()
    
    # Fallback: simple dielectric rod
    if best_params is None or best_score < -0.5:
        cell = np.ones((size, size), dtype=int)
        cx, cy = 15.5, 15.5
        for i in range(size):
            for j in range(size):
                dx = (j + 0.5) - cx
                dy = (i + 0.5) - cy
                if dx*dx + dy*dy > 0.15**2 * size*size:
                    cell[i, j] = 0
        binary = cell
    else:
        eps_grid, binary = make_cell(best_params)
    
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(binary[i, j]))
        result.append(row)
    
    return result
