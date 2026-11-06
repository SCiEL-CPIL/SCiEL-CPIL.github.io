import numpy as np
import time
import math

def _get_d4_orbits(n=32):
    """Get all D4 orbits of pixels in an n x n grid."""
    orbits = []
    seen = set()
    for i in range(n):
        for j in range(n):
            if (i, j) in seen:
                continue
            orbit = set()
            # D4 operations:
            # Identity: (i, j)
            # Rot 90: (j, n-1-i)
            # Rot 180: (n-1-i, n-1-j)
            # Rot 270: (n-1-j, i)
            # Mirror X (horizontal flip): (n-1-i, j)
            # Mirror Y (vertical flip): (i, n-1-j)
            # Diag 1: (j, i)
            # Diag 2: (n-1-j, n-1-i)
            
            ops = [
                (i, j),
                (j, n-1-i),
                (n-1-i, n-1-j),
                (n-1-j, i),
                (n-1-i, j),
                (i, n-1-j),
                (j, i),
                (n-1-j, n-1-i)
            ]
            for op in ops:
                orbit.add(op)
            orbits.append(list(orbit))
            seen.update(orbit)
    return orbits

def _build_pw_solver(size=32, npw=13, n_bands=6, eps_high=12.0, eps_low=1.0):
    """Build a fast plane-wave band solver for the given pixel grid."""
    half = npw // 2  # For odd npw, we use -half to half
    
    # Reciprocal lattice vectors: G = (2*pi*m, 2*pi*n)
    # We use a diamond-shaped set of G vectors for efficiency
    G_vecs = []
    for m in range(-half, half+1):
        for n in range(-half, half+1):
            if abs(m) + abs(n) <= half:  # Diamond shape
                G_vecs.append((m, n))
    G_vecs = np.array(G_vecs, dtype=float) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute G^2
    G2 = (G_vecs[:, 0]**2 + G_vecs[:, 1]**2).astype(float)
    
    # K-points in the irreducible BZ
    # Irreducible BZ for square lattice: triangle with vertices (0,0), (pi,0), (pi,pi)
    # Path: Gamma(0,0) -> X(pi,0) -> M(pi,pi) -> Gamma(0,0)
    # Actually the standard path is Gamma -> X -> M -> Gamma
    # where X = (pi, 0) and M = (pi, pi)
    # But for square lattice with a=1, BZ is [-pi, pi] x [-pi, pi].
    # High symmetry points: Gamma=(0,0), X=(pi,0), M=(pi,pi).
    # Irreducible wedge is 1/8 of BZ.
    # Let's use a set of points along Gamma-X, X-M, M-Gamma.
    
    k_points = []
    n_k = 15  # Number of k-points along each segment
    # Segment 1: Gamma(0,0) to X(pi, 0)
    for i in range(n_k):
        t = i / (n_k - 1)
        k_points.append((t * np.pi, 0.0))
    # Segment 2: X(pi, 0) to M(pi, pi)
    for i in range(1, n_k):
        t = i / (n_k - 1)
        k_points.append((np.pi, t * np.pi))
    # Segment 3: M(pi, pi) to Gamma(0, 0)
    for i in range(1, n_k):
        t = i / (n_k - 1)
        k_points.append(((1 - t) * np.pi, (1 - t) * np.pi))
    
    k_points = np.array(k_points)
    n_kpts = len(k_points)
    
    def compute_eps_fourier(cell_grid):
        """Compute Fourier coefficients of epsilon from the pixel grid."""
        eps = eps_low + (eps_high - eps_low) * cell_grid
        # 2D FFT
        F = np.fft.fft2(eps)
        F = F / (size * size)  # Normalize
        return F
    
    def solve_bands_at_k(kx, ky, eps_F, G_vecs, G2):
        """Solve for the first n_bands eigenvalues at a given k-point."""
        nG = len(G_vecs)
        # k + G_m for all m
        kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
        kG2 = kG[:, 0]**2 + kG[:, 1]**2
        
        # Build the matrix for TM
        # The equation is: Σ_m [(ω/c)² ε_{n-m}] E_m = (k+G_n)² E_n
        # This is: A * E = D * E where A_{nm} = (ω/c)² ε_{n-m}, D = diag((k+G_n)²)
        # So: A_ε * E = (1/λ) D * E where λ = (ω/c)²
        # Or: D^{-1} A_ε * E = λ E
        
        # A_ε is the circulant matrix with first column = ε_F
        # A_ε = fft2(eps) gives the Fourier coefficients
        # The matrix element A_ε[n,m] = ε_F[(n-m) mod nG]
        # But we need to map G-vector indices to Fourier coefficient indices.
        
        # Let me create a lookup: for each pair (n,m), find the index q such that
        # G_q = G_n - G_m, then A_ε[n,m] = ε_F[q_index]
        
        # Actually, since the grid is 32x32 and we're using a diamond of G vectors,
        # the differences may not all be in our G set. Let me just use the FFT
        # coefficients directly.
        
        # Alternative approach: Since we have the pixel grid, let me compute the
        # matrix elements directly from the pixel data.
        # 
        # ε(r) = ε_low + (eps_high - eps_low) * f(r) where f(r) is 0/1
        # [ε]_q = (1/A) ∫ ε(r) e^{-i G_q · r} dr
        # For a pixel grid, this is just the 2D DFT of the pixel values.
        
        # Let me precompute the epsilon grid and its DFT once, then for each k-point
        # build the matrix.
        
        # For now, let me just form the matrix using the Fourier coefficients.
        # The key insight: A_ε[n,m] = [ε]_{G_n - G_m}
        
        # I need a mapping from (dm, dn) integers to the FFT index.
        # G_q = (2π*dm, 2π*dn) corresponds to FFT index (dm mod size, dn mod size)
        
        # Create a 2D array of G vectors for quick lookup
        G_lookup = {}
        for idx, (gm, gn) in enumerate(G_vecs):
            dm = round(gm / (2*np.pi))
            dn = round(gn / (2*np.pi))
            G_lookup[(dm, dn)] = idx
        
        # Build matrix M where M[n,m] = ε_F at (G_n - G_m)
        # Then solve: M * E = (1/ω²) * diag((k+G_n)²) * E
        # i.e., diag((k+G_n)²)^{-1} * M * E = ω² * E
        # i.e., solve eigenvalue problem: M * E = μ * diag((k+G_n)²) * E
        # where μ = 1/ω²
        
        M = np.zeros((nG, nG), dtype=complex)
        for n in range(nG):
            for m in range(nG):
                dm = round(G_vecs[n, 0] / (2*np.pi)) - round(G_vecs[m, 0] / (2*np.pi))
                dn = round(G_vecs[n, 1] / (2*np.pi)) - round(G_vecs[m, 1] / (2*np.pi))
                # Get FFT coefficient at (dm, dn)
                idx_m = dm % size
                idx_n = dn % size
                M[n, m] = eps_F[idx_n, idx_m]
        
        D = np.diag(kG2)
        
        # Solve generalized eigenvalue: M v = μ D v
        # Or: D^{-1} M v = μ v  (if D is invertible)
        # D has diagonal entries (k+G_n)² which can be zero for some k, G
        # Handle this by adding a small regularization or using scipy
        
        try:
            D_inv = np.diag(1.0 / kG2)
            A = D_inv @ M
            # A should be Hermitian if eps is real
            # Use eigvalsh for speed
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(eigvals.real)
            return eigvals[:n_bands]
        except:
            return None
    
    return k_points, G_vecs, G2, compute_eps_fourier, solve_bands_at_k

def _compute_gap_score(cell_grid, k_points, G_vecs, G2, compute_eps_fourier, solve_bands_at_k, n_bands=6):
    """Compute the 4-5 TM band gap score for a given cell grid."""
    eps_F = compute_eps_fourier(cell_grid.astype(float))
    
    omega4_vals = []
    omega5_vals = []
    
    for (kx, ky) in k_points:
        eigvals = solve_bands_at_k(kx, ky, eps_F, G_vecs, G2)
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
    
    if min_omega5 + max_omega4 == 0:
        return -1.0
    
    gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    
    return gap_ratio

def _is_symmetric(cell):
    """Check if cell has the required 4-fold symmetry."""
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
    return _is_symmetric_transpose(cell)

def _is_symmetric_transpose(cell):
    n = len(cell)
    for i in range(n):
        for j in range(n):
            if cell[i][j] != cell[j][i]:
                return False
    return True

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for evaluator's 230s limit
    
    # Build the solver infrastructure
    npw = 13  # Diamond shape, about 49 G vectors
    n_bands = 6
    
    k_points, G_vecs, G2, compute_eps_fourier, solve_bands_at_k = _build_pw_solver(
        size=size, npw=npw, n_bands=n_bands, eps_high=eps_high, eps_low=eps_low
    )
    
    # Get D4 orbits for symmetry-constrained search
    orbits = _get_d4_orbits(size)
    
    # Strategy:
    # 1. Start with a checkerboard-like pattern based on product of two square waves
    # 2. Optimize duty cycles and widths via analytic Fourier coefficient analysis
    # 3. Refine with D4-symmetric pixel flips using the PW solver
    
    # Initial design: product of two 1D square waves
    # For a square wave with duty cycle d and period 1:
    # f(x) = 1 if x mod 1 < d, else 0
    # The 2D pattern is f(x) * g(y) where f and g are independent
    
    def make_checkerboard_pattern(duty_x, duty_y, offset_x=0.0, offset_y=0.0):
        """Create a 2D pattern as product of two 1D square waves."""
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                # Square wave in x
                fx = 1.0 if ((x + offset_x) % 1.0) < duty_x else 0.0
                # Square wave in y
                fy = 1.0 if ((y + offset_y) % 1.0) < duty_y else 0.0
                cell[i][j] = int(fx * fy)
        return cell
    
    # Try several initial patterns
    best_cell = None
    best_score = -1.0
    
    # Sweep duty cycles
    duty_vals = [0.3, 0.35, 0.4, 0.45, 0.5]
    
    for dx in duty_vals:
        for dy in duty_vals:
            if time.time() > deadline - 30:
                break
            cell = make_checkerboard_pattern(dx, dy)
            # Ensure symmetry (checkerboard with duty 0.5 is symmetric)
            if not _is_symmetric(cell):
                continue
            score = _compute_gap_score(
                np.array(cell, dtype=float), k_points, G_vecs, G2,
                compute_eps_fourier, solve_bands_at_k, n_bands
            )
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
    
    # If no valid cell found, use a default
    if best_cell is None:
        best_cell = make_checkerboard_pattern(0.5, 0.5)
    
    # Now do local search with D4-symmetric pixel flips
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    
    # Precompute orbit indices for fast flipping
    orbit_indices = []
    for orbit in orbits:
        idxs = []
        for (i, j) in orbit:
            idxs.append((i, j))
        orbit_indices.append(idxs)
    
    # Hill climbing search
    max_iter = 20000
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < deadline - 10 and iter_count < max_iter:
        iter_count += 1
        
        # Pick a random orbit
        idx = np.random.randint(0, len(orbit_indices))
        orbit = orbit_indices[idx]
        
        # Try flipping all pixels in the orbit
        new_cell = [row[:] for row in current_cell]
        for (ri, ci) in orbit:
            new_cell[ri][ci] = 1 - new_cell[ri][ci]
        
        new_score = _compute_gap_score(
            np.array(new_cell, dtype=float), k_points, G_vecs, G2,
            compute_eps_fourier, solve_bands_at_k, n_bands
        )
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
        else:
            no_improve_count += 1
            # If stuck for too long, try a random restart
            if no_improve_count > 500:
                no_improve_count = 0
                # Perturb the current solution
                for _ in range(5):
                    idx2 = np.random.randint(0, len(orbit_indices))
                    orbit2 = orbit_indices[idx2]
                    for (ri, ci) in orbit2:
                        current_cell[ri][ci] = 1 - current_cell[ri][ci]
                current_score = _compute_gap_score(
                    np.array(current_cell, dtype=float), k_points, G_vecs, G2,
                    compute_eps_fourier, solve_bands_at_k, n_bands
                )
    
    # Verify symmetry of the best cell
    if not _is_symmetric(best_cell):
        # Force symmetry by averaging
        forced = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # Average over all D4 operations
                ops = [
                    (i, j),
                    (j, size-1-i),
                    (size-1-i, size-1-j),
                    (size-1-j, i),
                    (size-1-i, j),
                    (i, size-1-j),
                    (j, i),
                    (size-1-j, size-1-i)
                ]
                val = sum(best_cell[oi][oj] for oi, oj in ops) / len(ops)
                forced[i][j] = 1 if val > 0.5 else 0
        best_cell = forced
    
    return best_cell
