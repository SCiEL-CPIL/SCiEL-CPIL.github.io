import time
import math
import random
import numpy as np

def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]  # 32
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)
    
    start_time = time.time()
    deadline = 220.0
    margin = 15.0

    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline - margin

    def enforce_symmetry(cell):
        """Enforce full d4 symmetry by averaging over the symmetry group."""
        n = len(cell)
        # We just enforce it by mirroring.
        # The input is assumed to be generated symmetrically or we fix it.
        # For safety, we return the cell as is if it's already symmetric, 
        # but the generation process should ensure it.
        return cell

    def make_cell_from_radii(r_corner, r_edge, r_center):
        """
        Create a 32x32 cell from a 4x4 superlattice of 8x8 blocks.
        Radii are defined for the 3 unique block types due to D4 symmetry.
        """
        n = size
        block_size = n // 4  # 8
        cell = [[0] * n for _ in range(n)]
        
        # Define radii for the 4x4 grid of blocks
        # Indices (bi, bj) in 0..3
        # Symmetry:
        # (0,0) <-> (3,3) <-> (0,3) <-> (3,0) : Corner
        # (0,1) <-> (1,0) <-> (3,2) <-> (2,3) : Edge
        # (1,1) <-> (1,2) <-> (2,1) <-> (2,2) : Center
        
        radii = [[0.0]*4 for _ in range(4)]
        radii[0][0] = r_corner
        radii[3][3] = r_corner
        radii[0][3] = r_corner
        radii[3][0] = r_corner
        
        radii[0][1] = r_edge
        radii[1][0] = r_edge
        radii[3][2] = r_edge
        radii[2][3] = r_edge
        
        radii[1][1] = r_center
        radii[1][2] = r_center
        radii[2][1] = r_center
        radii[2][2] = r_center

        for bi in range(4):
            for bj in range(4):
                r = radii[bi][bj]
                if r <= 0:
                    continue
                # Block center in pixel coords
                ci = bi * block_size + block_size // 2 - 0.5
                cj = bj * block_size + block_size // 2 - 0.5
                # Draw circle
                for i in range(bi * block_size, (bi + 1) * block_size):
                    for j in range(bj * block_size, (bj + 1) * block_size):
                        dx = i - ci
                        dy = j - cj
                        if dx*dx + dy*dy < r*r:
                            cell[i][j] = 1
        return cell

    def compute_eps_fourier(cell, n_g):
        """Compute Fourier coefficients of 1/epsilon or epsilon?
        For TM, we need 1/epsilon in the matrix.
        Let's compute eps_fourier first, then invert the matrix?
        Actually, the matrix element is 1/eps_{G-G'}.
        It is easier to compute the matrix of 1/eps directly?
        No, eps is a convolution. 1/eps is not a simple convolution in real space if eps is binary?
        Wait, eps(x) is binary. 1/eps(x) is also binary (1/1 or 1/12).
        So we can compute the Fourier coefficients of the function f(x) = 1/eps(x).
        """
        n = size
        # f(x,y) = 1/eps(x,y)
        # eps = 12 if cell[i][j]==1 else 1
        # f = 1/12 if cell[i][j]==1 else 1
        
        # We need f_G for G in -n_g to n_g
        f_grid = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if cell[i][j] == 1:
                    f_grid[i][j] = 1.0 / eps_high
                else:
                    f_grid[i][j] = 1.0 / eps_low
        
        # FFT
        f_fft = np.fft.fftshift(np.fft.fft2(f_grid))
        f_fft /= (n * n)
        
        # Extract coefficients for G in [-n_g, n_g]
        coeffs = {}
        for gx in range(-n_g, n_g + 1):
            for gy in range(-n_g, n_g + 1):
                # Map G to FFT index
                # FFT index k corresponds to frequency k if k < n/2, k-n if k >= n/2
                # We use fftshift, so center is 0.
                # Index in shifted array: (gx + n_g, gy + n_g) is not correct.
                # fftshift moves zero freq to center.
                # Original index: (gx mod n, gy mod n)
                # Shifted index: (gx + n//2) mod n ?
                # Let's just use the unshifted FFT and map indices carefully.
                pass
        
        # Simpler: use unshifted FFT
        f_fft_unshifted = np.fft.fft2(f_grid) / (n * n)
        
        for gx in range(-n_g, n_g + 1):
            for gy in range(-n_g, n_g + 1):
                ix = gx % n
                iy = gy % n
                coeffs[(gx, gy)] = f_fft_unshifted[ix, iy]
        
        return coeffs

    def compute_band_gaps(cell, n_g=5, n_kpts=10):
        """
        Compute gap between band 4 and 5.
        Uses n_g x n_g plane waves.
        """
        n = size
        f_coeffs = compute_eps_fourier(cell, n_g)
        
        # Reciprocal lattice vectors
        # a = 1.0 (in units of lattice constant)
        # G = (2*pi/m, 2*pi/n)
        
        # List of G vectors
        g_vectors = []
        for gx in range(-n_g, n_g + 1):
            for gy in range(-n_g, n_g + 1):
                g_vectors.append((2.0 * math.pi * gx, 2.0 * math.pi * gy))
        
        n_waves = len(g_vectors)
        
        # K-points in irreducible BZ
        # Gamma (0,0), X (pi, 0), M (pi, pi)
        # Path: Gamma -> X -> M -> Gamma
        k_points = []
        # Gamma to X
        for i in range(n_kpts):
            kx = (i / (n_kpts - 1)) * math.pi
            ky = 0.0
            k_points.append((kx, ky))
        # X to M
        for i in range(1, n_kpts):
            kx = math.pi
            ky = (i / (n_kpts - 1)) * math.pi
            k_points.append((kx, ky))
        # M to Gamma
        for i in range(1, n_kpts):
            kx = math.pi * (1 - i / (n_kpts - 1))
            ky = math.pi * (1 - i / (n_kpts - 1))
            k_points.append((kx, ky))
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build matrix
            H = np.zeros((n_waves, n_waves), dtype=np.complex128)
            for i, (gx, gy) in enumerate(g_vectors):
                kx_g = kx + gx
                ky_g = ky + gy
                for j, (gx2, gy2) in enumerate(g_vectors):
                    kx_g2 = kx + gx2
                    ky_g2 = ky + gy2
                    
                    # TM matrix element:
                    # (k+G)^2 delta - (k+G).(k+G2) all divided by f_{G-G2}
                    # Wait, the formula is:
                    # M_{ij} = 1/f_{G_i - G_j} * ( |k+G_i|^2 delta_{ij} - (k+G_i).(k+G_j) )
                    
                    g_diff_x = gx - gx2
                    g_diff_y = gy - gy2
                    f_val = f_coeffs.get((g_diff_x, g_diff_y), 0.0)
                    if abs(f_val) < 1e-9:
                        f_val = 1.0 # Should not happen
                    
                    dot = kx_g * kx_g2 + ky_g * ky_g2
                    norm_sq = kx_g * kx_g + ky_g * ky_g
                    delta = 1.0 if i == j else 0.0
                    
                    val = (norm_sq * delta - dot) / f_val
                    H[i, j] = val
            
            # Solve eigenvalues
            # H is Hermitian
            eigs = np.linalg.eigvalsh(H)
            eigs = np.real(eigs)
            eigs = np.sort(eigs)
            
            # We want bands 4 and 5 (1-indexed)
            # eigs[3] is 4th, eigs[4] is 5th
            if len(eigs) >= 5:
                omega_4 = math.sqrt(max(0.0, eigs[3]))
                omega_5 = math.sqrt(max(0.0, eigs[4]))
                
                if omega_5 > 0 and omega_4 > 0:
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = gap / midgap
        return ratio

    # Phase 1: Search over 3 radii parameters
    best_ratio = -1.0
    best_cell = None
    
    # Grid search
    r_corner_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    r_edge_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    r_center_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    candidates = []
    for rc in r_corner_vals:
        for re in r_edge_vals:
            for rce in r_center_vals:
                candidates.append((rc, re, rce))
    
    # Shuffle to mix good/bad
    random.shuffle(candidates)
    
    for rc, re, rce in candidates:
        if not time_ok():
            break
        cell = make_cell_from_radii(rc, re, rce)
        ratio = compute_band_gaps(cell, n_g=4, n_kpts=5) # Fast eval
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell

    # Phase 2: Local search on best cell
    if best_cell is not None and time_ok():
        current_cell = [row[:] for row in best_cell]
        current_ratio = best_ratio
        n = size
        
        # Precompute orbit for each pixel
        orbits = {}
        for i in range(n):
            for j in range(n):
                orbit = set()
                orbit.add((i, j))
                orbit.add((n - 1 - i, n - 1 - j))
                orbit.add((j, i))
                orbit.add((n - 1 - j, n - 1 - i))
                orbits[(i, j)] = list(orbit)
        
        for iteration in range(200):
            if not time_ok():
                break
            
            improved = False
            # Try flipping a random orbit
            for _ in range(10):
                if not time_ok():
                    break
                ri = random.randint(0, n - 1)
                rj = random.randint(0, n - 1)
                orbit = orbits[(ri, rj)]
                
                new_cell = [row[:] for row in current_cell]
                for (oi, oj) in orbit:
                    new_cell[oi][oj] = 1 - new_cell[oi][oj]
                
                new_ratio = compute_band_gaps(new_cell, n_g=4, n_kpts=5)
                if new_ratio > current_ratio:
                    current_cell = new_cell
                    current_ratio = new_ratio
                    best_cell = current_cell
                    best_ratio = current_ratio
                    improved = True
                    break
            
            if not improved:
                # Try a random perturbation of multiple pixels
                new_cell = [row[:] for row in current_cell]
                for _ in range(3):
                    ri = random.randint(0, n - 1)
                    rj = random.randint(0, n - 1)
                    orbit = orbits[(ri, rj)]
                    for (oi, oj) in orbit:
                        new_cell[oi][oj] = 1 - new_cell[oi][oj]
                
                new_ratio = compute_band_gaps(new_cell, n_g=4, n_kpts=5)
                if new_ratio > current_ratio:
                    current_cell = new_cell
                    current_ratio = new_ratio
                    best_cell = current_cell
                    best_ratio = current_ratio

    if best_cell is None:
        # Fallback
        best_cell = make_cell_from_radii(2.0, 2.0, 2.0)

    return best_cell
