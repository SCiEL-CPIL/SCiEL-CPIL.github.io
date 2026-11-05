"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    # Deadline guard: keep ~10s margin before 230s wall-clock kill
    t0 = time.time()
    deadline = t0 + 215.0  # 215s of 230s
    
    # --- Symmetric parameterization ---
    # The cell must be invariant under:
    #   1. 180° rotation: cell[i][j] == cell[31-i][31-j]
    #   2. Transpose:     cell[i][j] == cell[j][i]
    # This means the cell is determined by its upper-left 16x16 quadrant
    # with the constraint that pixel (i,j) in the quadrant maps to:
    #   (i,j), (i, 31-j), (31-i, j), (31-i, 31-j)
    # Wait, let me think more carefully.
    # 
    # cell == cell[::-1] row-wise: cell[i][j] == cell[31-i][j]  (vertical flip)
    # cell == cell[::-1] column-wise: cell[i][j] == cell[i][31-j]  (horizontal flip)
    # cell == cell.T: cell[i][j] == cell[j][i]  (transpose)
    #
    # From vertical flip + horizontal flip + transpose, the cell is determined
    # by the 8x8 upper-left corner? Let's check:
    # cell[i][j] = cell[31-i][j] = cell[31-i][31-j] = cell[j][31-i] = ...
    # Actually with all three symmetries (D4), the fundamental domain is 1/8 of the cell.
    # For 32x32, that's a 16x16 quadrant, but with the diagonal constraint cell[i][j]=cell[j][i],
    # the independent region is the upper triangle of the 16x16 quadrant: i in 0..15, j in i..15.
    # That's 16*17/2 = 136 independent pixels.
    #
    # Actually, let me reconsider. The symmetries are:
    # - cell[::-1] row-wise means reversing the order of rows: new[i] = old[31-i]
    #   So cell[i][j] == cell[31-i][j]
    # - cell[::-1] column-wise: within each row, reverse: cell[i][j] == cell[i][31-j]
    # - cell == cell.T: cell[i][j] == cell[j][i]
    #
    # Combined: cell[i][j] = cell[31-i][j] = cell[31-i][31-j] = cell[j][31-i] = cell[j][i] = ...
    # The orbit of (i,j) under D4 has up to 8 elements.
    # For a 32x32 grid, the fundamental domain is pixels where 0 <= i <= 15 and i <= j <= 15
    # (upper triangle of the top-left 16x16 quadrant), but we need to be careful about
    # the center lines.
    #
    # Let me just use the 16x16 top-left quadrant and enforce symmetry.
    # Independent pixels: for i in 0..15, j in i..15 (upper triangle including diagonal)
    # That's sum_{i=0}^{15} (16-i) = 16+15+...+1 = 136 pixels.
    
    # --- Fast band-structure evaluator using plane-wave method ---
    # For the TM mode in 2D, the eigenvalue problem is:
    #   sum_k' epsilon^{-1}(G-G') (k+G)^2 (k+G') c_{k'} = omega^2 c_k
    # where G are reciprocal lattice vectors, and epsilon^{-1} has Fourier coefficients
    # computed from the pixel map.
    
    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from 16x16 upper-triangle quadrant."""
        N = size
        half = N // 2  # 16
        cell = [[0]*N for _ in range(N)]
        for i in range(half):
            for j in range(i, half):
                val = quadrant[i][j]
                # D4 orbit:
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
                cell[j][i] = val
                cell[j][N-1-i] = val
                cell[N-1-j][i] = val
                cell[N-1-j][N-1-i] = val
        return cell
    
    def compute_eps_inv_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon for the pixel map.
        epsilon(x) = 1 + 11 * f(x) where f is 1 for dielectric, 0 for air.
        1/epsilon(x) = 1 for air, 1/12 for dielectric.
        We need the Fourier series of 1/epsilon over the unit cell.
        """
        N = size
        eps_inv = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps_inv[i, j] = 1.0 / eps_high if cell[i][j] == 1 else 1.0
        
        # Compute Fourier coefficients: eps_inv(G) = (1/N^2) sum_{i,j} eps_inv[i,j] exp(-i G . r_{ij})
        # G = (2*pi*mx/N, 2*pi*my/N) for mx, my in integers
        # r_{ij} = (j/N, i/N)  [x = j/N, y = i/N]
        # 
        # We need coefficients for G vectors up to some cutoff.
        # The evaluator uses 289 plane waves. 289 = 17^2, so likely |mx|,|my| <= 8.
        # Let's compute for mx, my in -8..8.
        
        max_g = 8
        # Use FFT for efficiency
        # eps_inv_fourier[mx, my] = sum_{i,j} eps_inv[i,j] * exp(-2*pi*i*(mx*j/N + my*i/N)) / N^2
        # This is essentially the 2D DFT of eps_inv (with appropriate indexing).
        
        # np.fft.fft2 computes sum_{i,j} f[i,j] exp(-2*pi*i*(m*i/N + n*j/N))
        # We need exp(-2*pi*i*(mx*j/N + my*i/N))
        # So if we use fft2, the first index corresponds to y (row i) and second to x (column j)
        # fft2(f)[my, mx] = sum_{i,j} f[i,j] exp(-2*pi*i*(my*i/N + mx*j/N))
        # That's exactly what we want with the sign convention.
        
        fft_coeff = np.fft.fft2(eps_inv) / (N * N)
        
        # fft_coeff[my, mx] gives the coefficient for G = (2*pi*mx/N, 2*pi*my/N)
        # But numpy FFT wraps: indices 0..N-1 correspond to frequencies 0,1,...,N/2-1,-N/2,...,-1
        # We want mx, my in -8..8.
        
        def get_coeff(mx, my):
            # Wrap to 0..N-1
            mx_w = mx % N
            my_w = my % N
            return fft_coeff[my_w, mx_w]
        
        return get_coeff
    
    def compute_tm_bands(cell, num_pw=17, num_kpts=30):
        """Compute TM bands using plane-wave method.
        Returns list of (k, [omega_1, ..., omega_5]) for each k-point.
        """
        N = size
        get_coeff = compute_eps_inv_fourier(cell)
        
        # Reciprocal lattice vectors: G = (2*pi*mx/a, 2*pi*my/a) where a=1 (unit cell)
        # We use G vectors with |mx|,|my| <= 8 (total 17x17 = 289)
        max_g = 8
        g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
        num_g = len(g_indices)
        
        # Build the matrix A where A[p,q] = eps_inv(G_p - G_q) * |k + G_q|^2
        # Actually the standard form: for TM, the eigenvalue equation is
        #   sum_q eps^{-1}(G_p - G_q) (k + G_q) . (k + G_q) c_q = omega^2 c_p
        # Wait, I need to be more careful. Let me use the standard PWE formulation.
        
        # For TM polarization in 2D:
        #   (1/epsilon(r)) * (-nabla^2) H_z = omega^2 H_z
        # In plane waves: H_z = sum_k c_k exp(i(k+G).r)
        # -nabla^2 H_z = sum_k |k+G|^2 c_k exp(i(k+G).r)
        # (1/epsilon) * that: expand 1/epsilon = sum_G' eps_inv(G') exp(i G'.r)
        # Product: sum_{G,G'} eps_inv(G') |k+G|^2 c_k exp(i(k+G+G').r)
        # Match to sum_{G''} eps_inv(G'') |k+G''|^2 c_{G''} exp(i(k+G'').r)
        # So: omega^2 c_{G''} = sum_{G,G'} eps_inv(G') |k+G|^2 c_k delta(G'' = G+G')
        #   = sum_G eps_inv(G''-G) |k+G|^2 c_G
        # 
        # So the matrix is: M[p,q] = eps_inv(G_p - G_q) * |k + G_q|^2
        # and we solve M c = omega^2 c
        
        # Precompute G vectors
        G_vecs = np.array([(2*math.pi*mx, 2*math.pi*my) for mx, my in g_indices])
        
        # Precompute eps_inv differences: eps_inv(G_p - G_q)
        # G_p - G_q = (2*pi*(mx_p-mx_q), 2*pi*(my_p-my_q))
        # So we need get_coeff(mx_p-mx_q, my_p-my_q)
        
        # Build eps_inv matrix (num_g x num_g)
        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                eps_inv_mat[p, q] = get_coeff(mx_p - mx_q, my_p - my_q)
        
        # K-points in the irreducible BZ of the square lattice
        # The full BZ is [-pi, pi] x [-pi, pi]
        # Irreducible BZ (1/8): 0 <= kx <= ky <= pi  (or similar)
        # Actually for D4 symmetry, the irreducible BZ is the triangle with vertices
        # (0,0), (pi/2, pi/2), (0, pi)  -- wait, that's not right either.
        # The irreducible BZ of the square lattice under D4 is the region
        # 0 <= kx <= ky, kx + ky <= pi  (a triangle)
        # No wait. The BZ is [-pi,pi]^2. The D4 orbit of a point (kx,ky) has 8 points.
        # The fundamental domain is: 0 <= kx <= ky <= pi  -- no.
        # Let me think: the Wigner-Seitz cell of the square lattice is [-pi,pi]^2.
        # The point group D4 has 8 elements. The irreducible BZ is 1/8 of the BZ.
        # A common choice: 0 <= kx <= ky, |kx| + |ky| <= pi... 
        # Actually, the standard irreducible BZ for square lattice is the triangle
        # with vertices (0,0), (pi/2, pi/2), (0, pi). Wait no.
        # 
        # Let me just use a set of 30 k-points that cover the BZ boundary and interior.
        # The task says "30 wave vectors covering the boundary and interior of the irreducible Brillouin zone."
        # I'll use a reasonable set.
        
        # Let me define k-points along the high-symmetry lines and some interior points.
        # The irreducible BZ of the square lattice: I'll use the triangle
        # Gamma (0,0), X (pi, 0), M (pi, pi) but that's 1/4. 
        # For D4 (8-fold), the irreducible BZ is 1/8: triangle Gamma-X-M where
        # X = (pi/2, 0)? No...
        
        # Let me just use a grid of k-points in [0, pi] x [0, pi] with kx <= ky,
        # and also include the boundary. Actually, let me use the standard set:
        # The irreducible BZ for the square lattice (D4h point group in 2D, which is D4)
        # is the region: 0 <= kx <= ky, kx + ky <= pi.
        # Vertices: (0,0), (pi/2, pi/2), (0, pi).
        
        # Let me generate 30 points in this triangle.
        k_points = []
        # Along Gamma-X: (t, 0) for t in [0, pi/2]  -- wait, X should be (pi, 0) in full BZ
        # In the irreducible BZ with 0<=kx<=ky and kx+ky<=pi:
        # Gamma = (0,0)
        # X = (pi/2, pi/2)? No. Let me reconsider.
        
        # The full BZ of the square lattice with lattice constant a=1 is [-pi, pi] x [-pi, pi].
        # The point group D4 maps (kx, ky) -> (±kx, ±ky), (±ky, ±kx).
        # The fundamental domain: 0 <= kx <= |ky|, kx + ky <= pi (for ky >= 0)
        # So: 0 <= kx <= ky <= pi - kx, i.e., 0 <= kx, 0 <= ky, kx <= ky, kx+ky <= pi.
        # Vertices: (0,0), (pi/2, pi/2), (0, pi).
        
        # Generate points:
        # Along Gamma to M: (t, t) for t in [0, pi/2]
        # Along Gamma to X: (0, t) for t in [0, pi]
        # Along M to X: (pi/2 - t, pi/2 + t) for t in [0, pi/2]... 
        # Actually M=(pi/2,pi/2), X=(0,pi) in this parametrization? No.
        # X in the full BZ is at (pi, 0) or (0, pi). In the irreducible BZ, 
        # X corresponds to (0, pi) since 0 <= 0 <= pi and 0+pi = pi <= pi.
        # M = (pi/2, pi/2).
        
        # Let me just use a set of points:
        n_per_line = 5
        # Gamma to X: (0, t) t = 0, pi/4, pi/2, 3pi/4, pi
        for t in np.linspace(0, math.pi, n_per_line):
            k_points.append((0.0, t))
        # Gamma to M: (t, t) t = 0, pi/8, pi/4, 3pi/8, pi/2
        for t in np.linspace(0, math.pi/2, n_per_line):
            k_points.append((t, t))
        # M to X: (pi/2 - t, pi/2 + t) t = 0, pi/8, pi/4, 3pi/8, pi/2
        for t in np.linspace(0, math.pi/2, n_per_line):
            kx = math.pi/2 - t
            ky = math.pi/2 + t
            k_points.append((kx, ky))
        # Interior points
        k_points.append((math.pi/6, math.pi/3))
        k_points.append((math.pi/4, math.pi/2))
        k_points.append((math.pi/8, 3*math.pi/8))
        k_points.append((math.pi/5, 2*math.pi/5))
        k_points.append((math.pi/3, 2*math.pi/3))
        
        # Trim to 30
        k_points = k_points[:30]
        
        # Compute bands
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            # Build matrix M[p,q] = eps_inv_mat[p,q] * |k + G_q|^2
            # |k + G_q|^2 = (kx + Gx_q)^2 + (ky + Gy_q)^2
            kG_sq = np.zeros(num_g)
            for q in range(num_g):
                kG_sq[q] = (kx + G_vecs[q, 0])**2 + (ky + G_vecs[q, 1])**2
            
            M = eps_inv_mat * kG_sq[None, :]  # M[p,q] = eps_inv_mat[p,q] * kG_sq[q]
            
            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            # Sort eigenvalues (omega^2)
            eigvals = np.sort(eigvals)
            if len(eigvals) < 5:
                continue
            
            omega4 = math.sqrt(max(eigvals[3], 0))
            omega5 = math.sqrt(max(eigvals[4], 0))
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap
    
    # --- Local search over the 136 independent pixels ---
    # Use a simple local search: start from a good initial guess, then flip pixels
    # that improve the score.
    
    # Initial guess: a ring (annulus) which is known to give good TM gaps
    def make_ring_quadrant(r_inner, r_outer):
        """Create a ring in the 16x16 quadrant."""
        half = size // 2  # 16
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                # Pixel center at (j+0.5)/32, (i+0.5)/32 in the full cell
                # But in the quadrant, we work in the 16x16 space
                # The full cell coordinate: x = (j+0.5)/32, y = (i+0.5)/32
                # But due to symmetry, the quadrant covers [0, 0.5] x [0, 0.5]
                # Pixel (i,j) in quadrant corresponds to full cell pixel (i,j)
                # Center: x = (j+0.5)/32, y = (i+0.5)/32
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = math.sqrt(x*x + y*y)
                if r_inner <= r <= r_outer:
                    q[i][j] = 1
        return q
    
    # Try a few initial designs
    initial_designs = [
        make_ring_quadrant(0.15, 0.35),
        make_ring_quadrant(0.2, 0.4),
        make_ring_quadrant(0.1, 0.3),
        make_ring_quadrant(0.25, 0.45),
    ]
    
    # Also try a solid disk
    def make_disk_quadrant(radius):
        half = size // 2
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = math.sqrt(x*x + y*y)
                if r <= radius:
                    q[i][j] = 1
        return q
    
    initial_designs.append(make_disk_quadrant(0.2))
    initial_designs.append(make_disk_quadrant(0.3))
    
    # Try a cross pattern
    def make_cross_quadrant(width=0.15):
        half = size // 2
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                # Cross along x and y axes
                if x < width or y < width:
                    q[i][j] = 1
        return q
    
    initial_designs.append(make_cross_quadrant(0.1))
    initial_designs.append(make_cross_quadrant(0.15))
    
    # Evaluate initial designs
    best_score = -1.0
    best_quadrant = initial_designs[0]
    
    for q in initial_designs:
        if time.time() > deadline - 30:
            break
        cell = build_cell_from_quadrant(q)
        score = compute_tm_bands(cell)
        if score > best_score:
            best_score = score
            best_quadrant = [row[:] for row in q]
    
    # Local search: flip pixels in the quadrant
    half = size // 2
    # List of all independent (i,j) positions
    positions = [(i, j) for i in range(half) for j in range(i, half)]
    
    current_quadrant = [row[:] for row in best_quadrant]
    current_score = best_score
    no_improve_count = 0
    
    max_iterations = 500
    for iteration in range(max_iterations):
        if time.time() > deadline - 20:
            break
        if no_improve_count > 100:
            break
        
        improved = False
        # Randomly shuffle positions each iteration
        import random
        random.seed(iteration * 12345 + 67890)
        shuffled = positions[:]
        random.shuffle(shuffled)
        
        for (i, j) in shuffled:
            if time.time() > deadline - 15:
                break
            
            # Try flipping
            old_val = current_quadrant[i][j]
            current_quadrant[i][j] = 1 - old_val
            
            cell = build_cell_from_quadrant(current_quadrant)
            score = compute_tm_bands(cell)
            
            if score > current_score:
                current_score = score
                best_score = score
                best_quadrant = [row[:] for row in current_quadrant]
                improved = True
                no_improve_count = 0
            else:
                current_quadrant[i][j] = old_val
                no_improve_count += 1
        
        if not improved and no_improve_count > 50:
            break
    
    # Build final cell
    final_cell = build_cell_from_quadrant(best_quadrant)
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert final_cell[i][j] == final_cell[size-1-i][j], f"Vertical symmetry broken at ({i},{j})"
            assert final_cell[i][j] == final_cell[i][size-1-j], f"Horizontal symmetry broken at ({i},{j})"
            assert final_cell[i][j] == final_cell[j][i], f"Transpose symmetry broken at ({i},{j})"
    
    return final_cell
# EVOLVE-BLOCK-END
