import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell with 4 air holes in a dielectric background.
    Optimizes hole radius, position, and shape (superellipse exponent) to maximize
    the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s budget
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute for fast evaluation
    # 289 plane waves: N_pw = 8, indices -8 to 8
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # Precompute G differences
    Gx_flat = G_vectors[:, 0]
    Gy_flat = G_vectors[:, 1]
    Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
    Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
    
    # Map differences to FFT indices (0 to 31)
    # FFT index for difference d is (d % size)
    idx_x = (Diff_x + size) % size
    idx_y = (Diff_y + size) % size
    
    # K-points: 30 points in irreducible BZ
    # BZ boundaries: -pi to pi (or 0 to pi for irreducible)
    # Let's use 10 points on each of the 3 segments
    k_points = []
    # Gamma-X: (k, 0) for k in [0, pi]
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        k_points.append((kx, 0))
    # X-M: (pi, k) for k in [0, pi]
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        k_points.append((np.pi, ky))
    # M-Gamma: (k, k) for k in [0, pi/sqrt(2)? No, M is (pi, pi) in this convention?
    # Irreducible BZ of square lattice: 0 <= kx <= ky <= pi?
    # Standard irreducible BZ: Triangle Gamma (0,0), X (pi, 0), M (pi, pi).
    # Wait, X is (pi, 0) and M is (pi, pi).
    # Segment M-X is kx=pi, ky in [0, pi].
    # Segment Gamma-M is kx=ky in [0, pi].
    # Segment Gamma-X is ky=0, kx in [0, pi].
    # This covers the irreducible wedge.
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        k_points.append((t, t))
        
    k_points = np.array(k_points)
    
    # Precompute (k + G)^2 for all k points and G vectors
    # k is in rad/length units (a=1). G is in rad/length units (2*pi*n).
    # Wait, standard PW expansion uses G = 2*pi * n.
    # If k is in [0, pi], then G should be 2*pi * n?
    # Let's check units.
    # Eigenvalue: (k + G)^2.
    # If a=1, reciprocal lattice vector b = 2*pi.
    # So G_x = 2*pi * n_x.
    # k_x is in [0, pi].
    # So (k_x + 2*pi*n_x)^2.
    
    KG_SQ = np.zeros((len(k_points), num_pw))
    for i, k in enumerate(k_points):
        kx, ky = k
        kg_x = kx + 2 * np.pi * Gx_flat
        kg_y = ky + 2 * np.pi * Gy_flat
        KG_SQ[i] = kg_x**2 + kg_y**2
        
    # Helper to evaluate a cell
    def evaluate_cell(cell):
        """
        cell: 32x32 array of 0s and 1s.
        Returns gap-midgap ratio.
        """
        eps = np.where(cell == 1, eps_high, eps_low)
        mu = 1.0 / eps
        
        # FFT of mu
        mu_fft = np.fft.fft2(mu)
        mu_fft /= (size * size)
        
        # Extract coefficients for G-G' differences
        # coeff[i, j] = mu_fft[idx_x[i,j], idx_y[i,j]]
        mu_diff = mu_fft[idx_x, idx_y]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for i_k in range(len(k_points)):
            kg_sq = KG_SQ[i_k]
            # H = mu_diff * kg_sq[:, None]
            # Wait, the equation is sum_j mu_{G_i - G_j} (k + G_j)^2 c_j = omega^2 c_i
            # So H_{ij} = mu_{G_i - G_j} * (k + G_j)^2
            # Note: mu_{G_i - G_j} is symmetric (mu_{-G} = mu_{G}^* = mu_G for real mu).
            # So H is Hermitian.
            H = mu_diff * kg_sq[None, :]
            
            # Get 5 smallest eigenvalues
            evals = np.linalg.eigvalsh(H)
            # Ensure real
            evals = np.real(evals)
            
            if len(evals) >= 5:
                w4 = np.sqrt(evals[3])
                w5 = np.sqrt(evals[4])
                
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
                
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        return 2 * gap / midgap

    def generate_cell(params):
        """
        params: [r, d, n_exp]
        r: radius of holes
        d: distance of hole centers from axis (center at 0.5-d)
        n_exp: superellipse exponent
        """
        r, d, n_exp = params
        
        # Create 8x8 quadrant
        # Pixel centers in quadrant: i in 0..7, j in 0..7
        # Coords: x = (i+0.5)/32, y = (j+0.5)/32
        # Hole centers in quadrant: (0.5-d, 0.5-d) is in Q2?
        # Wait, 0.5 is the center of the 32x32 cell.
        # Quadrant 0 (top-left) covers x in [0, 0.5], y in [0, 0.5].
        # Hole centers are at (0.5-d, 0.5-d), (0.5+d, 0.5-d), etc.
        # The hole in the top-left quadrant is centered at (0.5-d, 0.5-d).
        
        # Initialize quadrant as dielectric (1)
        q = np.ones((8, 8), dtype=int)
        
        # Grid points in the quadrant
        # i=0 is top row? Or bottom?
        # Standard image: (0,0) is top-left.
        # x = (j+0.5)/32, y = (i+0.5)/32
        # Let's stick to array indices: row i, col j.
        # x = (j+0.5)/32, y = (i+0.5)/32
        
        # Hole center in this quadrant:
        # The full cell has holes at (0.5-d, 0.5-d), (0.5+d, 0.5-d), (0.5-d, 0.5+d), (0.5+d, 0.5+d).
        # In the top-left quadrant (x<0.5, y<0.5), the hole is at (0.5-d, 0.5-d).
        
        cx = 0.5 - d
        cy = 0.5 - d
        
        # Superellipse distance
        # |x-cx|^n + |y-cy|^n < r^n
        
        # Vectorized check for 8x8
        ii = np.arange(8)
        jj = np.arange(8)
        I, J = np.meshgrid(ii, jj, indexing='ij')
        
        x_coords = (J + 0.5) / 32.0
        y_coords = (I + 0.5) / 32.0
        
        dx = np.abs(x_coords - cx)
        dy = np.abs(y_coords - cy)
        
        # Avoid 0^0 issues if n_exp is small, but n_exp >= 2 usually
        dist = (dx**n_exp + dy**n_exp)**(1.0/n_exp)
        
        # If dist < r, it's air (0)
        air_mask = dist < r
        q[air_mask] = 0
        
        # Enforce transpose symmetry in the quadrant
        # q[i][j] == q[j][i]
        # Since we defined it based on a single hole at (0.5-d, 0.5-d),
        # and the quadrant is x,y in [0, 0.5],
        # the hole is symmetric about y=x if the center is on y=x.
        # (0.5-d, 0.5-d) is on y=x.
        # So the shape is naturally symmetric about y=x.
        
        # Construct full 32x32 cell
        cell = np.zeros((32, 32), dtype=int)
        
        # Top-Left: q
        cell[0:8, 0:8] = q
        # Top-Right: q flipped horizontally
        cell[0:8, 24:32] = q[:, ::-1]
        # Bottom-Left: q flipped vertically
        cell[24:32, 0:8] = q[::-1, :]
        # Bottom-Right: q flipped both
        cell[24:32, 24:32] = q[::-1, ::-1]
        
        # Wait, the quadrants are 8x8?
        # 32 / 4 = 8.
        # Indices:
        # TL: 0:8, 0:8
        # TR: 0:8, 24:32 (since 32-8=24)
        # BL: 24:32, 0:8
        # BR: 24:32, 24:32
        
        # Check symmetry:
        # cell[i][j] == cell[j][i]
        # TL[i][j] = q[i][j]
        # TR[i][j] = q[i][31-j]
        # If i=0, j=31, cell[0][31] = q[0][0].
        # cell[31][0] = q[7][0] (from BL: cell[24:32, 0:8] = q[::-1, :])
        # q[7][0] vs q[0][0].
        # Are they equal?
        # The hole is at (0.5-d, 0.5-d).
        # In TL, pixel (0,0) is at (0.5/32, 0.5/32).
        # In BL, pixel (31,0) is at (0.5/32, 31.5/32).
        # y-coord: 0.015 vs 0.98.
        # Distance to center (0.5-d, 0.5-d).
        # If d=0.25, center is (0.25, 0.25).
        # Dist(0.015, 0.015) to (0.25, 0.25) is same as Dist(0.015, 0.98) to (0.25, 0.75)?
        # No. The reflection is about y=0.5.
        # y' = 1 - y.
        # Center y_c = 0.5 - d.
        # Reflected center y_c' = 1 - (0.5 - d) = 0.5 + d.
        # So the hole in BL is at (0.5-d, 0.5+d).
        # This is correct for the 4-hole motif.
        
        # However, we need to ensure the *entire* cell is transpose symmetric.
        # cell[i][j] == cell[j][i].
        # This requires the pattern in TL to be such that when mirrored, it matches.
        # Specifically, q[i][j] (TL) must equal the value at (j, i) in the full cell.
        # If i,j < 8, (j,i) is in TL. So q[i][j] == q[j][i].
        # If i < 8, j >= 24. (j,i) is in BL? No.
        # Let i=0, j=31. cell[0][31] is in TR. Value q[0][0].
        # cell[31][0] is in BL. Value q[7][0].
        # We need q[0][0] == q[7][0].
        # This implies the TL quadrant must be symmetric about the line i+j = 7?
        # Or rather, the whole cell must be symmetric.
        # The 4-hole motif with holes at (0.5-d, 0.5-d) etc. IS symmetric under transpose.
        # So if we generate the TL quadrant correctly, and mirror it, the full cell should be symmetric.
        # Let's verify q[0][0] == q[7][0] for the hole at (0.25, 0.25).
        # Pixel (0,0) center (0.015, 0.015).
        # Pixel (7,0) center (0.234, 0.015).
        # Dist to (0.25, 0.25):
        # (0,0): dx=0.235, dy=0.235. d^2 = 2*0.235^2.
        # (7,0): dx=0.016, dy=0.235. d^2 = 0.016^2 + 0.235^2.
        # These are NOT equal.
        # So q[0][0] != q[7][0] generally.
        # This means my mirroring logic for TR/BL/BR might break transpose symmetry if q is not symmetric in a specific way.
        
        # Actually, the standard way to enforce C4v symmetry is to define the fundamental domain (1/8 of the cell) and reflect.
        # The fundamental domain is the triangle 0 <= x <= y <= 0.5.
        # Or simply, define the top-left 8x8, and enforce q[i][j] = q[j][i].
        # If we enforce q[i][j] = q[j][i], then:
        # cell[0][31] = q[0][0].
        # cell[31][0] = q[7][0] (from BL mirroring).
        # For cell[0][31] == cell[31][0], we need q[0][0] == q[7][0].
        # This is a constraint on q.
        # Does the hole at (0.5-d, 0.5-d) satisfy this?
        # No, as shown above.
        
        # So we must define the cell such that it is symmetric.
        # The 4-hole motif is symmetric.
        # Let's just generate the full 32x32 grid directly using the distance to the 4 centers.
        # This is safer and still fast enough for 32x32.
        
        cell_full = np.ones((32, 32), dtype=int)
        I, J = np.meshgrid(np.arange(32), np.arange(32), indexing='ij')
        x = (J + 0.5) / 32.0
        y = (I + 0.5) / 32.0
        
        centers = [
            (0.5 - d, 0.5 - d),
            (0.5 + d, 0.5 - d),
            (0.5 - d, 0.5 + d),
            (0.5 + d, 0.5 + d)
        ]
        
        # Superellipse distance to any center
        min_dist = np.inf
        for cx, cy in centers:
            dx = np.abs(x - cx)
            dy = np.abs(y - cy)
            d_super = (dx**n_exp + dy**n_exp)**(1.0/n_exp)
            min_dist = np.minimum(min_dist, d_super)
            
        cell_full[min_dist < r] = 0
        
        return cell_full

    # Optimization
    best_score = -np.inf
    best_params = None
    
    # Coarse Grid Search
    # r: 0.1 to 0.4
    # d: 0.15 to 0.35
    # n_exp: 2 to 6
    
    r_vals = np.linspace(0.15, 0.40, 10)
    d_vals = np.linspace(0.15, 0.35, 10)
    n_vals = [2.0, 3.0, 4.0, 6.0]
    
    for r in r_vals:
        for d in d_vals:
            for n in n_vals:
                if time.time() > deadline - 5:
                    break
                params = [r, d, n]
                cell = generate_cell(params)
                score = evaluate_cell(cell)
                if score > best_score:
                    best_score = score
                    best_params = params
    
    # Fine Grid Search around best
    if best_params is not None:
        r0, d0, n0 = best_params
        
        r_fine = np.linspace(r0 - 0.05, r0 + 0.05, 10)
        d_fine = np.linspace(d0 - 0.05, d0 + 0.05, 10)
        n_fine = np.linspace(n0 - 1, n0 + 1, 5)
        
        for r in r_fine:
            for d in d_fine:
                for n in n_fine:
                    if time.time() > deadline - 5:
                        break
                    params = [r, d, n]
                    cell = generate_cell(params)
                    score = evaluate_cell(cell)
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
    # Finalize
    if best_params is None:
        # Fallback to Inspiration 1
        best_params = [0.28, 0.25, 2.0]
        
    final_cell = generate_cell(best_params)
    
    # Convert to nested list
    result = []
    for i in range(32):
        row = []
        for j in range(32):
            row.append(int(final_cell[i, j]))
        result.append(row)
        
    return result
