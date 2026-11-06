import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses a Gray-Scott reaction-diffusion pattern to initialize a complex symmetric structure,
    then refines it with local search using a fast NumPy-based plane-wave solver.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem.get("eps_low", 1.0)
    
    # 1. Plane Wave Setup
    # 289 plane waves -> 17x17 grid -> G range -8 to 8
    N_g = 8
    g_vals = np.arange(-N_g, N_g + 1)
    Gx, Gy = np.meshgrid(g_vals, g_vals, indexing='ij')
    Gx = Gx.flatten()
    Gy = Gy.flatten()
    Npw = len(Gx)
    
    # Precompute G vectors
    G_vecs = np.vstack([Gx, Gy]).T
    
    # 2. k-points in the Irreducible Brillouin Zone (IBZ)
    # Square lattice IBZ: 0 <= kx <= ky, kx + ky <= 1 (in units of 2pi/a)
    # We need ~30 points.
    k_points = []
    # Gamma-M: (0, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0, t))
    # M-X: (t, 0.5-t) for t in [0, 0.5] -- Wait, standard IBZ for square is triangle (0,0)-(0.5,0)-(0.5,0.5)
    # Actually, for square lattice, the IBZ is the triangle with vertices Gamma(0,0), M(0.5,0), X(0.5,0.5) is wrong.
    # Square lattice reciprocal lattice is square. IBZ is the triangle (0,0), (0.5, 0), (0.5, 0.5)? No.
    # The IBZ of a square lattice (lattice constant a) is the triangle with vertices (0,0), (1,0)/2pi?
    # Let's stick to the standard: Gamma (0,0), M (0.5, 0.0) [midpoint of edge], X (0.5, 0.5) [corner].
    # The IBZ is the triangle (0,0) - (0.5, 0) - (0.5, 0.5).
    # Wait, the square BZ is [-0.5, 0.5] x [-0.5, 0.5].
    # The IBZ is 1/8 of the square. Vertices: (0,0), (0.5, 0), (0.5, 0.5).
    # Let's use points along the edges:
    # 1. Gamma to M: (0, t), t in [0, 0.5]
    # 2. M to X: (0.5, t), t in [0, 0.5] ? No, M is (0.5, 0). X is (0.5, 0.5).
    #    So edge is (0.5, t) for t in [0, 0.5].
    # 3. X to Gamma: (t, t), t in [0, 0.5].
    
    k_list = []
    for t in np.linspace(0, 0.5, 10):
        k_list.append((0, t))       # Gamma-M
    for t in np.linspace(0, 0.5, 10):
        k_list.append((0.5, t))     # M-X
    for t in np.linspace(0, 0.5, 10):
        k_list.append((t, t))       # X-Gamma
    k_points = np.array(k_list)
    # Remove duplicates if any (Gamma appears 3 times, M appears 2, X appears 2)
    # We can just use all, or unique them. Let's use unique to save time.
    k_points = np.unique(k_points, axis=0)
    Nk = len(k_points)

    # 3. Fast PWE Solver
    def solve_bands(eps_coeffs):
        """
        eps_coeffs: array of shape (size, size) containing Fourier coefficients of 1/epsilon.
        Returns array of shape (Nk, Npw) with omega values.
        """
        bands = np.zeros((Nk, Npw))
        
        # Precompute k+G for all k and G
        # k_points: (Nk, 2)
        # G_vecs: (Npw, 2)
        # kG: (Nk, Npw, 2)
        kG = k_points[:, np.newaxis, :] + G_vecs[np.newaxis, :, :]
        kxG = kG[:, :, 0]
        kyG = kG[:, :, 1]
        
        # Precompute the "dot product" matrix structure?
        # H[i, j] = sum_{G-G'} (k+G)_x (k+G')_x eps_{G-G'} + ...
        # This is a circulant-like structure but 2D.
        # We can build the matrix for each k.
        
        # To speed up, we can precompute the indices for G-G'
        # But building the 289x289 matrix 30 times per eval is ~87k entries.
        # Python loop is too slow. We need vectorized or C-speed.
        # Since we can't use scipy, we must be clever with NumPy.
        
        # Alternative: Use the fact that H is symmetric.
        # For each k, we construct H.
        
        for ik in range(Nk):
            kx = k_points[ik, 0]
            ky = k_points[ik, 1]
            
            # kxG and kyG are already computed for this ik
            # We need H[i, j] = (kxG[i]*kxG[j] + kyG[i]*kyG[j]) * eps_coeffs[(Gx[i]-Gx[j]) % size, (Gy[i]-Gy[j]) % size]
            
            # This is hard to fully vectorize in pure NumPy without creating huge intermediate arrays (Npw x Npw x size x size).
            # However, Npw=289. 289x289 = 83521.
            # size=32.
            # We can loop over the 32x32 Fourier components and accumulate into H.
            # H = sum_{dm, dn} eps[dm, dn] * M_{dm, dn}
            # where M_{dm, dn}[i, j] = (k+G_i) dot (k+G_j) if G_i - G_j == (dm, dn) else 0.
            
            H = np.zeros((Npw, Npw), dtype=float)
            
            # Get k+G vectors for this k
            kx_cur = kx + Gx
            ky_cur = ky + Gy
            
            # Loop over unique G differences? No, loop over dm, dn in [-N_g, N_g]
            for dm in range(-N_g, N_g + 1):
                for dn in range(-N_g, N_g + 1):
                    # Get the coefficient
                    # Note: eps_coeffs is indexed by positive modulo size
                    em = dm % size
                    en = dn % size
                    coeff = eps_coeffs[em, en]
                    if coeff == 0:
                        continue
                    
                    # Find all pairs (i, j) such that Gx[i] - Gx[j] == dm and Gy[i] - Gy[j] == dn
                    # This is still slow if done naively.
                    
                    # Optimization: 
                    # The matrix H is a "block" circulant in 2D.
                    # Actually, for a fixed k, H is not circulant because of the (k+G) terms.
                    # However, we can compute the outer product of vectors?
                    # No, H_{ij} = (kx_i kx_j + ky_i ky_j) * eps_{ij}
                    # Let A = diag(kx), B = diag(ky).
                    # H = A * (K_x * eps_matrix) * A^T + B * (K_y * eps_matrix) * B^T ?
                    # No, eps depends on i-j.
                    
                    # Let's just build the eps_matrix for this k?
                    # eps_matrix[i, j] = eps_coeffs[(Gx[i]-Gx[j])%size, (Gy[i]-Gy[j])%size]
                    # We can compute this using broadcasting if we precompute Gx and Gy arrays.
                    
                    # Let's try to vectorize the matrix construction for a single k.
                    # diff_x = Gx[:, None] - Gx[None, :]
                    # diff_y = Gy[:, None] - Gy[None, :]
                    # indices_x = diff_x % size
                    # indices_y = diff_y % size
                    # eps_mat = eps_coeffs[indices_x, indices_y]
                    
                    # This creates 289x289 arrays. 30 k-points.
                    # 30 * (289*289*4 bytes) ~ 2.5 MB. Manageable.
                    # The operation is fast in NumPy.
                    
                    pass # We will do this inside the loop below, but precomputed indices are better.

            # Pre-compute index maps for G differences
            # diff_x[i, j] = Gx[i] - Gx[j]
            # We can compute this once outside the k-loop.
            
            # Let's move the diff calculation outside.
            pass

        # RESTRUCTURE: Precompute index maps for G differences
        diff_x = Gx[:, None] - Gx[None, :]
        diff_y = Gy[:, None] - Gy[None, :]
        idx_x = diff_x % size
        idx_y = diff_y % size
        
        # Now inside the k loop:
        for ik in range(Nk):
            kx_cur = k_points[ik, 0] + Gx
            ky_cur = k_points[ik, 1] + Gy
            
            # Get the epsilon matrix for this k (same for all k, actually! 
            # Wait, eps_coeffs is constant for a given pixel map.
            # The matrix structure H_{ij} = (kx_i kx_j + ky_i ky_j) * eps_{ij_diff}
            # The eps_{ij_diff} part is constant for a given pixel map.
            # Let's precompute E_mat = eps_coeffs[idx_x, idx_y]
            
            # We can compute E_mat once per pixel map evaluation.
            # But kx_cur and ky_cur change.
            
            # H = E_mat * (kx_cur[:, None] * kx_cur[None, :] + ky_cur[:, None] * ky_cur[None, :])
            
            # Let's do it.
            E_mat = eps_coeffs[idx_x, idx_y]
            
            # Outer products
            # kx_outer = np.outer(kx_cur, kx_cur)
            # ky_outer = np.outer(ky_cur, ky_cur)
            # H = E_mat * (kx_outer + ky_outer)
            
            # This is a 289x289 element-wise multiplication. Very fast in NumPy.
            
            kx_outer = np.outer(kx_cur, kx_cur)
            ky_outer = np.outer(ky_cur, ky_cur)
            
            H = E_mat * (kx_outer + ky_outer)
            
            # Solve eigenvalues
            # H is symmetric real.
            vals = np.linalg.eigvalsh(H)
            vals = np.sort(vals)
            # omega = sqrt(omega^2)
            # Ensure non-negative
            vals = np.maximum(vals, 0)
            bands[ik, :] = np.sqrt(vals)
            
        return bands

    def get_eps_coeffs(cell):
        """Compute Fourier coefficients of 1/epsilon."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        # FFT
        # The PWE method uses coefficients of 1/epsilon.
        # G_{mn} = (1/N^2) sum_{i,j} (1/eps(i,j)) exp(-2pi i (m x_i + n y_j))
        # Using FFT:
        # np.fft.fft2 returns sum_{i,j} f(i,j) exp(-2pi i (k_i i + l_j j) / N)
        # So we just need to normalize by N^2.
        coeffs = np.fft.fft2(inv_eps) / (size * size)
        return coeffs

    def compute_gap(bands):
        """
        Compute gap-midgap ratio between band 4 and 5.
        bands: (Nk, Npw)
        """
        # Band indices: 0-based. Band 4 is index 3, Band 5 is index 4.
        # We need max of band 4 and min of band 5 across all k-points.
        # Note: We must ensure we are looking at the correct bands.
        # The lowest band is 1. So index 0 is Band 1.
        # Index 3 is Band 4.
        # Index 4 is Band 5.
        
        # Check if we have enough bands
        if bands.shape[1] < 5:
            return -1.0
            
        omega_4 = bands[:, 3]
        omega_5 = bands[:, 4]
        
        max_4 = np.max(omega_4)
        min_5 = np.min(omega_5)
        
        gap = min_5 - max_4
        midgap = (min_5 + max_4) / 2.0
        
        if midgap <= 0:
            return -1.0
            
        return gap / midgap

    # 4. Gray-Scott Initialization
    def generate_gray_scott(size, steps=50, F=0.0367, k=0.0649, dt=1.0, noise_scale=0.1):
        """
        Generate a D4-symmetric pattern using Gray-Scott model.
        """
        # Initialize with a symmetric seed
        # Use a few symmetric blobs or a symmetric noise
        u = np.ones((size, size))
        v = np.zeros((size, size))
        
        # Symmetric seed: center blob + 4 corner blobs?
        # Let's use a symmetric noise pattern.
        # Create a base noise, then symmetrize it.
        base = np.random.randn(size, size)
        # Symmetrize: average with flips and transpose
        sym_base = (base + base[::-1, :] + base[:, ::-1] + base[::-1, ::-1] +
                    base.T + base.T[::-1, :] + base.T[:, ::-1] + base.T[::-1, ::-1]) / 8.0
        
        # Threshold to create initial spots
        v[sym_base > np.percentile(sym_base, 70)] = 1.0
        u[v == 1.0] = 0.0
        
        # Evolve
        # Laplacian using finite differences (periodic boundary conditions)
        def laplacian(arr):
            return (np.roll(arr, 1, 0) + np.roll(arr, -1, 0) +
                    np.roll(arr, 1, 1) + np.roll(arr, -1, 1) - 4 * arr)
        
        for _ in range(steps):
            lap_u = laplacian(u)
            lap_v = laplacian(v)
            
            # Reaction terms
            uvv = u * v * v
            u_new = u + dt * (1.0 * lap_u - uvv + F * (1.0 - u))
            v_new = v + dt * (1.0 * lap_v + uvv - (F + k) * v)
            
            u = np.clip(u_new, 0, 1)
            v = np.clip(v_new, 0, 1)
            
            # Enforce symmetry at each step to prevent drift
            sym_v = (v + v[::-1, :] + v[:, ::-1] + v[::-1, ::-1] +
                     v.T + v.T[::-1, :] + v.T[:, ::-1] + v.T[::-1, ::-1]) / 8.0
            v = sym_v
            # u is determined by conservation? No, in Gray-Scott u+v is not conserved.
            # But we can just symmetrize u too, or recompute u from v?
            # Actually, just symmetrizing v is enough if we assume u is fast variable or we symmetrize both.
            sym_u = (u + u[::-1, :] + u[:, ::-1] + u[::-1, ::-1] +
                     u.T + u.T[::-1, :] + u.T[:, ::-1] + u.T[::-1, ::-1]) / 8.0
            u = sym_u
            
        # Threshold to binary
        # Use a threshold that ensures some dielectric
        # We want ~30-50% dielectric?
        # Let's pick the top 40%
        threshold = np.percentile(v, 60)
        cell = (v > threshold).astype(int)
        
        # Ensure symmetry
        cell = ((cell + cell[::-1, :] + cell[:, ::-1] + cell[::-1, ::-1] +
                 cell.T + cell.T[::-1, :] + cell.T[:, ::-1] + cell.T[::-1, ::-1]) / 8.0).astype(int)
        cell[cell > 0.5] = 1
        cell[cell <= 0.5] = 0
        
        return cell

    # 5. Main Loop
    # Initialize
    np.random.seed(42)
    cell = generate_gray_scott(size, steps=30, F=0.037, k=0.065)
    
    # Ensure strict symmetry
    cell = ((cell + cell[::-1, :] + cell[:, ::-1] + cell[::-1, ::-1] +
             cell.T + cell.T[::-1, :] + cell.T[:, ::-1] + cell.T[::-1, ::-1]) / 8.0).astype(int)
    cell[cell > 0.5] = 1
    cell[cell <= 0.5] = 0
    
    # Evaluate initial
    eps_coeffs = get_eps_coeffs(cell)
    bands = solve_bands(eps_coeffs)
    best_gap = compute_gap(bands)
    best_cell = cell.copy()
    
    # Local Search
    # Symmetric pixels: only modify the fundamental domain (1/8 of the cell)
    # Fundamental domain: i <= j, i + j <= size/2 ?
    # Actually, just pick random pixels and update their symmetric counterparts.
    
    n_iter = 0
    max_iter = 1000
    improvement_count = 0
    
    while time.time() < deadline and n_iter < max_iter:
        n_iter += 1
        
        # Pick a random pixel in the fundamental domain
        # i in [0, 16), j in [i, 16-i]? 
        # Simpler: pick random i, j.
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        # Define symmetric set
        # (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)
        # (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
        
        sym_indices = [
            (i, j),
            (i, size - 1 - j),
            (size - 1 - i, j),
            (size - 1 - i, size - 1 - j),
            (j, i),
            (j, size - 1 - i),
            (size - 1 - j, i),
            (size - 1 - j, size - 1 - i)
        ]
        
        # Get current value
        # If any is 1, set all to 0. If all 0, set all to 1?
        # Or just flip the value of (i,j) and propagate.
        # Let's flip (i,j) and set all symmetric ones to the new value of (i,j).
        
        old_val = cell[i, j]
        new_val = 1 - old_val
        
        # Apply change
        for (r, c) in sym_indices:
            cell[r, c] = new_val
        
        # Evaluate
        eps_coeffs = get_eps_coeffs(cell)
        bands = solve_bands(eps_coeffs)
        gap = compute_gap(bands)
        
        if gap > best_gap:
            best_gap = gap
            best_cell = cell.copy()
            improvement_count = 0
        else:
            # Revert
            for (r, c) in sym_indices:
                cell[r, c] = old_val
            improvement_count += 1
            
            # If stuck for too long, maybe restart with new Gray-Scott params?
            # For now, just continue.
            if improvement_count > 50:
                # Try a different Gray-Scott parameter set
                cell = generate_gray_scott(size, steps=30, F=np.random.uniform(0.03, 0.04), k=np.random.uniform(0.06, 0.07))
                cell = ((cell + cell[::-1, :] + cell[:, ::-1] + cell[::-1, ::-1] +
                         cell.T + cell.T[::-1, :] + cell.T[:, ::-1] + cell.T[::-1, ::-1]) / 8.0).astype(int)
                cell[cell > 0.5] = 1
                cell[cell <= 0.5] = 0
                
                eps_coeffs = get_eps_coeffs(cell)
                bands = solve_bands(eps_coeffs)
                gap = compute_gap(bands)
                if gap > best_gap:
                    best_gap = gap
                    best_cell = cell.copy()
                improvement_count = 0

    return best_cell.tolist()
