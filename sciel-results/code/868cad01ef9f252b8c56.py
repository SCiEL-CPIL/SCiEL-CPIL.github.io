"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill
    
    def time_ok():
        return time.time() - t_start < DEADLINE

    # Plane wave setup: 17x17 = 289 plane waves
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289
    
    # K-points in irreducible BZ (30 points as specified)
    pi = np.pi
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)
    
    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2
    
    # Precompute difference indices for Fourier coupling
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            idx_x = int(dGx) + half
            idx_y = int(dGy) + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)
    
    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs

    def compute_band_ratio(coeffs, k_indices=None):
        """Compute gap-midgap ratio."""
        if k_indices is None:
            k_indices = range(len(k_points))
        
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_indices:
            k2 = k2_all[k_idx]
            # Build H matrix efficiently
            H_vals = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                if eigvals[3] > max_band4:
                    max_band4 = eigvals[3]
                if eigvals[4] < min_band5:
                    eigvals[4]
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def generate_voronoi_cell(seeds):
        """
        Generate a 32x32 symmetric binary cell from a set of seeds.
        Seeds are defined in the first quadrant (0.5x0.5 box) and mirrored.
        Pixels are assigned to the nearest seed (Euclidean distance).
        If the nearest seed is in the 'dielectric' group, pixel=1, else 0.
        Here we simplify: We define 8 seeds (4 in Q1, mirrored). 
        Actually, to ensure symmetry and allow complex shapes, we define 
        4 seeds in the top-left 16x16 quadrant? No, Voronoi on full domain 
        with symmetric seeds.
        
        Let's define 4 seeds in the (0,0) to (0.5, 0.5) region.
        The full set of seeds is their 4-fold rotation/reflection.
        Pixel (i,j) is 1 if it is closer to any of the 'dielectric' seeds 
        than to any 'air' seeds? 
        Simpler: The cell is defined by which Voronoi cell it falls into.
        Let's use 4 seeds. The Voronoi tessellation partitions the square.
        We assign 2 seeds to be Dielectric and 2 to be Air? 
        Or better: The cell is 1 if the pixel is closer to a Dielectric seed 
        than to the center? No.
        
        Standard approach: 
        Define N seeds. The domain is partitioned into N Voronoi cells.
        We label each seed as 0 or 1.
        To maintain 4-fold symmetry, we define 4 seeds in the fundamental 
        domain (1/8 of the square, e.g., 0<=x<=y<=0.5).
        Then mirror them to get 8 seeds (or 4 if on axes).
        Let's use 4 seeds in the quadrant [0, 0.5]x[0, 0.5].
        Mirror to get 16 seeds total? 
        Symmetry group D4 has 8 elements. 
        If we place seeds in the fundamental wedge (1/8), mirroring gives 8 seeds.
        
        Let's place 2 seeds in the fundamental wedge [0, 0.5] x [0, 0.5] where x<=y.
        Seed 1: Dielectric
        Seed 2: Air
        Mirror these to 8 seeds.
        Pixel is 1 if closest seed is Dielectric, 0 if Air.
        """
        # Seeds in fundamental region (0 <= x <= y <= 0.5)
        # We will use 2 seeds.
        # s1 = (x1, y1), s2 = (x2, y2)
        # Constraints: 0 <= x1 <= y1 <= 0.5, 0 <= x2 <= y2 <= 0.5
        
        x1, y1, x2, y2 = seeds
        
        # Generate all 8 symmetric seeds
        # Transformations for D4 on coordinates (x,y) in [0,1]x[0,1] centered at 0.5?
        # The problem defines cell indices 0..31. Center is 15.5.
        # Let's work in normalized coordinates [0,1]x[0,1].
        # Fundamental domain for D4 (square symmetry): 0 <= x <= 0.5, 0 <= y <= 0.5, x <= y?
        # No, the standard fundamental domain for a square centered at origin 
        # with 4-fold symmetry is a triangle.
        
        # Let's use a simpler symmetry enforcement:
        # Define seeds in the top-left 16x16 quadrant (i in 0..15, j in 0..15).
        # Then mirror to fill the rest.
        # But Voronoi needs global seeds.
        
        # Let's define 4 seeds in the full [0,1]x[0,1] box, constrained to be symmetric.
        # Seed A: (a, a) - on diagonal
        # Seed B: (b, 1-b) - on anti-diagonal? 
        # If we use 4 seeds: (a,a), (a,1-a), (1-a,a), (1-a,1-a).
        # This creates 4-fold symmetry.
        # Label: A and C are Dielectric? B and D are Air?
        # Or A is Dielectric, B is Air, C is Dielectric, D is Air.
        
        # Let's try 4 seeds:
        # S1: (s1, s1)  [Dielectric]
        # S2: (s2, 1-s2) [Air]
        # S3: (1-s1, 1-s1) [Dielectric]
        # S4: (1-s2, s2) [Air]
        
        # This might be too restrictive. Let's allow 2 independent parameters 
        # for the dielectric seed and 2 for the air seed in the quadrant.
        
        # Better: 4 seeds in the first quadrant (0,0)-(0.5,0.5).
        # S1: (x1, y1) [Dielectric]
        # S2: (x2, y2) [Air]
        # Mirror to get 8 seeds.
        # S1' = (y1, x1) [Dielectric]
        # S1'' = (1-x1, 1-y1) [Dielectric]
        # S1''' = (1-y1, 1-x1) [Dielectric]
        # ... and similarly for Air.
        
        # Total 8 seeds. 4 Dielectric, 4 Air.
        
        seeds_list = []
        # Dielectric seeds
        seeds_list.append((x1, y1, 1))
        seeds_list.append((y1, x1, 1))
        seeds_list.append((1-x1, 1-y1, 1))
        seeds_list.append((1-y1, 1-x1, 1))
        # Air seeds
        seeds_list.append((x2, y2, 0))
        seeds_list.append((y2, x2, 0))
        seeds_list.append((1-x2, 1-y2, 0))
        seeds_list.append((1-y2, 1-x2, 0))
        
        cell = np.zeros((size, size), dtype=int)
        
        # Vectorized distance calculation
        # Grid points
        ys, xs = np.mgrid[0:size, 0:size]
        # Normalize to [0,1]
        xs = (xs + 0.5) / size
        ys = (ys + 0.5) / size
        
        min_dist = np.full((size, size), np.inf)
        min_label = np.zeros((size, size), dtype=int)
        
        for sx, sy, label in seeds_list:
            dist_sq = (xs - sx)**2 + (ys - sy)**2
            mask = dist_sq < min_dist
            min_dist[mask] = dist_sq[mask]
            min_label[mask] = label
            
        cell = min_label.astype(int)
        
        # Enforce exact symmetry (in case of floating point issues or tie-breaking)
        # The construction above should be symmetric, but let's verify/fix.
        # cell == cell[::-1, ::-1] (180 rot)
        # cell == cell.T (transpose)
        # cell == cell[:, ::-1] (hflip)
        # cell == cell[::-1, :] (vflip)
        
        # If there are ties, the order in seeds_list matters. 
        # To be safe, we can average the symmetric versions.
        sym_cell = (cell + cell.T + cell[:, ::-1] + cell[::-1, :] + 
                    cell[::-1, ::-1] + cell.T[:, ::-1] + cell.T[::-1, :] + cell[::-1, :])
        cell = (sym_cell >= 4).astype(int)
        
        return cell

    def objective(seeds):
        """Objective function to minimize (negative gap ratio)."""
        if not time_ok():
            return 1e10 # Penalty for timeout
        try:
            cell = generate_voronoi_cell(seeds)
            coeffs = get_fourier_coeffs(cell)
            ratio = compute_band_ratio(coeffs)
            return -ratio
        except Exception:
            return 1e10

    # Initial guesses for seeds (x1, y1, x2, y2)
    # x1, y1 in [0, 0.5], x2, y2 in [0, 0.5]
    # Constraints: 0 <= x <= 0.5, 0 <= y <= 0.5
    
    best_ratio = -np.inf
    best_cell = None
    
    # Try a few random starting points
    np.random.seed(42)
    n_starts = 5
    
    for k in range(n_starts):
        if not time_ok():
            break
            
        # Random seeds in [0.05, 0.45]
        x1 = np.random.uniform(0.05, 0.45)
        y1 = np.random.uniform(0.05, 0.45)
        x2 = np.random.uniform(0.05, 0.45)
        y2 = np.random.uniform(0.05, 0.45)
        
        # Ensure they are distinct enough
        if np.hypot(x1-x2, y1-y2) < 0.1:
            continue
            
        init_seeds = np.array([x1, y1, x2, y2])
        
        # Bounds
        bounds = [(0.01, 0.49), (0.01, 0.49), (0.01, 0.49), (0.01, 0.49)]
        
        # Optimize
        res = minimize(objective, init_seeds, method='Nelder-Mead', 
                       bounds=bounds, 
                       options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        
        if res.success and not time_ok() == False: # Check if we ran out of time during optimization
             pass
             
        if not time_ok():
            break
            
        # Evaluate the result
        best_seeds = res.x
        cell = generate_voronoi_cell(best_seeds)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()

    # Fallback if no valid cell found or ratio is bad
    if best_cell is None or best_ratio < 0.1:
        # Fallback to a simple 4-rod pattern which is known to work decently
        # We can simulate this by setting seeds to create rod-like shapes?
        # Or just use the previous best from a quick heuristic.
        # Let's try a fixed good configuration:
        # Rods at (0.25, 0.25) etc.
        # Seed 1 (Dielectric) at (0.25, 0.25)
        # Seed 2 (Air) at (0.5, 0.5) ?? No, Air should be in the middle?
        # If we want rods, the Dielectric seeds should be at the corners of the quadrant.
        # Let's just run one more specific config.
        
        # Config: Dielectric at (0.2, 0.2), Air at (0.4, 0.4)
        seeds = np.array([0.2, 0.2, 0.4, 0.4])
        cell = generate_voronoi_cell(seeds)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
            
        if best_cell is None:
             # Absolute fallback: checkerboard? No, symmetry.
             # Just return a zero cell (all air) - gap will be 0, but valid.
             # Or better, a single rod.
             cell = np.zeros((size, size), dtype=int)
             # Center rod
             center = size // 2
             radius = 4
             for i in range(size):
                 for j in range(size):
                     if (i-center)**2 + (j-center)**2 <= radius**2:
                         cell[i,j] = 1
             best_cell = cell
             best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))

    # Final check
    if not time_ok():
        return best_cell.tolist()

    # Optional: Quick local refinement of the best cell found by Voronoi
    # Flip pixels in the independent quadrant to see if we can squeeze more.
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    indep_pixels = [(i, j) for i in range(16) for j in range(16)]
    no_improve = 0
    max_no_improve = 20
    
    while no_improve < max_no_improve and time_ok():
        improved = False
        np.random.shuffle(indep_pixels)
        for (i, j) in indep_pixels:
            if not time_ok():
                break
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            # Enforce symmetry
            sym_cell = (new_cell + new_cell.T + new_cell[:, ::-1] + new_cell[::-1, :] + 
                        new_cell[::-1, ::-1] + new_cell.T[:, ::-1] + new_cell.T[::-1, :] + new_cell[::-1, :])
            new_cell = (sym_cell >= 4).astype(int)
            
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                improved = True
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > 10: # Break early if stuck
                break
        if not improved:
            break

    return current_cell.tolist()
