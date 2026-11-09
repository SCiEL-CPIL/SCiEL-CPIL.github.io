import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management with safety margin
    start_time = time.time()
    time_limit = 200.0  # Evaluator kills at 230s, leave 30s margin
    
    # PWM Setup - 289 plane waves (17x17 grid, G from -8 to 8)
    N_G = 17
    g_range = np.arange(-N_G // 2, N_G // 2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # Precompute diff indices for H matrix
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    valid_diffs = (np.abs(dx) <= N_G // 2) & (np.abs(dy) <= N_G // 2)
    diff_indices = np.where(valid_diffs, (dx + N_G // 2) * N_G + (dy + N_G // 2), 0)
    
    # K points in irreducible BZ (30 points matching evaluator)
    # Irreducible BZ of square lattice: 0 <= ky <= kx, 0 <= kx <= 0.5 (in units of 2*pi/a)
    k_points = []
    # Along X axis (ky=0, kx from 0 to 0.5)
    for i in range(10):
        t = i / 9.0 * 0.5
        k_points.append([t, 0.0])
    # Along diagonal (kx=ky, from 0 to 0.5)
    for i in range(1, 10):
        t = i / 9.0 * 0.5
        k_points.append([t, t])
    # Along M-X edge (kx=0.5, ky from 0 to 0.5)
    for i in range(1, 10):
        t = i / 9.0 * 0.5
        k_points.append([0.5, t])
    k_points = np.array(k_points)
    
    def get_fourier_eps_inv(cell_arr):
        """Compute Fourier coefficients of 1/epsilon for the pixel map."""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        coeffs_2d = np.fft.fft2(eps_inv) / (size * size)
        # Map G indices to FFT indices
        Gx_grid = G_vectors[:, 0] % size
        Gy_grid = G_vectors[:, 1] % size
        return coeffs_2d[Gx_grid, Gy_grid]
    
    def evaluate(cell_arr):
        """Evaluate gap-midgap ratio for TM bands 4 and 5."""
        eps_inv_coeffs = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # Build H matrix: H[m,n] = eps_inv(G_m - G_n) * |k + G_n|^2
            H = eps_inv_coeffs[diff_indices] * k_plus_G_sq[None, :]
            H[~valid_diffs] = 0.0
            
            # For TM polarization, the eigenvalue problem is:
            # sum_n eps_inv(G_m - G_n) (k+G_n)^2 A_n = (k/a)^2 A_m
            # So eigenvalues are (omega*a/c)^2, and omega is sqrt of eigenvalue
            # But for gap ratio, we can work with eigenvalues directly
            # since ratio is scale-invariant
            evals = np.linalg.eigvalsh(H)
            evals = np.sort(np.real(evals))
            
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap
    
    # Generate candidate initial designs
    def make_cell_from_function(func):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if func(x, y):
                    cell[i, j] = 1
        return cell
    
    # 4-circle cluster (inspiration 1 - score 5.7865)
    def four_circles(x, y):
        r = 0.15
        centers = [(0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)]
        for cx, cy in centers:
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy < r * r:
                return True
        return False
    
    # Single centered rod (original seed)
    def single_rod(x, y):
        return x * x + y * y < 0.2**2
    
    # Ring/annulus
    def ring(x, y):
        r = np.sqrt(x * x + y * y)
        return 0.15 < r < 0.35
    
    # Cross shape
    def cross(x, y):
        return abs(x) < 0.12 or abs(y) < 0.12
    
    # Square with hole
    def square_hole(x, y):
        return abs(x) < 0.4 and abs(y) < 0.4 and not (abs(x) < 0.15 and abs(y) < 0.15)
    
    # Evaluate all candidates
    candidates = [
        make_cell_from_function(four_circles),
        make_cell_from_function(single_rod),
        make_cell_from_function(ring),
        make_cell_from_function(cross),
        make_cell_from_function(square_hole),
    ]
    
    best_score = -float('inf')
    best_cell = None
    for c in candidates:
        s = evaluate(c)
        if s > best_score:
            best_score = s
            best_cell = c.copy()
    
    # Local search with symmetry-preserving orbit flips
    def get_orbit(i, j):
        """Get all symmetrically equivalent positions under D4 symmetry."""
        N = size
        points = [
            (i, j),
            (j, N - 1 - i),
            (N - 1 - i, N - 1 - j),
            (N - 1 - j, i),
            (i, N - 1 - j),
            (N - 1 - i, j),
            (j, i),
            (N - 1 - j, N - 1 - i)
        ]
        return list(set(points))
    
    def flip_orbit(cell, i, j):
        """Flip all pixels in the orbit of (i,j)."""
        orbit = get_orbit(i, j)
        for (r, c) in orbit:
            cell[r, c] = 1 - cell[r, c]
    
    # Simulated annealing-style local search
    cell = best_cell.copy()
    current_score = best_score
    temperature = 1.0
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() - start_time < time_limit:
        # Pick random pixel in fundamental domain (top-left 8x8 for efficiency)
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)
        
        # Flip the orbit
        flip_orbit(cell, i, j)
        
        new_score = evaluate(cell)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = current_score - new_score
        if new_score > current_score or np.random.random() < np.exp(-delta / temperature):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = cell.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            flip_orbit(cell, i, j)
            no_improve_count += 1
        
        # Cool down
        if no_improve_count > 50:
            temperature = max(0.1, temperature * 0.95)
        
        # Restart from best if stuck
        if no_improve_count > max_no_improve:
            cell = best_cell.copy()
            current_score = best_score
            # Perturb with a few random flips
            for _ in range(np.random.randint(2, 6)):
                ri = np.random.randint(0, 8)
                rj = np.random.randint(0, 8)
                flip_orbit(cell, ri, rj)
            current_score = evaluate(cell)
            no_improve_count = 0
            temperature = max(0.5, temperature * 0.8)
    
    # Return best_cell as nested list
    return best_cell.tolist()
