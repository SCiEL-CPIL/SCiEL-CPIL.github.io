import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 230s wall-clock kill
    
    # PWM Setup - use larger basis for better accuracy
    N_G = 21
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # K points in irreducible BZ - denser sampling
    k_points = []
    # Along X-axis: (0,0) to (0.5, 0)
    for i in range(11):
        t = i / 10.0 * 0.5
        k_points.append([t, 0.0])
    # Along M-X: (0.5, 0) to (0.5, 0.5)
    for i in range(1, 11):
        t = i / 10.0 * 0.5
        k_points.append([0.5, t])
    # Along M-G: (0.5, 0.5) to (0, 0)
    for i in range(1, 11):
        t = i / 10.0 * 0.5
        k_points.append([t, t])
    # Interior points for better gap estimation
    for t1 in [0.1, 0.2, 0.3, 0.4]:
        for t2 in [0.05, 0.1, 0.15, 0.2]:
            if t1 <= t2 and t1 + t2 <= 0.5:
                k_points.append([t1, t2])
    k_points = np.array(k_points)
    
    # Precompute diff indices for H matrix
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    valid_diffs = (np.abs(dx) <= N_G//2) & (np.abs(dy) <= N_G//2)
    diff_indices = np.where(valid_diffs, (dx + N_G//2) * N_G + (dy + N_G//2), 0)
    
    def get_fourier_eps_inv(cell_arr):
        """Compute Fourier coefficients of 1/eps for the given cell."""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        # FFT gives coefficients indexed by frequency 0..N-1
        # We need to map G_vectors indices to FFT indices
        coeffs_2d = np.fft.fft2(eps_inv) / (size * size)
        # Map each G vector to its FFT index
        # G vector (gx, gy) corresponds to FFT index (gx % size, gy % size)
        Gx_grid = G_vectors[:, 0] % size
        Gy_grid = G_vectors[:, 1] % size
        return coeffs_2d[Gx_grid, Gy_grid]
    
    def evaluate(cell_arr):
        """Evaluate the gap-midgap ratio using proper Hermitian eigenvalue solver."""
        eps_inv_coeffs = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k, dtype=float)
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # Build H matrix: H_ij = eps_inv[G_i - G_j] * |k + G_j|^2
            # This should be Hermitian for real eps_inv coefficients
            H = np.zeros((N_pw, N_pw), dtype=float)
            for i in range(N_pw):
                for j in range(N_pw):
                    if valid_diffs[i, j]:
                        H[i, j] = eps_inv_coeffs[diff_indices[i, j]] * k_plus_G_sq[j]
            
            # Symmetrize to ensure Hermitian (should be real symmetric)
            H = (H + H.T) / 2.0
            
            # Use eigvalsh for symmetric matrix
            evals = np.linalg.eigvalsh(H)
            evals = np.sort(evals)
            
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return 0.0
        return gap / midgap

    def generate_superellipse_cell(radius, exponent, hole_radius=0.0, hole_exponent=None):
        """Generate a symmetric superellipse cell with optional central hole."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Superellipse: |x|^n + |y|^n < r^n
                val = abs(x)**exponent + abs(y)**exponent
                if val < radius**exponent:
                    cell[i, j] = 1
                # Optional hole
                if hole_radius > 0 and hole_exponent is not None:
                    hole_val = abs(x)**hole_exponent + abs(y)**hole_exponent
                    if hole_val < hole_radius**hole_exponent:
                        cell[i, j] = 0
        return cell

    # Parameterized search over superellipse designs
    best_score = -float('inf')
    best_cell = None
    
    # Phase 1: Coarse grid search over parameters
    radii = np.linspace(0.15, 0.45, 12)
    exponents = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    hole_configs = [(0.0, None), (0.05, 2.0), (0.1, 2.0), (0.08, 3.0), (0.12, 2.0)]
    
    for r in radii:
        for n in exponents:
            for hr, hn in hole_configs:
                if time.time() - start_time > time_limit:
                    break
                cell = generate_superellipse_cell(r, n, hr, hn)
                score = evaluate(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
    
    # Phase 2: Local refinement around best parameter set
    if best_cell is not None and time.time() - start_time < time_limit:
        # Try small perturbations of the best cell
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
            cell = best_cell.copy()
            # Flip a few random symmetric orbits
            for _ in range(np.random.randint(1, 4)):
                i = np.random.randint(size)
                j = np.random.randint(size)
                # Get full symmetry orbit
                N = size
                orbit = set()
                for (a, b) in [(i, j), (j, N-1-i), (N-1-i, N-1-j), (N-1-j, i),
                               (i, N-1-j), (N-1-i, j), (j, i), (N-1-j, N-1-i)]:
                    orbit.add((a, b))
                for (r, c) in orbit:
                    cell[r, c] = 1 - cell[r, c]
            
            score = evaluate(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
    
    # Phase 3: More refined parameter search if time allows
    if best_cell is not None and time.time() - start_time < time_limit - 30:
        # Fine-tune around best radius and exponent
        for r in np.linspace(0.15, 0.45, 20):
            for n in np.linspace(1.0, 6.0, 10):
                if time.time() - start_time > time_limit:
                    break
                for hr, hn in [(0.0, None), (0.06, 2.0), (0.1, 2.0), (0.08, 4.0)]:
                    cell = generate_superellipse_cell(r, n, hr, hn)
                    score = evaluate(cell)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
    
    # Fallback: if no valid cell found, return a simple circle
    if best_cell is None:
        best_cell = generate_superellipse_cell(0.25, 2.0)
    
    return best_cell.tolist()
