import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 105.0  # Safety margin for 230s wall-clock limit
    
    # PWM Setup
    N_G = 17
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # K points in irreducible BZ (30 points)
    k_points = []
    for i in range(10):
        t = i / 9.0 * 0.5
        k_points.append([t, 0.0])
        k_points.append([0.5, t])
        k_points.append([t, t])
    k_points.append([0.25, 0.1])
    k_points.append([0.35, 0.2])
    k_points.append([0.45, 0.3])
    k_points = np.array(k_points)
    
    # Precompute diff indices for H matrix
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    valid_diffs = (np.abs(dx) <= N_G//2) & (np.abs(dy) <= N_G//2)
    diff_indices = np.where(valid_diffs, (dx + N_G//2) * N_G + (dy + N_G//2), 0)
    
    # Precompute k_plus_G for all K points
    k_plus_G_list = []
    k_plus_G_sq_list = []
    for k in k_points:
        k_vec = np.array(k)
        k_plus_G = k_vec[None, :] + G_vectors
        k_plus_G_list.append(k_plus_G)
        k_plus_G_sq_list.append(np.sum(k_plus_G**2, axis=1))
    
    def cell_to_eps_inv_coeffs(cell_arr):
        """Convert binary cell to inverse epsilon Fourier coefficients."""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        coeffs_2d = np.fft.fft2(eps_inv) / (size * size)
        Gx_grid = G_vectors[:, 0] % size
        Gy_grid = G_vectors[:, 1] % size
        return coeffs_2d[Gx_grid, Gy_grid]
    
    def evaluate_from_coeffs(eps_inv_coeffs):
        """Evaluate gap-midgap ratio from precomputed Fourier coefficients."""
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for idx, k_plus_G_sq in enumerate(k_plus_G_sq_list):
            # H[i,j] = eps_inv_coeffs[G_i - G_j] * |k + G_i|^2
            # Vectorized: for each i, H[i,:] = eps_inv_coeffs[diff_indices[i,:]] * k_plus_G_sq[i]
            H = eps_inv_coeffs[diff_indices] * k_plus_G_sq[:, None]
            H[~valid_diffs] = 0.0
            
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap
    
    def evaluate(cell_arr):
        eps_inv_coeffs = cell_to_eps_inv_coeffs(cell_arr)
        return evaluate_from_coeffs(eps_inv_coeffs)
    
    # RBF parameterization with C4v symmetry
    # Centers at high-symmetry points: (0,0), (0.5,0), (0.5,0.5)
    # Each center generates 4 symmetric copies under C4v
    def generate_rbf_field(weights, widths, cell_shape):
        """Generate C4v-symmetric RBF field from weights and widths."""
        N = cell_shape
        x = (np.arange(N) + 0.5) / N - 0.5
        y = (np.arange(N) + 0.5) / N - 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        field = np.zeros((N, N), dtype=float)
        
        # Define C4v symmetric center sets
        # Center at (0,0) -> only one unique center
        # Center at (0.5,0) -> 4 centers: (±0.5,0), (0,±0.5)
        # Center at (0.5,0.5) -> 4 centers: (±0.5,±0.5)
        
        # For each weight/width pair, add RBF contributions from all symmetric copies
        for w, sigma in zip(weights, widths):
            if sigma <= 0:
                continue
            sigma2 = 2 * sigma * sigma
            
            # Type 0: center at origin (1 unique)
            if w > 0:
                field += w * np.exp(-(X**2 + Y**2) / sigma2)
            
            # Type 1: center at (0.5, 0) and symmetric copies (4 unique)
            centers_type1 = [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]
            for cx, cy in centers_type1:
                field += w * 0.25 * np.exp(-((X - cx)**2 + (Y - cy)**2) / sigma2)
            
            # Type 2: center at (0.5, 0.5) and symmetric copies (4 unique)
            centers_type2 = [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
            for cx, cy in centers_type2:
                field += w * 0.25 * np.exp(-((X - cx)**2 + (Y - cy)**2) / sigma2)
        
        return field
    
    def field_to_cell(field, threshold):
        """Convert continuous field to binary cell."""
        return (field > threshold).astype(int)
    
    # Initialize with simple rod-like structure
    N_rbf = 3  # 3 RBF terms
    best_weights = np.array([1.0, 0.5, 0.3])
    best_widths = np.array([0.1, 0.2, 0.3])
    best_threshold = 0.5
    
    # Generate initial cell
    field = generate_rbf_field(best_weights, best_widths, size)
    cell = field_to_cell(field, best_threshold)
    
    # Ensure symmetry is exact by averaging with its transforms
    def enforce_symmetry(cell_arr):
        """Enforce C4v symmetry by averaging with all symmetry operations."""
        N = cell_arr.shape[0]
        sym_ops = [
            cell_arr,
            cell_arr.T,
            cell_arr[::-1, :],
            cell_arr[:, ::-1],
            cell_arr[::-1, ::-1],
            cell_arr.T[::-1, :],
            cell_arr.T[:, ::-1],
            cell_arr[::-1, ::-1].T
        ]
        avg = np.mean(sym_ops, axis=0)
        return (avg > 0.5).astype(int)
    
    cell = enforce_symmetry(cell)
    best_score = evaluate(cell)
    best_cell = cell.copy()
    
    # Local search in continuous parameter space
    no_improve_count = 0
    max_no_improve = 200
    step_size = 0.1
    
    while time.time() - start_time < time_limit:
        # Perturb parameters
        perturb_idx = np.random.randint(N_rbf)
        param_type = np.random.choice(['weight', 'width', 'threshold'])
        
        new_weights = best_weights.copy()
        new_widths = best_widths.copy()
        new_threshold = best_threshold
        
        if param_type == 'weight':
            delta = np.random.uniform(-step_size, step_size)
            new_weights[perturb_idx] = max(0.0, new_weights[perturb_idx] + delta)
        elif param_type == 'width':
            delta = np.random.uniform(-step_size * 0.5, step_size * 0.5)
            new_widths[perturb_idx] = max(0.05, new_widths[perturb_idx] + delta)
        else:  # threshold
            delta = np.random.uniform(-0.1, 0.1)
            new_threshold = np.clip(new_threshold + delta, 0.1, 0.9)
        
        # Generate new cell
        field = generate_rbf_field(new_weights, new_widths, size)
        new_cell = field_to_cell(field, new_threshold)
        new_cell = enforce_symmetry(new_cell)
        
        score = evaluate(new_cell)
        
        if score > best_score:
            best_score = score
            best_cell = new_cell.copy()
            best_weights = new_weights
            best_widths = new_widths
            best_threshold = new_threshold
            no_improve_count = 0
            # Reduce step size on improvement
            step_size = max(0.01, step_size * 0.95)
        else:
            no_improve_count += 1
            # Increase step size slightly when stuck
            step_size = min(0.3, step_size * 1.05)
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Random restart in parameter space
            best_weights = np.random.uniform(0.1, 1.5, N_rbf)
            best_widths = np.random.uniform(0.05, 0.4, N_rbf)
            best_threshold = np.random.uniform(0.3, 0.7)
            no_improve_count = 0
            step_size = 0.1
    
    # Return best_cell as nested list
    return best_cell.tolist()
