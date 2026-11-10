import numpy as np
import time

def design(problem):
    size = 32
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin for 200s limit

    # --- 1. Fast Surrogate Evaluator Setup ---
    # We use a smaller plane-wave basis for the search loop to maximize iterations.
    # The evaluator uses 289 PWs (17x17). We use 81 PWs (9x9) for speed.
    N_PW_SURROGATE = 9
    G_range = np.arange(-N_PW_SURROGATE, N_PW_SURROGATE + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    Gx_flat = G_vectors[:, 0]
    Gy_flat = G_vectors[:, 1]
    Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
    Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
    idx_x = (Diff_x + 16) % 32
    idx_y = (Diff_y + 16) % 32

    # Generate k-points in the irreducible Brillouin zone
    def generate_k_points(num_points=30):
        points = []
        # Gamma-X: kx in [0, pi], ky = 0
        for i in range(10):
            kx = (i + 0.5) * (np.pi / 10)
            points.append((kx, 0))
        # X-M: kx = pi, ky in [0, pi]
        for i in range(10):
            ky = (i + 0.5) * (np.pi / 10)
            points.append((np.pi, ky))
        # M-Gamma: kx = ky
        for i in range(10):
            t = (i + 0.5) * (np.pi / 10)
            points.append((t, t))
        return np.array(points)

    K_POINTS = generate_k_points()
    KG_SQ = []
    for k in K_POINTS:
        kx, ky = k
        kg = k[None, :] + G_vectors * (2 * np.pi)
        kg_sq = np.sum(kg**2, axis=1)
        KG_SQ.append(kg_sq)
    KG_SQ = np.array(KG_SQ)
    KG_INV_SQRT = [1.0 / np.sqrt(kg_sq) for kg_sq in KG_SQ]

    def evaluate_cell_fast(cell_np):
        """
        Fast surrogate evaluation using 81 PWs.
        cell_np: numpy array of 0/1, shape (32, 32). 1 = dielectric, 0 = air.
        """
        eps = np.where(cell_np == 1, 12.0, 1.0)
        mu = 1.0 / eps
        
        # Compute Fourier coefficients of mu
        mu_fft = np.fft.fft2(mu)
        mu_fft_shifted = np.fft.fftshift(mu_fft)
        mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for ki in range(len(K_POINTS)):
            B_inv_sqrt = KG_INV_SQRT[ki]
            S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
            S = (S + S.T) / 2.0
            evals = np.linalg.eigvalsh(S)
            w4 = np.sqrt(evals[3])
            w5 = np.sqrt(evals[4])
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # --- 2. Structural Representation: Fourier Coefficients ---
    # We represent the dielectric map via its Fourier coefficients.
    # Due to C4v symmetry (centered at 15.5, 15.5), we can restrict the search space.
    # However, direct optimization of complex Fourier coefficients is tricky with binary constraints.
    # Instead, we use a "Superposition of Harmonics" approach.
    # V(x,y) = sum A_k cos(2*pi*(h*x + k*y) + phase_k)
    # We enforce symmetry by only using even harmonics or symmetric pairs.
    
    # Let's define a set of basis functions that are inherently C4v symmetric.
    # Basis 0: Constant (DC)
    # Basis 1: cos(2*pi*x)*cos(2*pi*y)  -> (1,1)
    # Basis 2: cos(2*pi*x) + cos(2*pi*y) -> (1,0)+(0,1)
    # Basis 3: cos(4*pi*x)*cos(4*pi*y)  -> (2,2)
    # Basis 4: cos(4*pi*x) + cos(4*pi*y) -> (2,0)+(0,2)
    # Basis 5: cos(6*pi*x)*cos(6*pi*y)  -> (3,3)
    # Basis 6: cos(6*pi*x) + cos(6*pi*y) -> (3,0)+(0,1)
    
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    xc = (X - (size - 1) / 2.0) / size
    yc = (Y - (size - 1) / 2.0) / size
    
    # Precompute basis functions
    basis_funcs = []
    # (1,1)
    basis_funcs.append(np.cos(2 * np.pi * xc) * np.cos(2 * np.pi * yc))
    # (1,0) + (0,1)
    basis_funcs.append(np.cos(2 * np.pi * xc) + np.cos(2 * np.pi * yc))
    # (2,2)
    basis_funcs.append(np.cos(4 * np.pi * xc) * np.cos(4 * np.pi * yc))
    # (2,0) + (0,2)
    basis_funcs.append(np.cos(4 * np.pi * xc) + np.cos(4 * np.pi * yc))
    # (3,3)
    basis_funcs.append(np.cos(6 * np.pi * xc) * np.cos(6 * np.pi * yc))
    # (3,0) + (0,1)
    basis_funcs.append(np.cos(6 * np.pi * xc) + np.cos(6 * np.pi * yc))
    
    basis_matrix = np.stack(basis_funcs, axis=-1) # Shape (32, 32, 6)
    
    def coeffs_to_cell(coeffs, threshold=0.0):
        """
        Convert Fourier coefficients to a binary cell.
        coeffs: array of shape (6,)
        threshold: offset for binarization
        """
        V = np.sum(basis_matrix * coeffs[None, None, :], axis=-1)
        # Normalize V to have zero mean and unit variance for stable thresholding?
        # Or just use raw values. Let's normalize to prevent saturation.
        V_mean = np.mean(V)
        V_std = np.std(V)
        if V_std > 1e-9:
            V = (V - V_mean) / V_std
        else:
            V = V - V_mean
            
        cell_np = (V > threshold).astype(int)
        
        # Enforce exact symmetry by averaging with symmetric counterparts
        # and re-thresholding? No, the basis functions are mathematically symmetric.
        # However, floating point errors might break it.
        # Let's enforce it strictly:
        # Symmetry: cell[i][j] == cell[i][31-j] == cell[31-i][j] == cell[31-i][31-j]
        # And transpose symmetry: cell[i][j] == cell[j][i]
        
        # We will just take the top-left 8x8 quadrant, enforce symmetry there, and expand.
        # This guarantees the output is valid.
        q = cell_np[:8, :8].copy()
        
        # Enforce transpose symmetry in the quadrant
        for i in range(8):
            for j in range(i, 8):
                val = (q[i][j] + q[j][i]) // 2
                q[i][j] = val
                q[j][i] = val
                
        # Expand to 32x32
        full_cell = np.zeros((size, size), dtype=int)
        for i in range(8):
            for j in range(8):
                val = q[i][j]
                full_cell[i][j] = val
                full_cell[i][size-1-j] = val
                full_cell[size-1-i][j] = val
                full_cell[size-1-i][size-1-j] = val
                
        return full_cell

    # --- 3. Optimization: Simulated Annealing on Coefficients ---
    # We optimize the 6 coefficients + 1 threshold = 7 parameters.
    # This is a much smaller space than 64 pixels (8x8 quadrant).
    
    def random_init():
        # Initialize with some structure
        # Strong (1,1) and (1,0) components often lead to holes
        coeffs = np.array([
            np.random.uniform(-1, 1), # (1,1)
            np.random.uniform(-1, 1), # (1,0)
            np.random.uniform(-0.5, 0.5), # (2,2)
            np.random.uniform(-0.5, 0.5), # (2,0)
            np.random.uniform(-0.2, 0.2), # (3,3)
            np.random.uniform(-0.2, 0.2)  # (3,0)
        ])
        thresh = np.random.uniform(-0.5, 0.5)
        return coeffs, thresh

    # Start with a known good geometric seed converted to coeffs?
    # Or just start with a few random restarts.
    
    best_cell = None
    best_score = -float('inf')
    
    # Run multiple short SA chains
    n_restarts = 5
    steps_per_restart = 200
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        if r == 0:
            # Try a specific promising structure: 4 holes
            # This corresponds to strong (1,1) and (1,0) with specific phases?
            # Actually, let's just use a random start for simplicity, 
            # but bias the first one towards the "4 holes" geometry.
            # The "4 holes" geometry has high (1,1) and (1,0) content.
            c0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            t0 = 0.0
            curr_coeffs, curr_thresh = c0, t0
        else:
            curr_coeffs, curr_thresh = random_init()
            
        curr_cell = coeffs_to_cell(curr_coeffs, curr_thresh)
        curr_score = evaluate_cell_fast(curr_cell)
        
        local_best_coeffs = curr_coeffs.copy()
        local_best_thresh = curr_thresh
        local_best_score = curr_score
        
        T = 0.5
        T_min = 0.01
        cooling = 0.99
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Perturb coefficients
            new_coeffs = curr_coeffs + np.random.normal(0, 0.1, size=6)
            new_thresh = curr_thresh + np.random.normal(0, 0.05)
            
            new_cell = coeffs_to_cell(new_coeffs, new_thresh)
            new_score = evaluate_cell_fast(new_cell)
            
            delta = new_score - curr_score
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                curr_coeffs = new_coeffs
                curr_thresh = new_thresh
                curr_cell = new_cell
                curr_score = new_score
                
                if curr_score > local_best_score:
                    local_best_score = curr_score
                    local_best_coeffs = curr_coeffs.copy()
                    local_best_thresh = curr_thresh
                    local_best_cell = curr_cell
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
        # After SA chain, update global best
        if local_best_score > best_score:
            best_score = local_best_score
            best_cell = local_best_cell

    # --- 4. Final Refinement: Pixel-wise SA on the Best Candidate ---
    # The Fourier representation might miss fine details.
    # Do a quick pixel-wise refinement on the best cell found.
    
    if best_cell is not None and time.time() < deadline - 20:
        curr_cell = best_cell.copy()
        curr_score = best_score
        T = 0.05
        T_min = 0.001
        cooling = 0.995
        
        # Limit steps to fit in remaining time
        max_steps = 500
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Flip a random pixel in the 8x8 quadrant
            i = np.random.randint(0, 8)
            j = np.random.randint(0, 8)
            
            new_cell = curr_cell.copy()
            # Flip pixel and its symmetric counterparts
            idxs = [
                (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
                (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
            ]
            seen = set()
            for r, c in idxs:
                if (r, c) not in seen:
                    seen.add((r, c))
                    new_cell[r][c] = 1 - new_cell[r][c]
                    
            new_score = evaluate_cell_fast(new_cell)
            delta = new_score - curr_score
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                curr_cell = new_cell
                curr_score = new_score
                if curr_score > best_score:
                    best_score = curr_score
                    best_cell = new_cell.copy()
                    
            T *= cooling
            if T < T_min:
                T = T_min

    # Convert best_cell (numpy array) to list of lists
    return best_cell.tolist() if best_cell is not None else [[0]*size for _ in range(size)]

# Fallback if design fails or returns None
def design_fallback(problem):
    size = 32
    cell = []
    centers = [
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (-0.25, -0.25)
    ]
    r = 0.28
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            is_air = 0
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < r * r:
                    is_air = 1
                    break
            row.append(0 if is_air else 1)
        cell.append(row)
    return cell

# Wrap design to ensure it never crashes
_orig_design = design
def design(problem):
    try:
        return _orig_design(problem)
    except Exception:
        return design_fallback(problem)
