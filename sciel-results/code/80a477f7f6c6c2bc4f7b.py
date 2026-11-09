"""
Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
Strategy:
1. Use a C4v-symmetric harmonic potential (Zernike-like) to parameterize the dielectric mask.
2. This creates high-contrast, multi-lobed structures that maximize spatial frequency content.
3. Optimize the amplitudes and phases of these harmonic components.
4. Use a fast plane-wave evaluator for feedback during optimization.
"""
import numpy as np
import time
import math

def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- 1. Precompute Grid Coordinates ---
    # Pixel centers
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5
    R = np.sqrt(Xc**2 + Yc**2)
    Theta = np.arctan2(Yc, Xc)

    # --- 2. Define C4v Symmetric Basis Functions ---
    # We use a set of basis functions that are invariant under C4v symmetry.
    # f(r, theta) = sum c_i * basis_i(r, theta)
    # Basis functions are constructed from r^n * cos(m*theta) or r^n * sin(m*theta)
    # where m is a multiple of 4 for C4 symmetry.
    
    # Let's define a few specific basis functions that create interesting patterns.
    # We want high spatial frequency content, so we include higher radial orders.
    
    # Basis 0: Constant (DC offset)
    basis_0 = np.ones((size, size))
    
    # Basis 1: r^4 * cos(4*theta) -> Four-lobed pattern
    # cos(4*theta) = 8x^4 - 8x^2y^2 + y^4 normalized? No, just use trig.
    basis_1 = (R**4) * np.cos(4 * Theta)
    
    # Basis 2: r^2 * cos(4*theta) -> Different radial profile
    basis_2 = (R**2) * np.cos(4 * Theta)
    
    # Basis 3: r^6 * cos(4*theta) -> Sharper lobes
    basis_3 = (R**6) * np.cos(4 * Theta)
    
    # Basis 4: r^4 * cos(8*theta) -> Eight-lobed pattern (still C4 symmetric)
    basis_4 = (R**4) * np.cos(8 * Theta)
    
    # Basis 5: r^2 * cos(8*theta)
    basis_5 = (R**2) * np.cos(8 * Theta)
    
    # Basis 6: r^8 * cos(4*theta) -> Very sharp central feature
    basis_6 = (R**8) * np.cos(4 * Theta)
    
    # Basis 7: r^2 * cos(12*theta)
    basis_7 = (R**2) * np.cos(12 * Theta)

    # Normalize basis functions to have zero mean and unit variance for stable optimization
    def normalize_basis(b):
        b_mean = np.mean(b)
        b_std = np.std(b)
        if b_std < 1e-10:
            return np.zeros_like(b)
        return (b - b_mean) / b_std

    bases = [
        normalize_basis(basis_0),
        normalize_basis(basis_1),
        normalize_basis(basis_2),
        normalize_basis(basis_3),
        normalize_basis(basis_4),
        normalize_basis(basis_5),
        normalize_basis(basis_6),
        normalize_basis(basis_7)
    ]
    
    # Stack into a matrix for fast linear combination
    # B[i, j] is the value of basis i at pixel (j)
    # Let's flatten for matrix operations
    B_flat = np.array([b.flatten() for b in bases]) # Shape: (num_bases, size*size)
    
    def generate_cell(params):
        """
        Generate a binary cell from continuous potential parameters.
        params: array of coefficients for each basis function.
        """
        # Continuous potential
        potential = np.dot(params, B_flat)
        potential = potential.reshape((size, size))
        
        # Binarize: 1 if potential > 0 (dielectric), 0 if air
        # We can adjust the threshold, but 0 is a good start for symmetric zero-mean bases
        cell = (potential > 0).astype(int)
        
        # Enforce C4v symmetry exactly (just in case of numerical noise breaking symmetry slightly)
        # The basis functions are C4v symmetric, so this should be a no-op or fix tiny errors
        cell = enforce_symmetry(cell)
        
        return cell

    # --- 3. Symmetry Enforcement ---
    def enforce_symmetry(c):
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, j),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                s = sum(c[pi, pj] for pi, pj in points)
                sym[i, j] = 1 if s >= 4 else 0
        return sym

    # --- 4. Fast Evaluator ---
    # Use 7x7 plane waves (49 modes) for speed
    pw_half = 3
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    # Precompute G vector arrays
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    # Precompute G differences
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
    
    # FFT index lookup
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size
    
    # K-points in irreducible BZ (subset for speed)
    k_points = []
    k_points.append((0.0, 0.0))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.0, t))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.5 - t))
    k_points.append((0.5, 0.5))
    
    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        """
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 5. Optimization: Coarse-to-Fine Grid Search + Local Refinement ---
    
    # Initial parameters
    # Try a few random initializations and pick the best one
    num_params = len(bases)
    
    best_params = np.zeros(num_params)
    best_score = -float('inf')
    best_cell = None
    
    # Coarse Search: Random restarts with larger steps
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 30: break
        
        # Random initial parameters
        params = np.random.randn(num_params) * 0.5
        
        # Local optimization (Hill Climbing)
        current_params = params.copy()
        current_cell = generate_cell(current_params)
        current_score = evaluate_gap(current_cell)
        
        step_size = 0.2
        n_iterations = 200
        
        for it in range(n_iterations):
            if time.time() > deadline - 10: break
            
            # Pick a random parameter to perturb
            idx = np.random.randint(num_params)
            
            # Try positive and negative steps
            for direction in [1, -1]:
                trial_params = current_params.copy()
                trial_params[idx] += direction * step_size
                trial_cell = generate_cell(trial_params)
                trial_score = evaluate_gap(trial_cell)
                
                if trial_score > current_score:
                    current_params = trial_params
                    current_cell = trial_cell
                    current_score = trial_score
                    break # Accept and move on
            
            # Reduce step size slightly
            step_size *= 0.98
            if step_size < 0.01:
                step_size = 0.01
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            best_cell = current_cell.copy()

    # Fine Search: Refine the best parameters found
    if best_cell is not None:
        current_params = best_params.copy()
        current_cell = best_cell.copy()
        current_score = best_score
        
        step_size = 0.05
        n_iterations = 500
        
        for it in range(n_iterations):
            if time.time() > deadline - 5: break
            
            idx = np.random.randint(num_params)
            
            for direction in [1, -1]:
                trial_params = current_params.copy()
                trial_params[idx] += direction * step_size
                trial_cell = generate_cell(trial_params)
                trial_score = evaluate_gap(trial_cell)
                
                if trial_score > current_score:
                    current_params = trial_params
                    current_cell = trial_cell
                    current_score = trial_score
                    break
            
            step_size *= 0.99
            if step_size < 0.005:
                step_size = 0.005

        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            best_cell = current_cell.copy()

    # --- 6. Fallback: If harmonic approach didn't yield a good result, try disk initialization ---
    # This ensures we don't return a terrible result if the harmonic search fails.
    if best_score < 0.05:
        cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        radius = 4.0
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist < radius:
                        cell[i, j] = 1
        cell = enforce_symmetry(cell)
        disk_score = evaluate_gap(cell)
        if disk_score > best_score:
            best_score = disk_score
            best_cell = cell

    # --- 7. Finalize ---
    final_cell = enforce_symmetry(best_cell)
    return final_cell.tolist()
