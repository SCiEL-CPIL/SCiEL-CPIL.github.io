"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Use a continuous field representation with C4v symmetry.
    2. Optimize via gradient descent on a differentiable plane-wave surrogate.
    3. Multi-scale: coarse-to-fine with symmetry-adapted parameterization.
    4. Return the best design found within the time budget.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 230s limit

    # --- Symmetry-adapted fundamental domain ---
    # C4v symmetry: the fundamental domain is 1/8 of the cell
    # For a 32x32 cell centered at (15.5, 15.5), the fundamental domain
    # covers i in [0, 16], j in [0, i] (upper-left triangle of the left half)
    # Actually, for C4v with center at (15.5, 15.5):
    # The fundamental domain is: 0 <= i <= 15, 0 <= j <= i (triangle)
    # But we need to be careful with the center line.
    
    # Let's define the fundamental domain more carefully.
    # C4v operations: identity, 90deg rot, 180deg rot, 270deg rot,
    # and 4 reflections (horizontal, vertical, two diagonals)
    # The fundamental domain for a 32x32 grid centered at (15.5, 15.5):
    # We can use: i in [0, 15], j in [0, i] (this is 128 pixels)
    # But we need to handle the center lines carefully.
    
    # Actually, let's use a simpler approach: represent the full cell but
    # enforce symmetry after each update. The key is to work with continuous values.
    
    # --- Plane wave setup for surrogate ---
    # Use 9x9 plane waves (81 modes) for better accuracy in surrogate
    pw_half = 4
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors], dtype=float)
    Gy_arr = np.array([g[1] for g in G_vectors], dtype=float)
    
    # Precompute G difference indices for FFT lookup
    G_diff_h = np.zeros((N_pw, N_pw), dtype=int)
    G_diff_k = np.zeros((N_pw, N_pw), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diff_h[i, j] = (G_vectors[i][0] - G_vectors[j][0]) % size
            G_diff_k[i, j] = (G_vectors[i][1] - G_vectors[j][1]) % size
    
    # K-points in irreducible BZ
    k_points = []
    k_points.append((0.0, 0.0))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.0, t))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.5 - t))
    k_points.append((0.5, 0.5))
    
    def compute_fourier_coeffs(eps_grid):
        """Compute Fourier coefficients of 1/eps using FFT."""
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_h, G_diff_k] / (size * size)
        return coeffs
    
    def evaluate_gap_continuous(field):
        """
        Evaluate gap ratio for a continuous field (values in [0,1]).
        Uses soft interpolation: eps = eps_low + field * (eps_high - eps_low)
        Returns the gap-midgap ratio.
        """
        # Soft epsilon: interpolate between eps_low and eps_high
        eps_grid = eps_low + field * (eps_high - eps_low)
        eps_grid = np.clip(eps_grid, eps_low, eps_high)
        
        coeffs = compute_fourier_coeffs(eps_grid)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = (H + H.T) / 2.0  # Ensure Hermitian
            
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0.0, w4_sq))
                w5 = np.sqrt(max(0.0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2.0 * gap / midgap
        return ratio
    
    def enforce_symmetry_continuous(field):
        """Enforce C4v symmetry on a continuous field."""
        sym = np.zeros_like(field)
        count = np.zeros_like(field)
        
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
                vals = [field[pi, pj] for pi, pj in points]
                avg = np.mean(vals)
                for pi, pj in points:
                    sym[pi, pj] = avg
                    count[pi, pj] += 1
        
        # Average over all symmetric points
        result = np.zeros_like(field)
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
                unique_points = list(set(points))
                avg = np.mean([field[pi, pj] for pi, pj in unique_points])
                result[i, j] = avg
        return result
    
    def vectorize_symmetry(field):
        """Vectorized C4v symmetrization."""
        result = np.zeros_like(field)
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
                unique_points = list(set(points))
                avg = np.mean([field[pi, pj] for pi, pj in unique_points])
                result[i, j] = avg
        return result
    
    # --- Initialize with smooth symmetric field ---
    # Start with a Gaussian mixture at the four quarter-points
    field = np.zeros((size, size), dtype=float)
    centers = [(8.0, 8.0), (8.0, 24.0), (24.0, 8.0), (24.0, 24.0)]
    sigma = 3.0
    
    for cy, cx in centers:
        for i in range(size):
            for j in range(size):
                dist_sq = (i - cy)**2 + (j - cx)**2
                field[i, j] += math.exp(-dist_sq / (2 * sigma * sigma))
    
    # Normalize to [0, 1]
    field_max = np.max(field)
    if field_max > 0:
        field /= field_max
    
    # Enforce symmetry
    field = vectorize_symmetry(field)
    
    # --- Gradient-based optimization ---
    # Use finite differences to compute gradients of the surrogate loss
    # Then use Adam-like optimizer
    
    # Precompute gradient mask for efficiency
    # We'll perturb individual pixels and measure the change
    
    current_field = field.copy()
    current_score = evaluate_gap_continuous(current_field)
    best_field = current_field.copy()
    best_score = current_score
    
    # Adam optimizer state
    m = np.zeros_like(current_field)
    v = np.zeros_like(current_field)
    beta1 = 0.9
    beta2 = 0.999
    epsilon_adam = 1e-8
    lr = 0.1
    
    t_step = 0
    
    # Precompute which pixels to perturb for gradient estimation
    # Use a subset for speed
    all_pixels = [(i, j) for i in range(size) for j in range(size)]
    
    while time.time() < deadline:
        t_step += 1
        
        # Compute gradient via finite differences
        # Perturb a random subset of pixels for efficiency
        n_perturb = min(50, size * size)
        perturbation_size = 0.05
        
        # Select random pixels to perturb
        perm = np.random.permutation(size * size)
        selected = perm[:n_perturb]
        
        grad = np.zeros_like(current_field)
        
        for idx in selected:
            i = idx // size
            j = idx % size
            
            # Central difference
            field_plus = current_field.copy()
            field_plus[i, j] = min(1.0, current_field[i, j] + perturbation_size)
            score_plus = evaluate_gap_continuous(field_plus)
            
            field_minus = current_field.copy()
            field_minus[i, j] = max(0.0, current_field[i, j] - perturbation_size)
            score_minus = evaluate_gap_continuous(field_minus)
            
            # Gradient estimate
            grad[i, j] = (score_plus - score_minus) / (2 * perturbation_size)
            
            # Also set gradient for symmetric partners
            sym_points = [
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            for si, sj in sym_points:
                if (si, sj) != (i, j):
                    grad[si, sj] = grad[i, j]
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t_step)
        v_hat = v / (1 - beta2 ** t_step)
        
        # Update field
        update = m_hat / (np.sqrt(v_hat) + epsilon_adam)
        new_field = current_field + lr * update
        
        # Clip to [0, 1]
        new_field = np.clip(new_field, 0.0, 1.0)
        
        # Enforce symmetry
        new_field = vectorize_symmetry(new_field)
        
        # Evaluate
        new_score = evaluate_gap_continuous(new_field)
        
        # Accept if better, or with some probability (annealing-like)
        if new_score >= current_score or (np.random.random() < 0.1 and new_score > current_score - 0.01):
            current_field = new_field
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_field = current_field.copy()
        
        # Decay learning rate
        lr *= 0.995
        if lr < 0.005:
            lr = 0.005
        
        # Periodic restart: if stuck, try a different initialization
        if t_step % 200 == 0 and best_score < 0.5:
            # Try a different initial pattern
            alt_field = np.zeros((size, size), dtype=float)
            
            # Try a ring pattern
            for i in range(size):
                for j in range(size):
                    dist_center = math.sqrt((i - 15.5)**2 + (j - 15.5)**2)
                    if 6 < dist_center < 10:
                        alt_field[i, j] = 1.0
            alt_field = vectorize_symmetry(alt_field)
            
            alt_score = evaluate_gap_continuous(alt_field)
            if alt_score > best_score:
                current_field = alt_field
                current_score = alt_score
                best_field = alt_field.copy()
                best_score = alt_score
                m = np.zeros_like(current_field)
                v = np.zeros_like(current_field)
                lr = 0.1
                t_step = 0
    
    # --- Finalize: binarize the best field ---
    # Use a threshold that maximizes the gap
    # Try multiple thresholds
    best_binary = None
    best_binary_score = -float('inf')
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        binary = (best_field > threshold).astype(int)
        binary = vectorize_symmetry(binary.astype(float)).astype(int)
        
        # Evaluate with exact binary
        eps_grid = np.where(binary == 1, eps_high, eps_low)
        coeffs = compute_fourier_coeffs(eps_grid)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = (H + H.T) / 2.0
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4 = np.sqrt(max(0.0, eigenvalues[3]))
                w5 = np.sqrt(max(0.0, eigenvalues[4]))
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 < float('inf') and max_omega_4 > 0:
            gap = min_omega_5 - max_omega_4
            midgap = (min_omega_5 + max_omega_4) / 2.0
            ratio = 2.0 * gap / midgap if midgap > 0 else 0.0
            
            if ratio > best_binary_score:
                best_binary_score = ratio
                best_binary = binary
    
    if best_binary is None:
        best_binary = (best_field > 0.5).astype(int)
        best_binary = vectorize_symmetry(best_binary.astype(float)).astype(int)
    
    # Final symmetry check and return
    final_cell = vectorize_symmetry(best_binary.astype(float)).astype(int)
    
    # Verify symmetry
    assert np.array_equal(final_cell, final_cell[::-1, :]), "Row symmetry failed"
    assert np.array_equal(final_cell, final_cell[:, ::-1]), "Col symmetry failed"
    assert np.array_equal(final_cell, final_cell.T), "Transpose symmetry failed"
    
    return final_cell.tolist()

# EVOLVE-BLOCK-END
