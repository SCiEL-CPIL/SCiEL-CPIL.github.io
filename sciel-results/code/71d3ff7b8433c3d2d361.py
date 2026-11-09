"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Initialize the design with four circular rods at the quarter-points 
       (superlattice configuration known to open larger gaps).
    2. Represent the dielectric field using a low-order Chebyshev series 
       constrained to C4v symmetry.
    3. Fit the initial Chebyshev coefficients to the 4-rod target shape.
    4. Optimize the coefficients using stochastic hill-climbing with a 
       9x9 plane-wave evaluator for better correlation with the 17x17 evaluator.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- 1. Parameterization Setup ---
    # Chebyshev orders to use. Even orders ensure symmetry about the center.
    # (m, n) pairs with m <= n ensure symmetry under transpose.
    cheb_orders = [0, 2, 4, 6]
    param_indices = []
    for m in cheb_orders:
        for n in cheb_orders:
            if m <= n:
                param_indices.append((m, n))
    num_params = len(param_indices)

    # Grid coordinates
    x_coords = np.array([(i + 0.5) / size for i in range(size)])
    y_coords = np.array([(j + 0.5) / size for j in range(size)])
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    U = 2 * X - 1
    V = 2 * Y - 1

    def cheb_poly(t, k):
        if k == 0:
            return np.ones_like(t)
        if k == 1:
            return t.copy()
        T_prev2 = np.ones_like(t)
        T_prev1 = t.copy()
        for _ in range(2, k + 1):
            T_curr = 2 * t * T_prev1 - T_prev2
            T_prev2 = T_prev1
            T_prev1 = T_curr
        return T_prev1

    # Precompute basis functions
    basis = []
    for m, n in param_indices:
        Tm_U = cheb_poly(U, m)
        Tn_V = cheb_poly(V, n)
        if m == n:
            basis.append(Tm_U * Tn_V)
        else:
            Tn_U = cheb_poly(U, n)
            Tm_V = cheb_poly(V, m)
            basis.append(Tm_U * Tn_V + Tn_U * Tm_V)
    
    # Stack basis for fast linear combination
    # basis_stack shape: (num_params, size, size)
    basis_stack = np.stack(basis, axis=0)

    def get_eps_grid(params):
        """
        Computes the 32x32 epsilon grid from Chebyshev parameters.
        """
        # field = sum(params[i] * basis[i])
        field = np.tensordot(params, basis_stack, axes=(0, 0))
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    # --- 2. Initialization: 4 Circular Rods ---
    # Create a target shape: 4 circles at (1/4, 1/4), (1/4, 3/4), (3/4, 1/4), (3/4, 3/4)
    # Radius ~ 3.5 pixels (approx 0.11 of cell width)
    centers = [
        (size * 0.25, size * 0.25),
        (size * 0.25, size * 0.75),
        (size * 0.75, size * 0.25),
        (size * 0.75, size * 0.75)
    ]
    radius = 3.5
    
    target_field = np.zeros((size, size))
    for cx, cy in centers:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        # Smooth step function to make it differentiable-ish for fitting
        target_field += np.where(dist < radius, 1.0, 0.0)
    
    # Fit Chebyshev coefficients to target_field
    # We want params such that sum(params[i] * basis[i]) approximates target_field
    # This is a least squares problem: A @ params = target_field_flat
    A = np.array([b.flatten() for b in basis]) # shape (num_params, size*size)
    b_vec = target_field.flatten()
    
    try:
        initial_params, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    except:
        initial_params = np.zeros(num_params)
        initial_params[0] = 1.0

    # --- 3. Fast Evaluator ---
    # Use 9x9 plane waves (81 modes) for better accuracy than 7x7
    pw_half = 4 # range [-4, 4]
    G_vectors = []
    for h in range(-pw_half, pw_half + 1):
        for k in range(-pw_half, pw_half + 1):
            G_vectors.append((h, k))
    N_pw = len(G_vectors)

    # Precompute G vector differences for the Hamiltonian construction
    # We need coeff for (G_i - G_j)
    # G_diffs[i][j] = (G_i[0] - G_j[0], G_i[1] - G_j[1])
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
            
    # Precompute indices for FFT lookup
    # idx_h = (h % size), idx_k = (k % size)
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size

    def evaluate_gap(eps_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        """
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)
        
        # Extract Fourier coefficients for all G_diffs at once
        # coeff[i, j] = fft_1eps[G_diff_idx_h[i,j], G_diff_idx_k[i,j]] / (size*size)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        # K-points in irreducible BZ
        # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
        # Path: Gamma -> M -> X
        k_points = []
        # Gamma to M: (0, t)
        for t in np.linspace(0, 0.5, 4):
            k_points.append((0.0, t))
        # M to X: (t, 0.5-t)
        for t in np.linspace(0.25, 0.5, 3): # Skip M (already covered) and start closer to X
            k_points.append((t, 0.5 - t))
        # Add X
        k_points.append((0.5, 0.5))
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # K_i = G_i + k
            # Kx_i = G_vectors[i][0] + kx
            # Ky_i = G_vectors[i][1] + ky
            
            Gx = np.array([g[0] for g in G_vectors])
            Gy = np.array([g[1] for g in G_vectors])
            
            Kx = Gx + kx
            Ky = Gy + ky
            
            # Dot products: K_i dot K_j = Kx_i*Kx_j + Ky_i*Ky_j
            # Matrix form: outer(Kx, Kx) + outer(Ky, Ky)
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            
            # H = dots * coeffs
            H = dots * coeffs
            
            # H should be Hermitian. For real symmetric eps, coeffs are real (conjugate symmetric).
            # dots are real. So H is real symmetric.
            # Due to numerical noise, take real part.
            H_real = H.real
            
            # Solve eigenvalue problem
            # We only need the 5th smallest eigenvalue (index 4) and 4th (index 3)
            # Using eigvalsh is fast for small N
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

    # --- 4. Optimization Loop ---
    params = initial_params.copy()
    current_score = evaluate_gap(get_eps_grid(params))
    best_score = current_score
    best_params = params.copy()
    
    step_size = 0.5
    num_iterations = 0
    max_iterations = 10000
    
    # Track time
    last_time = time.time()
    
    while time.time() < deadline and num_iterations < max_iterations:
        num_iterations += 1
        
        # Pick a random parameter to perturb
        idx = np.random.randint(num_params)
        
        # Try a perturbation
        delta = np.random.choice([-1, 1]) * step_size
        
        new_params = params.copy()
        new_params[idx] += delta
        
        new_eps_grid = get_eps_grid(new_params)
        new_score = evaluate_gap(new_eps_grid)
        
        if new_score > best_score:
            params = new_params
            best_score = new_score
            best_params = params.copy()
            # Increase step size slightly on success
            step_size *= 1.02
            if step_size > 2.0:
                step_size = 2.0
        else:
            # Decrease step size on failure
            step_size *= 0.95
            if step_size < 0.05:
                step_size = 0.05
        
        # Occasionally, do a larger jump (restarting) to escape local minima
        if num_iterations % 1000 == 0:
            # Perturb all parameters
            jump_params = best_params + np.random.normal(0, 1.0, num_params)
            jump_score = evaluate_gap(get_eps_grid(jump_params))
            if jump_score > best_score:
                best_score = jump_score
                best_params = jump_params.copy()
                params = jump_params.copy()
                step_size = 1.0

    # --- 5. Finalize Design ---
    final_params = best_params
    final_eps_grid = get_eps_grid(final_params)
    
    # Threshold to binary 0/1
    threshold = (eps_low + eps_high) / 2.0
    cell = (final_eps_grid > threshold).astype(int)
    
    # Enforce symmetry exactly
    sym_cell = np.zeros_like(cell)
    for i in range(size):
        for j in range(size):
            # C4v symmetry group points
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
            s = 0
            for pi, pj in points:
                s += cell[pi, pj]
            avg = s / 8.0
            sym_cell[i, j] = 1 if avg >= 0.5 else 0
    
    return sym_cell.tolist()

# EVOLVE-BLOCK-END
