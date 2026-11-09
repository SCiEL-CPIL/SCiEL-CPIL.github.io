"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Combines:
    1. Continuous Chebyshev level-set representation with C4v symmetry (from inspiration).
    2. Full 289-plane-wave exact evaluation (from current program) for high-fidelity scoring.
    3. Stochastic hill-climbing with adaptive steps on the continuous parameter space.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- 1. Parameterization ---
    # We represent the dielectric field as:
    # eps(x, y) = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * tanh(f(x, y))
    # where f(x, y) is a sum of Chebyshev polynomials T_m(2x-1) * T_n(2y-1)
    # constrained by C4v symmetry.
    
    # Chebyshev polynomials T_k(t) are defined on [-1, 1].
    # Let u = 2*x - 1, v = 2*y - 1.
    # Symmetry requirements:
    # f(x, y) = f(1-x, y) => f(u, v) is even in u.
    # f(x, y) = f(x, 1-y) => f(u, v) is even in v.
    # f(x, y) = f(y, x) => f(u, v) = f(v, u).
    
    # So we sum over m, n >= 0 where m, n are even.
    # T_m(-u) = (-1)^m T_m(u). For evenness in u, m must be even.
    # For evenness in v, n must be even.
    # Symmetry f(u,v)=f(v,u) implies coefficients A[m,n] = A[n,m].
    
    # Let's use m, n in {0, 2, 4, 6, 8, 10}.
    # Number of independent parameters:
    # Pairs (m, n) with m, n in {0, 2, 4, 6, 8, 10} and m <= n.
    # This gives us 21 parameters.
    
    max_cheb = 10
    cheb_orders = [0, 2, 4, 6, 8, 10]
    param_indices = []
    for i, m in enumerate(cheb_orders):
        for j, n in enumerate(cheb_orders):
            if m <= n:
                param_indices.append((m, n))
    num_params = len(param_indices)

    # Precompute Chebyshev values on the grid
    # Grid points: pixel centers (i + 0.5) / size
    x_coords = np.array([(i + 0.5) / size for i in range(size)])
    y_coords = np.array([(j + 0.5) / size for j in range(size)])
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Map to Chebyshev domain [-1, 1]
    U = 2 * X - 1
    V = 2 * Y - 1
    
    # Precompute T_m(U) and T_n(V) for m, n in cheb_orders
    # T_0 = 1
    # T_1 = t
    # T_k = 2*t*T_{k-1} - T_{k-2}
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

    # Store basis functions for each parameter
    # basis[i] is a 32x32 array corresponding to param_indices[i]
    basis = []
    for m, n in param_indices:
        Tm_U = cheb_poly(U, m)
        Tn_V = cheb_poly(V, n)
        # The term is T_m(U) * T_n(V) + T_n(U) * T_m(V) if m != n, else T_m(U) * T_m(V)
        # Wait, the field f(u,v) = sum A_mn T_m(u) T_n(v).
        # If we enforce A_mn = A_nm, then the term for pair (m,n) with m<n contributes
        # A_mn [ T_m(u)T_n(v) + T_n(u)T_m(v) ].
        # For m=n, it's A_mm T_m(u)T_m(v).
        if m == n:
            basis.append(Tm_U * Tn_V)
        else:
            Tn_U = cheb_poly(U, n)
            Tm_V = cheb_poly(V, m)
            basis.append(Tm_U * Tn_V + Tn_U * Tm_V)
    
    # --- 2. Full Evaluator (289 PWs) ---
    # Precompute plane-wave vectors (289 = 17x17 grid, |G| <= 8.5)
    N = 17
    half = N // 2  # 8
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            if gx * gx + gy * gy <= (half + 0.5) ** 2:  # |G| <= 8.5
                G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors, dtype=float)
    n_pw = len(G_vectors)
    
    # Precompute 30 k-points in irreducible BZ
    k_points = []
    # Along Gamma-X: kx from 0 to 0.5, ky=0
    for i in range(11):
        k_points.append((i * 0.5 / 10.0, 0.0))
    # Along X-M: kx=0.5, ky from 0 to 0.5
    for i in range(1, 11):
        k_points.append((0.5, i * 0.5 / 10.0))
    # Along M-Gamma: kx from 0.5 to 0, ky from 0.5 to 0
    for i in range(1, 11):
        t = i / 10.0
        k_points.append((0.5 * (1 - t), 0.5 * (1 - t)))
    # Remove duplicates
    unique_k = []
    for k in k_points:
        if not any(abs(k[0] - uk[0]) < 1e-9 and abs(k[1] - uk[1]) < 1e-9 for uk in unique_k):
            unique_k.append(k)
    k_points = np.array(unique_k[:30], dtype=float)
    
    # Precompute coefficient index matrix
    coeff_lookup = {}
    for idx, (gx, gy) in enumerate(G_vectors):
        coeff_lookup[(int(gx), int(gy))] = idx
        
    coeff_idx_mat = np.full((n_pw, n_pw), -1, dtype=int)
    for i in range(n_pw):
        for j in range(n_pw):
            dx = int(G_vectors[i, 0] - G_vectors[j, 0])
            dy = int(G_vectors[i, 1] - G_vectors[j, 1])
            key = (dx, dy)
            if key in coeff_lookup:
                coeff_idx_mat[i, j] = coeff_lookup[key]
                
    # Precompute cross product matrices for each k-point to save time
    cross_mats = []
    for kx, ky in k_points:
        kG_x = kx + G_vectors[:, 0]
        kG_y = ky + G_vectors[:, 1]
        cross = np.outer(kG_x, kG_y) - np.outer(kG_y, kG_x)
        cross_mats.append(cross)

    def compute_inv_eps_fourier(cell_arr):
        """Compute Fourier coefficients of 1/eps for the given cell."""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv = np.fft.fft2(inv_eps)
        c = fft_inv / (size * size)
        coeffs = np.zeros(n_pw, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            ix = int(gx) % size
            iy = int(gy) % size
            coeffs[idx] = c[iy, ix]
        return np.real(coeffs)

    def score_cell(cell_arr):
        """Compute the gap-midgap ratio."""
        coeffs = compute_inv_eps_fourier(cell_arr)
        
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for idx_k, cross in enumerate(cross_mats):
            # Build coefficient matrix using precomputed indices
            C = np.zeros((n_pw, n_pw), dtype=float)
            valid = coeff_idx_mat >= 0
            C[valid] = coeffs[coeff_idx_mat[valid]]
            
            M = C * cross
            M = (M + M.T) / 2.0
            
            eigvals = np.linalg.eigvalsh(M)
            
            if len(eigvals) >= 5:
                w4 = math.sqrt(max(0, eigvals[3]))
                w5 = math.sqrt(max(0, eigvals[4]))
                if w4 > max_w4: max_w4 = w4
                if w5 < min_w5: min_w5 = w5
        
        if min_w5 <= 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)

    def params_to_cell(params):
        """Convert parameter vector to binary cell."""
        f = np.zeros((size, size))
        for i, p in enumerate(params):
            f += p * basis[i]
        # Threshold at 0
        cell = (f > 0).astype(int)
        return cell

    # --- 3. Optimization ---
    rng = np.random.default_rng(42)
    
    # Initialize with random parameters
    best_params = rng.normal(0, 0.5, num_params)
    best_cell = params_to_cell(best_params)
    best_score = score_cell(best_cell)
    
    # Adaptive step size
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 20
    
    # Hill climbing on continuous parameters
    while time.time() < deadline - 5.0:
        # Perturb parameters
        new_params = best_params + rng.normal(0, step_size, num_params)
        new_cell = params_to_cell(new_params)
        new_score = score_cell(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_params = new_params
            best_cell = new_cell
            no_improve_count = 0
            # Increase step size slightly if improving
            step_size = min(step_size * 1.05, 1.0)
        else:
            no_improve_count += 1
            # Decrease step size if not improving
            step_size = max(step_size * 0.95, 0.01)
            
            # If stuck for too long, try a larger jump (simulated annealing-like)
            if no_improve_count > max_no_improve:
                # Random restart within a neighborhood
                new_params = best_params + rng.normal(0, 0.5, num_params)
                new_cell = params_to_cell(new_params)
                new_score = score_cell(new_cell)
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params
                    best_cell = new_cell
                no_improve_count = 0
                step_size = 0.1

    return [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
# EVOLVE-BLOCK-END
