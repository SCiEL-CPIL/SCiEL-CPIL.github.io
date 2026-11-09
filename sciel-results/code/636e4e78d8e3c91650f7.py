"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Parameterize the design using superellipses with C4v symmetry.
    2. Use a proper TM plane-wave eigenvalue solver with symmetrized matrices.
    3. Optimize continuous parameters with Nelder-Mead, then binarize.
    4. Enforce C4v symmetry throughout.
    5. Return the best design found within the time budget.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 230s limit

    # --- C4v symmetry enforcement ---
    def enforce_symmetry(c):
        """Enforce C4v symmetry: cell[i,j] must equal all 8 symmetric counterparts."""
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

    # --- Superellipse parameterization ---
    # A superellipse: |x/a|^n + |y/b|^n <= 1
    # With C4v symmetry, we place superellipses at 4 quarter-points.
    # Parameters: [cx, cy, a, b, n] for the primary superellipse in Q1,
    # then mirror to other quadrants.
    # We also allow a second smaller superellipse for more flexibility.
    
    def superellipse_mask(cx, cy, a, b, n, grid_size):
        """Create a boolean mask for a superellipse centered at (cx, cy)."""
        xs = np.arange(grid_size) - cx
        ys = np.arange(grid_size) - cy
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        # Superellipse: |X/a|^n + |Y/b|^n <= 1
        val = np.abs(X / a) ** n + np.abs(Y / b) ** n
        return val <= 1.0

    def params_to_cell(params):
        """
        Convert continuous parameters to a 32x32 binary cell.
        params: [cx1, cy1, a1, b1, n1, cx2, cy2, a2, b2, n2]
        Two superellipses in Q1, mirrored to all quadrants.
        """
        cell = np.zeros((size, size), dtype=int)
        
        for idx in range(2):
            cx = params[idx * 5 + 0]
            cy = params[idx * 5 + 1]
            a = params[idx * 5 + 2]
            b = params[idx * 5 + 3]
            n = params[idx * 5 + 4]
            
            # Clamp parameters to valid ranges
            cx = np.clip(cx, 0, size / 2)
            cy = np.clip(cy, 0, size / 2)
            a = np.clip(a, 0.5, size / 2)
            b = np.clip(b, 0.5, size / 2)
            n = np.clip(n, 0.5, 10.0)
            
            mask = superellipse_mask(cx, cy, a, b, n, size)
            
            # Apply C4v symmetry: mirror to all 4 quadrants
            # Q1: (cx, cy), Q2: (size-1-cx, cy), Q3: (size-1-cx, size-1-cy), Q4: (cx, size-1-cy)
            # Also transpose for full C4v
            for (mx, my) in [(cx, cy), (size-1-cx, cy), (cx, size-1-cy), (size-1-cx, size-1-cy)]:
                m = superellipse_mask(mx, my, a, b, n, size)
                cell = np.maximum(cell, m.astype(int))
            
            # Transpose symmetry
            for (mx, my) in [(cy, cx), (size-1-cy, cx), (cy, size-1-cx), (size-1-cy, size-1-cx)]:
                m = superellipse_mask(mx, my, b, a, n, size)  # swap a,b for transpose
                cell = np.maximum(cell, m.astype(int))
        
        return cell

    # --- TM Plane-wave evaluator ---
    # For TM polarization, the eigenvalue problem is:
    # sum_G' (k+G)·(k+G') * (1/eps)_{G-G'} * c_{G'} = omega^2 * c_G
    # We symmetrize the matrix for numerical stability.
    
    # Use 17x17 = 289 plane waves (matching evaluator)
    pw_half = 8
    G_vectors = np.array([(h, k) for h in range(-pw_half, pw_half + 1) 
                          for k in range(-pw_half, pw_half + 1)], dtype=float)
    N_pw = len(G_vectors)
    
    # Precompute G-G' differences
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (N_pw, N_pw, 2)
    G_diff_idx_h = G_diff[:, :, 0].astype(int) % size
    G_diff_idx_k = G_diff[:, :, 1].astype(int) % size
    
    # K-points in irreducible BZ (30 points as specified)
    k_points = []
    # Gamma to M: (0, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.0, t))
    # M to X: (t, 0.5-t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, 0.5 - t))
    # Gamma to X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, t))
    # Additional interior points
    for t in [0.1, 0.2, 0.3, 0.4]:
        for s in [0.1, 0.2, 0.3, 0.4]:
            if s <= t and t <= 0.5 and s <= 0.5:
                k_points.append((t, s))
    
    # Remove duplicates
    seen = set()
    unique_k = []
    for kx, ky in k_points:
        key = (round(kx, 10), round(ky, 10))
        if key not in seen:
            seen.add(key)
            unique_k.append((kx, ky))
    
    def evaluate_gap(cell_grid):
        """
        Computes the TM band gap between bands 4 and 5 using plane-wave method.
        cell_grid: 32x32 array of 0/1
        """
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # FFT for Fourier coefficients of 1/eps
        fft_1eps = np.fft.fft2(inv_eps) / (size * size)
        
        # Build coefficient matrix
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k]  # (N_pw, N_pw)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors  # (N_pw, 2)
            omega_sq_diag = np.sum(k_plus_G ** 2, axis=1)  # (N_pw,)
            
            # Matrix: M[i,j] = omega_sq_diag[i] * coeffs[i,j]
            # For TM, the standard formulation uses (k+G)^2 * (1/eps)_{G-G'}
            # Symmetrize for numerical stability
            M = omega_sq_diag[:, np.newaxis] * coeffs
            M = 0.5 * (M + M.conj().T)
            M_real = M.real
            
            # Get eigenvalues
            try:
                eigenvalues = np.linalg.eigvalsh(M_real)
            except np.linalg.LinAlgError:
                continue
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                if w4_sq < 0 or w5_sq < 0:
                    continue
                    
                w4 = np.sqrt(w4_sq)
                w5 = np.sqrt(w5_sq)
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return -1.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- Optimization using Nelder-Mead ---
    from scipy.optimize import minimize
    
    # Initial parameters: one large superellipse at (8,8) with n=2 (circle)
    # and one smaller at (16,16)
    initial_params = np.array([
        8.0, 8.0, 4.0, 4.0, 2.0,    # First superellipse: center (8,8), a=b=4, n=2
        16.0, 16.0, 3.0, 3.0, 2.0    # Second superellipse: center (16,16), a=b=3, n=2
    ])
    
    def objective(params):
        if time.time() > deadline - 5:
            return 100.0  # Penalize timeout
        cell = params_to_cell(params)
        score = evaluate_gap(cell)
        return -score  # Minimize negative score
    
    # Run Nelder-Mead optimization
    best_params = initial_params.copy()
    best_score = -objective(initial_params)
    
    # Try multiple initializations
    initializations = [
        # Circle-like
        np.array([8.0, 8.0, 4.0, 4.0, 2.0, 16.0, 16.0, 3.0, 3.0, 2.0]),
        # Square-like
        np.array([8.0, 8.0, 4.0, 4.0, 1.0, 16.0, 16.0, 3.0, 3.0, 1.0]),
        # Intermediate
        np.array([8.0, 8.0, 4.0, 4.0, 1.5, 16.0, 16.0, 3.0, 3.0, 1.5]),
        # Offset
        np.array([6.0, 6.0, 5.0, 3.0, 2.0, 18.0, 18.0, 4.0, 2.0, 2.0]),
        # Large single
        np.array([8.0, 8.0, 5.0, 5.0, 2.0, 8.0, 8.0, 1.0, 1.0, 2.0]),
    ]
    
    for init in initializations:
        if time.time() > deadline - 30:
            break
        
        current_params = init.copy()
        current_score = -objective(current_params)
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
        
        # Nelder-Mead refinement
        try:
            result = minimize(objective, current_params, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 0.1, 'fatol': 0.01,
                                   'adaptive': True})
            if result.fun < -best_score:
                best_params = result.x.copy()
                best_score = -result.fun
        except Exception:
            pass
        
        # Local random search around best
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            perturbed = best_params + np.random.normal(0, 0.5, len(best_params))
            s = -objective(perturbed)
            if s > best_score:
                best_score = s
                best_params = perturbed.copy()
    
    # Final refinement with smaller steps
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        perturbed = best_params + np.random.normal(0, 0.2, len(best_params))
        s = -objective(perturbed)
        if s > best_score:
            best_score = s
            best_params = perturbed.copy()

    # Convert to final cell
    final_cell = params_to_cell(best_params)
    final_cell = enforce_symmetry(final_cell)
    
    return final_cell.tolist()

# EVOLVE-BLOCK-END
