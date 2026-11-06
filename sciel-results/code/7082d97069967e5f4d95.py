import numpy as np
import time
import sys
import scipy.optimize
import scipy.fft

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5.
    Strategy: Inverse design using gradient-based optimization of continuous Fourier coefficients.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Symmetry constraints are handled by the evaluator's check, but we must output a symmetric grid.
    # We will optimize a continuous field, binarize it, and then enforce symmetry by averaging 
    # the orbit members if necessary, or simply ensure the binarized result is symmetric 
    # by construction (e.g., optimizing only independent pixels is hard with gradients, 
    # so we optimize the full field and then symmetrize the final result).
    # Actually, a better approach for symmetry in continuous optimization:
    # We can optimize the full 32x32 field, but the objective function should be 
    # invariant or we just rely on the fact that the band gap of a symmetric 
    # structure is often a local maximum. 
    # However, the task requires the OUTPUT to be symmetric.
    # So: Optimize continuous field -> Binarize -> Symmetrize (average over orbits) -> Re-binarize.
    
    # 1. Setup for Fast Evaluation (Reduced PWE)
    # We use a smaller plane-wave basis for the gradient calculation to keep it fast.
    # The evaluator uses 289 PWs (17x17). We use 9x9=81 PWs for optimization.
    N_pw = 9
    G_x = np.arange(-N_pw//2, N_pw//2 + 1)
    G_y = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y])
    N_modes = len(G_vectors)
    
    # K-points in the irreducible Brillouin zone (same as seed)
    k_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (0.5, 0.0),
        (0.25, 0.25),
        (0.25, 0.0),
        (0.5, 0.25),
        (0.125, 0.125),
        (0.375, 0.125),
        (0.125, 0.375),
        (0.375, 0.375)
    ]

    def compute_gap_continuous(eps_grid):
        """
        Compute the gap-midgap ratio for a continuous epsilon grid.
        Returns (gap_ratio, max_omega_4, min_omega_5)
        """
        # Compute Fourier coefficients of epsilon
        # Note: We need the coefficients of 1/epsilon for TM mode in PWE?
        # Standard PWE for TM:
        # (k+G)^2 delta - (k+G).(k+G') / epsilon_{G-G'}
        # So we need FFT of 1/eps.
        # eps_grid is in [eps_low, eps_high].
        
        # To avoid division by zero or instability, we ensure eps is bounded away from 0.
        # eps_low is 1, so we are safe.
        
        inv_eps_grid = 1.0 / eps_grid
        inv_eps_fft = np.fft.fft2(inv_eps_grid) / (size * size)
        
        global_max_4 = 0.0
        global_min_5 = float('inf')
        
        for kx, ky in k_points:
            # Build PWE matrix
            kG = G_vectors + np.array([kx, ky])
            kG_sq = np.sum(kG**2, axis=1)
            
            # Dot product matrix
            dot = np.dot(kG, kG.T)
            
            # Epsilon matrix (actually 1/epsilon matrix)
            G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
            idx_x = G_diff[:, :, 0] % size
            idx_y = G_diff[:, :, 1] % size
            inv_eps_mat = inv_eps_fft[idx_x, idx_y].real
            
            # M = diag(kG_sq) - dot * inv_eps_mat
            # Note: The term is (k+G).(k+G') / epsilon. 
            # In matrix form, M_{ij} = (k+G_i)^2 delta_{ij} - (k+G_i).(k+G_j) * (1/eps)_{G_i-G_j}
            M = np.diag(kG_sq) - dot * inv_eps_mat
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                return -1.0, 0, 0
                
            if len(eigvals) < 5:
                return -1.0, 0, 0
                
            # Filter out negative eigenvalues if any (numerical noise)
            # Usually the first few are positive.
            omega_4 = np.sqrt(eigvals[3]) if eigvals[3] > 0 else 0
            omega_5 = np.sqrt(eigvals[4]) if eigvals[4] > 0 else 0
            
            if omega_4 > global_max_4:
                global_max_4 = omega_4
            if omega_5 < global_min_5:
                global_min_5 = omega_5
                
        if global_min_5 == float('inf'):
            return -1.0, 0, 0
            
        gap = global_min_5 - global_max_4
        midgap = (global_min_5 + global_max_4) / 2.0
        if midgap == 0:
            ratio = 0.0
        else:
            ratio = gap / midgap
            
        return ratio, global_max_4, global_min_5

    def objective(params):
        """
        Objective function for scipy.optimize.
        We minimize the negative of the gap ratio.
        params are the real and imaginary parts of the Fourier coefficients.
        We exclude the DC term (index 0,0) because it fixes the average epsilon.
        Actually, it's easier to work in real space for the grid, but gradients are easier in Fourier?
        Let's stick to optimizing the real-space grid directly using finite differences?
        No, scipy.optimize.minimize with 'L-BFGS-B' can handle 1024 variables, but it's slow.
        
        Alternative: Optimize a low-resolution continuous field and upscale?
        Or, optimize the Fourier coefficients directly.
        
        Let's optimize the real-space grid values, but use a smooth mapping.
        Let z be the vector of 32x32 values.
        We want to map z to eps_grid.
        eps_grid = eps_low + (eps_high - eps_low) * sigmoid(z)
        
        However, 1024 variables is a lot for L-BFGS.
        Let's try optimizing a smaller set of parameters: The first 20x20 Fourier coefficients.
        Then reconstruct the grid.
        
        Let's go with Fourier coefficients.
        We define a mask for which Fourier components to optimize.
        We keep the low-frequency components.
        """
        # Reconstruct grid from params
        # params: [real_part, imag_part] of shape (size, size) flattened?
        # Let's assume params is a flat array of length 2 * size * size.
        # But we only need to optimize a subset to keep it fast.
        # Let's optimize all of them but use a very fast objective.
        
        # Unpack params
        real_fft = params[:size*size].reshape((size, size))
        imag_fft = params[size*size:].reshape((size, size))
        
        # Ensure symmetry of FFT coefficients: C[-k] = C[k]*
        # If we optimize both, we might break symmetry.
        # It is safer to optimize the real-space grid directly if we can make it fast enough.
        # 1024 variables, ~1000 function evaluations? Might be too slow.
        
        # Let's try a different parameterization:
        # Optimize a low-res 8x8 grid of "control points" and bilinearly interpolate to 32x32.
        # This reduces dimensionality to 64.
        
        # Let's switch strategy to Low-Res Interpolation.
        # This is a structural change in representation.
        
        # We will redefine the objective to take 8x8 params.
        pass

    # --- REVISED STRATEGY: Low-Resolution Parameterization ---
    # The search space of 32x32 is too large for gradient descent within 120s if we use a decent PW basis.
    # We parameterize the dielectric distribution using a low-resolution grid (e.g., 8x8) 
    # and upsample it to 32x32 using bilinear interpolation.
    # This reduces the optimization variables to 64.
    
    low_res_size = 8
    
    def params_to_grid(params):
        """
        Map low-res params to 32x32 grid.
        params: 64 values in [0, 1].
        """
        low_res_grid = params.reshape((low_res_size, low_res_size))
        # Bilinear interpolation to 32x32
        # We can use scipy.ndimage.zoom or manual.
        # Manual is faster and dependency-free (mostly).
        # Actually, np.kron or simple repeat won't work for bilinear.
        # Let's use scipy.ndimage.zoom if available, or a simple loop.
        # Since we are in a sandbox, scipy is available.
        from scipy.ndimage import zoom
        # Zoom factor: 32/8 = 4
        high_res_grid = zoom(low_res_grid, (32/8.0, 32/8.0), order=1)
        # Clip to [0, 1]
        high_res_grid = np.clip(high_res_grid, 0, 1)
        return high_res_grid

    def objective(params):
        """
        Minimize negative gap ratio.
        """
        grid = params_to_grid(params)
        # Map [0,1] to [eps_low, eps_high]
        eps_grid = eps_low + (eps_high - eps_low) * grid
        
        ratio, _, _ = compute_gap_continuous(eps_grid)
        
        # If ratio is negative, it's bad. We want to maximize it.
        # So we minimize -ratio.
        # Add a small penalty for non-binary values to encourage sharp interfaces?
        # Not necessary if we binarize at the end.
        
        return -ratio

    # Initial guess: A rod
    # Create an 8x8 grid with a central blob
    init_params = np.zeros((low_res_size, low_res_size))
    center = low_res_size / 2.0
    for i in range(low_res_size):
        for j in range(low_res_size):
            dist = np.sqrt((i - center + 0.5)**2 + (j - center + 0.5)**2)
            if dist < 1.5:
                init_params[i, j] = 1.0
    init_params = init_params.flatten()

    # Optimization
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s for post-processing
    
    best_params = init_params.copy()
    best_score = objective(best_params)
    
    try:
        # Use L-BFGS-B
        # maxiter is a safety limit, but we have a time deadline.
        # We will run in a loop, checking time.
        
        # Since scipy.optimize doesn't have a built-in time limit callback that stops cleanly 
        # and returns the best so far easily in all versions, we will run it 
        # and hope it finishes, or we can use a manual gradient descent if scipy is too slow.
        
        # Let's try scipy first. It's robust.
        # If it takes too long, we might need to reduce maxiter or use a coarser PW basis.
        
        res = scipy.optimize.minimize(
            objective, 
            init_params, 
            method='L-BFGS-B', 
            bounds=[(0, 1)] * (low_res_size * low_res_size),
            options={'maxiter': 200, 'ftol': 1e-4}
        )
        
        if res.success:
            best_params = res.x
            best_score = -res.fun
        else:
            # Check if the result is better than initial
            current_score = -res.fun
            if current_score > best_score:
                best_params = res.x
                best_score = current_score
                
    except Exception as e:
        # If optimization fails, stick with initial
        pass

    # Check time
    if time.time() > deadline - 5:
        # We are running out of time. Return best so far.
        pass

    # Post-processing:
    # 1. Convert best_params to 32x32 grid
    # 2. Binarize (threshold at 0.5)
    # 3. Enforce Symmetry
    
    grid_cont = params_to_grid(best_params)
    grid_bin = (grid_cont > 0.5).astype(int)
    
    # Enforce Symmetry
    # The grid must be invariant under square symmetry group.
    # We can average the orbit members.
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            orbit = set()
            x, y = i, j
            points = [
                (x, y),
                (y, size-1-x),
                (size-1-x, size-1-y),
                (size-1-y, x),
                (x, size-1-y),
                (size-1-x, y),
                (y, x),
                (size-1-y, size-1-x)
            ]
            for p in points:
                if p not in orbit:
                    orbit.add(p)
            orbits.append(orbit)
            seen.update(orbit)
            
    sym_grid = np.zeros((size, size), dtype=int)
    for orbit in orbits:
        # Sum the values in the orbit
        val_sum = 0
        for (i, j) in orbit:
            val_sum += grid_bin[i, j]
        # Average and threshold
        avg_val = val_sum / len(orbit)
        if avg_val >= 0.5:
            for (i, j) in orbit:
                sym_grid[i, j] = 1
        else:
            for (i, j) in orbit:
                sym_grid[i, j] = 0
                
    # Final sanity check: Is the gap positive?
    # We can do one final evaluation of the symmetric grid to ensure it's valid.
    # If the symmetric grid is worse, we might want to try a few random perturbations?
    # For now, return the symmetric grid.
    
    return sym_grid.tolist()

# Helper for testing locally
if __name__ == "__main__":
    problem = {
        "pixels": 32,
        "eps_low": 1,
        "eps_high": 12,
        "lower_band": 4,
        "polarization": "TM"
    }
    result = design(problem)
    print("Result shape:", len(result), "x", len(result[0]))
    print("Symmetry check:", result == [row[::-1] for row in result] and result == [list(row) for row in zip(*result)])
