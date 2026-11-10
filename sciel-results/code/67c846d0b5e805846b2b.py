# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N
    t0 = time.time()
    DEADLINE = 590.0  # Safety margin for return overhead

    def elapsed():
        return time.time() - t0

    # Pre-compute indices for tubes
    # k is in [0, 64)
    # i = k // 8, j = k % 8
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N

    # Pre-compute shifts for x and y as functions of c
    # x_k(c) = x_k + c * (i_k * w)
    # y_k(c) = y_k + c * (j_k * w)

    # Loss function: Union Volume via Monte Carlo
    # We use a fixed set of random points for deterministic evaluation during optimization.
    # The final score is computed by the evaluator using its own rng, but this proxy is good enough for optimization.
    
    rng_opt = np.random.default_rng(12345)
    n_mc = 20000
    u = rng_opt.uniform(0, 1, n_mc)
    v = rng_opt.uniform(0, 1, n_mc)
    c = rng_opt.uniform(0, 1, n_mc)

    def volume_loss(params):
        """
        params: 128D vector [x_0...x_63, y_0...y_63]
        Returns: Estimated union volume
        """
        x = params[:64]
        y = params[64:]
        
        # Box bounds
        x_max = x + (i_idx + 1) * w
        y_max = y + (j_idx + 1) * w
        
        X0 = np.min(x)
        X1 = np.max(x_max)
        Y0 = np.min(y)
        Y1 = np.max(y_max)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return 1e10
            
        # Map MC points to box
        a = X0 + u * (X1 - X0)
        b = Y0 + v * (Y1 - Y0)
        
        # Vectorized check:
        # Tube k covers [x_k + c*i_k*w, x_k + c*i_k*w + w] x [y_k + c*j_k*w, y_k + c*j_k*w + w]
        # Condition: 0 <= a - (x_k + c*i_k*w) <= w AND 0 <= b - (y_k + c*j_k*w) <= w
        
        # shift_x[m, k] = c_m * i_k * w
        # shift_y[m, k] = c_m * j_k * w
        shift_x = c[:, None] * (i_idx[None, :] * w)
        shift_y = c[:, None] * (j_idx[None, :] * w)
        
        diff_x = a[:, None] - shift_x - x[None, :]
        diff_y = b[:, None] - shift_y - y[None, :]
        
        valid_x = (diff_x >= 0) & (diff_x <= w)
        valid_y = (diff_y >= 0) & (diff_y <= w)
        
        covered = np.any(valid_x & valid_y, axis=1)
        fraction = np.mean(covered)
        
        return area * fraction

    # Initial Guess: Keich Construction
    # This is a known lower bound for the Kakeya problem in Z^d.
    # It is separable: x depends only on i, y depends only on j.
    # We will use this as the starting point for the optimizer.
    
    # Generate Keich positions for N=8
    def keich_positions(n):
        pos = int(np.log2(n))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(n, dtype=np.float64)
        for idx in range(n):
            eps = np.array(list(bin(idx)[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich - keich.min()

    base_1d = keich_positions(N)
    
    # Create 2D grid for i, j
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x_init = base_1d[i_flat]
    y_init = base_1d[j_flat]
    
    x0 = np.concatenate([x_init, y_init])
    
    best_params = x0.copy()
    best_loss = volume_loss(x0)
    
    # Optimization Strategy:
    # 1. L-BFGS-B is good for smooth landscapes. The MC loss is noisy but 20k samples make it reasonably smooth.
    # 2. We will run L-BFGS-B from the Keich start.
    # 3. We will also try a few random restarts if time permits, but the Keich start is strong.
    # 4. We might also try to break symmetry by adding small noise and optimizing.
    
    # Since the problem is non-convex, we might get stuck.
    # However, the Keich construction is a local minimum (or close to it) for the separable case.
    # Can we do better than separable?
    # The "union volume" is minimized when tubes overlap as much as possible.
    # The Keich construction maximizes overlap in a specific algebraic sense.
    
    # Let's run L-BFGS-B.
    # We need to be careful with the time budget.
    
    options = {
        'maxiter': 100,
        'gtol': 1e-4,
        'ftol': 1e-6,
        'maxls': 20,
        'maxcor': 10
    }
    
    # Run 1: From Keich
    if elapsed() < DEADLINE - 100:
        res = minimize(volume_loss, x0, method='L-BFGS-B', options=options)
        if res.fun < best_loss:
            best_params = res.x
            best_loss = res.fun
            
    # Run 2: From Keich + small noise (to break symmetry if separable is not global min)
    if elapsed() < DEADLINE - 100:
        rng_perturb = np.random.default_rng(42)
        noise = rng_perturb.normal(0, 0.05, 128)
        x_perturbed = x0 + noise
        res = minimize(volume_loss, x_perturbed, method='L-BFGS-B', options=options)
        if res.fun < best_loss:
            best_params = res.x
            best_loss = res.fun

    # Run 3: Random restarts if time permits
    # The search space is 128D. Random restarts are expensive.
    # But L-BFGS-B is fast per iteration.
    
    # Let's try a few random starts with smaller maxiter
    rng_search = np.random.default_rng(999)
    
    # We will generate random starts within a reasonable box.
    # The optimal box size is likely around 1.0 to 2.0.
    # Keich box size is roughly 1.0.
    
    for restart in range(5):
        if elapsed() > DEADLINE - 50:
            break
            
        # Random start: uniform in [0, 1] for all coordinates
        # This is a very rough start, but L-BFGS-B might pull it towards a good solution.
        # Actually, random uniform in [0,1] is likely to produce a large volume (little overlap).
        # Better: Start near Keich but with larger noise.
        
        noise = rng_search.normal(0, 0.2, 128)
        x_start = x0 + noise
        
        res = minimize(volume_loss, x_start, method='L-BFGS-B', options={'maxiter': 50})
        if res.fun < best_loss:
            best_params = res.x
            best_loss = res.fun

    # Final Refinement:
    # Use the best parameters found so far and run a final L-BFGS-B with tighter tolerances.
    if elapsed() < DEADLINE - 20:
        options_final = {
            'maxiter': 50,
            'gtol': 1e-6,
            'ftol': 1e-8
        }
        res = minimize(volume_loss, best_params, method='L-BFGS-B', options=options_final)
        if res.fun < best_loss:
            best_params = res.x
            best_loss = res.fun

    x_final = best_params[:64]
    y_final = best_params[64:]
    
    positions = np.stack([x_final, y_final], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
