# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).
    
    Strategy: Minimize V = integral_0^1 Ux(c) * Uy(c) dc.
    Uses exact 1D union length decomposition.
    Optimizes x and y vectors independently via alternating Nelder-Mead,
    followed by coordinate descent polish.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    
    # Increase samples to match evaluator and reduce objective noise
    rng = np.random.default_rng(42)
    M_obj = 500_000
    c_samples = rng.uniform(0, 1, M_obj)
    
    i_vals = np.arange(N, dtype=np.float64)
    shear_coeffs = i_vals / N  # shape (N,)
    
    def compute_1d_union_lens(x, c_samples):
        """
        Computes U(c) for all c_samples.
        x: shape (N,) - base positions
        c_samples: shape (M,)
        Returns: shape (M,) union lengths
        """
        L = x[:, None] + c_samples[None, :] * shear_coeffs[:, None]
        R = L + 1.0 / N
        
        # Transpose to (M, N)
        L_t = L.T
        R_t = R.T
        
        # Sort intervals by start point
        order = np.argsort(L_t, axis=1, kind='mergesort')
        L_sorted = np.take_along_axis(L_t, order, axis=1)
        R_sorted = np.take_along_axis(R_t, order, axis=1)
        
        # Compute union length via sweep
        max_r_up_to = np.maximum.accumulate(R_sorted, axis=1)
        max_r_before = np.empty_like(R_sorted)
        max_r_before[:, 0] = -np.inf
        max_r_before[:, 1:] = max_r_up_to[:, :-1]
        
        effective_start = np.maximum(L_sorted, max_r_before)
        contributions = np.maximum(0, R_sorted - effective_start)
        union_lens = np.sum(contributions, axis=1)
        
        return union_lens

    def make_positions(x, y):
        """Convert 1D positions x, y (shape N,) to 64x2 array."""
        positions = np.zeros((64, 2), dtype=np.float64)
        for k in range(64):
            i = k // 8
            j = k % 8
            positions[k, 0] = x[i]
            positions[k, 1] = y[j]
        return positions

    def validate_positions(pos):
        """Check if positions are valid (finite, V > 0)."""
        if not np.all(np.isfinite(pos)):
            return False
        x_k = pos[:, 0]
        y_k = pos[:, 1]
        i_vals = np.arange(64) // 8
        j_vals = np.arange(64) % 8
        
        X0 = np.min(x_k)
        X1 = np.max(x_k + (i_vals + 1) / 8.0)
        Y0 = np.min(y_k)
        Y1 = np.max(y_k + (j_vals + 1) / 8.0)
        
        if (X1 - X0) * (Y1 - Y0) <= 1e-9:
            return False
        return True

    def objective_x(x, U_y):
        """Evaluate objective for candidate x given fixed U_y."""
        if time.time() > deadline:
            return 1e9
        U_x = compute_1d_union_lens(x, c_samples)
        val = np.sum(U_x * U_y)
        if not np.isfinite(val):
            return 1e9
        return val

    def objective_y(y, U_x):
        """Evaluate objective for candidate y given fixed U_x."""
        if time.time() > deadline:
            return 1e9
        U_y = compute_1d_union_lens(y, c_samples)
        val = np.sum(U_x * U_y)
        if not np.isfinite(val):
            return 1e9
        return val

    def nelder_mead_1d(x0, obj_func, max_iters=500):
        """Nelder-Mead optimization for 8D vector."""
        x = x0.copy()
        
        # Initialize simplex
        alpha = 1.0
        beta = 0.5
        gamma = 2.0
        eps = 1e-4
        
        simplex = [x.copy()]
        for i in range(N):
            xi = x.copy()
            xi[i] += eps if x[i] == 0 else eps * x[i]
            simplex.append(xi)
        
        vals = [obj_func(s) for s in simplex]
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Sort simplex by objective value
            order = np.argsort(vals)
            simplex = [simplex[i] for i in order]
            vals = [vals[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            val_r = obj_func(x_r)
            
            if val_r < vals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                val_e = obj_func(x_e)
                if val_e < val_r:
                    simplex[-1] = x_e
                    vals[-1] = val_e
                else:
                    simplex[-1] = x_r
                    vals[-1] = val_r
            elif val_r < vals[-2]:
                simplex[-1] = x_r
                vals[-1] = val_r
            else:
                # Contraction
                x_c = centroid + beta * (simplex[-1] - centroid)
                val_c = obj_func(x_c)
                if val_c < vals[-1]:
                    simplex[-1] = x_c
                    vals[-1] = val_c
                else:
                    # Shrink
                    for i in range(1, N + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        vals[i] = obj_func(simplex[i])
        
        # Return best point
        best_idx = np.argmin(vals)
        return simplex[best_idx], vals[best_idx]

    # Generate diverse initial candidates for x and y
    k = np.arange(N, dtype=np.float64)
    
    # Centered shear alignment with slight spread
    x_init = -k / (2.0 * N) + 0.01 * np.sin(2 * np.pi * k / N)
    y_init = -k / (2.0 * N) + 0.01 * np.cos(2 * np.pi * k / N)
    
    best_x = x_init.copy()
    best_y = y_init.copy()
    
    # Compute initial U_y
    U_y = compute_1d_union_lens(best_y, c_samples)
    
    # Alternating Minimization
    max_outer_iters = 20
    for outer_it in range(max_outer_iters):
        if time.time() > deadline - 60:
            break
        
        # Optimize x
        def obj_x_wrapper(x):
            return objective_x(x, U_y)
        
        best_x, val_x = nelder_mead_1d(best_x, obj_x_wrapper, max_iters=300)
        
        # Update U_x
        U_x = compute_1d_union_lens(best_x, c_samples)
        
        # Optimize y
        def obj_y_wrapper(y):
            return objective_y(y, U_x)
        
        best_y, val_y = nelder_mead_1d(best_y, obj_y_wrapper, max_iters=300)
        
        # Update U_y
        U_y = compute_1d_union_lens(best_y, c_samples)
        
        # Check for convergence or improvement
        current_val = np.sum(U_x * U_y)
        
        if outer_it % 5 == 4:
            # Periodic random restarts to escape local minima
            if time.time() < deadline - 30:
                x_trial = best_x + np.random.normal(0, 0.05, N)
                y_trial = best_y + np.random.normal(0, 0.05, N)
                
                U_x_trial = compute_1d_union_lens(x_trial, c_samples)
                U_y_trial = compute_1d_union_lens(y_trial, c_samples)
                val_trial = np.sum(U_x_trial * U_y_trial)
                
                if val_trial < current_val:
                    best_x = x_trial
                    best_y = y_trial
                    U_x = U_x_trial
                    U_y = U_y_trial

    # Final polish: coordinate descent with minimize_scalar
    def coordinate_descent_polish(pos_x, pos_y, max_rounds=5):
        """Per-dimension 1D optimization using scipy minimize_scalar."""
        px = pos_x.copy()
        py = pos_y.copy()
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 15:
                break
            
            # Optimize x dimensions
            U_y = compute_1d_union_lens(py, c_samples)
            for dim in range(N):
                if time.time() > deadline - 15:
                    break
                
                def obj_1d_x(delta):
                    trial = px.copy()
                    trial[dim] += delta
                    U_x_trial = compute_1d_union_lens(trial, c_samples)
                    return np.sum(U_x_trial * U_y)
                
                res = minimize_scalar(obj_1d_x, bounds=(-0.1, 0.1),
                                      method='bounded',
                                      options={'xatol': 1e-6, 'maxiter': 50})
                if res.success and res.fun < obj_1d_x(0):
                    px[dim] += res.x
            
            # Optimize y dimensions
            U_x = compute_1d_union_lens(px, c_samples)
            for dim in range(N):
                if time.time() > deadline - 15:
                    break
                
                def obj_1d_y(delta):
                    trial = py.copy()
                    trial[dim] += delta
                    U_y_trial = compute_1d_union_lens(trial, c_samples)
                    return np.sum(U_x * U_y_trial)
                
                res = minimize_scalar(obj_1d_y, bounds=(-0.1, 0.1),
                                      method='bounded',
                                      options={'xatol': 1e-6, 'maxiter': 50})
                if res.success and res.fun < obj_1d_y(0):
                    py[dim] += res.x
        
        return px, py
    
    if time.time() < deadline - 30:
        best_x, best_y = coordinate_descent_polish(best_x, best_y, max_rounds=4)
    
    # One more Nelder-Mead pass if time allows
    if time.time() < deadline - 20:
        U_y = compute_1d_union_lens(best_y, c_samples)
        def obj_x_final(x):
            return objective_x(x, U_y)
        best_x, _ = nelder_mead_1d(best_x, obj_x_final, max_iters=150)
        
        U_x = compute_1d_union_lens(best_x, c_samples)
        def obj_y_final(y):
            return objective_y(y, U_x)
        best_y, _ = nelder_mead_1d(best_y, obj_y_final, max_iters=150)
    
    # Final coordinate descent polish
    if time.time() < deadline - 10:
        best_x, best_y = coordinate_descent_polish(best_x, best_y, max_rounds=3)
    
    positions = make_positions(best_x, best_y)
    if not validate_positions(positions):
        # Fallback
        fallback_x = -np.arange(N, dtype=np.float64) / (2.0 * N)
        fallback_y = fallback_x.copy()
        positions = make_positions(fallback_x, fallback_y)
    
    return positions

# EVOLVE-BLOCK-END
