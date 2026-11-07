"""
Robust root-finder for 2D systems using a two-phase strategy:
1. Global Grid Search: Deterministically sample a neighborhood to find a good starting point.
2. Local Broyden Method: Use Broyden's method with strict merit-function line search for rapid convergence.

This approach avoids the instability of homotopy methods and the randomness of multi-start heuristics.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Global Search: Sample a grid around x0 to find a point close to the root.
    2. Local Solver: Broyden's method with line search on ||F(x)||^2.
    3. Fallback: Expand grid search if local solver fails or stalls.
    
    Call budget: < 500 calls.
    Time budget: < 150 seconds.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    x0_arr = np.array(x0, dtype=float).copy()
    best_x = x0_arr.copy()
    best_norm = float('inf')
    
    calls = 0
    max_calls = 490
    
    def safe_F(x):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError
        if calls >= max_calls:
            raise RuntimeError("Call limit exceeded")
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None
            calls += 1
            return Fx
        except Exception:
            return None

    def update_best(x, Fx):
        nonlocal best_x, best_norm
        if Fx is not None:
            norm = np.linalg.norm(Fx)
            if norm < best_norm:
                best_norm = norm
                best_x = x.copy()
            return norm
        return float('inf')

    def compute_jacobian_fd(x, Fx):
        """Finite difference Jacobian. Uses 2 calls (forward diff)."""
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_j = safe_F(x + e)
            if Fx_j is None:
                return None
            J[:, j] = (Fx_j - Fx) / h
        return J

    def broyden_solve(x_start, max_iter=100):
        """
        Solves F(x) = 0 using Broyden's method with merit function line search.
        Returns (x, norm, success).
        """
        nonlocal best_x, best_norm, calls
        
        x = x_start.copy()
        Fx = safe_F(x)
        if Fx is None:
            return x, best_norm, False
        
        norm = update_best(x, Fx)
        if norm <= 1e-8:
            return x, norm, True
            
        # Initial Jacobian
        J = compute_jacobian_fd(x, Fx)
        if J is None:
            return x, norm, False

        for _ in range(max_iter):
            if time.time() > deadline or calls >= max_calls - 5:
                break
                
            try:
                # Newton step
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    # Regularize if singular
                    J_reg = J + 1e-8 * np.eye(2)
                    dx = np.linalg.solve(J_reg, -Fx)
                except Exception:
                    break

            if not np.all(np.isfinite(dx)):
                break

            # Line search on merit function m(x) = ||F(x)||^2
            alpha = 1.0
            m_old = norm**2
            step_accepted = False
            
            for _ in range(20):
                x_try = x + alpha * dx
                if not np.all(np.isfinite(x_try)):
                    alpha *= 0.5
                    continue
                
                Fx_try = safe_F(x_try)
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                
                norm_try = np.linalg.norm(Fx_try)
                m_try = norm_try**2
                
                update_best(x_try, Fx_try)
                
                # Armijo condition: sufficient decrease
                if m_try < m_old - 1e-4 * alpha * np.dot(dx, Fx): 
                    # Update Broyden
                    s = alpha * dx
                    y = Fx_try - Fx
                    J_s = J @ s
                    diff = y - J_s
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(diff, s) / denom
                    else:
                        # Recompute Jacobian if step is too small
                        J_new = compute_jacobian_fd(x_try, Fx_try)
                        if J_new is not None:
                            J = J_new
                        else:
                            break
                    
                    x = x_try
                    Fx = Fx_try
                    norm = norm_try
                    step_accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not step_accepted:
                break
                
            if norm <= 1e-8:
                return x, norm, True

        return x, norm, norm <= 1e-8

    # --- Phase 1: Global Grid Search ---
    
    # Determine scale based on x0
    scale = max(1.0, np.linalg.norm(x0_arr))
    
    # Grid search parameters
    # Start with a small grid, expand if needed
    grid_sizes = [5, 9, 13]
    radii = [0.5 * scale, 2.0 * scale, 5.0 * scale]
    
    best_start_x = x0_arr.copy()
    best_start_norm = float('inf')
    
    # Initial evaluation
    Fx0 = safe_F(x0_arr)
    if Fx0 is not None:
        n0 = np.linalg.norm(Fx0)
        best_start_norm = n0
        update_best(x0_arr, Fx0)
        if n0 <= 1e-8:
            return x0_arr

    for size, radius in zip(grid_sizes, radii):
        if time.time() > deadline - 20 or calls >= max_calls - 50:
            break
            
        # Create grid points
        # We sample a box around x0
        # To keep calls low, we use a sparse grid
        # For 2D, a grid of size N x N has N^2 points.
        # 5x5 = 25 points. 9x9 = 81 points. 13x13 = 169 points.
        # Total calls for all grids: 1 + 25 + 81 + 169 = 276.
        # This leaves ~200 calls for local solvers.
        
        # Generate grid
        xs = np.linspace(-radius, radius, size)
        for dx in xs:
            for dy in xs:
                x_cand = x0_arr + np.array([dx, dy])
                
                # Skip if we already evaluated x0 (dx=0, dy=0)
                if np.allclose(x_cand, x0_arr):
                    continue
                
                Fx_cand = safe_F(x_cand)
                if Fx_cand is None:
                    continue
                
                norm_cand = np.linalg.norm(Fx_cand)
                update_best(x_cand, Fx_cand)
                
                if norm_cand < best_start_norm:
                    best_start_norm = norm_cand
                    best_start_x = x_cand.copy()
                    
                if norm_cand <= 1e-8:
                    return x_cand

    # --- Phase 2: Local Refinement from Best Global Point ---
    
    # Try Broyden from the best point found in Phase 1
    x_sol, norm_sol, success = broyden_solve(best_start_x, max_iter=50)
    if success:
        return x_sol
    
    # If that didn't work, try from x0 directly (in case global search was bad)
    if not np.allclose(best_start_x, x0_arr):
        x_sol, norm_sol, success = broyden_solve(x0_arr, max_iter=50)
        if success:
            return x_sol

    # --- Phase 3: Fallback / Multi-Start ---
    # If we still haven't found a root, try a few more random or structured starts
    # using the remaining call budget.
    
    rng = np.random.RandomState(42)
    for _ in range(10):
        if time.time() > deadline - 5 or calls >= max_calls - 10:
            break
            
        # Random perturbation
        x_cand = x0_arr + rng.randn(2) * scale * 0.5
        
        Fx_cand = safe_F(x_cand)
        if Fx_cand is None:
            continue
            
        norm_cand = np.linalg.norm(Fx_cand)
        update_best(x_cand, Fx_cand)
        
        if norm_cand <= 1e-8:
            return x_cand
            
        # Try local solve from this point
        x_sol, norm_sol, success = broyden_solve(x_cand, max_iter=20)
        if success:
            return x_sol

    return best_x
# EVOLVE-BLOCK-END
