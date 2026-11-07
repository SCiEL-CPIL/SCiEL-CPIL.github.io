"""Root finder for 3 fixed 2D systems using Broyden's method with fallback to SciPy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with adaptive line search, falling back to SciPy's root finder.
    """
    x = np.array(x0, dtype=float).copy()
    if x.shape != (2,):
        x = x.reshape(2)
        
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def time_up():
        return time.time() > deadline

    # --- Phase 1: Broyden's Method ---
    # Initial evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        # Perturb if initial point is bad
        for dx in [1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1]:
            x_try = x + np.array([dx, -dx])
            Fx_try = F(x_try)
            if np.all(np.isfinite(Fx_try)):
                x = x_try
                Fx = Fx_try
                break
        else:
            return x # Return whatever we have, likely fail

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Initial Jacobian using central differences for better accuracy
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - F(x - e)) / (2 * h)

    best_x = x.copy()
    best_norm = norm_Fx

    # Convergence tracking for early termination
    prev_norm = norm_Fx
    fast_converging = False

    for iteration in range(40):
        if time_up():
            break

        # Check for fast convergence - if norm is decreasing rapidly, we're close
        if iteration > 0 and norm_Fx < 0.5 * prev_norm:
            fast_converging = True
        prev_norm = norm_Fx

        # Solve for step direction
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        # If step is too small, we're likely at the root or stuck
        if np.linalg.norm(d) < 1e-12:
            break

        # Line search with backtracking
        alpha = 1.0
        step_accepted = False
        
        for ls_iter in range(15):
            if time_up():
                break
            
            x_new = x + alpha * d
            Fx_new = F(x_new)
            
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue

            norm_Fx_new = np.linalg.norm(Fx_new)
            
            if norm_Fx_new <= 1e-8:
                return x_new

            # Track best
            if norm_Fx_new < best_norm:
                best_norm = norm_Fx_new
                best_x = x_new.copy()

            # Use a more aggressive Armijo condition for faster convergence
            # c = 1e-3 is more aggressive than 1e-4
            if norm_Fx_new < (1.0 - 1e-3 * alpha) * norm_Fx:
                # Update Broyden (good Broyden update)
                d_dot_d = np.dot(d, d)
                if d_dot_d > 1e-15:
                    delta_F = Fx_new - Fx - J @ d
                    J = J + np.outer(delta_F, d) / d_dot_d
                
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                step_accepted = True
                break
            else:
                alpha *= 0.5

        if not step_accepted:
            break

        # Early termination if we're converging fast and close to solution
        if fast_converging and norm_Fx < 1e-4:
            # Do a few more iterations to ensure convergence
            if iteration >= 3:
                continue
            break

    if norm_Fx <= 1e-8:
        return x

    # --- Phase 2: Fallback to SciPy ---
    # Use best_x from Broyden as starting point
    try:
        res = root(F, best_x, method='hybr', options={'maxfev': 500})
        if res.success or np.linalg.norm(res.fun) <= 1e-8:
            return res.x
        # Return the best of the two
        if np.linalg.norm(res.fun) < best_norm:
            return res.x
    except Exception:
        pass

    return best_x

# EVOLVE-BLOCK-END
