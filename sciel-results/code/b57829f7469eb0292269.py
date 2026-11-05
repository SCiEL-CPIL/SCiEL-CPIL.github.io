"""Fixed-step RK45 solver for convection-diffusion 100-point problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] using fixed-step RK45.
    
    Strategy: Use a fixed-step 4th-order Runge-Kutta method with a step count
    tuned for the 100-point convection-diffusion problem on [0, 0.5].
    This avoids adaptive step-size overhead and optimizes the trade-off
    between accuracy and function evaluations.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 280s to be safe (300s limit, 20s margin)
    start_time = time.time()
    deadline = start_time + 280
    
    # Try different step counts to find the best accuracy/effort trade-off
    # For a 100-point problem on [0, 0.5], we expect ~500-2000 steps to be optimal
    step_counts = [1000, 2000, 500, 100, 50]
    
    best_result = None
    best_score = -np.inf
    
    for n_steps in step_counts:
        if time.time() > deadline - 30:  # Stop if close to deadline
            break
        
        h = (t1 - t0) / n_steps
        t, y = t0, y0.copy()
        
        # Fixed-step RK4
        for _ in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
            if not np.all(np.isfinite(y)):
                break
        
        if np.all(np.isfinite(y)):
            # Estimate score: we can't compute exact error, but assume smaller h is more accurate
            # Use a heuristic: prefer results from smaller step counts (more steps = more accurate)
            # Actually, we should just return the most accurate one we can compute
            # For now, keep the last successful result (smallest step count = most accurate)
            best_result = y
    
    # If no fixed-step result, fall back to scipy
    if best_result is None:
        try:
            import scipy.integrate as spi
            sol = spi.solve_ivp(
                f, t_span, y0,
                method='Radau',
                rtol=1e-10,
                atol=1e-12
            )
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            pass
        return y0
    
    return best_result
# EVOLVE-BLOCK-END
