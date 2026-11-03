"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Objective function: sum of squared violations of radius r
    def objective(x, r):
        c = x.reshape(-1, 3)
        # Wall constraints: |c_ik| <= 0.5 - r
        wall_viol = np.sum(np.maximum(np.abs(c) - (0.5 - r), 0.0) ** 2)
        
        # Pairwise constraints: |c_i - c_j| >= 2r
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = dists[mask]
        pair_viol = np.sum(np.maximum(2.0 * r - dists, 0.0) ** 2)
        
        return wall_viol + pair_viol

    # Initial guess: jittered grid
    def initial_guess():
        m = 4  # 4x4x4 grid
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        np.random.shuffle(grid)
        noise = np.random.normal(0, 0.02, grid.shape)
        return (grid[:N] + noise).flatten()

    best_r = 0.12
    best_x = initial_guess()
    
    # Verify initial guess is valid for best_r
    # The grid with m=4 has spacing 0.25, so r=0.125 is valid. 0.12 is safe.
    
    delta_r = 0.0005
    max_iter = 300
    max_attempts = 15

    for attempt in range(max_attempts):
        if time.time() >= deadline:
            break
            
        # For the first attempt, use the grid-based initial guess.
        # For subsequent attempts, use a new random guess to escape local minima.
        if attempt == 0:
            x = best_x.copy()
        else:
            x = initial_guess()
            
        r_current = best_r
        last_improvement_time = time.time()

        # Continuation loop: try to increase r
        while time.time() < deadline:
            bounds = [(-0.5 + r_current, 0.5 - r_current)] * (N * 3)
            
            res = minimize(
                objective, 
                x, 
                args=(r_current,), 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < 1e-8:
                # Success: we found a valid packing for r_current
                best_r = r_current
                best_x = res.x.copy()
                last_improvement_time = time.time()
                
                # Increase r for next step
                r_current += delta_r
                x = res.x.copy()
            else:
                # Failure: could not pack for r_current
                # Try to find the exact boundary by stepping back
                r_test = r_current - delta_r / 2.0
                if r_test > best_r:
                    bounds_test = [(-0.5 + r_test, 0.5 - r_test)] * (N * 3)
                    res_test = minimize(
                        objective, 
                        x, 
                        args=(r_test,), 
                        method='L-BFGS-B', 
                        bounds=bounds_test, 
                        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res_test.fun < 1e-8:
                        best_r = r_test
                        best_x = res_test.x.copy()
                break

        # If we didn't improve in the last 15 seconds, we might be stuck.
        # The loop above breaks on failure or deadline.
        # If it broke on failure, we go to the next attempt (restart).
        # If it broke on deadline, the outer loop will also break.
        
        # Optional: If we are close to the deadline, stop.
        if time.time() > deadline - 10.0:
            break

    return best_x.reshape(N, 3)
# EVOLVE-BLOCK-END
