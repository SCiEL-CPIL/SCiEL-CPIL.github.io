import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    def is_valid(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= tol or h <= tol:
            return False
        for i in range(N):
            if r[i] <= tol:
                return False
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    def sum_radii(r):
        return float(np.sum(r))

    def objective(z):
        """
        Objective function for optimization.
        z = [x1..xN, y1..yN, r1..rN, w]
        Minimizes -sum(r) + penalties.
        """
        x = z[0:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        w = z[3*N]
        h = 2.0 - w

        # Basic bounds check
        if w < 0.2 or h < 0.2 or np.any(r < 0.01):
            return 1e6

        obj = -np.sum(r)
        penalty = 0.0

        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                overlap = max(r[i] + r[j] - d, 0.0)
                penalty += 100.0 * overlap * overlap

        # Boundary penalty
        for i in range(N):
            # x bounds
            if x[i] < r[i]:
                penalty += 100.0 * (r[i] - x[i]) ** 2
            if x[i] > w - r[i]:
                penalty += 100.0 * (x[i] - (w - r[i])) ** 2
            # y bounds
            if y[i] < r[i]:
                penalty += 100.0 * (r[i] - y[i]) ** 2
            if y[i] > h - r[i]:
                penalty += 100.0 * (y[i] - (h - r[i])) ** 2

        return obj + penalty

    def refine_candidate(x_init, y_init, r_init, w_init):
        """
        Local refinement using L-BFGS-B followed by Nelder-Mead.
        Returns best valid (x, y, r, w) or None.
        """
        if time.time() > deadline - 2.0:
            return None

        # Stage 1: L-BFGS-B with bounds
        # Variables: x, y, r, w
        # Bounds for x: [0, 2] (loose, penalty handles walls)
        # Bounds for y: [0, 2]
        # Bounds for r: [0.01, 1.0]
        # Bounds for w: [0.5, 1.5] (perimeter 4 => w+h=2, w in (0,2))
        
        z0 = np.concatenate([x_init, y_init, r_init, [w_init]])
        
        bounds = [(-0.1, 2.1)] * (2 * N) + [(0.01, 1.0)] * N + [(0.4, 1.6)]
        
        try:
            res1 = minimize(objective, z0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 50})
            z1 = res1.x
        except Exception:
            z1 = z0

        # Stage 2: Nelder-Mead for final polish (derivative-free, good for non-smooth penalties)
        if time.time() > deadline - 1.0:
            # Just use L-BFGS result if valid
            x_t = z1[0:N]
            y_t = z1[N:2*N]
            r_t = z1[2*N:3*N]
            w_t = z1[3*N]
            if is_valid(x_t, y_t, r_t, w_t):
                return x_t, y_t, r_t, w_t
            return None

        try:
            res2 = minimize(objective, z1, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-12})
            z2 = res2.x
        except Exception:
            z2 = z1

        x_t = z2[0:N]
        y_t = z2[N:2*N]
        r_t = z2[2*N:3*N]
        w_t = z2[3*N]
        
        # Ensure radii are positive and valid
        r_t = np.maximum(r_t, 0.01)
        
        if is_valid(x_t, y_t, r_t, w_t):
            return x_t, y_t, r_t, w_t
        else:
            # Try L-BFGS result if NM failed validity
            x_t1 = z1[0:N]
            y_t1 = z1[N:2*N]
            r_t1 = z1[2*N:3*N]
            w_t1 = z1[3*N]
            r_t1 = np.maximum(r_t1, 0.01)
            if is_valid(x_t1, y_t1, r_t1, w_t1):
                return x_t1, y_t1, r_t1, w_t1
            return None

    best_value = 0.0
    best_result = None

    # Global Search using Differential Evolution
    # Bounds:
    # x: [0, 2], y: [0, 2]
    # r: [0.05, 0.5] (Estimate max radius: area 1, 21 circles -> avg area 0.047 -> r~0.12. Max single r maybe 0.3-0.4)
    # w: [0.5, 1.5]
    
    bounds = [(0, 2.0)] * (2 * N) + [(0.05, 0.5)] * N + [(0.5, 1.5)]
    
    # Number of iterations depends on time. 
    # DE is expensive. We will run it for a fixed number of iterations or until time is up.
    # maxiter=100, popsize=10 might be ~1000 evals. Each eval is O(N^2) ~ 400 ops. Very fast.
    # We can afford more.
    
    max_iter = 200
    popsize = 15
    
    def de_callback(xk, convergence):
        # Check time
        if time.time() > deadline - 30.0:
            # We need to stop DE early if time is tight, but scipy DE doesn't support easy early stop via callback return value in all versions.
            # However, we can raise an exception or just let it run if it's fast enough.
            # With N=21, objective is very fast. 200 iters * 15 pop * ~1ms = 3s. 
            # So 200 iters is safe.
            pass

    try:
        res_de = differential_evolution(
            objective, 
            bounds=bounds, 
            maxiter=max_iter, 
            popsize=popsize,
            tol=1e-8,
            seed=42,
            polish=False, # We do our own polish
            init='latinhypercube'
        )
        
        # Extract best from DE
        if res_de.success or res_de.fun < 1e5:
            z_best_de = res_de.x
            x_init = z_best_de[0:N]
            y_init = z_best_de[N:2*N]
            r_init = z_best_de[2*N:3*N]
            w_init = z_best_de[3*N]
            
            # Refine
            refined = refine_candidate(x_init, y_init, r_init, w_init)
            if refined:
                x_t, y_t, r_t, w_t = refined
                val = sum_radii(r_t)
                if val > best_value:
                    best_value = val
                    best_result = (x_t.copy(), y_t.copy(), r_t.copy(), w_t)
                    
    except Exception as e:
        print(f"DE failed: {e}")

    # If DE didn't find a good result, try a few random restarts with local refinement
    if best_result is None or time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        
        # Try a few random initializations
        for i in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Random aspect ratio
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            
            # Random positions
            x0 = rng.uniform(0.1, w0 - 0.1, N)
            y0 = rng.uniform(0.1, h0 - 0.1, N)
            # Random radii, slightly smaller than estimate
            r0 = np.full(N, 0.1)
            
            refined = refine_candidate(x0, y0, r0, w0)
            if refined:
                x_t, y_t, r_t, w_t = refined
                val = sum_radii(r_t)
                if val > best_value:
                    best_value = val
                    best_result = (x_t.copy(), y_t.copy(), r_t.copy(), w_t)

    # Final fallback if nothing worked
    if best_result is None:
        # Simple grid fallback
        w0 = 1.0
        h0 = 1.0
        cols = 5
        rows = 5
        cell_w = w0 / cols
        cell_h = h0 / rows
        r0 = min(cell_w, cell_h) / 2.0 * 0.8
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, r0)
        for i in range(N):
            row, col = divmod(i, cols)
            x[i] = (col + 0.5) * cell_w
            y[i] = (row + 0.5) * cell_h
        best_result = (x, y, r, w0)

    x_final, y_final, r_final, w_final = best_result

    # Safety shrink to ensure validity due to floating point errors
    for _ in range(10):
        r_trial = r_final * (1 - 1e-8)
        if is_valid(x_final, y_final, r_trial, w_final):
            r_final = r_trial
        else:
            break

    return np.column_stack((x_final, y_final, r_final))
