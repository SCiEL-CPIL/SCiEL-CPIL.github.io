"""
Circle packing in unit square using simulated annealing with repulsion forces.
Centers are optimized; radii are derived from local geometry.
SA provides global exploration to escape local minima.
Smooth soft-min surrogate used in gradient polish for better convergence.
Added: radius growth post-pass, structured initializations, tangent-plane local solver.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # Safety margin from 150s limit


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        return r

    def compute_radii_smooth(x, y, beta=50.0):
        """Smooth approximation of radii using log-sum-exp soft-min.
        beta controls sharpness: large beta -> hard min, small beta -> smooth average.
        """
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        # Soft-min of distances: -logsumexp(-beta * dist) / beta
        # For numerical stability, use the trick: min(a,b) ≈ -(1/beta)*log(exp(-beta*a)+exp(-beta*b))/2
        # Generalized: softmin(d_i) = -(1/beta) * log(sum(exp(-beta*d_i)))
        # But we need per-row soft-min, so:
        neg_beta_dist = -beta * dist
        # Subtract max for stability
        max_neg = np.max(neg_beta_dist, axis=1, keepdims=True)
        log_sum = np.log(np.sum(np.exp(neg_beta_dist - max_neg), axis=1, keepdims=True))
        soft_min_dist = -(max_neg + log_sum) / beta  # shape (N, 1)
        soft_min_dist = soft_min_dist[:, 0]  # shape (N,)

        # Also need soft-min with boundary: combine dist-to-boundary as another "neighbor"
        # Boundary distances: x, 1-x, y, 1-y -> min of these 4
        # We can treat boundary as 4 additional "circles" at infinity distance, but easier:
        # soft-min of [dist_to_neighbors..., dist_to_bound]
        # Let's just do: r = softmin(0.5*dist_to_neighbors, r_bound)
        # Actually, let's compute soft-min over all constraints for each circle:
        # For circle i: constraints are 0.5*d(i,j) for all j!=i, and r_bound[i]
        # Stack them:
        # dist matrix has inf on diagonal, so 0.5*dist has inf on diagonal
        half_dist = 0.5 * dist  # (N, N), inf on diagonal
        # Add boundary as an extra column
        # Constraints for circle i: half_dist[i, j] for j!=i, and r_bound[i]
        # Build full constraint matrix: (N, N) where row i has all constraints
        # Replace diagonal with r_bound[i]
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        # Soft-min per row
        neg_beta = -beta * full_constraints
        max_neg_row = np.max(neg_beta, axis=1, keepdims=True)
        log_sum_row = np.log(np.sum(np.exp(neg_beta - max_neg_row), axis=1, keepdims=True))
        soft_min_all = -(max_neg_row + log_sum_row) / beta
        soft_min_all = soft_min_all[:, 0]

        r = np.maximum(soft_min_all, 1e-6)
        return r

    def check_validity(x, y, r):
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def make_initial_guess(seed, mode='grid'):
        rng = np.random.RandomState(seed)
        if mode == 'grid':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.08, 0.08, N)
            y += rng.uniform(-0.08, 0.08, N)
        elif mode == 'hex':
            # Hexagonal lattice initialization
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        elif mode == 'spiral':
            # Archimedean spiral initialization
            x = np.zeros(N)
            y = np.zeros(N)
            for k in range(N):
                t = k / (N - 1)
                angle = t * 6 * np.pi  # 3 full turns
                radius = 0.45 * t
                x[k] = 0.5 + radius * np.cos(angle)
                y[k] = 0.5 + radius * np.sin(angle)
            x += rng.uniform(-0.03, 0.03, N)
            y += rng.uniform(-0.03, 0.03, N)
        elif mode == 'diagonal':
            # Diagonal band initialization
            x = np.zeros(N)
            y = np.zeros(N)
            for k in range(N):
                t = k / (N - 1)
                x[k] = 0.1 + 0.8 * t
                y[k] = 0.1 + 0.8 * (1.0 - t)
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=3000):
        """Simulated annealing to optimize center positions with vectorized forces.
        Uses a simpler acceptance criterion and fewer iterations to save time for polish."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        # Cooling schedule
        T_start = 0.03
        T_end = 0.0003
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        # Repulsion force parameters
        repulse_k = 3.0
        boundary_k = 8.0

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 20:
                break

            T = T_start * (cool_rate ** it)

            # Compute forces (vectorized)
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Boundary repulsion (vectorized)
            fx += boundary_k * np.clip(0.1 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.9, 0, None)
            fy += boundary_k * np.clip(0.1 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.9, 0, None)

            # Pairwise repulsion (vectorized)
            dx = x[:, None] - x[None, :]  # (N, N)
            dy = y[:, None] - y[None, :]  # (N, N)
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)  # Avoid division by zero on diagonal

            # Force magnitude: repulse_k / (d^2 + eps)
            f_mag = repulse_k / (d**2 + 0.005)
            np.fill_diagonal(f_mag, 0)

            # Force on i from j
            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            # Propose move: combine force direction with random perturbation
            scale = 0.015 * T / T_start + 0.003
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            # Clip to valid range
            x_new = np.clip(x_new, 0.03, 0.97)
            y_new = np.clip(y_new, 0.03, 0.97)

            # Evaluate new state
            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)
            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            # Acceptance criterion
            delta = new_val - curr_val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                x = x_new
                y = y_new
                r = r_new
                if new_val > best_val:
                    best_val = new_val
                    best_x = x.copy()
                    best_y = y.copy()

        return best_x, best_y, compute_radii(best_x, best_y), best_val

    def gradient_polish(x0, y0, use_smooth=True, beta_start=10.0, beta_end=2000.0, num_steps=8):
        """Final gradient-based refinement using L-BFGS-B.
        Uses smooth soft-min objective for better gradient signal.
        Iteratively increases sharpness (beta) to converge to hard constraints.
        More aggressive beta schedule and more iterations for better convergence.
        """
        params0 = np.concatenate([x0, y0])
        bounds = [(0.005, 0.995)] * (2 * N)
        
        # Geometric progression of betas - wider range, more steps
        betas = np.geomspace(beta_start, beta_end, num_steps)
        
        current_params = params0.copy()
        best_result = None
        best_val = -np.inf
        
        def get_val_from_params(params):
            x = params[:N]
            y = params[N:2*N]
            r = compute_radii(x, y)
            if check_validity(x, y, r):
                return np.sum(r)
            # Try scaling down to find a valid subset if invalid
            for scale in [0.9999, 0.999, 0.995, 0.99, 0.98, 0.95]:
                r_test = r * scale
                if check_validity(x, y, r_test):
                    return np.sum(r_test)
            return -np.inf

        def store_best(params, val):
            nonlocal best_result, best_val
            if val > best_val:
                best_val = val
                r_hard = compute_radii(params[:N], params[N:])
                if check_validity(params[:N], params[N:], r_hard):
                    best_result = (params[:N].copy(), params[N:].copy(), r_hard, val)
                else:
                    for scale in [0.9999, 0.999, 0.998, 0.995, 0.99]:
                        r_test = r_hard * scale
                        if check_validity(params[:N], params[N:], r_test):
                            best_result = (params[:N].copy(), params[N:].copy(), r_test, np.sum(r_test))
                            break

        # Initial check
        init_val = get_val_from_params(current_params)
        if init_val > best_val:
            store_best(current_params, init_val)

        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            def objective_smooth(params):
                x = params[:N]
                y = params[N:2*N]
                r = compute_radii_smooth(x, y, beta=beta)
                return -np.sum(r)

            try:
                res = minimize(objective_smooth, current_params, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                current_params = res.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    store_best(current_params, val)
            except Exception:
                pass

        # Final hard polish with multiple restarts from nearby points
        if time.time() - start_time < DEADLINE - 3:
            def objective_hard(params):
                x = params[:N]
                y = params[N:2*N]
                r = compute_radii(x, y)
                return -np.sum(r)

            # Try multiple small perturbations to escape local minima
            rng_polish = np.random.RandomState(42)
            for attempt in range(3):
                if time.time() - start_time > DEADLINE - 2:
                    break
                if attempt == 0:
                    start_params = current_params.copy()
                else:
                    start_params = current_params + rng_polish.normal(0, 0.001, 2*N)
                    start_params = np.clip(start_params, 0.005, 0.995)
                
                try:
                    res2 = minimize(objective_hard, start_params, method='L-BFGS-B',
                                    bounds=bounds,
                                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
                    current_params = res2.x
                    val = get_val_from_params(current_params)
                    if val > best_val:
                        store_best(current_params, val)
                except Exception:
                    pass

        if best_result is not None:
            return best_result
        return None

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start with SA
    modes = ['grid', 'hex', 'random', 'spiral', 'diagonal']
    num_starts = 20
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 25:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # Run SA with fewer iterations to save time for polish
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=2000)

        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa
            best_y = y_sa
            best_r = r_sa

        # Gradient polish on SA result - more aggressive
        if time.time() - start_time < DEADLINE - 15:
            polished = gradient_polish(x_sa, y_sa, use_smooth=True, 
                                       beta_start=10.0, beta_end=2000.0, num_steps=8)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp

    # Final high-precision polish: run gradient polish on the best solution found
    if time.time() - start_time < DEADLINE - 10:
        final_polish = gradient_polish(best_x, best_y, use_smooth=True, 
                                       beta_start=20.0, beta_end=5000.0, num_steps=10)
        if final_polish is not None:
            xp, yp, rp, vp = final_polish
            if vp > best_val:
                best_val = vp
                best_x = xp
                best_y = yp
                best_r = rp
        
        # Additional polish pass with even higher precision
        if time.time() - start_time < DEADLINE - 5:
            final_polish2 = gradient_polish(best_x, best_y, use_smooth=True, 
                                            beta_start=100.0, beta_end=10000.0, num_steps=6)
            if final_polish2 is not None:
                xp, yp, rp, vp = final_polish2
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp

    # Post-optimization radius growth: try to expand each radius slightly
    # while maintaining validity. This exploits slack in non-active constraints.
    if time.time() - start_time < DEADLINE - 2:
        for grow_iter in range(100):
            if time.time() - start_time > DEADLINE - 1:
                break
            improved = False
            for i in range(N):
                # Try to grow circle i's radius
                dx_all = best_x[i] - best_x
                dy_all = best_y[i] - best_y
                dist_all = np.sqrt(dx_all**2 + dy_all**2)
                dist_all[i] = np.inf
                # Max radius limited by neighbor distances
                max_r = np.minimum(
                    np.minimum(best_x[i], 1.0 - best_x[i]),
                    np.minimum(best_y[i], 1.0 - best_y[i])
                )
                # Check against all other circles
                for j in range(N):
                    if j == i:
                        continue
                    max_r = min(max_r, 0.5 * (dist_all[j] - best_r[j]))
                # Try growing more aggressively
                new_r_i = min(max_r, best_r[i] * 1.005)
                if new_r_i > best_r[i] + 1e-12:
                    old_r = best_r[i]
                    best_r[i] = new_r_i
                    if check_validity(best_x, best_y, best_r):
                        improved = True
                    else:
                        best_r[i] = old_r
            if not improved:
                break

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.999, 0.90, 30):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
