"""
Circle packing in unit square using simulated annealing with repulsion forces.
Centers are optimized; radii are derived from local geometry.
SA provides global exploration to escape local minima.
Smooth soft-min surrogate with analytic gradient for fast L-BFGS-B polish.
Vectorized gradient computation for faster L-BFGS-B convergence.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 118.0  # Safety margin from 150s limit (evaluator kills at 150s)


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
        Returns (radii, soft_min_values) where soft_min_values are the unclamped values.
        """
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        half_dist = 0.5 * dist  # (N, N), inf on diagonal
        # Constraints for circle i: half_dist[i, j] for j!=i, and r_bound[i]
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        # Soft-min per row
        neg_beta = -beta * full_constraints
        max_neg_row = np.max(neg_beta, axis=1, keepdims=True)
        log_sum_row = np.log(np.sum(np.exp(neg_beta - max_neg_row), axis=1, keepdims=True))
        soft_min_all = -(max_neg_row + log_sum_row) / beta
        soft_min_all = soft_min_all[:, 0]

        r = np.maximum(soft_min_all, 1e-6)
        return r, soft_min_all

    def smooth_objective_and_grad(params, beta=50.0):
        """Compute smooth objective value and analytic gradient (vectorized).
        params = [x_0..x_{N-1}, y_0..y_{N-1}]
        Objective: -sum(r_i) where r_i = softmin over constraints for circle i.
        Constraints for circle i: 0.5*dist(i,j) for j!=i, and r_bound[i]=min(x_i,1-x_i,y_i,1-y_i)
        """
        x = params[:N]
        y = params[N:2*N]

        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, 1e-10)

        half_dist = 0.5 * dist
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        # Soft-min: r_i = -(1/beta) * log(sum_j exp(-beta * c_{ij}))
        neg_beta_c = -beta * full_constraints
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        log_sum = np.log(np.sum(exp_terms, axis=1, keepdims=True))
        soft_min = -(max_neg + log_sum) / beta
        soft_min = soft_min[:, 0]

        objective_val = -np.sum(soft_min)

        # Weights w_{ij} = exp(-beta*c_{ij}) / sum_k exp(-beta*c_{ik})
        w = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)

        # Unit vectors (zero on diagonal)
        inv_dist = 1.0 / dist
        np.fill_diagonal(inv_dist, 0.0)
        ux = dx * inv_dist  # (x_i - x_j) / dist(i,j)
        uy = dy * inv_dist  # (y_i - y_j) / dist(i,j)

        # Vectorized gradient from neighbor constraints:
        # grad_x[i] = 0.5 * (sum_j w[i,j]*ux[i,j] + sum_j w[j,i]*ux[i,j])
        # Note: ux[j,i] = -ux[i,j], so sum_j w[j,i]*ux[j,i] = -sum_j w[j,i]*ux[i,j]
        # Thus: grad_x[i] = 0.5 * sum_j (w[i,j] - w[j,i]) * ux[i,j]
        # But this isn't quite right. Let me derive properly.
        #
        # grad_x[i] gets contribution from:
        #   - As circle i in pair (i,j): w[i,j] * 0.5 * ux[i,j] for all j != i
        #   - As circle j in pair (k,i): w[k,i] * (-0.5) * ux[k,i] for all k != i
        #     = w[k,i] * (-0.5) * (-ux[i,k]) = w[k,i] * 0.5 * ux[i,k] for all k != i
        #
        # So: grad_x[i] = 0.5 * sum_{j!=i} w[i,j]*ux[i,j] + 0.5 * sum_{k!=i} w[k,i]*ux[i,k]
        #   = 0.5 * sum_j (w[i,j] + w[j,i]) * ux[i,j]   (since ux[i,i]=0)
        #
        # Wait, let me recheck. For the second sum, let j=k:
        # sum_{k!=i} w[k,i]*ux[i,k] = sum_j w[j,i]*ux[i,j]
        #
        # So grad_x[i] = 0.5 * sum_j (w[i,j] + w[j,i]) * ux[i,j]
        #
        # In matrix form: grad_x = 0.5 * ((W + W.T) * Ux).sum(axis=1)
        # where (W*Ux)[i,j] = w[i,j]*ux[i,j] and (W.T*Ux) doesn't work directly.
        #
        # Let me use: S1[i,j] = w[i,j]*ux[i,j], then sum_j S1[i,j] = (W*Ux) row sum
        # S2[i,j] = w[j,i]*ux[i,j], then sum_j S2[i,j] = (W.T * Ux) ... no.
        # sum_j w[j,i]*ux[i,j] = (W.T @ diag(1) ... ) 
        # Actually: sum_j w[j,i]*ux[i,j] = (W.T * Ux)[i,i]? No.
        # 
        # Let A[i,j] = w[i,j]*ux[i,j]. Then sum_j A[i,j] = (W*Ux).sum(1)
        # Let B[i,j] = w[j,i]*ux[i,j]. Then B = W.T * Ux (element-wise? No, W.T[i,j]=w[j,i])
        # So B[i,j] = W.T[i,j] * Ux[i,j] = w[j,i]*ux[i,j]. Yes!
        # So sum_j B[i,j] = (W.T * Ux).sum(1)
        #
        # grad_x = 0.5 * ((W*Ux).sum(1) + (W.T*Ux).sum(1))
        #        = 0.5 * ((W + W.T) * Ux).sum(1)  -- NO, this is wrong because * is element-wise
        #
        # Correct: grad_x = 0.5 * ((W * Ux).sum(axis=1) + (W.T * Ux).sum(axis=1))

        W_ux = w * ux
        W_T_ux = w.T * ux
        W_uy = w * uy
        W_T_uy = w.T * uy

        grad_x = 0.5 * (W_ux.sum(axis=1) + W_T_ux.sum(axis=1))
        grad_y = 0.5 * (W_uy.sum(axis=1) + W_T_uy.sum(axis=1))

        # Boundary contributions from diagonal: w[i,i] * d(r_bound[i])/d(x_i or y_i)
        # Use soft-min of 4 boundary distances for smooth gradient
        beta_bound = 500.0
        # Shape: (N, 4)
        boundaries = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
        neg_bb = -beta_bound * boundaries
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        softmax_b = exp_bb / np.sum(exp_bb, axis=1, keepdims=True)  # (N, 4)

        wii = w[np.arange(N), np.arange(N)]  # diagonal weights, shape (N,)
        # d softmin / d x_i = softmax_b[i,0] - softmax_b[i,1]
        # d softmin / d y_i = softmax_b[i,2] - softmax_b[i,3]
        grad_x += wii * (softmax_b[:, 0] - softmax_b[:, 1])
        grad_y += wii * (softmax_b[:, 2] - softmax_b[:, 3])

        grad = np.concatenate([grad_x, grad_y])
        return objective_val, grad

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
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=8000):
        """Simulated annealing to optimize center positions with vectorized forces."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        # Cooling schedule
        T_start = 0.04
        T_end = 0.0003
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        # Repulsion force parameters
        repulse_k = 2.0
        boundary_k = 5.0

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break

            T = T_start * (cool_rate ** it)

            # Compute current energy
            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            # Compute forces (vectorized)
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Boundary repulsion (vectorized)
            fx += boundary_k * np.clip(0.08 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.92, 0, None)
            fy += boundary_k * np.clip(0.08 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.92, 0, None)

            # Pairwise repulsion (vectorized)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)

            # Force magnitude: repulse_k / (d^2 + eps)
            f_mag = repulse_k / (d**2 + 0.008)
            np.fill_diagonal(f_mag, 0)

            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            # Propose move: combine force direction with random perturbation
            scale = 0.015 * T / T_start + 0.004
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            x_new = np.clip(x_new, 0.03, 0.97)
            y_new = np.clip(y_new, 0.03, 0.97)

            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

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

    def gradient_polish(x0, y0, use_smooth=True, beta_start=20.0, beta_end=500.0, num_steps=5):
        """Final gradient-based refinement using L-BFGS-B with analytic gradient.
        Uses smooth soft-min objective for better gradient signal.
        Iteratively increases sharpness (beta) to converge to hard constraints.
        """
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        
        # Geometric progression of betas
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
            for scale in [0.999, 0.99, 0.95, 0.90]:
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
                    for scale in [0.999, 0.99, 0.98, 0.95]:
                        r_test = r_hard * scale
                        if check_validity(params[:N], params[N:], r_test):
                            best_result = (params[:N].copy(), params[N:].copy(), r_test, np.sum(r_test))
                            break

        init_val = get_val_from_params(current_params)
        if init_val > best_val:
            store_best(current_params, init_val)

        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            def objective_smooth(params):
                val, _ = smooth_objective_and_grad(params, beta=beta)
                return val

            def grad_smooth(params):
                _, g = smooth_objective_and_grad(params, beta=beta)
                return g

            try:
                res = minimize(objective_smooth, current_params, method='L-BFGS-B',
                               jac=grad_smooth,
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
                current_params = res.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    store_best(current_params, val)
            except Exception:
                pass

        # Final hard polish with more iterations
        if time.time() - start_time < DEADLINE - 2:
            def objective_hard(params):
                x = params[:N]
                y = params[N:2*N]
                r = compute_radii(x, y)
                return -np.sum(r)

            try:
                res2 = minimize(objective_hard, current_params, method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
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
    modes = ['grid', 'hex', 'random']
    num_starts = 40
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 20:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # Run SA with more iterations (vectorized forces are fast)
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=8000)

        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa
            best_y = y_sa
            best_r = r_sa

        # Gradient polish on SA result
        if time.time() - start_time < DEADLINE - 12:
            polished = gradient_polish(x_sa, y_sa, use_smooth=True)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp

    # Final high-precision polish: run gradient polish on the best solution found
    if time.time() - start_time < DEADLINE - 5:
        final_polish = gradient_polish(best_x, best_y, use_smooth=True, beta_start=50.0, beta_end=3000.0, num_steps=10)
        if final_polish is not None:
            xp, yp, rp, vp = final_polish
            if vp > best_val:
                best_val = vp
                best_x = xp
                best_y = yp
                best_r = rp

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
