import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    best_result = None
    best_score = -np.inf
    
    def _elapsed():
        return time.time() - start_time

    def _remaining():
        return deadline - _elapsed()

    def _softplus(z, eps=1e-4):
        """Smooth approximation of max(z, 0) using log-sum-exp."""
        return np.logaddexp(0, z / eps) * eps

    def _softplus_grad(z, eps=1e-4):
        """Derivative of softplus: sigmoid(z/eps)."""
        z_scaled = z / eps
        sig = np.where(z_scaled > 0, 
                       1.0 / (1.0 + np.exp(-z_scaled)), 
                       np.exp(z_scaled) / (1.0 + np.exp(z_scaled)))
        return sig / eps

    def _validate(circles, tol=1e-7):
        """Check circles are inside unit square and non-overlapping."""
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
            return False
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def _fix_boundaries(circles):
        """Project circles inside unit square and ensure non-overlap by shrinking if necessary."""
        circles = circles.copy()
        for i in range(N):
            r = max(0.0, circles[i, 2])
            r = min(r, 0.5)
            circles[i, 2] = r
            circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)
            
        for _iter in range(15):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = circles[i, 2] + circles[j, 2]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        shrink = overlap / 2.0 + 1e-5
                        circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                        circles[j, 2] = max(0.0, circles[j, 2] - shrink)
                        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
                        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
                        circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], 1.0 - circles[j, 2])
                        circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], 1.0 - circles[j, 2])
                        changed = True
            if not changed:
                break
        return circles

    def _objective_penalty(params, penalty_weight, smooth_eps=1e-4):
        """
        Vectorized objective with soft penalty for constraint violations.
        params: flat array [x0, y0, r0, x1, y1, r1, ...]
        """
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]

        penalty = 0.0

        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0

        penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
        penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
        penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
        penalty += np.sum(_softplus(uy, smooth_eps) ** 2)

        penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

        return -np.sum(r) + penalty_weight * penalty

    def _objective_penalty_grad(params, penalty_weight, smooth_eps=1e-4):
        """
        Analytic gradient of the penalty objective.
        """
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        n = N
        
        grad = np.zeros(3 * n)
        
        # d(-sum(r))/dr = -1
        grad[2::3] -= 1.0
        
        # Boundary penalties
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0
        
        # d(softplus(lx)^2)/dx = -2*softplus(lx)*sigmoid(lx/eps)
        sp_lx = _softplus(lx, smooth_eps)
        sig_lx = _softplus_grad(lx, smooth_eps)
        grad[0::3] += 2 * penalty_weight * sp_lx * (-sig_lx)
        grad[2::3] += 2 * penalty_weight * sp_lx * sig_lx
        
        sp_ux = _softplus(ux, smooth_eps)
        sig_ux = _softplus_grad(ux, smooth_eps)
        grad[0::3] += 2 * penalty_weight * sp_ux * sig_ux
        grad[2::3] += 2 * penalty_weight * sp_ux * sig_ux
        
        sp_ly = _softplus(ly, smooth_eps)
        sig_ly = _softplus_grad(ly, smooth_eps)
        grad[1::3] += 2 * penalty_weight * sp_ly * (-sig_ly)
        grad[2::3] += 2 * penalty_weight * sp_ly * sig_ly
        
        sp_uy = _softplus(uy, smooth_eps)
        sig_uy = _softplus_grad(uy, smooth_eps)
        grad[1::3] += 2 * penalty_weight * sp_uy * sig_uy
        grad[2::3] += 2 * penalty_weight * sp_uy * sig_uy
        
        # Negative radius penalty
        sp_nr = _softplus(-r, smooth_eps)
        sig_nr = _softplus_grad(-r, smooth_eps)
        grad[2::3] += 2 * penalty_weight * sp_nr * (-sig_nr)
        
        # Pairwise overlap penalties
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        
        sp_ov = _softplus(overlap, smooth_eps)
        sig_ov = _softplus_grad(overlap, smooth_eps)
        
        # d(softplus(overlap)^2)/dx_i = 2*sp*sig * (-dx/dist)
        # d(softplus(overlap)^2)/dx_j = 2*sp*sig * (dx/dist)
        coeff = 2 * penalty_weight * sp_ov * sig_ov / dist
        
        np.add.at(grad[0::3], idx_i, -coeff * dx)
        np.add.at(grad[0::3], idx_j, coeff * dx)
        np.add.at(grad[1::3], idx_i, -coeff * dy)
        np.add.at(grad[1::3], idx_j, coeff * dy)
        np.add.at(grad[2::3], idx_i, 2 * penalty_weight * sp_ov * sig_ov)
        np.add.at(grad[2::3], idx_j, 2 * penalty_weight * sp_ov * sig_ov)
        
        return grad

    def _objective_slsqp(params):
        r = params[2::3]
        return -np.sum(r)

    def _constraints_slsqp(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        c_x_min = x - r
        c_x_max = 1.0 - x - r
        c_y_min = y - r
        c_y_max = 1.0 - y - r
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        r_sum = r[:, None] + r[None, :]
        overlap_constr = dist_sq - r_sum**2
        
        iu = np.triu_indices(N, k=1)
        pair_constr = overlap_constr[iu]
        
        all_constr = np.concatenate([c_x_min, c_x_max, c_y_min, c_y_max, pair_constr])
        return all_constr

    cons_slsqp = ({
        'type': 'ineq',
        'fun': _constraints_slsqp
    },)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    def _initial_hex():
        cols = 5
        rows = 6
        positions = []
        r = 0.09
        for row in range(rows):
            for col in range(cols):
                if row % 2 == 1:
                    x = (col + 0.5) / cols + 0.5 / cols
                else:
                    x = (col + 0.5) / cols
                y = (row + 0.5) / rows
                if x > 1 or y > 1:
                    continue
                positions.append([x, y, r])
                if len(positions) >= N:
                    break
            if len(positions) >= N:
                break
        while len(positions) < N:
            positions.append([0.5, 0.5, r * 0.5])
        return np.array(positions[:N])

    def _initial_grid():
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        r = 0.5 / side * 0.9
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y, r])
                if len(positions) >= N:
                    break
            if len(positions) >= N:
                break
        while len(positions) < N:
            positions.append([0.5, 0.5, r * 0.5])
        return np.array(positions[:N])

    def _initial_corner_weighted(seed=0):
        rng = np.random.default_rng(seed)
        positions = []
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
                   [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
        for cx, cy in corners:
            positions.append([cx, cy, 0.08])
        for _ in range(N - len(positions)):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            positions.append([x, y, 0.08])
        return np.array(positions[:N])

    def _force_directed_relax(circles, iterations=200, dt=0.005, k_repel=5.0, k_attract=0.05, boundary_k=50.0):
        """
        Vectorized physics-based force-directed relaxation.
        """
        circles = circles.copy()
        x = circles[:, 0].copy()
        y = circles[:, 1].copy()
        r = circles[:, 2].copy()
        
        # Precompute index arrays for vectorized pairwise operations
        idx_i, idx_j = np.triu_indices(N, k=1)
        
        for it in range(iterations):
            if _remaining() < 5.0:
                break
            
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            min_dist = r[idx_i] + r[idx_j]
            
            # Repulsive force when overlapping or close
            close_mask = dist < min_dist * 1.5
            force_repel = np.zeros(len(idx_i))
            force_repel[close_mask] = k_repel / (dist_sq[close_mask] + 1e-6)
            
            # Attractive force to maintain packing
            force_att = k_attract * (dist - min_dist)
            
            # Accumulate forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            fx[idx_i] += (force_repel - force_att) * dx / dist
            fy[idx_i] += (force_repel - force_att) * dy / dist
            fx[idx_j] -= (force_repel - force_att) * dx / dist
            fy[idx_j] -= (force_repel - force_att) * dy / dist
            
            # Boundary forces
            fx += np.where(x < r, boundary_k * (r - x), 0.0)
            fx += np.where(x > 1.0 - r, -boundary_k * (x - (1.0 - r)), 0.0)
            fy += np.where(y < r, boundary_k * (r - y), 0.0)
            fy += np.where(y > 1.0 - r, -boundary_k * (y - (1.0 - r)), 0.0)
            
            # Adaptive step size
            max_force = np.max(np.sqrt(fx**2 + fy**2))
            if max_force > 10:
                dt_eff = dt * 10 / max_force
            else:
                dt_eff = dt
            
            x += dt_eff * fx
            y += dt_eff * fy
            
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
            
            circles[:, 0] = x
            circles[:, 1] = y
            
        return circles

    def _grow_radii(circles, steps=50, step_size=0.003):
        """
        Vectorized attempt to grow radii while maintaining validity.
        """
        circles = circles.copy()
        x, y = circles[:, 0].copy(), circles[:, 1].copy()
        r = circles[:, 2].copy()
        
        idx_i, idx_j = np.triu_indices(N, k=1)
        
        for _ in range(steps):
            if _remaining() < 5.0:
                break
            
            new_r = r + step_size
            
            # Boundary check
            if np.any(x - new_r < 0) or np.any(x + new_r > 1) or np.any(y - new_r < 0) or np.any(y + new_r > 1):
                max_r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
                new_r = np.minimum(new_r, max_r_bound)
                if np.any(new_r <= r + 1e-10):
                    break
            
            # Overlap check (vectorized)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist = np.sqrt(dx * dx + dy * dy)
            if np.any(dist < new_r[idx_i] + new_r[idx_j] - 1e-7):
                break
            
            r = new_r
        
        circles[:, 2] = r
        return circles

    def _polish_slsqp(params0, maxiter=300, ftol=1e-10):
        """High-precision SLSQP polish."""
        try:
            res = minimize(
                _objective_slsqp, 
                params0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons_slsqp,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            if res.success or res.status == 0:
                x_res = res.x[0::3]
                y_res = res.x[1::3]
                r_res = res.x[2::3]
                candidate = np.column_stack((x_res, y_res, r_res))
                candidate = _fix_boundaries(candidate)
                if _validate(candidate):
                    return candidate
        except Exception:
            pass
        return None

    def _polish_lbfgsb(params0, penalty_weight=100.0, maxiter=500):
        """High-precision L-BFGS-B polish with analytic gradient."""
        try:
            res = minimize(
                lambda p: _objective_penalty(p, penalty_weight, 1e-4),
                params0,
                jac=lambda p: _objective_penalty_grad(p, penalty_weight, 1e-4),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            x_pen = res.x[0::3]
            y_pen = res.x[1::3]
            r_pen = res.x[2::3]
            candidate = np.column_stack((x_pen, y_pen, r_pen))
            candidate = _fix_boundaries(candidate)
            if _validate(candidate):
                return candidate
        except Exception:
            pass
        return None

    # Multi-start strategy with better time allocation
    num_starts = 8
    initializers = [_initial_hex, _initial_grid, _initial_corner_weighted]
    
    for k in range(num_starts):
        if _remaining() < 20.0:
            break
            
        init_func = initializers[k % len(initializers)]
        if init_func == _initial_corner_weighted:
            circles0 = init_func(seed=k * 7 + 3)
        else:
            circles0 = init_func()
        
        # Add random jitter
        circles0 = circles0.copy()
        circles0[:, 0] += np.random.uniform(-0.03, 0.03, N)
        circles0[:, 1] += np.random.uniform(-0.03, 0.03, N)
        circles0 = _fix_boundaries(circles0)
        
        # 1. Force-directed relaxation
        circles_relaxed = _force_directed_relax(circles0, iterations=200)
        circles_relaxed = _fix_boundaries(circles_relaxed)
        
        # 2. Grow radii
        circles_grown = _grow_radii(circles_relaxed, steps=50, step_size=0.003)
        
        params0 = np.zeros(3 * N)
        params0[0::3] = circles_grown[:, 0]
        params0[1::3] = circles_grown[:, 1]
        params0[2::3] = circles_grown[:, 2]
        
        # 3. L-BFGS-B polish with analytic gradient (faster than SLSQP)
        if _remaining() > 15.0:
            candidate_lbfgs = _polish_lbfgsb(params0, penalty_weight=100.0, maxiter=300)
            if candidate_lbfgs is not None:
                score = np.sum(candidate_lbfgs[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = candidate_lbfgs
        
        # 4. SLSQP polish for constraint accuracy
        if _remaining() > 10.0:
            # Start from best available
            start_params = params0.copy()
            if best_result is not None and _remaining() > 12.0:
                start_params[0::3] = best_result[:, 0]
                start_params[1::3] = best_result[:, 1]
                start_params[2::3] = best_result[:, 2]
            
            candidate_slsqp = _polish_slsqp(start_params, maxiter=200, ftol=1e-9)
            if candidate_slsqp is not None:
                score = np.sum(candidate_slsqp[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = candidate_slsqp

    # Final high-precision polish on the best result
    if best_result is not None and _remaining() > 10.0:
        final_params = np.zeros(3 * N)
        final_params[0::3] = best_result[:, 0]
        final_params[1::3] = best_result[:, 1]
        final_params[2::3] = best_result[:, 2]
        
        # L-BFGS-B with tighter tolerance
        final_lbfgs = _polish_lbfgsb(final_params, penalty_weight=200.0, maxiter=500)
        if final_lbfgs is not None:
            score = np.sum(final_lbfgs[:, 2])
            if score > best_score:
                best_score = score
                best_result = final_lbfgs
        
        # SLSQP with tightest tolerance
        if _remaining() > 5.0:
            final_slsqp = _polish_slsqp(final_params, maxiter=400, ftol=1e-11)
            if final_slsqp is not None:
                score = np.sum(final_slsqp[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = final_slsqp

    if best_result is not None:
        return best_result
    else:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-4
                    cells.append((x, y, r))
        return np.array(cells)

# EVOLVE-BLOCK-END
