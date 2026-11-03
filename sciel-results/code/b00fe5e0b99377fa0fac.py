import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
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
        # Numerically stable sigmoid
        z_scaled = z / eps
        # For large positive z_scaled, sigmoid -> 1
        # For large negative z_scaled, sigmoid -> 0
        # Use np.where to avoid overflow in exp
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
        # First pass: Clip positions to keep circles inside the square
        for i in range(N):
            r = max(0.0, circles[i, 2])
            # Ensure radius is small enough to fit in the square
            r = min(r, 0.5)
            circles[i, 2] = r
            circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)
            
        # Second pass: Resolve overlaps by shrinking radii or moving. 
        # Simple approach: Iteratively shrink radii of overlapping circles.
        for _iter in range(10):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = circles[i, 2] + circles[j, 2]
                    if dist < min_dist:
                        # Shrink both radii slightly to resolve overlap
                        overlap = min_dist - dist
                        shrink = overlap / 2.0 + 1e-5
                        circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                        circles[j, 2] = max(0.0, circles[j, 2] - shrink)
                        # Re-clip positions after shrinking
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

        # Boundary violations (smooth)
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0

        penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
        penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
        penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
        penalty += np.sum(_softplus(uy, smooth_eps) ** 2)

        # Negative radius
        penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

        # Full pairwise overlaps (vectorized)
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
        params: flat array [x0, y0, r0, x1, y1, r1, ...]
        """
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        n = len(r)
        grad = np.zeros(3 * n)

        # 1. Objective term: -sum(r)
        # d(-sum(r))/dr_i = -1
        grad[2::3] -= 1.0

        # 2. Penalty terms
        # General form: penalty_weight * sum(softplus(v)^2)
        # d/dv (softplus(v)^2) = 2 * softplus(v) * softplus_grad(v)
        # d/dparam = 2 * penalty_weight * softplus(v) * softplus_grad(v) * dv/dparam

        # Boundary violations
        # lx = r - x  (violation if lx > 0)
        # ux = x + r - 1 (violation if ux > 0)
        # ly = r - y
        # uy = y + r - 1
        
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0
        
        slx = _softplus(lx, smooth_eps)
        glx = _softplus_grad(lx, smooth_eps)
        # d(lx)/dx = -1, d(lx)/dr = 1
        grad[0::3] += 2 * penalty_weight * slx * glx * (-1.0)
        grad[2::3] += 2 * penalty_weight * slx * glx * (1.0)
        
        sux = _softplus(ux, smooth_eps)
        gux = _softplus_grad(ux, smooth_eps)
        # d(ux)/dx = 1, d(ux)/dr = 1
        grad[0::3] += 2 * penalty_weight * sux * gux * (1.0)
        grad[2::3] += 2 * penalty_weight * sux * gux * (1.0)
        
        sly = _softplus(ly, smooth_eps)
        gly = _softplus_grad(ly, smooth_eps)
        # d(ly)/dy = -1, d(ly)/dr = 1
        grad[1::3] += 2 * penalty_weight * sly * gly * (-1.0)
        grad[2::3] += 2 * penalty_weight * sly * gly * (1.0)
        
        suy = _softplus(uy, smooth_eps)
        guy = _softplus_grad(uy, smooth_eps)
        # d(uy)/dy = 1, d(uy)/dr = 1
        grad[1::3] += 2 * penalty_weight * suy * guy * (1.0)
        grad[2::3] += 2 * penalty_weight * suy * guy * (1.0)
        
        # Negative radius
        # v = -r
        # d(-r)/dr = -1
        sneg = _softplus(-r, smooth_eps)
        gneg = _softplus_grad(-r, smooth_eps)
        grad[2::3] += 2 * penalty_weight * sneg * gneg * (-1.0)
        
        # Pairwise overlaps
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        
        s_ov = _softplus(overlap, smooth_eps)
        g_ov = _softplus_grad(overlap, smooth_eps)
        
        # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
        # d(overlap)/dy_i = -dy/dist, d(overlap)/dy_j = dy/dist
        # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
        
        coeff = 2 * penalty_weight * s_ov * g_ov
        
        # Accumulate into grad
        # For each pair (i, j):
        # grad_x[i] += coeff * (-dx/dist)
        # grad_x[j] += coeff * (dx/dist)
        # grad_y[i] += coeff * (-dy/dist)
        # grad_y[j] += coeff * (dy/dist)
        # grad_r[i] += coeff * 1
        # grad_r[j] += coeff * 1
        
        # Vectorized accumulation using np.add.at
        # grad_x is at indices 0, 3, 6... in grad
        # grad[0::3] corresponds to x
        
        # We need to map idx_i to flat index 3*idx_i + 0
        # idx_i to 3*idx_i + 0
        # idx_j to 3*idx_j + 0
        
        # Let's compute the delta gradients
        dx_dist = dx / dist
        dy_dist = dy / dist
        
        gx_i = -coeff * dx_dist
        gx_j = coeff * dx_dist
        gy_i = -coeff * dy_dist
        gy_j = coeff * dy_dist
        gr_i = coeff
        gr_j = coeff
        
        # Accumulate x gradients
        np.add.at(grad, 3 * idx_i + 0, gx_i)
        np.add.at(grad, 3 * idx_j + 0, gx_j)
        
        # Accumulate y gradients
        np.add.at(grad, 3 * idx_i + 1, gy_i)
        np.add.at(grad, 3 * idx_j + 1, gy_j)
        
        # Accumulate r gradients
        np.add.at(grad, 3 * idx_i + 2, gr_i)
        np.add.at(grad, 3 * idx_j + 2, gr_j)
        
        return grad

    def _objective_slsqp(params):
        # Maximize sum of radii -> Minimize negative sum
        r = params[2::3]
        return -np.sum(r)

    def _constraints_slsqp(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # 1. x_i - r_i >= 0
        c_x_min = x - r
        # 2. 1 - x_i - r_i >= 0
        c_x_max = 1.0 - x - r
        # 3. y_i - r_i >= 0
        c_y_min = y - r
        # 4. 1 - y_i - r_i >= 0
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
    
    # Bounds: r_i > 0, x,y in [0,1]
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    def _initial_hex():
        """Hexagonal packing initial config."""
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
        """Grid initial config."""
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
        """Random config with corner preference."""
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

    def _force_directed_relax(circles, iterations=100, dt=0.01, k_repel=10.0, k_attract=0.1, boundary_k=100.0):
        """
        Physics-based force-directed relaxation.
        Returns improved circles.
        """
        circles = circles.copy()
        x = circles[:, 0].copy()
        y = circles[:, 1].copy()
        r = circles[:, 2].copy()
        
        for _ in range(iterations):
            if _remaining() < 5.0:
                break
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsion (Coulomb-like)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(max(dist_sq, 1e-12))
                    
                    # Repulsive force if overlapping or close
                    min_dist = r[i] + r[j]
                    if dist < min_dist * 1.5:
                        force = k_repel / (dist_sq + 1e-6)
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                    
                    # Weak attractive force to keep them close (packing)
                    force_att = k_attract * (dist - min_dist)
                    if dist > 1e-6:
                        fx[i] -= force_att * dx / dist
                        fy[i] -= force_att * dy / dist
                        fx[j] += force_att * dx / dist
                        fy[j] += force_att * dy / dist
            
            # Boundary forces (push back if outside or too close to edge)
            for i in range(N):
                # Left
                if x[i] < r[i]:
                    fx[i] += boundary_k * (r[i] - x[i])
                # Right
                if x[i] > 1.0 - r[i]:
                    fx[i] -= boundary_k * (x[i] - (1.0 - r[i]))
                # Bottom
                if y[i] < r[i]:
                    fy[i] += boundary_k * (r[i] - y[i])
                # Top
                if y[i] > 1.0 - r[i]:
                    fy[i] -= boundary_k * (y[i] - (1.0 - r[i]))
            
            # Update positions
            x += dt * fx
            y += dt * fy
            
            # Hard clip to stay within bounds (with radius margin)
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
            
            circles[:, 0] = x
            circles[:, 1] = y
            
        return circles

    def _grow_radii(circles, steps=20, step_size=0.005):
        """
        Attempt to grow radii uniformly or non-uniformly while maintaining validity.
        """
        circles = circles.copy()
        for _ in range(steps):
            if _remaining() < 5.0:
                break
            # Try growing all radii by step_size
            new_r = circles[:, 2] + step_size
            # Check if this is valid
            valid = True
            x, y = circles[:, 0], circles[:, 1]
            # Boundary check
            if np.any(x - new_r < 0) or np.any(x + new_r > 1) or np.any(y - new_r < 0) or np.any(y + new_r > 1):
                # Shrink growth to fit boundaries
                max_r = np.min(np.minimum(x, 1.0 - x))
                max_r = np.minimum(max_r, np.min(np.minimum(y, 1.0 - y)))
                new_r = np.minimum(new_r, max_r)
                if np.any(new_r <= circles[:, 2]):
                    break
            # Overlap check
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < new_r[i] + new_r[j] - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                circles[:, 2] = new_r
            else:
                # Try smaller growth or break
                break
        
        return circles

    # Multi-start strategy
    num_starts = 10 # Max number of random starts to try
    initializers = [_initial_hex, _initial_grid, _initial_corner_weighted]
    
    for k in range(num_starts):
        if _remaining() < 15.0:
            break
            
        # Cycle through initializers
        init_func = initializers[k % len(initializers)]
        if init_func == _initial_corner_weighted:
            circles0 = init_func(seed=k)
        else:
            circles0 = init_func()
        
        # Add random jitter to break symmetry
        circles0 = circles0.copy()
        circles0[:, 0] += np.random.uniform(-0.02, 0.02, N)
        circles0[:, 1] += np.random.uniform(-0.02, 0.02, N)
        circles0 = _fix_boundaries(circles0)
        
        # 1. Force-directed relaxation to get a good local minimum
        circles_relaxed = _force_directed_relax(circles0, iterations=150)
        circles_relaxed = _fix_boundaries(circles_relaxed)
        
        # 2. Grow radii
        circles_grown = _grow_radii(circles_relaxed, steps=30, step_size=0.005)
        
        # 3. Refine with SLSQP (starting from grown state)
        params0 = np.zeros(3 * N)
        params0[0::3] = circles_grown[:, 0]
        params0[1::3] = circles_grown[:, 1]
        params0[2::3] = circles_grown[:, 2]
        
        try:
            res = minimize(
                _objective_slsqp, 
                params0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons_slsqp,
                options={'maxiter': 150, 'ftol': 1e-8}
            )
            
            if res.success or res.status == 0:
                x_res = res.x[0::3]
                y_res = res.x[1::3]
                r_res = res.x[2::3]
                candidate = np.column_stack((x_res, y_res, r_res))
                candidate = _fix_boundaries(candidate)
                
                if _validate(candidate):
                    score = np.sum(candidate[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = candidate
                        
        except Exception as e:
            continue
            
        # 4. Also try penalty-based optimization from the relaxed state
        if _remaining() > 10.0:
            try:
                # Use analytic gradient for faster convergence
                res_pen = minimize(
                    lambda p: _objective_penalty(p, penalty_weight=100.0),
                    params0,
                    jac=lambda p: _objective_penalty_grad(p, penalty_weight=100.0),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'gtol': 1e-6}
                )
                x_pen = res_pen.x[0::3]
                y_pen = res_pen.x[1::3]
                r_pen = res_pen.x[2::3]
                candidate_pen = np.column_stack((x_pen, y_pen, r_pen))
                candidate_pen = _fix_boundaries(candidate_pen)
                
                if _validate(candidate_pen):
                    score_pen = np.sum(candidate_pen[:, 2])
                    if score_pen > best_score:
                        best_score = score_pen
                        best_result = candidate_pen
            except Exception:
                pass

    if best_result is not None:
        return best_result
    else:
        # Fallback to a known valid configuration
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
