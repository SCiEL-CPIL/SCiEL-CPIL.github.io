"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 110.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def validate(x, y, r, w, h):
        if np.any(r < -1e-9):
            return False
        if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
            return False
        # Check overlaps vectorized
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        # Upper triangle check
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        if np.any((dist[mask] < r_sum[mask] - 1e-9)):
            return False
        return True

    def compute_penalty(x, y, r, w, h):
        """
        Compute a penalty for constraint violations.
        """
        # Overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlaps = np.maximum(r_sum - dist, 0)
        overlap_pen = np.sum(overlaps[mask] ** 2)
        
        # Boundary penalty
        bound_pen = 0.0
        if np.any(x < r): bound_pen += np.sum((x - r)[x < r] ** 2)
        if np.any(x > w - r): bound_pen += np.sum((x - (w - r))[x > w - r] ** 2)
        if np.any(y < r): bound_pen += np.sum((y - r)[y < r] ** 2)
        if np.any(y > h - r): bound_pen += np.sum((y - (h - r))[y > h - r] ** 2)
        
        # Radius positivity penalty
        neg_r = np.maximum(-r, 0)
        r_pen = np.sum(neg_r ** 2)
        
        return overlap_pen + bound_pen + r_pen

    def force_directed_relax(x, y, r, w, h, max_iters=200, lr=0.05):
        """
        Force-directed relaxation to resolve overlaps and expand radii.
        This acts as a robust local solver that can escape some shallow minima.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        # We want to maximize sum(r) subject to constraints.
        # Strategy: 
        # 1. If overlapping, push apart.
        # 2. If not overlapping, expand radii slightly and push apart if needed.
        # We use a penalty gradient approach implicitly by moving towards "equilibrium".
        
        current_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_local_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute forces on positions
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Overlap forces
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            dist[dist < 1e-8] = 1e-8
            
            r_sum = r[:, None] + r[None, :]
            
            # Overlap amount (positive if overlapping)
            overlap = r_sum - dist
            mask = overlap > 0
            mask = mask & np.triu(np.ones((N, N), dtype=bool), k=1) # only upper triangle for force calc symmetry
            
            # Force is proportional to overlap. Direction is along the line connecting centers.
            # F_ij = overlap * (dx/dx, dy/dx)
            # We sum forces on i from all j > i, and on j from all i < j.
            # Let's just compute pairwise and add.
            for i in range(N):
                for j in range(i+1, N):
                    if mask[i, j]:
                        d = dist[i, j]
                        # Unit vector from j to i
                        ux = dx[i, j] / d
                        uy = dy[i, j] / d
                        
                        # Force magnitude
                        f_mag = overlap[i, j] * 10.0 # Stiffness
                        
                        # i is pushed away from j (positive direction)
                        fx[i] += f_mag * ux
                        fy[i] += f_mag * uy
                        
                        # j is pushed away from i (negative direction)
                        fx[j] -= f_mag * ux
                        fy[j] -= f_mag * uy
            
            # Boundary forces (push inward if outside)
            # x < r -> push right
            # x > w-r -> push left
            fx += np.where(x < r, (r - x) * 10.0, 0.0)
            fx += np.where(x > w - r, (w - r - x) * 10.0, 0.0)
            fy += np.where(y < r, (r - y) * 10.0, 0.0)
            fy += np.where(y > h - r, (h - r - y) * 10.0, 0.0)
            
            # Update positions
            x += lr * fx
            y += lr * fy
            
            # Clip to valid range (hard constraint)
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            
            # Try to expand radii
            # Check if there is "room". 
            # Simple heuristic: if max overlap is small, try to increase all r by a small amount.
            # A more robust way: binary search on a global scale factor for r?
            # Or just try r * 1.01 and see if penalty increases significantly.
            
            # Let's try expanding radii proportional to distance to nearest neighbor/wall.
            # This is complex. Let's just do a simple uniform expansion if penalty is low.
            penalty = compute_penalty(x, y, r, w, h)
            
            if penalty < 1e-4:
                # Try to expand
                new_r = r * 1.005
                new_pen = compute_penalty(x, y, new_r, w, h)
                if new_pen < 1e-3:
                    r = new_r
                    # If expansion caused overlaps, the next iteration's force step will fix them
                    # but we might need to reduce lr or do more position iterations.
                    # For now, just accept and let next loop handle it.
                
            # Track best
            if validate(x, y, r, w, h):
                score = np.sum(r)
                if score > best_local_score:
                    best_local_score = score
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            
            # Decay learning rate
            lr *= 0.995
            
        return best_x, best_y, best_r

    def hex_init(w, h, est_r, seed):
        rng = np.random.default_rng(seed)
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))
        
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1

        xs = np.zeros(N)
        ys = np.zeros(N)
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            
            x = (col + 0.5) * w / n_cols
            y = (row + 0.5) * h / n_rows
            
            if row % 2 == 1:
                x += 0.5 * w / n_cols
                if x > w - est_r:
                    x -= w / n_cols
                elif x < est_r:
                    x += w / n_cols

            xs[i] = x
            ys[i] = y

        jitter_scale = 0.2 * est_r
        xs += rng.normal(0, jitter_scale, N)
        ys += rng.normal(0, jitter_scale, N)
        
        xs = np.clip(xs, est_r * 0.5, w - est_r * 0.5)
        ys = np.clip(ys, est_r * 0.5, h - est_r * 0.5)
        
        return xs, ys

    # Aspect ratios to try
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.6, 1.3, 0.75, 1.1, 0.9, 1.8, 0.55]
    aspect_ratios.sort(key=lambda ar: abs(ar - 1.0))

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        est_r = np.sqrt(0.9 * w * h / (N * np.pi))
        init_r = est_r * 0.8

        # Determine restarts
        time_left = deadline - time.time()
        max_restarts = max(1, min(4, int(time_left / 5)))

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            seed = restart * 1000 + int(ar * 100)
            xs, ys = hex_init(w, h, init_r, seed)
            
            r0 = np.full(N, init_r)
            
            x0 = np.zeros(3 * N)
            x0[0::3] = xs
            x0[1::3] = ys
            x0[2::3] = r0

            def objective(vars_):
                x, y, r = unpack(vars_)
                sum_r = np.sum(r)
                
                # Soft penalty
                x_diff = x[:, None] - x[None, :]
                y_diff = y[:, None] - y[None, :]
                dist_matrix = np.sqrt(x_diff**2 + y_diff**2)
                r_sum_matrix = r[:, None] + r[None, :]
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                overlaps = np.where(r_sum_matrix - dist_matrix > 0, r_sum_matrix - dist_matrix, 0)
                overlap_pen = np.sum(overlaps[mask] ** 2)
                
                # Boundary penalties
                b_pen = 0.0
                b_pen += np.sum(np.maximum(r - x, 0) ** 2)
                b_pen += np.sum(np.maximum(x + r - w, 0) ** 2)
                b_pen += np.sum(np.maximum(r - y, 0) ** 2)
                b_pen += np.sum(np.maximum(y + r - h, 0) ** 2)
                
                # Radius positivity
                r_pen = np.sum(np.maximum(-r, 0) ** 2)
                
                return -sum_r + 100.0 * (overlap_pen + b_pen + r_pen)

            def make_constraints(w, h):
                def cons_inside(vars_):
                    x, y, r = unpack(vars_)
                    return np.concatenate([
                        x - r,
                        w - x - r,
                        y - r,
                        h - y - r
                    ])
                def cons_positive_r(vars_):
                    r = unpack(vars_)[2]
                    return r
                return [
                    {'type': 'ineq', 'fun': cons_inside},
                    {'type': 'ineq', 'fun': cons_positive_r}
                ]

            constraints = make_constraints(w, h)
            
            try:
                # 1. SLSQP Optimization
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
                )
                
                x, y, r = unpack(result.x)
                r = np.maximum(r, 0)
                
                # 2. Force-Directed Refinement
                # This is the key structural change: using physics-based relaxation
                # to fine-tune positions and expand radii, which is often more robust
                # than pure gradient-based local search in high-dimensional non-convex spaces.
                x_fd, y_fd, r_fd = force_directed_relax(x, y, r, w, h, max_iters=100, lr=0.02)
                
                # Choose the best of SLSQP result and Force-Directed result
                score_slsqp = np.sum(r) if validate(x, y, r, w, h) else 0.0
                score_fd = np.sum(r_fd) if validate(x_fd, y_fd, r_fd, w, h) else 0.0
                
                if score_fd > score_slsqp:
                    x, y, r = x_fd, y_fd, r_fd
                    score = score_fd
                else:
                    score = score_slsqp

                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x, y, r])

            except Exception:
                continue

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END
