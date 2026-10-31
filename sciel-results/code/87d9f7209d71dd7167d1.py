"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    # Aspect ratios to try: w + h = 2
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.6, 1.3, 0.75, 1.1, 0.9, 1.8, 0.55, 1.4, 0.7, 1.6]
    
    def validate(x, y, r, w, h):
        if np.any(r < -1e-9):
            return False
        if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
            return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def force_directed_relax(cx, cy, r, W, H, max_iter=500, dt=0.005):
        """Force-directed relaxation to fix overlaps and wall violations.
        This serves as both a pre-relaxation and a repair mechanism."""
        n = len(cx)
        cx = cx.copy()
        cy = cy.copy()
        
        for _ in range(max_iter):
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Wall repulsion (soft)
            for i in range(n):
                if cx[i] - r[i] < 0:
                    fx[i] += (cx[i] - r[i]) * 50
                if cy[i] - r[i] < 0:
                    fy[i] += (cy[i] - r[i]) * 50
                if cx[i] + r[i] > W:
                    fx[i] -= (cx[i] + r[i] - W) * 50
                if cy[i] + r[i] > H:
                    fy[i] -= (cy[i] + r[i] - H) * 50
            
            # Pairwise repulsion for overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    d_sq = dx * dx + dy * dy
                    min_d = r[i] + r[j]
                    if d_sq < min_d * min_d and d_sq > 1e-12:
                        d = np.sqrt(d_sq)
                        overlap = min_d - d
                        force_mag = overlap * 10
                        fx[i] += (dx / d) * force_mag
                        fy[i] += (dy / d) * force_mag
                        fx[j] -= (dx / d) * force_mag
                        fy[j] -= (dy / d) * force_mag
            
            cx += dt * fx
            cy += dt * fy
            
            # Clamp to rectangle
            cx = np.clip(cx, r, W - r)
            cy = np.clip(cy, r, H - r)
        
        return cx, cy

    def hex_init(w, h, est_r, seed):
        """Generate a hexagonal-like grid initialization with jitter."""
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

    def grid_init(w, h, est_r, seed, cols=None):
        """Simple grid initialization, optionally with specified columns."""
        rng = np.random.default_rng(seed)
        
        if cols is None:
            cols = max(1, int(np.round(np.sqrt(N * w / h))))
        rows = max(1, int(np.ceil(N / cols)))
        
        while cols * rows < N:
            if cols <= rows:
                cols += 1
            else:
                rows += 1

        xs = np.zeros(N)
        ys = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            
            x = (col + 0.5) * w / cols
            y = (row + 0.5) * h / rows
            
            xs[i] = x
            ys[i] = y

        jitter_scale = 0.15 * est_r
        xs += rng.normal(0, jitter_scale, N)
        ys += rng.normal(0, jitter_scale, N)
        
        xs = np.clip(xs, est_r * 0.5, w - est_r * 0.5)
        ys = np.clip(ys, est_r * 0.5, h - est_r * 0.5)
        
        return xs, ys

    def optimize_once(w, h, x0, maxiter=300):
        """Run SLSQP optimization for a single configuration."""
        
        def objective(vars_):
            x = vars_[0::3]
            y = vars_[1::3]
            r = vars_[2::3]
            
            sum_r = np.sum(r)
            
            x_diff = x[:, None] - x[None, :]
            y_diff = y[:, None] - y[None, :]
            dist_matrix = np.sqrt(x_diff**2 + y_diff**2)
            
            r_sum_matrix = r[:, None] + r[None, :]
            
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            overlaps = r_sum_matrix - dist_matrix
            positive_overlaps = np.where(overlaps > 0, overlaps, 0)
            positive_overlaps = positive_overlaps[mask]
            
            penalty = np.sum(positive_overlaps**2)
            
            # Wall penalties
            wall_pen = 0.0
            wall_pen += np.sum(np.maximum(r - x, 0)**2)
            wall_pen += np.sum(np.maximum(r - y, 0)**2)
            wall_pen += np.sum(np.maximum(x + r - w, 0)**2)
            wall_pen += np.sum(np.maximum(y + r - h, 0)**2)
            
            return -sum_r + 500.0 * penalty + 200.0 * wall_pen

        def make_constraints(w, h):
            def cons_inside(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                return np.concatenate([
                    x - r,
                    w - x - r,
                    y - r,
                    h - y - r
                ])
            def cons_positive_r(vars_):
                r = vars_[2::3]
                return r
            return [
                {'type': 'ineq', 'fun': cons_inside},
                {'type': 'ineq', 'fun': cons_positive_r}
            ]

        constraints = make_constraints(w, h)
        
        try:
            result = minimize(
                objective, x0, method='SLSQP',
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-9, 'disp': False}
            )
            
            x, y, r = result.x[0::3], result.x[1::3], result.x[2::3]
            r = np.maximum(r, 0)
            
            # Force-directed repair after SLSQP
            x, y = force_directed_relax(x, y, r, w, h, max_iter=200, dt=0.002)
            
            if validate(x, y, r, w, h):
                score = np.sum(r)
                return x, y, r, score
            else:
                # Try without repair
                x2, y2, r2 = result.x[0::3], result.x[1::3], result.x[2::3]
                r2 = np.maximum(r2, 0)
                if validate(x2, y2, r2, w, h):
                    score = np.sum(r2)
                    return x2, y2, r2, score
        except Exception:
            pass
        
        return None, None, None, 0.0

    # Main search loop
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        est_r = np.sqrt(0.9 * w * h / (N * np.pi))
        init_r = est_r * 0.8

        # Determine restarts based on remaining time
        time_left = deadline - time.time()
        max_restarts = max(1, int(time_left / 3))
        max_restarts = min(max_restarts, 15)

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            seed = restart * 1000 + int(ar * 100)
            
            # Alternate between hex and grid inits
            if restart % 3 == 0:
                xs, ys = hex_init(w, h, init_r, seed)
            elif restart % 3 == 1:
                cols = max(1, int(np.round(np.sqrt(N * w / h))))
                xs, ys = grid_init(w, h, init_r, seed, cols=cols)
            else:
                cols = max(1, int(np.round(np.sqrt(N * w / h))) + 1)
                xs, ys = grid_init(w, h, init_r, seed, cols=cols)
            
            r0 = np.full(N, init_r)
            
            # Pre-relax with force-directed
            xs, ys = force_directed_relax(xs, ys, r0, w, h, max_iter=300, dt=0.003)
            
            x0 = np.zeros(3 * N)
            x0[0::3] = xs
            x0[1::3] = ys
            x0[2::3] = r0

            x_opt, y_opt, r_opt, score = optimize_once(w, h, x0, maxiter=300)
            
            if x_opt is not None and score > best_score:
                best_score = score
                best_result = np.column_stack([x_opt, y_opt, r_opt])

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
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
