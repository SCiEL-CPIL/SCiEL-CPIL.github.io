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
        Used for soft-constraint optimization or local repair.
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

    def local_search(x, y, r, w, h, max_iters=100, step=0.01):
        """
        Simple local search to refine a packing.
        Tries to move circles to reduce overlaps and expand radii.
        Returns improved x, y, r.
        """
        best_local_score = np.sum(r)
        best_local_x, best_local_y, best_local_r = x.copy(), y.copy(), r.copy()
        current_pen = compute_penalty(x, y, r, w, h)
        
        # We want to minimize penalty and maximize sum(r).
        # Objective: -sum(r) + lambda * penalty
        # Let's just do a simple gradient-free hill climbing on sum(r) subject to penalty==0
        # Or better: if penalty > 0, reduce penalty. If penalty == 0, increase r.
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try increasing radii slightly if no overlap
            if current_pen < 1e-6:
                # Try expanding all radii by a small factor
                new_r = r * 1.005
                new_pen = compute_penalty(x, y, new_r, w, h)
                if new_pen < 1e-6:
                    r = new_r
                    current_pen = new_pen
                    improved = True
                else:
                    # Try expanding individual radii
                    for i in range(N):
                        trial_r = r.copy()
                        trial_r[i] *= 1.01
                        trial_pen = compute_penalty(x, y, trial_r, w, h)
                        if trial_pen < 1e-6:
                            r = trial_r
                            current_pen = trial_pen
                            improved = True
                            break
            
            # If still overlapping or didn't improve, try moving circles
            if not improved or current_pen > 1e-6:
                # Pick a random circle and nudge it
                i = np.random.randint(N)
                dx = np.random.uniform(-step, step)
                dy = np.random.uniform(-step, step)
                
                new_x = x.copy()
                new_y = y.copy()
                new_x[i] += dx
                new_y[i] += dy
                
                # Keep inside bounds roughly
                new_x[i] = np.clip(new_x[i], r[i], w - r[i])
                new_y[i] = np.clip(new_y[i], r[i], h - r[i])
                
                new_pen = compute_penalty(new_x, new_y, r, w, h)
                
                # Accept if penalty decreases or (penalty similar and we can expand r later)
                # Simple rule: accept if new_pen < current_pen
                if new_pen < current_pen:
                    x, y = new_x, new_y
                    current_pen = new_pen
                    improved = True
            
            if not improved:
                # Try a larger step or different strategy? For now, break if stuck too long
                # Or just try to shrink radii slightly to allow movement?
                # Let's try shrinking the most constrained circle
                # Find circle with most overlap?
                # Simple: shrink all radii by 5%
                new_r = r * 0.95
                new_pen = compute_penalty(x, y, new_r, w, h)
                if new_pen < current_pen:
                    r = new_r
                    current_pen = new_pen
                    # Then try moving again
                else:
                    break

        # Final check
        if validate(x, y, r, w, h):
            score = np.sum(r)
            if score > best_local_score:
                return x, y, r
        
        return best_local_x, best_local_y, best_local_r

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
    # We found that non-square can be better.
    # w+h=2. 
    # Square: w=1, h=1.
    # Let's try a range.
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.6, 1.3, 0.75, 1.1, 0.9, 1.8, 0.55]
    
    # Sort by how "square" they are? Or just iterate.
    # Let's prioritize near-square first as they are robust.
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
        # Allocate time based on remaining budget
        # Each restart takes ~1-3s for SLSQP + Local Search
        # Let's do at least 2 restarts per AR if time permits
        max_restarts = max(1, min(5, int(time_left / 4)))

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
                
                # Soft penalty for overlaps and boundaries
                # This allows SLSQP to handle infeasible starts better
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
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-8, 'disp': False}
                )
                
                x, y, r = unpack(result.x)
                r = np.maximum(r, 0)
                
                # Clip positions to be strictly inside based on current r
                # This helps if SLSQP drifted slightly out
                x = np.clip(x, r, w - r)
                y = np.clip(y, r, h - r)
                
                # Local search refinement
                # This is the "refinement machinery"
                x_ls, y_ls, r_ls = local_search(x, y, r, w, h, max_iters=50, step=0.02)
                
                # Check if local search improved
                score_ls = np.sum(r_ls)
                score_slsqp = np.sum(r)
                
                if score_ls > score_slsqp:
                    x, y, r = x_ls, y_ls, r_ls
                
                if validate(x, y, r, w, h):
                    score = np.sum(r)
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
