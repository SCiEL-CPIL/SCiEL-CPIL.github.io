"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    best_value = -np.inf
    best_result = None

    def _strictly_valid(circles, w, h, tol=1e-8):
        """Full-pair strict validity check. circles: (N,3)."""
        n = circles.shape[0]
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        for i in range(n):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[i + 1:] - tol
            if np.any(d2 < need * need):
                return False
        return True

    def _evaluate(circles, w, h, tol=1e-7):
        """Return sum of radii if valid, else -inf."""
        if not _strictly_valid(circles, w, h, tol):
            return -np.inf
        return float(np.sum(circles[:, 2]))

    def check_and_update(circles, w, h):
        nonlocal best_value, best_result
        if time_left() < 1.0:
            return
        val = _evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = circles.copy()

    def _softplus(x, beta=100.0):
        xb = beta * x
        return np.where(xb > 0,
                        x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def _softplus_grad(x, beta=100.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
        """Softplus-penalized objective with analytical gradients (vectorized)."""
        c = x.reshape(-1, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        # Equalization pressure
        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[2::3] += -2.0 * eq_weight * r_dev

        # Boundary penalties
        v1 = rs - xs
        obj += np.sum(_softplus(v1, beta)); g1 = _softplus_grad(v1, beta)
        grad[0::3] -= g1; grad[2::3] += g1
        v2 = xs + rs - w
        obj += np.sum(_softplus(v2, beta)); g2 = _softplus_grad(v2, beta)
        grad[0::3] += g2; grad[2::3] += g2
        v3 = rs - ys
        obj += np.sum(_softplus(v3, beta)); g3 = _softplus_grad(v3, beta)
        grad[1::3] -= g3; grad[2::3] += g3
        v4 = ys + rs - h
        obj += np.sum(_softplus(v4, beta)); g4 = _softplus_grad(v4, beta)
        grad[1::3] += g4; grad[2::3] += g4
        v5 = -rs
        obj += np.sum(_softplus(v5, beta)); g5 = _softplus_grad(v5, beta)
        grad[2::3] -= g5

        # Pairwise overlap penalties
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(_softplus(v, beta))
            g = _softplus_grad(v, beta)
            grad[3 * i]     += -np.sum(g * dx / d)
            grad[3 * i + 1] += -np.sum(g * dy / d)
            grad[3 * i + 2] += np.sum(g)
            idx = np.arange(i + 1, N)
            grad[3 * idx]     += g * dx / d
            grad[3 * idx + 1] += g * dy / d
            grad[3 * idx + 2] += g

        grad[2::3] += -1.0
        return obj, grad

    def _hex_init(w, h, seed):
        rng = random.Random(seed)
        # Try various row configurations
        row_configs = [
            [3, 4, 4, 4, 3, 3],
            [4, 4, 4, 4, 3, 2],
            [2, 4, 4, 4, 4, 3],
            [3, 4, 4, 4, 4, 2],
            [4, 3, 4, 4, 4, 2],
            [2, 3, 4, 4, 4, 4],
            [3, 3, 4, 4, 4, 3],
            [4, 4, 3, 4, 4, 2],
            [2, 4, 3, 4, 4, 4],
            [3, 4, 3, 4, 4, 3],
            [5, 4, 4, 4, 4],
            [4, 5, 4, 4, 4],
            [3, 5, 4, 4, 5],
            [4, 4, 5, 4, 4],
            [2, 5, 4, 5, 5],
            [5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5],
            [3, 4, 5, 4, 5],
            [5, 4, 5, 4, 3],
            [7, 7, 7],
            [7, 7, 4, 3],
            [4, 4, 4, 4, 5],
            [5, 4, 4, 4, 4],
            [3, 4, 4, 4, 3, 3],
            [2, 4, 4, 4, 4, 3],
        ]
        best = None
        best_val = -np.inf
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            n_rows = len(row_counts)
            max_cols = max(row_counts)
            est_r_h = w / (max_cols + 0.5)
            est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
            r_est = min(est_r_h, est_r_v) * 0.95
            if r_est < 0.01:
                continue
            circles = []
            for row_idx, n_cols in enumerate(row_counts):
                y = (row_idx + 0.5) * r_est * np.sqrt(3)
                x_offset = r_est if row_idx % 2 == 1 else 0.0
                row_width = n_cols * 2 * r_est
                center_x = w / 2
                for col_idx in range(n_cols):
                    x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                    circles.append([x, y, r_est])
            circles = np.array(circles)
            valid = True
            for i in range(N):
                xi, yi, ri = circles[i]
                if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
                        min_dist = ri + rj
                        if dist_sq < (min_dist - 1e-7) ** 2:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                val = float(np.sum(circles[:, 2]))
                if val > best_val:
                    best_val = val
                    best = circles.copy()
        if best is None:
            # Fallback: simple grid
            a = 7
            b = 3
            r = min(w / (a + 1), h / (b + 1)) * 0.9
            circles = np.zeros((N, 3))
            for i in range(N):
                col = i % a
                row = i // a
                circles[i, 0] = (col + 0.5) * w / a
                circles[i, 1] = (row + 0.5) * h / b
                circles[i, 2] = r
            best = circles
        return best

    def _grid_init(w, h, cols, rows, seed=0):
        rng = random.Random(seed)
        r = min(w / (cols + 1), h / (rows + 1)) * 0.9
        circles = np.zeros((N, 3))
        for i in range(N):
            col = i % cols
            row = i // cols
            circles[i, 0] = (col + 0.5) * w / cols
            circles[i, 1] = (row + 0.5) * h / rows
            circles[i, 2] = r
        return circles

    def _optimize_with_grad(circles, w, h, max_iter=2000, beta_start=50.0, beta_end=500.0):
        """Optimize using L-BFGS-B with softplus penalties."""
        x0 = circles.flatten()
        
        def obj_and_grad(params):
            c = params.reshape(-1, 3)
            obj, grad = _objective_and_grad(params, w, h, beta=beta_start, eq_weight=0.05)
            return obj, grad
        
        # First pass with lower beta
        try:
            res = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            x1 = res.x.reshape(-1, 3)
        except Exception:
            x1 = circles
        
        # Second pass with higher beta for tighter constraints
        if time_left() > 5.0:
            def obj_and_grad2(params):
                c = params.reshape(-1, 3)
                obj, grad = _objective_and_grad(params, w, h, beta=beta_end, eq_weight=0.02)
                return obj, grad
            
            try:
                res2 = minimize(obj_and_grad2, x1.flatten(), jac=True, method='L-BFGS-B',
                               options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                x1 = res2.x.reshape(-1, 3)
            except Exception:
                pass
        
        # Clip to valid range
        for i in range(N):
            x1[i, 2] = max(x1[i, 2], 1e-6)
        
        return x1

    def _local_search(circles, w, h, steps=500, lr=0.01):
        """Simple gradient-based local search to polish."""
        x = circles.copy().flatten()
        for step in range(steps):
            if time_left() < 2.0:
                break
            obj, grad = _objective_and_grad(x, w, h, beta=200.0, eq_weight=0.03)
            # Adaptive learning rate with backtracking
            if step % 50 == 0:
                lr = 0.01 * (0.9 ** (step // 50))
            x_new = x - lr * grad
            # Project back to feasible region
            c = x_new.reshape(-1, 3)
            for i in range(N):
                c[i, 2] = max(c[i, 2], 1e-6)
                c[i, 0] = min(max(c[i, 0], c[i, 2]), w - c[i, 2])
                c[i, 1] = min(max(c[i, 1], c[i, 2]), h - c[i, 2])
            x_new = c.flatten()
            # Check if objective improved; if not, reduce lr
            obj_new, _ = _objective_and_grad(x_new, w, h, beta=200.0, eq_weight=0.03)
            if obj_new > obj:
                x = x_new
            else:
                lr *= 0.5
        return x.reshape(-1, 3)

    def _high_precision_polish(circles, w, h):
        """Final high-precision polish with very high beta and tight tolerances."""
        if time_left() < 5.0:
            return circles
        
        x0 = circles.flatten()
        
        def obj_grad_hp(params):
            obj, grad = _objective_and_grad(params, w, h, beta=2000.0, eq_weight=0.01)
            return obj, grad
        
        try:
            res = minimize(obj_grad_hp, x0, jac=True, method='L-BFGS-B',
                          options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
            x1 = res.x.reshape(-1, 3)
            # Clip to valid range
            for i in range(N):
                x1[i, 2] = max(x1[i, 2], 1e-6)
                x1[i, 0] = min(max(x1[i, 0], x1[i, 2]), w - x1[i, 2])
                x1[i, 1] = min(max(x1[i, 1], x1[i, 2]), h - x1[i, 2])
            return x1
        except Exception:
            return circles

    def _optimize_aspect(circles, w, h):
        """Jointly optimize w (aspect ratio) and circle positions/radii."""
        if time_left() < 10.0:
            return circles, w, h
        
        # Parameterize: w is free, h = 2 - w, w in (0.2, 1.8)
        x0 = np.concatenate([circles.flatten(), [w]])
        
        def obj_aspect(params):
            w_val = params[-1]
            h_val = 2.0 - w_val
            if w_val <= 0.1 or w_val >= 1.9:
                return 1e6, np.zeros_like(params)
            c_params = params[:-1]
            obj, grad = _objective_and_grad(c_params, w_val, h_val, beta=500.0, eq_weight=0.02)
            # Gradient w.r.t. w: need to account for h = 2 - w
            # d(obj)/dw = sum of boundary penalties derivatives
            c = c_params.reshape(-1, 3)
            xs = c[:, 0]
            ys = c[:, 1]
            rs = c[:, 2]
            beta = 500.0
            dw = 0.0
            # v2 = xs + rs - w: d(v2)/dw = -1
            v2 = xs + rs - w_val
            g2 = _softplus_grad(v2, beta)
            dw += np.sum(g2)
            # v4 = ys + rs - h = ys + rs - (2-w): d(v4)/dw = +1
            v4 = ys + rs - h_val
            g4 = _softplus_grad(v4, beta)
            dw -= np.sum(g4)
            grad_full = np.concatenate([grad, [dw]])
            return obj, grad_full
        
        try:
            res = minimize(obj_aspect, x0, jac=True, method='L-BFGS-B',
                          bounds=[(0.2, 1.8)] * (N * 3) + [(0.3, 1.7)],
                          options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            x1 = res.x
            w_new = x1[-1]
            h_new = 2.0 - w_new
            circles_new = x1[:-1].reshape(-1, 3)
            # Clip
            for i in range(N):
                circles_new[i, 2] = max(circles_new[i, 2], 1e-6)
                circles_new[i, 0] = min(max(circles_new[i, 0], circles_new[i, 2]), w_new - circles_new[i, 2])
                circles_new[i, 1] = min(max(circles_new[i, 1], circles_new[i, 2]), h_new - circles_new[i, 2])
            return circles_new, w_new, h_new
        except Exception:
            return circles, w, h

    # Main optimization loop over aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0]
    
    for aspect in aspect_ratios:
        if time_left() < 10.0:
            break
        
        w = 2.0 * aspect / (1 + aspect)
        h = 2.0 - w
        
        # Generate initial configurations
        inits = []
        
        # Hexagonal init
        hex_circ = _hex_init(w, h, seed=42)
        inits.append(hex_circ)
        
        # Grid inits
        inits.append(_grid_init(w, h, 7, 3, seed=0))
        inits.append(_grid_init(w, h, 5, 5, seed=1))
        inits.append(_grid_init(w, h, 6, 4, seed=2))
        
        # Random perturbation of hex
        hex_pert = hex_circ.copy()
        hex_pert[:, 0] += np.random.randn(N) * 0.01
        hex_pert[:, 1] += np.random.randn(N) * 0.01
        inits.append(hex_pert)
        
        for init_circ in inits:
            if time_left() < 15.0:
                break
            
            # Check initial validity
            if _strictly_valid(init_circ, w, h, tol=1e-6):
                check_and_update(init_circ, w, h)
            
            # Optimize with gradient method
            opt_circ = _optimize_with_grad(init_circ, w, h, max_iter=1000)
            check_and_update(opt_circ, w, h)
            
            # Local search polish
            if time_left() > 10.0:
                polished = _local_search(opt_circ, w, h, steps=200, lr=0.005)
                check_and_update(polished, w, h)
    
    # If no valid result found, use a simple grid
    if best_result is None:
        w, h = 1.0, 1.0
        best_result = _grid_init(w, h, 7, 3, seed=0)
        # Ensure validity
        for i in range(N):
            best_result[i, 2] = min(best_result[i, 2], 0.1)
            best_result[i, 0] = min(max(best_result[i, 0], best_result[i, 2]), w - best_result[i, 2])
            best_result[i, 1] = min(max(best_result[i, 1], best_result[i, 2]), h - best_result[i, 2])
    
    # High-precision polish on the best result
    if best_result is not None and time_left() > 15.0:
        # Find the w, h that gave the best result by re-deriving from aspect ratios
        # We need to track which w,h gave the best result
        # For now, try polishing with the aspect ratio that was used
        # Re-optimize with best known w,h
        for aspect in [0.8, 0.9, 1.0, 1.1, 1.2]:
            if time_left() < 10.0:
                break
            w_test = 2.0 * aspect / (1 + aspect)
            h_test = 2.0 - w_test
            polished = _high_precision_polish(best_result, w_test, h_test)
            check_and_update(polished, w_test, h_test)
    
    # Joint aspect ratio optimization on the best result
    if best_result is not None and time_left() > 10.0:
        for aspect in [0.8, 0.9, 1.0, 1.1, 1.2]:
            if time_left() < 8.0:
                break
            w_test = 2.0 * aspect / (1 + aspect)
            h_test = 2.0 - w_test
            circ_new, w_new, h_new = _optimize_aspect(best_result, w_test, h_test)
            check_and_update(circ_new, w_new, h_new)
    
    # Final format
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[i]
    
    return result
# EVOLVE-BLOCK-END
