"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_value = -np.inf
    best_result = None
    
    def evaluate(circles, w, h):
        """Return sum of radii if valid, else -inf."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                return -np.inf
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if d < ri + rj - 1e-6:
                    return -np.inf
        return float(np.sum(circles[:, 2]))
    
    def check_and_update(x, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        circles = np.column_stack([xs, ys, rs])
        val = evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = x.copy()
    
    def softplus(x, beta=100.0):
        """Numerically stable softplus."""
        x_beta = beta * x
        result = np.where(x_beta > 0,
                         x + np.log1p(np.exp(-x_beta)) / beta,
                         np.log1p(np.exp(x_beta)) / beta)
        return result
    
    def softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)."""
        return 1.0 / (1.0 + np.exp(-beta * x))
    
    def objective_and_grad(x, w, h, beta=100.0):
        """
        Force-directed penalty objective with analytical gradients (vectorized).
        x: [x0, y0, r0, x1, y1, r1, ...]
        """
        c = x.reshape(-1, 3)
        n = len(c)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]
        
        obj = -np.sum(rs)
        grad = np.zeros_like(x)
        
        # Boundary penalties (vectorized)
        # x - r < 0
        v1 = rs - xs
        obj += np.sum(softplus(v1, beta))
        g1 = softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1
        
        # x + r > w
        v2 = xs + rs - w
        obj += np.sum(softplus(v2, beta))
        g2 = softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2
        
        # y - r < 0
        v3 = rs - ys
        obj += np.sum(softplus(v3, beta))
        g3 = softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3
        
        # y + r > h
        v4 = ys + rs - h
        obj += np.sum(softplus(v4, beta))
        g4 = softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4
        
        # r < 0
        v5 = -rs
        obj += np.sum(softplus(v5, beta))
        g5 = softplus_grad(v5, beta)
        grad[2::3] -= g5
        
        # Pairwise overlap penalties (vectorized)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            
            v = rs[i] + rs[i+1:] - d
            obj += np.sum(softplus(v, beta))
            g = softplus_grad(v, beta)
            
            # Gradient contribution to circle i
            grad[3*i] += -np.sum(g * dx / d)
            grad[3*i+1] += -np.sum(g * dy / d)
            grad[3*i+2] += np.sum(g)
            
            # Gradient contribution to circles j > i
            idx = np.arange(i+1, n)
            grad[3*idx] += g * dx / d
            grad[3*idx+1] += g * dy / d
            grad[3*idx+2] += g
        
        # Gradient of -sum(r)
        grad[2::3] += -1.0
        
        return obj, grad
    
    def optimize_from(circles0, w, h, max_iter=400, beta=100.0, ftol=1e-12, gtol=1e-8):
        """L-BFGS-B optimization with softplus penalties."""
        x0 = circles0.flatten()
        
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
        
        try:
            res = minimize(
                lambda p: objective_and_grad(p, w, h, beta),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            return res.x
        except Exception:
            return x0
    
    def polish(circles0, w, h, max_iter=500, beta=200.0):
        """High-precision polish pass with tighter tolerance and higher beta."""
        return optimize_from(circles0, w, h, max_iter=max_iter, beta=beta, ftol=1e-14, gtol=1e-10)
    
    def hex_lattice_init(w, h):
        """
        Generate hexagonal lattice initialization for N=21 circles.
        Tries multiple row configurations and returns best initial config.
        """
        best_circles = None
        best_val = -np.inf
        
        # Row configurations summing to 21
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
        ]
        
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            
            n_rows = len(row_counts)
            max_cols = max(row_counts)
            
            # Estimate radius from available space
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
            val = evaluate(circles, w, h)
            
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
        
        if best_circles is None:
            a = int(np.ceil(np.sqrt(N * w / h)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            circles = []
            for i in range(N):
                col = i % a
                row = i // a
                r = min(w / a, h / b) * 0.45
                x = (col + 0.5) * w / a
                y = (row + 0.5) * h / b
                circles.append([x, y, r])
            best_circles = np.array(circles)
        
        return best_circles
    
    def grid_init(w, h):
        """Simple grid initialization."""
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.95
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        return np.array(circles)
    
    def random_init(w, h, rng, n_attempts=20):
        """Generate random initial configurations and return the best."""
        best_circles = None
        best_val = -np.inf
        
        for _ in range(n_attempts):
            circles = []
            for _ in range(N):
                r = rng.uniform(0.02, 0.15)
                x = rng.uniform(r, w - r)
                y = rng.uniform(r, h - r)
                circles.append([x, y, r])
            circles = np.array(circles)
            val = evaluate(circles, w, h)
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
        
        return best_circles if best_circles is not None else grid_init(w, h)
    
    # Try multiple aspect ratios
    aspects = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
               1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
    
    rng = np.random.default_rng(42)
    
    for aspect in aspects:
        if time.time() > deadline - 5:
            break
        
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Try multiple initializations
        inits = [
            hex_lattice_init(w, h),
            grid_init(w, h),
            random_init(w, h, rng, n_attempts=10)
        ]
        
        best_x_for_aspect = None
        best_val_for_aspect = -np.inf
        
        for circles0 in inits:
            if time.time() > deadline - 8:
                break
            
            # Check initial config
            val0 = evaluate(circles0, w, h)
            if val0 > best_val_for_aspect:
                best_val_for_aspect = val0
                best_x_for_aspect = circles0.flatten().copy()
            
            # Phase 1: Coarse optimization
            if time.time() < deadline - 10:
                x1 = optimize_from(circles0, w, h, max_iter=250, beta=50.0)
                val1 = evaluate(x1.reshape(-1, 3), w, h)
                if val1 > best_val_for_aspect:
                    best_val_for_aspect = val1
                    best_x_for_aspect = x1.copy()
            
            # Phase 2: Medium optimization
            if time.time() < deadline - 8:
                x2 = optimize_from(x1, w, h, max_iter=250, beta=100.0)
                val2 = evaluate(x2.reshape(-1, 3), w, h)
                if val2 > best_val_for_aspect:
                    best_val_for_aspect = val2
                    best_x_for_aspect = x2.copy()
        
        if best_x_for_aspect is None:
            continue
        
        # Phase 3: Polish the best for this aspect
        if time.time() < deadline - 5:
            x3 = polish(best_x_for_aspect, w, h, max_iter=400, beta=200.0)
            check_and_update(x3, w, h)
        
        # Phase 4: Perturbed restart
        if time.time() < deadline - 3:
            x4 = best_x_for_aspect.copy()
            for i in range(N):
                x4[3 * i] += rng.normal(0, 0.008)
                x4[3 * i + 1] += rng.normal(0, 0.008)
                x4[3 * i + 2] += rng.normal(0, 0.004)
            for i in range(N):
                r = max(x4[3 * i + 2], 1e-6)
                x4[3 * i] = np.clip(x4[3 * i], r, w - r)
                x4[3 * i + 1] = np.clip(x4[3 * i + 1], r, h - r)
                x4[3 * i + 2] = np.clip(x4[3 * i + 2], 1e-6, min(w, h) / 2)
            
            x4 = polish(x4, w, h, max_iter=300, beta=200.0)
            check_and_update(x4, w, h)
        
        # Phase 5: Ultra-high-precision final polish
        if time.time() < deadline - 2:
            x5 = optimize_from(best_x_for_aspect, w, h, max_iter=400, beta=500.0, ftol=1e-15, gtol=1e-12)
            check_and_update(x5, w, h)
    
    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.099
        best_result = np.zeros(3 * N)
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[3 * idx] = (i + 0.5) * w / a
                best_result[3 * idx + 1] = (j + 0.5) * h / b
                best_result[3 * idx + 2] = r
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3 * i:3 * i + 3]
    
    return result
# EVOLVE-BLOCK-END
