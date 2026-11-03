"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    best_value = -np.inf
    best_result = None

    def check_and_update(x, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        # x is [x1,y1,r1, x2,y2,r2, ...]
        valid = True
        # Check bounds
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                valid = False
                break
        if not valid:
            return
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist = ri + rj
                if dist_sq < (min_dist - 1e-7)**2:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            val = np.sum(x[2::3])
            if val > best_value:
                best_value = val
                best_result = x.copy()

    def _softplus(x, beta=100.0):
        """Numerically stable softplus."""
        xb = beta * x
        return np.where(xb > 0,
                        x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def _softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)."""
        return 1.0 / (1.0 + np.exp(-beta * x))

    def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
        """
        Softplus-penalized objective with analytical gradients (vectorized).
        x: shape (3N,) = [x0, y0, r0, x1, y1, r1, ...]
        Minimizes: -sum(r) + boundary penalties + pairwise overlap penalties + equalization penalty.
        """
        c = x.reshape(-1, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        # Equalization pressure: penalize variance of radii to push toward symmetric packing
        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_var = np.mean((rs - r_mean) ** 2)
            obj += eq_weight * r_var
            # d/dr_i of eq_weight * mean((rs - r_mean)^2) = eq_weight * 2*(rs_i - r_mean)/N
            grad[2::3] += eq_weight * 2.0 * (rs - r_mean) / N

        # Boundary penalties
        v1 = rs - xs          # x - r < 0  ->  v1 > 0 means violation
        obj += np.sum(_softplus(v1, beta))
        g1 = _softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1

        v2 = xs + rs - w      # x + r > w
        obj += np.sum(_softplus(v2, beta))
        g2 = _softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2

        v3 = rs - ys          # y - r < 0
        obj += np.sum(_softplus(v3, beta))
        g3 = _softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3

        v4 = ys + rs - h      # y + r > h
        obj += np.sum(_softplus(v4, beta))
        g4 = _softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4

        v5 = -rs              # r < 0
        obj += np.sum(_softplus(v5, beta))
        g5 = _softplus_grad(v5, beta)
        grad[2::3] -= g5

        # Pairwise overlap penalties (vectorized per i)
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

        # Gradient of -sum(r)
        grad[2::3] += -1.0

        return obj, grad

    def _hex_init(w, h, seed):
        """Hexagonal lattice init over several row configurations; returns best valid (N,3)."""
        rng = random.Random(seed)
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
            # Check validity
            valid = True
            for i in range(N):
                xi, yi, ri = circles[i]
                if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist_sq = (xi-xj)**2 + (yi-yj)**2
                        min_dist = ri + rj
                        if dist_sq < (min_dist - 1e-7)**2:
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
            # Fallback grid
            a = int(np.ceil(np.sqrt(N * w / h)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            circles = []
            for i in range(N):
                col = i % a
                row = i // a
                r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
                x = (col + 0.5) * w / a
                y = (row + 0.5) * h / b
                circles.append([x, y, r])
            best = np.array(circles)
        return best

    def _grid_init(w, h, seed):
        rng = random.Random(seed)
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        return np.array(circles)

    def _random_init(w, h, seed):
        rng = np.random.RandomState(seed)
        r0 = min(w, h) / 2.0 * 0.25
        circles = np.zeros((N, 3))
        circles[:, 0] = rng.uniform(r0, w - r0, N)
        circles[:, 1] = rng.uniform(r0, h - r0, N)
        circles[:, 2] = r0 * rng.uniform(0.5, 1.0, N)
        return circles

    def _delaunay_neighbors(centers):
        """Return list of neighbor-index lists via Delaunay; fallback to all pairs."""
        try:
            tri = Delaunay(centers)
            nbrs = [set() for _ in range(N)]
            for i, j, k in tri.simplices:
                nbrs[i].add(j); nbrs[i].add(k)
                nbrs[j].add(i); nbrs[j].add(k)
                nbrs[k].add(i); nbrs[k].add(j)
            return [list(s) for s in nbrs]
        except Exception:
            return [list(range(N)) for _ in range(N)]

    def _fast_valid(circles, w, h, neighbors, tol=1e-9):
        """Fast validity using precomputed neighbor list (approximate, for polish steps)."""
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
            for j in neighbors[i]:
                if j <= i:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx * dx + dy * dy
                need = rs[i] + rs[j] - tol
                if d2 < need * need:
                    return False
        return True

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

    def _polish(circles, w, h, beta_start, beta_end, max_iter=50):
        """
        Multi-phase polish with increasing beta.
        circles: (N,3)
        Returns best valid circles found.
        """
        x = circles.flatten()
        best_x = x.copy()
        best_val = -np.inf
        
        # Check initial validity
        if _strictly_valid(circles, w, h):
            best_val = np.sum(circles[:, 2])
        
        # Phases of increasing beta
        betas = [beta_start, beta_start * 10, beta_start * 100, beta_start * 1000]
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            def obj_grad(params):
                return _objective_and_grad(params, w, h, beta=beta, eq_weight=0.05)
            
            # Bounds for L-BFGS-B
            bounds = []
            r_max = min(w, h) / 2.0
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((1e-4, r_max))
                
            try:
                res = minimize(obj_grad, x, method='L-BFGS-B', bounds=bounds, 
                               jac=True, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10})
                if time.time() > deadline:
                    break
                
                new_circles = res.x.reshape(-1, 3)
                
                # Check validity
                if _strictly_valid(new_circles, w, h):
                    val = np.sum(new_circles[:, 2])
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
                elif _fast_valid(new_circles, w, h, _delaunay_neighbors(new_circles[:, :2])):
                    # Accept if fast valid, but only update best if strictly valid
                    pass
                
                x = res.x
            except Exception:
                pass
        
        return best_x.reshape(-1, 3)

    # Loop over aspect ratios (widths)
    # w + h = 2, so h = 2 - w
    # Try a range of widths
    widths = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for w in widths:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Try multiple initializations
        inits = []
        
        # Hex init
        hex_circles = _hex_init(w, h, seed=42)
        inits.append(hex_circles)
        
        # Grid init
        grid_circles = _grid_init(w, h, seed=42)
        inits.append(grid_circles)
        
        # Random init
        rand_circles = _random_init(w, h, seed=42)
        inits.append(rand_circles)
        
        for init_circles in inits:
            if time.time() > deadline:
                break
            
            # Polish with multi-phase beta
            polished = _polish(init_circles, w, h, beta_start=10.0, beta_end=10000.0, max_iter=30)
            
            # Check and update
            check_and_update(polished.flatten(), w, h)
            
            # If we have a good result, try a final high-precision polish
            if best_value > 0 and time.time() < deadline - 10:
                final_polish = _polish(polished, w, h, beta_start=1000.0, beta_end=100000.0, max_iter=50)
                check_and_update(final_polish.flatten(), w, h)

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros(3*N)
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                best_result[3*idx] = (i + 0.5) * w / a
                best_result[3*idx+1] = (j + 0.5) * h / b
                best_result[3*idx+2] = r
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
    
    return result
# EVOLVE-BLOCK-END
