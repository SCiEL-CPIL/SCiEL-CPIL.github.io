"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    best_value = -np.inf
    best_result = None
    best_w = 1.0
    best_h = 1.0

    def is_valid(x, w, h, tol=1e-6):
        """Check if configuration is strictly valid with tolerance."""
        n = len(x) // 3
        # Check bounds
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9:
                return False
            if xi - ri < -tol or xi + ri > w + tol or yi - ri < -tol or yi + ri > h + tol:
                return False
        
        # Check disjointness
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i+1, n):
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist = ri + rj
                # Use squared distance to avoid sqrt and be precise
                if dist_sq < (min_dist - tol)**2:
                    return False
        return True

    def check_and_update(x, w, h):
        nonlocal best_value, best_result, best_w, best_h
        if time.time() > deadline:
            return
        
        if is_valid(x, w, h, tol=1e-5):
            val = np.sum(x[2::3])
            if val > best_value:
                best_value = val
                best_result = x.copy()
                best_w = w
                best_h = h

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
            [7, 7, 7],
            [7, 7, 6, 1],
            [6, 7, 8],
            [8, 7, 6],
            [5, 6, 5, 5],
            [6, 5, 6, 4],
            [4, 6, 5, 6],
            [6, 5, 6, 4],
            [5, 5, 5, 6],
            [6, 5, 5, 5],
            [5, 5, 6, 5],
            [5, 6, 5, 5],
        ]
        
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            
            n_rows = len(row_counts)
            max_cols = max(row_counts)
            
            # Estimate radius from available space
            # Horizontal: max_cols circles, hex spacing ~ 2r
            # Vertical: n_rows rows, hex spacing ~ r*sqrt(3)
            est_r_h = w / (max_cols + 0.5)
            est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
            r_est = min(est_r_h, est_r_v) * 0.95  # slight margin
            
            if r_est < 0.01:
                continue
            
            circles = []
            for row_idx, n_cols in enumerate(row_counts):
                y = (row_idx + 0.5) * r_est * np.sqrt(3)
                # Offset alternating rows for hex packing
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
            # Fallback: simple grid
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

    def delaunay_lp_refine(circles, w, h, max_lp_iters=50):
        """
        Structural change: Use Delaunay triangulation to identify geometrically relevant pairs.
        Use LP to maximize sum of radii given fixed centers, using only Delaunay edges.
        Use LP duals (shadow prices) to guide center adjustments.
        """
        if time.time() > deadline:
            return circles
        
        circles = circles.copy()
        xs = circles[:, 0].copy()
        ys = circles[:, 1].copy()
        rs = circles[:, 2].copy()
        
        for lp_iter in range(max_lp_iters):
            if time.time() > deadline:
                break
            
            # Compute Delaunay triangulation of centers
            try:
                points = np.column_stack([xs, ys])
                tri = Delaunay(points)
            except Exception:
                break
            
            # Build list of edges from Delaunay triangulation
            edges = set()
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            
            # Also include boundary constraints
            # LP formulation:
            # Maximize sum(r_i)
            # Subject to:
            #   r_i + r_j <= d_ij for (i,j) in edges
            #   r_i <= x_i (left wall)
            #   r_i <= w - x_i (right wall)
            #   r_i <= y_i (bottom wall)
            #   r_i <= h - y_i (top wall)
            #   r_i >= 0
            
            n_constraints = len(edges) + 4 * N
            A_ub = np.zeros((n_constraints, N))
            b_ub = np.zeros(n_constraints)
            
            constraint_idx = 0
            for (i, j) in edges:
                d_ij = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                A_ub[constraint_idx, i] = 1.0
                A_ub[constraint_idx, j] = 1.0
                b_ub[constraint_idx] = d_ij
                constraint_idx += 1
            
            for i in range(N):
                # Left wall
                A_ub[constraint_idx, i] = 1.0
                b_ub[constraint_idx] = xs[i]
                constraint_idx += 1
                # Right wall
                A_ub[constraint_idx, i] = 1.0
                b_ub[constraint_idx] = w - xs[i]
                constraint_idx += 1
                # Bottom wall
                A_ub[constraint_idx, i] = 1.0
                b_ub[constraint_idx] = ys[i]
                constraint_idx += 1
                # Top wall
                A_ub[constraint_idx, i] = 1.0
                b_ub[constraint_idx] = h - ys[i]
                constraint_idx += 1
            
            c_obj = -np.ones(N)  # Minimize -sum(r)
            
            bounds = [(0, None)] * N
            
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    rs = res.x
                    # Extract shadow prices (duals) for constraints
                    # res.ineqlin.marginals gives shadow prices for A_ub constraints
                    marginals = res.ineqlin.marginals if hasattr(res, 'ineqlin') else None
                    
                    if marginals is not None:
                        # Use shadow prices to adjust centers
                        forces_x = np.zeros(N)
                        forces_y = np.zeros(N)
                        
                        constraint_idx = 0
                        for (i, j) in edges:
                            # Shadow price for this constraint
                            price = marginals[constraint_idx] if constraint_idx < len(marginals) else 0.0
                            constraint_idx += 1
                            if price > 0:
                                dx = xs[i] - xs[j]
                                dy = ys[i] - ys[j]
                                d_ij = np.sqrt(dx*dx + dy*dy) + 1e-12
                                # Move apart proportional to price
                                forces_x[i] += price * dx / d_ij
                                forces_y[i] += price * dy / d_ij
                                forces_x[j] -= price * dx / d_ij
                                forces_y[j] -= price * dy / d_ij
                        
                        # Boundary forces
                        for i in range(N):
                            # Left wall
                            price = marginals[constraint_idx] if constraint_idx < len(marginals) else 0.0
                            constraint_idx += 1
                            forces_x[i] -= price
                            # Right wall
                            price = marginals[constraint_idx] if constraint_idx < len(marginals) else 0.0
                            constraint_idx += 1
                            forces_x[i] += price
                            # Bottom wall
                            price = marginals[constraint_idx] if constraint_idx < len(marginals) else 0.0
                            constraint_idx += 1
                            forces_y[i] -= price
                            # Top wall
                            price = marginals[constraint_idx] if constraint_idx < len(marginals) else 0.0
                            constraint_idx += 1
                            forces_y[i] += price
                        
                        # Apply small step
                        step_size = 0.01
                        xs += step_size * forces_x
                        ys += step_size * forces_y
                        
                        # Clamp to bounds
                        xs = np.clip(xs, 0, w)
                        ys = np.clip(ys, 0, h)
            
            except Exception:
                break
            
            circles = np.column_stack([xs, ys, rs])
        
        return circles

    # Try multiple aspect ratios and initialization strategies
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Hex lattice initialization
        circles0 = hex_lattice_init(w, h)
        x0 = circles0.flatten()
        
        # First pass: L-BFGS-B with softplus penalties
        relaxed = optimize_from(circles0, w, h, max_iter=300, beta=100.0)
        check_and_update(relaxed, w, h)
        
        # Second pass: Delaunay LP refinement
        if time.time() > deadline:
            continue
        relaxed_circles = relaxed.reshape(-1, 3)
        lp_refined = delaunay_lp_refine(relaxed_circles, w, h, max_lp_iters=30)
        check_and_update(lp_refined.flatten(), w, h)
        
        # Third pass: High-precision polish
        if time.time() > deadline:
            continue
        polished = polish(lp_refined, w, h, max_iter=400, beta=200.0)
        check_and_update(polished, w, h)
        
        # Fourth pass: Another Delaunay LP refinement after polish
        if time.time() > deadline:
            continue
        polished_circles = polished.reshape(-1, 3)
        lp_refined2 = delaunay_lp_refine(polished_circles, w, h, max_lp_iters=20)
        check_and_update(lp_refined2.flatten(), w, h)
        
        # Final polish
        if time.time() > deadline:
            continue
        final_polish = polish(lp_refined2, w, h, max_iter=300, beta=300.0)
        check_and_update(final_polish, w, h)

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.09
        best_result = np.zeros(3*N)
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                best_result[3*idx] = (i + 0.5) * w / a
                best_result[3*idx+1] = (j + 0.5) * h / b
                best_result[3*idx+2] = r
        best_w = w
        best_h = h

    # Final strict validation and correction
    if not is_valid(best_result, best_w, best_h, tol=1e-7):
        for i in range(N):
            best_result[3*i+2] *= 0.99
        
        if not is_valid(best_result, best_w, best_h, tol=1e-7):
            a, b = 7, 3
            w, h = 1.4, 0.6
            r = 0.05
            best_result = np.zeros(3*N)
            for i in range(a):
                for j in range(b):
                    idx = i*b + j
                    best_result[3*idx] = (i + 0.5) * w / a
                    best_result[3*idx+1] = (j + 0.5) * h / b
                    best_result[3*idx+2] = r
            best_w = w
            best_h = h

    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
    
    return result
# EVOLVE-BLOCK-END
