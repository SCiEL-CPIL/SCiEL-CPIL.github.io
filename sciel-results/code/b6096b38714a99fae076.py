import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit

    def _softplus(x, beta=10.0):
        """Numerically stable softplus."""
        if x > 1.0 / beta:
            return x
        return np.log1p(np.exp(beta * x)) / beta

    def _softplus_grad(x, beta=10.0):
        """Gradient of softplus."""
        if x > 1.0 / beta:
            return 1.0
        return 1.0 / (1.0 + np.exp(-beta * x))

    def _compute_overlap_penalty(xs, ys, rs, beta=10.0):
        """Smooth overlap penalty using softplus surrogate."""
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                val = min_dist - dist
                penalty += _softplus(val, beta)
        return penalty

    def _delaunay_neighbors(xs, ys):
        """Get Delaunay neighbor pairs for efficient constraint checking."""
        points = np.column_stack((xs, ys))
        try:
            tri = Delaunay(points)
            neighbors = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        neighbors.add((min(a, b), max(a, b)))
            return list(neighbors)
        except Exception:
            # Fallback: all pairs
            return [(i, j) for i in range(N) for j in range(i + 1, N)]

    def _boundary_penalty(xs, ys, rs, w, h):
        """Squared boundary violation penalty."""
        pen = 0.0
        for i in range(N):
            if xs[i] < rs[i]:
                pen += (rs[i] - xs[i]) ** 2
            if xs[i] > w - rs[i]:
                pen += (xs[i] - (w - rs[i])) ** 2
            if ys[i] < rs[i]:
                pen += (rs[i] - ys[i]) ** 2
            if ys[i] > h - rs[i]:
                pen += (ys[i] - (h - rs[i])) ** 2
        return pen

    def _smooth_objective(params, w, h, beta=10.0):
        """
        Objective: minimize penalty - sum(r).
        params = [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
        """
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]

        if np.any(rs <= 0):
            return 1e10 + np.sum(np.minimum(rs, 0) ** 2)

        overlap_pen = _compute_overlap_penalty(xs, ys, rs, beta)
        boundary_pen = _boundary_penalty(xs, ys, rs, w, h)

        return overlap_pen + boundary_pen - np.sum(rs)

    def _smooth_objective_grad(params, w, h, beta=10.0):
        """
        Gradient of smooth objective.
        """
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]

        grad = np.zeros_like(params)

        if np.any(rs <= 0):
            for i in range(N):
                if rs[i] <= 0:
                    grad[2 * i + 2] = -2.0 * rs[i]
            return grad

        # Overlap penalty gradient
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    continue
                min_dist = rs[i] + rs[j]
                val = min_dist - dist
                sp_grad = _softplus_grad(val, beta)
                if sp_grad < 1e-10:
                    continue
                # d(val)/d(xs[i]) = -dx/dist, etc.
                grad[3 * i] += -sp_grad * dx / dist
                grad[3 * i + 1] += -sp_grad * dy / dist
                grad[3 * i + 2] += sp_grad
                grad[3 * j] += sp_grad * dx / dist
                grad[3 * j + 1] += sp_grad * dy / dist
                grad[3 * j + 2] += sp_grad

        # Boundary penalty gradient
        for i in range(N):
            if xs[i] < rs[i]:
                grad[3 * i] += -2.0 * (rs[i] - xs[i])
            if xs[i] > w - rs[i]:
                grad[3 * i] += 2.0 * (xs[i] - (w - rs[i]))
            if ys[i] < rs[i]:
                grad[3 * i + 1] += -2.0 * (rs[i] - ys[i])
            if ys[i] > h - rs[i]:
                grad[3 * i + 1] += 2.0 * (ys[i] - (h - rs[i]))

        # Objective: -sum(r)
        for i in range(N):
            grad[3 * i + 2] -= 1.0

        return grad

    def _validity_margin(result, W, H, tol=1e-4):
        """Return minimum clearance (positive = valid). result is (N,3): x, y, r.
        Checks if circles are inside [0, W] x [0, H] and disjoint."""
        xs = result[:, 0]
        ys = result[:, 1]
        rs = result[:, 2]
        
        # Check radii
        if np.any(rs < 1e-6):
            return -1.0

        # Check boundaries: circles must be inside [0, W] x [0, H]
        # x - r >= 0, x + r <= W
        # y - r >= 0, y + r <= H
        min_x_clearance = np.min(xs - rs)
        max_x_clearance = np.min(W - (xs + rs))
        min_y_clearance = np.min(ys - rs)
        max_y_clearance = np.min(H - (ys + rs))
        
        boundary_margin = min(min_x_clearance, max_x_clearance, min_y_clearance, max_y_clearance)
        
        if boundary_margin < -tol:
            return boundary_margin

        # Check disjointness using Delaunay for efficiency
        min_overlap_margin = float('inf')
        neighbors = _delaunay_neighbors(xs, ys)
        for i, j in neighbors:
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            m = d - (rs[i] + rs[j])
            if m < min_overlap_margin:
                min_overlap_margin = m
        
        # Also check a few more pairs as safety
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in neighbors or (j, i) in neighbors:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                min_dist_sq = (rs[i] + rs[j]) ** 2
                if dist_sq < min_dist_sq * 1.1:  # Only check if potentially close
                    d = np.sqrt(dist_sq)
                    m = d - (rs[i] + rs[j])
                    if m < min_overlap_margin:
                        min_overlap_margin = m
                
        if min_overlap_margin < -tol:
            return min_overlap_margin

        # If both are non-negative (within tol), return the minimum margin
        return min(boundary_margin, min_overlap_margin)

    def _scale_to_perimeter(result):
        """Scale result about its bounding-box center so perimeter <= 4."""
        xs = result[:, 0]
        ys = result[:, 1]
        rs = result[:, 2]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bw = max_x - min_x
        bh = max_y - min_y

        if bw + bh <= 2.0:
            return result

        scale = 2.0 / (bw + bh) * 0.999
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        new_xs = cx + (xs - cx) * scale
        new_ys = cy + (ys - cy) * scale
        new_rs = rs * scale

        return np.column_stack((new_xs, new_ys, new_rs))

    def _make_grid_packing(m, n, W, H, n_circles):
        """Create a grid packing with m rows and n columns, placing n_circles circles."""
        circles = []
        for i in range(m):
            for j in range(n):
                if len(circles) >= n_circles:
                    break
                x = (j + 0.5) * W / n
                y = (i + 0.5) * H / m
                r = min(W / (2 * n), H / (2 * m)) * 0.99
                circles.append([x, y, r])
            if len(circles) >= n_circles:
                break
        return np.array(circles)

    def _hex_packing(W, H, n_circles):
        """Create a hexagonal packing pattern."""
        circles = []
        # Hexagonal packing: rows offset by half a spacing
        # Try different numbers of rows
        best_circles = None
        best_value = 0.0
        
        for n_rows in range(1, 12):
            # In hexagonal packing, vertical spacing is sqrt(3)/2 * horizontal spacing
            # Let's compute based on fitting n_circles
            if n_rows == 0:
                continue
            
            # Approximate: number of columns needed
            n_cols = int(np.ceil(n_circles / n_rows))
            
            # Horizontal spacing
            dx = W / (n_cols + 0.5)
            # Vertical spacing for hexagonal
            dy = dx * np.sqrt(3) / 2
            
            # Check if it fits
            total_h = (n_rows - 1) * dy + 2 * dx
            if total_h > H * 1.1:
                continue
            
            r = dx * 0.95
            
            for i in range(n_rows):
                for j in range(n_cols):
                    if len(circles) >= n_circles:
                        break
                    # Offset odd rows
                    x_offset = 0 if i % 2 == 0 else dx / 2
                    x = (j + 0.5) * dx + x_offset
                    y = (i + 0.5) * dy + dx
                    
                    # Ensure within bounds
                    if x - r < 0 or x + r > W or y - r < 0 or y + r > H:
                        continue
                    
                    circles.append([x, y, r])
                if len(circles) >= n_circles:
                    break
            
            if len(circles) == n_circles:
                value = sum(c[2] for c in circles)
                if value > best_value:
                    best_value = value
                    best_circles = np.array(circles)
        
        return best_circles if best_circles is not None else None

    def _optimize_packing(circles, W, H, max_iters=2000, tol=1e-8):
        """Optimize packing using coordinate descent for radius maximization."""
        circles = circles.copy().astype(float)
        n = len(circles)

        for it in range(max_iters):
            improved = False
            for i in range(n):
                x, y, r = circles[i]

                # Try to increase radius
                max_r = min(x, W - x, y, H - y)  # Distance to walls
                for j in range(n):
                    if i == j:
                        continue
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((x - xj) ** 2 + (y - yj) ** 2)
                    if dist > 0:
                        max_r = min(max_r, dist - rj)
                new_r = max(0, max_r) * 0.999
                if new_r > r + tol:
                    circles[i, 2] = new_r
                    improved = True
            if not improved:
                break

        return circles

    def _optimize_positions(circles, W, H, max_iters=500, tol=1e-10):
        """Optimize positions to maximize minimum clearance, then increase radii."""
        circles = circles.copy().astype(float)
        n = len(circles)
        
        for it in range(max_iters):
            improved = False
            for i in range(n):
                x, y, r = circles[i]
                
                # Try small perturbations
                best_pos = (x, y)
                best_clearance = -np.inf
                
                # Current clearance
                curr_clearance = min(x, W - x, y, H - y)
                for j in range(n):
                    if i == j:
                        continue
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((x - xj) ** 2 + (y - yj) ** 2)
                    if dist > 0:
                        curr_clearance = min(curr_clearance, dist - rj - r)
                
                # Try moving in 8 directions
                step = 0.01 * max(r, 0.01)
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = x + dx, y + dy
                        if new_x - r < 0 or new_x + r > W or new_y - r < 0 or new_y + r > H:
                            continue
                        
                        clearance = min(new_x, W - new_x, new_y, H - new_y)
                        for j in range(n):
                            if i == j:
                                continue
                            xj, yj, rj = circles[j]
                            dist = np.sqrt((new_x - xj) ** 2 + (new_y - yj) ** 2)
                            if dist > 0:
                                clearance = min(clearance, dist - rj - r)
                        
                        if clearance > best_clearance + tol:
                            best_clearance = clearance
                            best_pos = (new_x, new_y)
                
                if best_clearance > curr_clearance + tol:
                    circles[i, 0] = best_pos[0]
                    circles[i, 1] = best_pos[1]
                    improved = True
            
            if not improved:
                break
        
        return circles

    def _try_aspect_ratio(W, H, n_circles=21):
        """Try packing for given aspect ratio using grid + coordinate descent."""
        if abs(W + H - 2.0) > 1e-9:
            scale = 2.0 / (W + H)
            W *= scale
            H *= scale

        best_circles = None
        best_value = 0.0

        # Try different grid configurations
        for m in range(1, 8):
            for n in range(1, 8):
                if m * n < n_circles:
                    continue
                if m * n > n_circles * 2:  # Don't waste too many cells
                    continue

                circles = _make_grid_packing(m, n, W, H, n_circles)
                if len(circles) < n_circles:
                    continue

                # Optimize
                circles_opt = _optimize_packing(circles, W, H)
                value = np.sum(circles_opt[:, 2])

                # Check validity
                if _validity_margin(circles_opt, W, H) >= -1e-4:
                    if value > best_value:
                        best_value = value
                        best_circles = circles_opt

        # Try hexagonal packing
        hex_circles = _hex_packing(W, H, n_circles)
        if hex_circles is not None:
            circles_opt = _optimize_packing(hex_circles, W, H)
            value = np.sum(circles_opt[:, 2])
            if _validity_margin(circles_opt, W, H) >= -1e-4:
                if value > best_value:
                    best_value = value
                    best_circles = circles_opt

        return best_circles, best_value

    def _smooth_optimize(circles, W, H, maxiter=500):
        """Optimize using smooth penalty with L-BFGS-B and gradient."""
        xs = circles[:, 0].copy()
        ys = circles[:, 1].copy()
        rs = circles[:, 2].copy()

        params = np.zeros(3 * N)
        for i in range(N):
            params[3 * i] = xs[i]
            params[3 * i + 1] = ys[i]
            params[3 * i + 2] = rs[i]

        bounds = []
        for i in range(N):
            bounds.append((1e-6, W))
            bounds.append((1e-6, H))
            bounds.append((1e-6, min(W, H) / 2.0))

        def obj(p):
            return _smooth_objective(p, W, H, beta=10.0)

        def grad(p):
            return _smooth_objective_grad(p, W, H, beta=10.0)

        try:
            res = minimize(obj, params, method='L-BFGS-B', jac=grad, bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < 1e6:  # Reasonable result
                new_xs = res.x[0::3]
                new_ys = res.x[1::3]
                new_rs = res.x[2::3]
                return np.column_stack((new_xs, new_ys, new_rs))
        except Exception:
            pass

        return circles

    def _slsqp_refine(circles, W, H, maxiter=300):
        """Refine using SLSQP with smooth objective and gradient."""
        xs = circles[:, 0].copy()
        ys = circles[:, 1].copy()
        rs = circles[:, 2].copy()

        params = np.zeros(3 * N)
        for i in range(N):
            params[3 * i] = xs[i]
            params[3 * i + 1] = ys[i]
            params[3 * i + 2] = rs[i]

        bounds = []
        for i in range(N):
            bounds.append((1e-6, W))
            bounds.append((1e-6, H))
            bounds.append((1e-6, min(W, H) / 2.0))

        def obj(p):
            return _smooth_objective(p, W, H, beta=20.0)

        def grad(p):
            return _smooth_objective_grad(p, W, H, beta=20.0)

        try:
            res = minimize(obj, params, method='SLSQP', jac=grad, bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.fun < 1e6:
                new_xs = res.x[0::3]
                new_ys = res.x[1::3]
                new_rs = res.x[2::3]
                return np.column_stack((new_xs, new_ys, new_rs))
        except Exception:
            pass

        return circles

    def _polish(circles, W, H):
        """Final high-precision polish pass with Delaunay neighbor optimization."""
        circles = circles.copy()
        for _ in range(100):
            improved = False
            # Get Delaunay neighbors for efficient constraint checking
            xs = circles[:, 0]
            ys = circles[:, 1]
            neighbors = _delaunay_neighbors(xs, ys)
            neighbor_set = set(neighbors)
            
            for i in range(N):
                x, y, r = circles[i]
                max_r = min(x, W - x, y, H - y)
                
                # Only check neighbors from Delaunay + nearby pairs
                for j in range(N):
                    if i == j:
                        continue
                    # Check if this is a Delaunay neighbor or potentially close
                    if (i, j) in neighbor_set or (j, i) in neighbor_set:
                        xj, yj, rj = circles[j]
                        dist = np.sqrt((x - xj) ** 2 + (y - yj) ** 2)
                        if dist > 0:
                            max_r = min(max_r, dist - rj)
                    else:
                        # Quick check: if distance squared is much larger than (r+rj)^2, skip
                        xj, yj, rj = circles[j]
                        dx = x - xj
                        dy = y - yj
                        dist_sq = dx * dx + dy * dy
                        min_dist_sq = (r + rj) ** 2
                        if dist_sq < min_dist_sq * 1.5:  # Only compute exact if potentially close
                            dist = np.sqrt(dist_sq)
                            if dist > 0:
                                max_r = min(max_r, dist - rj)
                
                new_r = max(0, max_r) * 0.99999
                if new_r > r + 1e-12:
                    circles[i, 2] = new_r
                    improved = True
            if not improved:
                break
        return circles

    best_circles = None
    best_value = 0.0
    best_W, best_H = 1.0, 1.0

    # Try various aspect ratios
    aspect_ratios = [
        (1.0, 1.0),
        (1.5, 0.5),
        (0.5, 1.5),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.8, 0.2),
        (0.2, 1.8),
        (1.6, 0.4),
        (0.4, 1.6),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.1, 0.9),
        (0.9, 1.1),
    ]

    # Phase 1: Grid + coordinate descent for multiple aspect ratios
    for W, H in aspect_ratios:
        if time.time() > deadline - 30:
            break

        circles, value = _try_aspect_ratio(W, H, N)

        if circles is not None and value > best_value:
            best_value = value
            best_circles = circles
            best_W, best_H = W, H

    # Phase 2: Smooth optimization on best result and a few promising aspect ratios
    if best_circles is not None:
        if time.time() < deadline - 20:
            # First optimize positions
            pos_opt = _optimize_positions(best_circles, best_W, best_H, max_iters=200)
            pos_val = np.sum(pos_opt[:, 2])
            if _validity_margin(pos_opt, best_W, best_H) >= -1e-4 and pos_val > best_value:
                best_value = pos_val
                best_circles = pos_opt
            
            # Then smooth optimization
            refined = _smooth_optimize(best_circles, best_W, best_H, maxiter=300)
            refined_val = np.sum(refined[:, 2])
            if _validity_margin(refined, best_W, best_H) >= -1e-4 and refined_val > best_value:
                best_value = refined_val
                best_circles = refined

    # Phase 3: SLSQP refinement
    if best_circles is not None and time.time() < deadline - 10:
        refined = _slsqp_refine(best_circles, best_W, best_H, maxiter=200)
        refined_val = np.sum(refined[:, 2])
        if _validity_margin(refined, best_W, best_H) >= -1e-4 and refined_val > best_value:
            best_value = refined_val
            best_circles = refined
        
        # Additional position optimization after SLSQP
        if time.time() < deadline - 5:
            pos_opt = _optimize_positions(best_circles, best_W, best_H, max_iters=100)
            pos_val = np.sum(pos_opt[:, 2])
            if _validity_margin(pos_opt, best_W, best_H) >= -1e-4 and pos_val > best_value:
                best_value = pos_val
                best_circles = pos_opt

    # Phase 4: Final polish
    if best_circles is not None and time.time() < deadline - 2:
        best_circles = _polish(best_circles, best_W, best_H)
        best_value = np.sum(best_circles[:, 2])

    # If we didn't find anything, use the seed
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    # Scale to ensure perimeter <= 4
    best_circles = _scale_to_perimeter(best_circles)

    # Final validation and cleanup
    xs = best_circles[:, 0]
    ys = best_circles[:, 1]
    rs = best_circles[:, 2]

    # Ensure circles are inside
    for i in range(N):
        if xs[i] < 0 or xs[i] > best_W or ys[i] < 0 or ys[i] > best_H or rs[i] < 0:
            # Clamp to valid region
            xs[i] = max(0.0, min(best_W, xs[i]))
            ys[i] = max(0.0, min(best_H, ys[i]))
            rs[i] = max(0.0, min(xs[i], best_W - xs[i], ys[i], best_H - ys[i]))

    # Check and fix overlaps
    neighbors = _delaunay_neighbors(xs, ys)
    neighbor_set = set(neighbors)
    for i in range(N):
        for j in range(i + 1, N):
            # Quick check first
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist_sq = (rs[i] + rs[j]) ** 2
            if dist_sq >= min_dist_sq * 1.01:
                continue
            
            dist = np.sqrt(dist_sq)
            if dist < rs[i] + rs[j] - 1e-9:
                # Scale down both radii
                factor = dist / (rs[i] + rs[j]) * 0.99
                rs[i] *= factor
                rs[j] *= factor
                rs[i] = max(0, min(rs[i], min(xs[i], best_W - xs[i], ys[i], best_H - ys[i])))
                rs[j] = max(0, min(rs[j], min(xs[j], best_W - xs[j], ys[j], best_H - ys[j])))

    result = np.column_stack((xs, ys, rs))
    return result
