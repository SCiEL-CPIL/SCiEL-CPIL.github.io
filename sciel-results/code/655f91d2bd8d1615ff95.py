"""Circle packing in perimeter-4 rectangle using penalty-based NLP + force relaxation."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0

    def is_valid(circles, W, H, tol=1e-7):
        """Check if packing is valid: no overlaps, all inside rectangle."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0:
                return False
            if x < r - tol or x > W - r + tol:
                return False
            if y < r - tol or y > H - r + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx * dx + dy * dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist * min_dist - tol:
                    return False
        return True

    def force_relax(circles, W, H, max_iter=500, time_limit=None):
        """
        Force-directed relaxation to resolve overlaps and boundary violations.
        Operates on list of (x, y, r) tuples.
        """
        xs = np.array([c[0] for c in circles], dtype=np.float64)
        ys = np.array([c[1] for c in circles], dtype=np.float64)
        rs = np.array([c[2] for c in circles], dtype=np.float64)
        n = len(xs)
        dt = 0.01
        t0 = time.time()

        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall repulsion
            for i in range(n):
                if xs[i] - rs[i] < 0:
                    fx[i] += (xs[i] - rs[i]) * 50.0
                if xs[i] + rs[i] > W:
                    fx[i] -= (xs[i] + rs[i] - W) * 50.0
                if ys[i] - rs[i] < 0:
                    fy[i] += (ys[i] - rs[i]) * 50.0
                if ys[i] + rs[i] > H:
                    fy[i] -= (ys[i] + rs[i] - H) * 50.0

            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d_sq = dx * dx + dy * dy
                    min_d = rs[i] + rs[j]
                    if d_sq < min_d * min_d:
                        if d_sq > 1e-12:
                            d = np.sqrt(d_sq)
                            overlap = min_d - d
                            f = overlap * 20.0
                            fx[i] += (dx / d) * f
                            fy[i] += (dy / d) * f
                            fx[j] -= (dx / d) * f
                            fy[j] -= (dy / d) * f
                        else:
                            angle = random.uniform(0, 2 * np.pi)
                            f = min_d * 20.0
                            fx[i] += np.cos(angle) * f
                            fy[i] += np.sin(angle) * f
                            fx[j] -= np.cos(angle) * f
                            fy[j] -= np.sin(angle) * f

            # Update positions
            xs += dt * fx
            ys += dt * fy

            # Clamp positions to stay within bounds
            for i in range(n):
                xs[i] = np.clip(xs[i], rs[i], W - rs[i])
                ys[i] = np.clip(ys[i], rs[i], H - rs[i])

        circles_out = list(zip(xs, ys, rs))
        return circles_out

    def objective(z, W, H, penalty_weight=500.0):
        """
        Unified objective: minimize -sum(r) + penalty for violations.
        z = [x1, y1, r1, ..., xN, yN, rN]
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]

        if np.any(r < 0.01):
            return 1e6

        obj = -np.sum(r)

        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                dist_req = r[i] + r[j]
                if d_sq < dist_req * dist_req:
                    d = np.sqrt(max(d_sq, 1e-12))
                    overlap = dist_req - d
                    obj += penalty_weight * overlap * overlap

        # Boundary penalty
        for i in range(N):
            if x[i] < r[i]:
                obj += penalty_weight * (r[i] - x[i]) ** 2
            elif x[i] > W - r[i]:
                obj += penalty_weight * (x[i] - (W - r[i])) ** 2
            if y[i] < r[i]:
                obj += penalty_weight * (r[i] - y[i]) ** 2
            elif y[i] > H - r[i]:
                obj += penalty_weight * (y[i] - (H - r[i])) ** 2

        return obj

    def nlp_refine(circles, W, H, time_budget):
        """
        Use SLSQP to refine a valid packing, maximizing sum of radii.
        """
        if time_budget < 3:
            return circles

        xs = np.array([c[0] for c in circles], dtype=np.float64)
        ys = np.array([c[1] for c in circles], dtype=np.float64)
        rs = np.array([c[2] for c in circles], dtype=np.float64)
        n = N

        z0 = np.zeros(3 * n)
        for i in range(n):
            z0[3*i] = xs[i]
            z0[3*i+1] = ys[i]
            z0[3*i+2] = rs[i]

        # Objective function for SLSQP
        def obj_func(z):
            x = z[0::3][:n]
            y = z[1::3][:n]
            r = z[2::3][:n]
            if np.any(r < 0.01):
                return 1e6
            return -np.sum(r)

        # Constraints: non-overlap and boundary
        constraints = []

        # Boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[-1] - p[3*idx] - p[3*idx+2] if False else W - p[3*idx] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx+1] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: H - p[3*idx+1] - p[3*idx+2]})

        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, idx1=i, idx2=j: (
                        (p[3*idx1] - p[3*idx2])**2 + 
                        (p[3*idx1+1] - p[3*idx2+1])**2 - 
                        (p[3*idx1+2] + p[3*idx2+2])**2
                    )
                })

        # Bounds: r > 0.01, positions within [0, W] x [0, H]
        bounds = []
        for i in range(n):
            bounds.append((0.0, W))      # x_i
            bounds.append((0.0, H))      # y_i
            bounds.append((0.01, 0.5))   # r_i

        try:
            result = minimize(
                obj_func, z0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
            )

            if result.success or result.fun < obj_func(z0):
                z_opt = result.x
                new_circles = []
                for i in range(n):
                    new_circles.append((z_opt[3*i], z_opt[3*i+1], z_opt[3*i+2]))

                if is_valid(new_circles, W, H, tol=1e-6):
                    return new_circles

        except Exception:
            pass

        return circles

    def radius_growth_search(circles, W, H, time_budget):
        """
        Iteratively try to grow each circle's radius, using force relaxation
        to resolve any resulting overlaps.
        """
        xs = np.array([c[0] for c in circles], dtype=np.float64)
        ys = np.array([c[1] for c in circles], dtype=np.float64)
        rs = np.array([c[2] for c in circles], dtype=np.float64)
        n = N
        t0 = time.time()

        for iteration in range(500):
            if time.time() > t0 + time_budget:
                break

            # Try to grow each circle slightly
            improved = False
            for i in range(n):
                old_r = rs[i]
                new_r = old_r * 1.003

                # Check if new radius fits in bounds
                if new_r > W / 2 or new_r > H / 2:
                    continue

                rs[i] = new_r

                # Force relax to resolve overlaps
                temp_circles = list(zip(xs, ys, rs))
                relaxed = force_relax(temp_circles, W, H, max_iter=50, time_limit=t0 + time_budget)

                if is_valid(relaxed, W, H, tol=1e-6):
                    xs = np.array([c[0] for c in relaxed])
                    ys = np.array([c[1] for c in relaxed])
                    rs = np.array([c[2] for c in relaxed])
                    improved = True
                else:
                    rs[i] = old_r

            if not improved:
                break

        circles_out = list(zip(xs, ys, rs))
        return circles_out

    # Main optimization loop
    random.seed(42)
    np.random.seed(42)

    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]

    for W_ratio in aspect_ratios:
        if time.time() > deadline - 20:
            break

        W = W_ratio
        H = 2.0 - W_ratio
        if W <= 0.1 or H <= 0.1:
            continue

        # Generate initial configurations
        inits = []

        # Grid initialization
        cols = 5
        rows = 5
        grid_circles = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                x = (i + 0.5) * W / cols
                y = (j + 0.5) * H / rows
                r = min(W / cols, H / rows) * 0.4
                grid_circles.append((x, y, r))
                count += 1
        if len(grid_circles) == N:
            inits.append(grid_circles)

        # Random initializations
        for _ in range(3):
            random_circles = []
            for _ in range(N):
                r = random.uniform(0.04, 0.1)
                x = random.uniform(r, W - r)
                y = random.uniform(r, H - r)
                random_circles.append((x, y, r))
            inits.append(random_circles)

        # Hilbert curve initialization
        order = 8
        size = 1 << order
        hilbert_pts = []
        for i in range(N):
            h = int(i * (size - 1) / max(N - 1, 1))
            x_h = 0
            y_h = 0
            for s in range(1, size, 2):
                rx = 1 & (h // 2)
                ry = 1 & (h ^ rx)
                if ry == 0:
                    if rx == 1:
                        x_h = s - 1 - x_h
                        y_h = s - 1 - y_h
                    x_h, y_h = y_h, x_h
                x_h += s * rx
                y_h += s * ry
            hilbert_pts.append(((x_h + 0.5) / size, (y_h + 0.5) / size))

        hilbert_circles = []
        for (u, v) in hilbert_pts:
            x = u * W
            y = v * H
            r = 0.07
            hilbert_circles.append((x, y, r))
        inits.append(hilbert_circles)

        # Process each initial configuration
        for init_circles in inits:
            if time.time() > deadline - 10:
                break

            # Step 1: Force relax to get a valid packing
            relaxed = force_relax(init_circles, W, H, max_iter=300, time_limit=deadline - 5)

            if not is_valid(relaxed, W, H, tol=1e-6):
                continue

            # Step 2: NLP refinement
            time_left = deadline - time.time()
            if time_left > 5:
                refined = nlp_refine(relaxed, W, H, time_budget=time_left - 2)
                if is_valid(refined, W, H, tol=1e-6):
                    relaxed = refined

            # Step 3: Radius growth search
            time_left = deadline - time.time()
            if time_left > 3:
                grown = radius_growth_search(relaxed, W, H, time_budget=time_left - 1)
                if is_valid(grown, W, H, tol=1e-6):
                    relaxed = grown

            total_r = sum(c[2] for c in relaxed)
            if total_r > best_sum_r:
                best_sum_r = total_r
                best_config = np.array(relaxed)
                best_W = W
                best_H = H

    if best_config is not None and len(best_config) == N:
        return best_config

    # Fallback: simple grid
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.8
            cells.append((x, y, r))
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
