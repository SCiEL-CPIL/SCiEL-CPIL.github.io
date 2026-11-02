"""Circle packing in perimeter-4 rectangle: SOCP via ADMM + multi-start."""
import numpy as np
import time

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def admm_solve(x0, y0, r0, w0, h0, time_budget, rho=50.0, max_iter=2000):
        """
        ADMM solver for the SOCP formulation:
        Maximize sum(r) subject to:
          ||(x_i - x_j, y_i - y_j)||_2 >= r_i + r_j  (non-overlap)
          x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i (boundary)
        """
        x = np.array(x0, dtype=float)
        y = np.array(y0, dtype=float)
        r = np.array(r0, dtype=float)
        w = w0
        h = h0

        # Dual variables for non-overlap constraints
        # u_ij is the dual variable for pair (i,j)
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        n_pairs = len(pairs)
        u = np.zeros((n_pairs, 2))  # 2D dual for each pair

        # Dual variables for boundary constraints
        # For each circle i, 4 boundary constraints
        alpha_l = np.zeros(N)  # x_i >= r_i
        alpha_r = np.zeros(N)  # x_i <= w - r_i
        alpha_b = np.zeros(N)  # y_i >= r_i
        alpha_t = np.zeros(N)  # y_i <= h - r_i

        best_r_sum = float(np.sum(r))
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

        for it in range(max_iter):
            if time.time() > time_budget:
                break

            # --- x-update: minimize ||x - x_old + u/rho||^2 + ||y - y_old + u/rho||^2 + boundary terms
            # This is a QP. We solve it via a simple iterative approach or direct solve.
            # For simplicity, use a gradient step with projection.

            # Compute forces from dual variables
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Non-overlap dual forces
            for k, (i, j) in enumerate(pairs):
                fx[i] += u[k, 0] / rho
                fy[i] += u[k, 1] / rho
                fx[j] -= u[k, 0] / rho
                fy[j] -= u[k, 1] / rho

            # Boundary dual forces
            fx -= alpha_l / rho
            fx += alpha_r / rho
            fy -= alpha_b / rho
            fy += alpha_t / rho

            # x-update: x_new = x + fx (since the quadratic is (1/rho)||x - (x - u/rho)||^2)
            x_new = x + fx
            y_new = y + fy

            # Project x_new, y_new onto boundary constraints
            for i in range(N):
                x_new[i] = np.clip(x_new[i], r[i], w - r[i])
                y_new[i] = np.clip(y_new[i], r[i], h - r[i])

            # --- r-update: maximize sum(r) - (1/rho)*sum of constraint violations squared
            # The r-update is: minimize -sum(r) + (1/(2rho)) * sum_k ||(x_i-x_j, y_i-y_j) - u_k/rho||_2^2
            #               + (1/(2rho)) * sum of boundary violation terms
            # This is a separable problem in r_i.

            r_new = np.zeros(N)
            for i in range(N):
                # Objective for r_i: -r_i + (1/(2rho)) * sum_{j != i} ||d_ij - u_k/rho||_2^2
                #                  + (1/(2rho)) * (max(0, r_i - x_i)^2 + max(0, r_i - y_i)^2
                #                                  + max(0, r_i - (w - x_i))^2 + max(0, r_i - (h - y_i))^2)
                # This is piecewise quadratic in r_i. We solve via binary search on the subgradient.

                # The function is convex in r_i (sum of convex functions).
                # We find r_i >= 0 that minimizes it.
                # The subgradient is: -1 + (1/rho) * sum_{j != i} (d_ij - u_k/rho) . (d_ij direction)
                #                    + boundary terms
                # This is complex. Instead, use a simpler approach:
                # r_i_new = max(0, r_i + (1/rho) * (sum of dual contributions))

                # For simplicity, use a heuristic update:
                r_contrib = 0.0
                for k, (a, b) in enumerate(pairs):
                    if a == i:
                        dx = x_new[a] - x_new[b]
                        dy = y_new[a] - y_new[b]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 1e-12:
                            r_contrib += (u[k, 0] * dx / d + u[k, 1] * dy / d) / rho
                    elif b == i:
                        dx = x_new[a] - x_new[b]
                        dy = y_new[a] - y_new[b]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 1e-12:
                            r_contrib -= (u[k, 0] * dx / d + u[k, 1] * dy / d) / rho

                # Boundary contributions
                r_contrib += max(0, r[i] - x_new[i]) / rho
                r_contrib += max(0, r[i] - y_new[i]) / rho
                r_contrib += max(0, r[i] - (w - x_new[i])) / rho
                r_contrib += max(0, r[i] - (h - y_new[i])) / rho

                r_new[i] = max(0.0, r[i] + r_contrib - 1.0 / rho)

            # --- Dual update
            for k, (i, j) in enumerate(pairs):
                dx = x_new[i] - x_new[j]
                dy = y_new[i] - y_new[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-12:
                    u[k, 0] += rho * (dx - (r_new[i] + r_new[j]) * dx / d)
                    u[k, 1] += rho * (dy - (r_new[i] + r_new[j]) * dy / d)
                else:
                    u[k, 0] += rho * (dx - (r_new[i] + r_new[j]))
                    u[k, 1] += rho * (dy)

            alpha_l += rho * (x_new - r_new)
            alpha_r += rho * (r_new - (w - x_new))
            alpha_b += rho * (y_new - r_new)
            alpha_t += rho * (r_new - (h - y_new))

            # Clip dual variables to avoid divergence
            alpha_l = np.maximum(alpha_l, 0)
            alpha_r = np.maximum(alpha_r, 0)
            alpha_b = np.maximum(alpha_b, 0)
            alpha_t = np.maximum(alpha_t, 0)

            # Update
            x, y, r = x_new, y_new, r_new

            # Track best
            r_sum = float(np.sum(r))
            if r_sum > best_r_sum:
                best_r_sum = r_sum
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

            # Convergence check
            if it > 50 and np.max(np.abs(r_new - r)) < 1e-10:
                break

        # Final check
        valid, val = check_validity(best_x, best_y, best_r, w, h)
        if valid:
            return best_x, best_y, best_r, w, h, val
        return None

    def force_relax(x0, y0, r0, w, h, max_iter=1000):
        """Force-directed relaxation to resolve overlaps."""
        x = np.array(x0, dtype=float)
        y = np.array(y0, dtype=float)
        r = np.array(r0, dtype=float)
        dt = 0.01

        for it in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Wall repulsion
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 10
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 10

            # Pairwise repulsion
            max_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        max_overlap = max(max_overlap, overlap)
                        if d > 1e-12:
                            ux = dx / d
                            uy = dy / d
                            f = overlap * 5.0
                            fx[i] += ux * f
                            fy[i] += uy * f
                            fx[j] -= ux * f
                            fy[j] -= uy * f

            if max_overlap < 1e-8:
                break

            x += dt * fx
            y += dt * fy
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        return x, y, r

    def grow_radii(x, y, r, w, h, time_budget):
        """Iteratively grow radii."""
        for _ in range(500):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(N):
                # Max radius for circle i
                r_max = min(x[i], w - x[i], y[i], h - y[i])
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-7:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return r

    # Generate diverse initial guesses
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55]
    side = int(np.ceil(np.sqrt(N)))

    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0
            y[i] = (row + 0.5) / side * h0
            r[i] = min(w0 / side, h0 / side) * 0.4

        # Force relax
        x, y, r = force_relax(x, y, r, w0, h0, max_iter=500)

        # ADMM
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            result = admm_solve(x, y, r, w0, h0, time_budget=time.time() + min(remaining - 5, 30), rho=50.0, max_iter=2000)
            if result is not None:
                rx, ry, rr, rw, rh, rval = result
                # Grow radii
                rr = grow_radii(rx, ry, rr, rw, rh, time_budget=time.time() + 5)
                valid, val = check_validity(rx, ry, rr, rw, rh)
                if valid and val > best_value:
                    best_value = val
                    best_result = (rx.copy(), ry.copy(), rr.copy(), rw, rh)

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        x = rng.uniform(0.1, w0 - 0.1, N)
        y = rng.uniform(0.1, h0 - 0.1, N)
        r = rng.uniform(0.04, 0.1, N)

        x, y, r = force_relax(x, y, r, w0, h0, max_iter=300)

        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            result = admm_solve(x, y, r, w0, h0, time_budget=time.time() + min(remaining - 5, 20), rho=50.0, max_iter=1000)
            if result is not None:
                rx, ry, rr, rw, rh, rval = result
                rr = grow_radii(rx, ry, rr, rw, rh, time_budget=time.time() + 5)
                valid, val = check_validity(rx, ry, rr, rw, rh)
                if valid and val > best_value:
                    best_value = val
                    best_result = (rx.copy(), ry.copy(), rr.copy(), rw, rh)

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        x = np.array([(i + 0.5) / side for i in range(N)])
        y = np.array([(j + 0.5) / side for j in range(N)])
        r = np.full(N, 0.5 / side - 1e-3)
        return np.column_stack((x, y, r))

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
