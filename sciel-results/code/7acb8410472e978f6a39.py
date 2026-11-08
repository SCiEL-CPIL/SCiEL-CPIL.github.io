import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    # --- Helper functions ---

    def initial_grid(n):
        """Create initial grid placement with slightly smaller radii for feasibility."""
        side = int(np.ceil(np.sqrt(n)))
        xs = np.zeros(n)
        ys = np.zeros(n)
        rs = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            xs[i] = (col + 0.5) / side
            ys[i] = (row + 0.5) / side
            rs[i] = 0.5 / side - 1e-4
        return xs, ys, rs

    def check_validity(xs, ys, rs, tol=1e-9):
        """Check if circles are valid (inside square, non-overlapping)."""
        for i in range(N):
            if xs[i] < rs[i] - tol or xs[i] > 1.0 - rs[i] + tol:
                return False
            if ys[i] < rs[i] - tol or ys[i] > 1.0 - rs[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist * min_dist - tol * (min_dist * min_dist):
                    return False
        return True

    def relax(xs, ys, rs, max_iter=80):
        """
        Position-based dynamics relaxation: project onto boundary and non-overlap constraints.
        Returns (xs, ys, converged).
        """
        for _ in range(max_iter):
            moved = False

            # Boundary constraints
            for i in range(N):
                r = rs[i]
                if xs[i] < r:
                    xs[i] = r
                    moved = True
                elif xs[i] > 1.0 - r:
                    xs[i] = 1.0 - r
                    moved = True
                if ys[i] < r:
                    ys[i] = r
                    moved = True
                elif ys[i] > 1.0 - r:
                    ys[i] = 1.0 - r
                    moved = True

            # Pairwise non-overlap constraints
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist_sq = dx * dx + dy * dy
                    min_dist = rs[i] + rs[j]

                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-15 else 1e-8
                        overlap = min_dist - dist
                        if dist < 1e-9:
                            angle = np.random.rand() * 2 * np.pi
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist

                        push = overlap * 0.5 * 1.02  # slight over-push
                        xs[i] -= dx * push
                        ys[i] -= dy * push
                        xs[j] += dx * push
                        ys[j] += dy * push
                        moved = True

            if not moved:
                return xs, ys, True

        return xs, ys, False

    def grow_and_relax(xs, ys, rs, growth_factor, max_iter=60):
        """
        Try to grow all radii by growth_factor, then relax.
        Returns (xs, ys, rs, success).
        """
        prev_xs = xs.copy()
        prev_ys = ys.copy()
        new_rs = rs * growth_factor

        # Check if any radius exceeds 0.5
        if np.max(new_rs) > 0.5:
            return xs, ys, rs, False

        rs_new = new_rs.copy()
        xs_new = xs.copy()
        ys_new = ys.copy()

        # Relax with new radii
        xs_new, ys_new, converged = relax(xs_new, ys_new, rs_new, max_iter)

        if converged:
            # Verify validity
            if check_validity(xs_new, ys_new, rs_new):
                return xs_new, ys_new, rs_new, True

        # Revert
        return xs, ys, rs, False

    # --- Main optimization loop ---

    xs, ys, rs = initial_grid(N)
    xs, ys, _ = relax(xs, ys, rs, max_iter=100)

    best_score = np.sum(rs)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_rs = rs.copy()

    growth_factor = 1.003
    min_growth = 1.0001

    # Phase 1: Steady growth
    while time.time() < deadline - 20:
        xs, ys, rs, success = grow_and_relax(xs, ys, rs, growth_factor, max_iter=50)

        if success:
            current_score = np.sum(rs)
            if current_score > best_score:
                best_score = current_score
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_rs = rs.copy()
            # Slightly increase growth factor to speed up
            growth_factor = min(1.005, growth_factor * 1.01)
        else:
            # Reduce growth factor
            growth_factor *= 0.92
            if growth_factor < min_growth:
                break

    # Phase 2: Final polish with smaller growth steps
    growth_factor = 1.0005
    polish_deadline = deadline - 5
    while time.time() < polish_deadline:
        xs, ys, rs, success = grow_and_relax(xs, ys, rs, growth_factor, max_iter=30)
        if success:
            current_score = np.sum(rs)
            if current_score > best_score:
                best_score = current_score
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_rs = rs.copy()
            growth_factor = min(1.0008, growth_factor * 1.005)
        else:
            growth_factor *= 0.95
            if growth_factor < 1.0001:
                break

    # Final validation
    if not check_validity(best_xs, best_ys, best_rs):
        # Fallback: return a valid grid
        xs_f, ys_f, rs_f = initial_grid(N)
        xs_f, ys_f, _ = relax(xs_f, ys_f, rs_f, max_iter=200)
        best_xs, best_ys, best_rs = xs_f, ys_f, rs_f

    return np.column_stack((best_xs, best_ys, best_rs))
