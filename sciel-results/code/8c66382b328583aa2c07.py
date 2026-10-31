"""Combined program: Sequential greedy construction + force-directed relaxation."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    def validate(W, H, x, y, r, tol=1e-8):
        if W < 0.1 or H < 0.1:
            return False
        if np.any(r < 0):
            return False
        if np.any(x - r < -tol) or np.any(W - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(H - y - r < -tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < (r[i] + r[j] - tol) ** 2:
                    return False
        return True

    def place_circle_greedy(circles, W, H, max_attempts=300):
        """Place a new circle at maximal radius that fits without overlap."""
        best_r = 0.0
        best_pos = None

        for attempt in range(max_attempts):
            px = np.random.uniform(0.05 * W, 0.95 * W)
            py = np.random.uniform(0.05 * H, 0.95 * H)

            r_candidate = min(px, W - px, py, H - py)
            for (xi, yi, ri) in circles:
                dx = px - xi
                dy = py - yi
                dist = np.sqrt(dx * dx + dy * dy)
                r_candidate = min(r_candidate, dist - ri)

            if r_candidate > best_r:
                best_r = r_candidate
                best_pos = (px, py)

        if best_pos is None or best_r <= 1e-6:
            return None

        # Local hill climbing to refine position
        px, py = best_pos
        for _ in range(30):
            improved = False
            for dx in np.linspace(-0.03, 0.03, 5):
                for dy in np.linspace(-0.03, 0.03, 5):
                    npx = px + dx
                    npy = py + dy
                    if npx < 0 or npx > W or npy < 0 or npy > H:
                        continue
                    r_new = min(npx, W - npx, npy, H - npy)
                    for (xi, yi, ri) in circles:
                        ddx = npx - xi
                        ddy = npy - yi
                        dist = np.sqrt(ddx * ddx + ddy * ddy)
                        r_new = min(r_new, dist - ri)
                    if r_new > best_r + 1e-8:
                        best_r = r_new
                        best_pos = (npx, npy)
                        px, py = npx, npy
                        improved = True
            if not improved:
                break

        return (best_pos[0], best_pos[1], best_r)

    def greedy_packing(W, H, seed=42):
        """Build a packing by placing circles one at a time."""
        rng = np.random.RandomState(seed)
        np.random.seed(seed)
        circles = []

        for i in range(N):
            placed = place_circle_greedy(circles, W, H, max_attempts=200)
            if placed is None:
                return None
            circles.append(placed)

        x = np.array([c[0] for c in circles])
        y = np.array([c[1] for c in circles])
        r = np.array([c[2] for c in circles])
        return x, y, r

    def force_relax(W, H, x, y, r, max_iter=300):
        """Force-directed relaxation with radius growth."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        fx = np.zeros(N)
        fy = np.zeros(N)
        step_size = 0.05
        radius_growth = 0.005

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            fx.fill(0.0)
            fy.fill(0.0)

            # Circle-Circle Repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                    min_dist = r[i] + r[j]

                    if dist < min_dist and dist > 1e-6:
                        overlap = min_dist - dist
                        force = overlap * 10.0
                        fx[i] += force * (dx / dist)
                        fy[i] += force * (dy / dist)
                        fx[j] -= force * (dx / dist)
                        fy[j] -= force * (dy / dist)
                    elif dist <= 1e-6:
                        fx[i] += np.random.uniform(-1, 1)
                        fy[i] += np.random.uniform(-1, 1)
                        fx[j] -= np.random.uniform(-1, 1)
                        fy[j] -= np.random.uniform(-1, 1)

            # Wall Repulsion
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10.0
                if W - x[i] < r[i]:
                    fx[i] -= (r[i] - (W - x[i])) * 10.0
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10.0
                if H - y[i] < r[i]:
                    fy[i] -= (r[i] - (H - y[i])) * 10.0

            # Update positions
            x += fx * step_size * 0.1
            y += fy * step_size * 0.1
            x = np.clip(x, 0, W)
            y = np.clip(y, 0, H)

            # Try to increase radii
            new_r = r * (1.0 + radius_growth)
            if validate(W, H, x, y, new_r, tol=1e-8):
                r = new_r

        return x, y, r

    best_result = None
    best_score = -np.inf

    def update_best(W, H, x, y, r):
        nonlocal best_result, best_score
        if validate(W, H, x, y, r, tol=1e-8):
            x = np.clip(x, 0, W)
            y = np.clip(y, 0, H)
            r = np.clip(r, 0, None)
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])
                return True
        return False

    # Try multiple aspect ratios and seeds
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.15, 0.8, 1.25]
    seeds = [42, 123, 7, 99, 2024]

    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W

        for seed in seeds[:2]:  # Limit seeds per aspect ratio
            if time.time() > deadline - 10.0:
                break

            # Greedy construction
            result = greedy_packing(W, H, seed=seed)
            if result is None:
                continue
            x, y, r = result

            if not validate(W, H, x, y, r, tol=1e-8):
                continue

            # Force relaxation
            x, y, r = force_relax(W, H, x, y, r, max_iter=200)

            update_best(W, H, x, y, r)

    # If we still have time, do a final polish on the best result
    if best_result is not None and time.time() < deadline - 15.0:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        # Estimate W, H from the best result
        # We need to track W, H; let's re-derive from the aspect ratio that gave best
        # Actually, let's just do a small polish with the current best
        # Re-run force relax with smaller steps
        # We need W, H - let's use the one that produced best_result
        # This is a limitation; for now, skip if we don't track W,H
        pass

    # Fallback: simple grid packing
    if best_result is None:
        W = 1.0
        H = 1.0
        n_cols = 5
        n_rows = 5
        cell_w = W / n_cols
        cell_h = H / n_rows
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * cell_w
                y[idx] = (i + 0.5) * cell_h
                r[idx] = min(cell_w, cell_h) * 0.45
                idx += 1
        best_result = np.column_stack([x, y, r])

    return best_result
# EVOLVE-BLOCK-END
