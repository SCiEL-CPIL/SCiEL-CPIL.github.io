import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def _elapsed():
        return time.time() - start_time

    def _remaining():
        return max(0.0, deadline - time.time())

    def _is_valid(xy, r, tol=1e-7):
        n = len(xy)
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if x - ri < -tol or 1.0 - x - ri < -tol or y - ri < -tol or 1.0 - y - ri < -tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def _score(xy, r):
        if not _is_valid(xy, r):
            return 0.0
        return float(np.sum(r))

    def _largest_empty_circle(xy, r, n_existing):
        """
        Find the point in [0,1]^2 maximizing min distance to walls and to existing circles.
        Uses circle-circle intersection vertices + random sampling + golden-section refinement.
        """
        best_pt = np.array([0.5, 0.5])
        best_val = -1.0

        def _clearance(pt):
            x, y = pt[0], pt[1]
            val = min(x, 1.0 - x, y, 1.0 - y)
            for i in range(n_existing):
                dx = x - xy[i, 0]
                dy = y - xy[i, 1]
                d = np.sqrt(dx * dx + dy * dy)
                val = min(val, d - r[i])
            return val

        # Candidate 1: grid sampling
        grid_n = 40
        gx = np.linspace(0, 1, grid_n)
        gy = np.linspace(0, 1, grid_n)
        for i in range(grid_n):
            for j in range(grid_n):
                pt = np.array([gx[i], gy[j]])
                v = _clearance(pt)
                if v > best_val:
                    best_val = v
                    best_pt = pt.copy()

        # Candidate 2: circle-circle intersection vertices
        for i in range(n_existing):
            for j in range(i + 1, n_existing):
                dx = xy[j, 0] - xy[i, 0]
                dy = xy[j, 1] - xy[i, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-12:
                    continue
                r1, r2 = r[i], r[j]
                # External tangent point (between the two circles)
                if d > r1 + r2:
                    # Point on line between centers at distance r1 from center i
                    t = r1 / d
                    pt = np.array([xy[i, 0] + t * dx, xy[i, 1] + t * dy])
                    if 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1:
                        v = _clearance(pt)
                        if v > best_val:
                            best_val = v
                            best_pt = pt.copy()
                # Intersection points of the two circles (if they overlap or are close)
                if d > abs(r1 - r2) and d < r1 + r2:
                    a = (r1**2 - r2**2 + d**2) / (2 * d)
                    h2 = r1**2 - a**2
                    if h2 > 0:
                        h = np.sqrt(h2)
                        mx = xy[i, 0] + a * dx / d
                        my = xy[i, 1] + a * dy / d
                        px = -dy / d
                        py = dx / d
                        pt1 = np.array([mx + h * px, my + h * py])
                        pt2 = np.array([mx - h * px, my - h * py])
                        for pt in [pt1, pt2]:
                            if 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1:
                                v = _clearance(pt)
                                if v > best_val:
                                    best_val = v
                                    best_pt = pt.copy()

        # Candidate 3: random sampling
        for _ in range(200):
            pt = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1)])
            v = _clearance(pt)
            if v > best_val:
                best_val = v
                best_pt = pt.copy()

        # Candidate 4: corners and edge midpoints
        special_pts = [
            [0.01, 0.01], [0.99, 0.01], [0.01, 0.99], [0.99, 0.99],
            [0.5, 0.01], [0.5, 0.99], [0.01, 0.5], [0.99, 0.5],
            [0.5, 0.5],
        ]
        for pt in special_pts:
            pt = np.array(pt)
            v = _clearance(pt)
            if v > best_val:
                best_val = v
                best_pt = pt.copy()

        # Local refinement: Nelder-Mead on clearance (negated for minimization)
        if _remaining() > 2.0:
            def _neg_clearance(pt):
                return -_clearance(pt)
            try:
                res = minimize(_neg_clearance, best_pt, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.success or -res.fun > best_val:
                    v = -res.fun
                    if v > best_val:
                        best_val = v
                        best_pt = np.clip(res.x, 0, 1)
            except Exception:
                pass

        if best_val < 0:
            best_val = 1e-4
        return best_pt, best_val

    def _build_sequential(xy, r, n_total, seed=0):
        """Sequentially place circles using largest-empty-circle."""
        rng = np.random.RandomState(seed)
        xy = np.zeros((n_total, 2))
        r = np.zeros(n_total)
        n_existing = 0
        for i in range(n_total):
            if _remaining() < 1.0:
                break
            pt, rad = _largest_empty_circle(xy, r, n_existing)
            xy[i] = pt
            r[i] = max(rad, 0.001)
            n_existing += 1
        return xy, r

    def _polish_positions(xy, r, max_sweeps=8):
        """
        Coordinate descent: for each circle, try small perturbations in x and y
        to see if it can fit with a larger radius (i.e., move to a better spot).
        Only moves if the new position allows a strictly larger radius.
        """
        n = len(xy)
        improved = True
        sweep = 0
        while improved and sweep < max_sweeps and _remaining() > 5.0:
            improved = False
            sweep += 1
            for i in range(n):
                if _remaining() < 2.0:
                    break
                best_x, best_y, best_rad = xy[i, 0], xy[i, 1], r[i]
                # Try perturbations
                for dx in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                        nx = xy[i, 0] + dx
                        ny = xy[i, 1] + dy
                        if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                            continue
                        # Compute max radius at (nx, ny) given all other circles
                        rad = min(nx, 1.0 - nx, ny, 1.0 - ny)
                        for j in range(n):
                            if j == i:
                                continue
                            d = np.sqrt((nx - xy[j, 0])**2 + (ny - xy[j, 1])**2)
                            rad = min(rad, d - r[j])
                        if rad > best_rad + 1e-8:
                            best_rad = rad
                            best_x = nx
                            best_y = ny
                if best_rad > r[i] + 1e-8:
                    xy[i, 0] = best_x
                    xy[i, 1] = best_y
                    r[i] = best_rad
                    improved = True
        return xy, r

    def _grow_radii(xy, r):
        """
        After positions are fixed, try to increase each radius slightly
        by checking if there's room.
        """
        n = len(xy)
        for i in range(n):
            # Max possible radius at current position
            max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            for j in range(n):
                if j == i:
                    continue
                d = np.sqrt((xy[i, 0] - xy[j, 0])**2 + (xy[i, 1] - xy[j, 1])**2)
                max_r = min(max_r, d - r[j])
            r[i] = min(r[i], max(0.001, max_r))
        return r

    # --- Main: multi-start with sequential construction + polish ---
    best_total = 0.0
    best_xy = None
    best_r = None
    best_valid = False

    # Diverse seeds for multi-start
    seeds = list(range(20))
    np.random.shuffle(seeds)

    for seed in seeds:
        if _remaining() < 15.0:
            break

        # Sequential construction
        xy, r = _build_sequential(np.zeros((N, 2)), np.zeros(N), N, seed=seed)

        # Validate
        if not _is_valid(xy, r):
            # Project: shrink radii to fix violations
            for i in range(N):
                max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
                for j in range(N):
                    if j != i:
                        d = np.sqrt((xy[i, 0] - xy[j, 0])**2 + (xy[i, 1] - xy[j, 1])**2)
                        max_r = min(max_r, d - r[j])
                r[i] = max(0.001, max_r)
            if not _is_valid(xy, r):
                continue

        total = _score(xy, r)

        # Polish
        if _remaining() > 10.0:
            xy, r = _polish_positions(xy, r, max_sweeps=6)
            r = _grow_radii(xy, r)
            if _is_valid(xy, r):
                total = _score(xy, r)

        if total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

    # Final fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Build final output
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final validation
    final_xy = result[:, :2]
    final_r = result[:, 2]
    if not _is_valid(final_xy, final_r):
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
