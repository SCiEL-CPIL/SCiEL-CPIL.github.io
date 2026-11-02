"""Circle packing in perimeter-4 rectangle: SLSQP + Voronoi-based expansion + polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 21


def _smooth_objective(z, beta=50.0):
    """Smooth log-sum-exp penalty objective for better gradient behavior."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap: smooth penalty via log-sum-exp
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    # violation = r_sum - d, penalize positive violations smoothly
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    violations = r_sum_pairs - d_pairs  # positive when overlapping

    # Smooth max penalty: use soft-plus approximation
    # softplus(x) = log(1 + exp(x/beta)) * beta
    scaled = violations * beta
    # clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    soft_pen = np.logaddexp(0, scaled) / beta  # ≈ max(0, violation)
    obj += 100.0 * np.sum(soft_pen ** 2)

    # Boundary violations: smooth penalty
    left_v = r - x
    right_v = x + r - w
    bottom_v = r - y
    top_v = y + r - h

    for v in [left_v, right_v, bottom_v, top_v]:
        sv = np.clip(v * beta, -50, 50)
        sp = np.logaddexp(0, sv) / beta
        obj += 100.0 * np.sum(sp ** 2)

    # Aspect ratio penalty
    if w < 0.4:
        obj += 200.0 * (0.4 - w) ** 2
    elif w > 1.6:
        obj += 200.0 * (w - 1.6) ** 2

    return obj


def _epigraph_radius_growth(z, time_budget):
    """
    Epigraph-style radius growth: iteratively grow all radii simultaneously
    by solving a small feasibility problem for the growth factor.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Binary search on uniform scale factor
    lo, hi = 1.0, 1.5
    for _ in range(40):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2.0
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h, tol=1e-9):
            lo = mid
        else:
            hi = mid

    if lo > 1.0 + 1e-6:
        r = r * lo

    # Now try per-circle growth with position adjustment
    for round_i in range(80):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            # Try growing circle i
            r_max = _max_radius_simple(x, y, r, w, h, i)
            if r_max > r[i] + 1e-8:
                # Binary search for max valid radius for circle i
                lo_i = r[i]
                hi_i = r_max
                for _ in range(20):
                    mid_i = (lo_i + hi_i) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid_i
                    if _is_valid(x, y, r_test, w, h, tol=1e-9):
                        lo_i = mid_i
                    else:
                        hi_i = mid_i
                if lo_i > r[i] + 1e-9:
                    r[i] = lo_i
                    improved = True
            else:
                # Try moving circle i to free space
                # Find best direction to move
                best_gain = 0.0
                best_pos = None
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    dx_dir = np.cos(angle)
                    dy_dir = np.sin(angle)
                    for step in [0.02, 0.01, 0.005, 0.002]:
                        new_x = x[i] + dx_dir * step
                        new_y = y[i] + dy_dir * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            # Check if this allows larger radius
                            r_new_max = _max_radius_simple(
                                np.where(np.arange(n) == i, new_x, x),
                                np.where(np.arange(n) == i, new_y, y),
                                r, w, h, i
                            )
                            gain = r_new_max - r[i]
                            if gain > best_gain:
                                best_gain = gain
                                best_pos = (new_x, new_y)
                            break
                    if best_pos is not None:
                        break

                if best_pos is not None and best_gain > 1e-7:
                    x[i] = best_pos[0]
                    y[i] = best_pos[1]
                    # Now grow
                    r_max2 = _max_radius_simple(x, y, r, w, h, i)
                    if r_max2 > r[i] + 1e-8:
                        lo_i = r[i]
                        hi_i = r_max2
                        for _ in range(20):
                            mid_i = (lo_i + hi_i) / 2.0
                            r_test = r.copy()
                            r_test[i] = mid_i
                            if _is_valid(x, y, r_test, w, h, tol=1e-9):
                                lo_i = mid_i
                            else:
                                hi_i = mid_i
                        if lo_i > r[i] + 1e-9:
                            r[i] = lo_i
                            improved = True

        if not improved:
            break

    if _is_valid(x, y, r, w, h, tol=1e-9):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _max_radius_voronoi(x, y, r, w, h, i):
    """Compute maximum radius for circle i using Voronoi cell inradius."""
    # Build point set: all circle centers + rectangle corners and edge midpoints
    pts = list(zip(x, y))
    # Add rectangle boundary points to constrain Voronoi cells
    margin = 0.01
    boundary_pts = [
        (0, 0), (w, 0), (0, h), (w, h),
        (0, h / 2), (w, h / 2), (w / 2, 0), (w / 2, h),
        (w * 0.25, 0), (w * 0.75, 0), (w * 0.25, h), (w * 0.75, h),
        (0, h * 0.25), (0, h * 0.75), (w, h * 0.25), (w, h * 0.75),
    ]
    all_pts = np.array(pts + boundary_pts)
    n_circles = len(pts)

    try:
        vor = Voronoi(all_pts)
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            # Fallback: use simple distance-based method
            return _max_radius_simple(x, y, r, w, h, i)

        # Compute inradius of the Voronoi cell (max distance from center to cell boundary)
        # The cell is defined by vertices; inradius = min distance from center to edges
        # Simpler: use the vertices and find max inscribed circle
        vertices = vor.vertices[region]
        if len(vertices) < 3:
            return _max_radius_simple(x, y, r, w, h, i)

        # The inradius is the minimum distance from the center to any edge of the cell
        # For a convex polygon, this is min over edges of distance from point to line segment
        min_dist = float('inf')
        for k in range(len(vertices)):
            v1 = vertices[k]
            v2 = vertices[(k + 1) % len(vertices)]
            dist = _point_to_segment_dist(x[i], y[i], v1[0], v1[1], v2[0], v2[1])
            if dist < min_dist:
                min_dist = dist

        r_voronoi = min_dist
    except Exception:
        return _max_radius_simple(x, y, r, w, h, i)

    # Also consider other circles' radii (they can't overlap)
    r_max = r_voronoi
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed

    return max(0.0, r_max)


def _max_radius_simple(x, y, r, w, h, i):
    """Fallback: compute max radius by distance to walls and other circles."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _force_relax(z, max_iter=500):
    """Force-directed relaxation to resolve initial overlaps (vectorized)."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-8:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        if max_overlap < 1e-8:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _voronoi_expansion_polish(z, time_budget):
    """
    Polish using Voronoi-based radius computation for accurate max radius estimation.
    Iteratively expands each circle to its Voronoi-limited maximum, then adjusts positions.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            # Compute max radius using Voronoi
            r_max = _max_radius_voronoi(x, y, r, w, h, i)

            if r_max > r[i] + 1e-7:
                # Binary search for true max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(22):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h, tol=1e-10):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True
            else:
                # Try to move circle i to a position allowing larger radius
                # Find most constraining neighbor/wall
                min_gap = float('inf')
                blocker_idx = -1
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r[i] + r[j])
                    if gap < min_gap:
                        min_gap = gap
                        blocker_idx = j

                wall_gaps = [x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i]]
                min_wall_gap = min(wall_gaps)

                if min_gap < 0.015 or min_wall_gap < 0.015:
                    # Generate candidate move directions
                    directions = []
                    if blocker_idx >= 0:
                        dx = x[i] - x[blocker_idx]
                        dy = y[i] - y[blocker_idx]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 1e-8:
                            directions.append((dx / d, dy / d))

                    if wall_gaps[0] < 0.015:
                        directions.append((1.0, 0.0))
                    if wall_gaps[1] < 0.015:
                        directions.append((-1.0, 0.0))
                    if wall_gaps[2] < 0.015:
                        directions.append((0.0, 1.0))
                    if wall_gaps[3] < 0.015:
                        directions.append((0.0, -1.0))

                    # Diagonal directions
                    for dx_dir, dy_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        norm = np.sqrt(2.0)
                        directions.append((dx_dir / norm, dy_dir / norm))

                    for dx_dir, dy_dir in directions:
                        for step in [0.03, 0.015, 0.008, 0.004]:
                            new_x = x[i] + dx_dir * step
                            new_y = y[i] + dy_dir * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                break
                        if improved:
                            break

        if not improved:
            break

    # Final uniform expansion
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            r_trial = r * 1.00002
            if _is_valid(x, y, r_trial, w, h, tol=1e-10):
                r = r_trial
            else:
                break

    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _polish_result(z, time_budget):
    """Polish with force-directed position adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(3000, time_budget * 150))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 18.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    grow_factor = 1.0003
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _high_precision_polish(z, time_budget):
    """
    High-precision polish: iterative radius growth with position adjustment.
    Uses Voronoi for accurate radius bounds and adaptive step sizing.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Phase 1: Force-directed position refinement
    dt = 0.0005
    max_iter = int(min(6000, time_budget * 250))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-12:
            f_pairs = overlap * 30.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = 0.0002
        elif max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        if max_overlap < 1e-10 and max_force < 0.01:
            break

    # Phase 2: Voronoi-guided individual radius expansion
    for _ in range(60):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            r_max = _max_radius_voronoi(x, y, r, w, h, i)

            if r_max > r[i] + 1e-9:
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h, tol=1e-10):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-10:
                    r[i] = lo
                    improved = True
            else:
                # Try to move circle i
                min_gap = float('inf')
                blocker_idx = -1
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r[i] + r[j])
                    if gap < min_gap:
                        min_gap = gap
                        blocker_idx = j

                wall_gaps = [x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i]]
                min_wall_gap = min(wall_gaps)

                if min_gap < 0.01 or min_wall_gap < 0.01:
                    directions = []
                    if blocker_idx >= 0:
                        dx = x[i] - x[blocker_idx]
                        dy = y[i] - y[blocker_idx]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 1e-8:
                            directions.append((dx / d, dy / d))

                    if wall_gaps[0] < 0.01:
                        directions.append((1.0, 0.0))
                    if wall_gaps[1] < 0.01:
                        directions.append((-1.0, 0.0))
                    if wall_gaps[2] < 0.01:
                        directions.append((0.0, 1.0))
                    if wall_gaps[3] < 0.01:
                        directions.append((0.0, -1.0))

                    for dx_dir, dy_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        norm = np.sqrt(2.0)
                        directions.append((dx_dir / norm, dy_dir / norm))

                    for dx_dir, dy_dir in directions:
                        for step in [0.02, 0.01, 0.005]:
                            new_x = x[i] + dx_dir * step
                            new_y = y[i] + dy_dir * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                break
                        if improved:
                            break

        if not improved:
            break

    # Phase 3: Final simultaneous squeeze
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * 1.00005
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            r_trial = r * 1.00001
            if _is_valid(x, y, r_trial, w, h, tol=1e-10):
                r = r_trial
            else:
                break

    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = []

    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    for i in range(n):
        bounds_i = 3 * i
        constraints.append({'type': 'ineq', 'fun': lambda p, bi=bounds_i: p[bi] - p[bi + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, bi=bounds_i: p[-1] - p[bi] - p[bi + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, bi=bounds_i: p[bi + 1] - p[bi + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, bi=bounds_i: (2.0 - p[-1]) - p[bi + 1] - p[bi + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = 3 * i, 3 * j
            constraints.append({'type': 'ineq', 'fun': lambda p, bi=bi, bj=bj:
                                (p[bi] - p[bj]) ** 2 + (p[bi + 1] - p[bj + 1]) ** 2 -
                                (p[bi + 2] + p[bj + 2]) ** 2})

    maxiter = int(min(600, time_budget * 120))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - z_opt[-1]

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 142.0

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def smooth_obj(z):
        return _smooth_objective(z, beta=50.0)

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

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
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for seed in range(20):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.14)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Hexagonal packing
        for w0 in [0.9, 1.0, 1.1]:
            h0 = 2.0 - w0
            if h0 < 0.4:
                continue
            z = np.zeros(3 * N + 1)
            rows = 5
            cols = 5
            spacing_x = w0 / (cols + 0.5)
            spacing_y = h0 / (rows + 0.5)
            for i in range(N):
                row, col = divmod(i, cols)
                cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
                cy = (row + 0.5) * spacing_y
                r0 = min(spacing_x, spacing_y) * 0.48
                cx = min(max(cx, r0), w0 - r0)
                cy = min(max(cy, r0), h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Add a few more diverse structured starts
        # 4x6 grid with staggered rows
        for w0 in [1.0, 1.1, 0.95]:
            h0 = 2.0 - w0
            if h0 < 0.4:
                continue
            z = np.zeros(3 * N + 1)
            rows = 4
            cols = 6
            spacing_x = w0 / (cols + 0.3)
            spacing_y = h0 / (rows + 0.3)
            for i in range(N):
                row, col = divmod(i, cols)
                cx = (col + 0.5 + (0.4 if row % 2 == 1 else 0)) * spacing_x
                cy = (row + 0.5) * spacing_y
                r0 = min(spacing_x, spacing_y) * 0.46
                cx = min(max(cx, r0), w0 - r0)
                cy = min(max(cy, r0), h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Concentric ring packing
        for w0 in [1.0, 1.1]:
            h0 = 2.0 - w0
            if h0 < 0.4:
                continue
            z = np.zeros(3 * N + 1)
            cx_center = w0 / 2
            cy_center = h0 / 2
            # Center circle
            r_center = min(w0, h0) * 0.15
            z[0] = cx_center
            z[1] = cy_center
            z[2] = r_center
            # Ring 1: 6 circles
            ring1_r = min(w0, h0) * 0.12
            ring1_dist = r_center + ring1_r
            for k in range(6):
                idx = 1 + k
                angle = k * np.pi / 3
                cx = cx_center + ring1_dist * np.cos(angle)
                cy = cy_center + ring1_dist * np.sin(angle)
                cx = min(max(cx, ring1_r), w0 - ring1_r)
                cy = min(max(cy, ring1_r), h0 - ring1_r)
                z[3 * idx] = cx
                z[3 * idx + 1] = cy
                z[3 * idx + 2] = ring1_r
            # Ring 2: 14 circles
            ring2_r = min(w0, h0) * 0.09
            ring2_dist = ring1_dist + ring1_r + ring2_r
            for k in range(14):
                idx = 7 + k
                angle = k * 2 * np.pi / 14 + np.pi / 14
                cx = cx_center + ring2_dist * np.cos(angle)
                cy = cy_center + ring2_dist * np.sin(angle)
                cx = min(max(cx, ring2_r), w0 - ring2_r)
                cy = min(max(cy, ring2_r), h0 - ring2_r)
                z[3 * idx] = cx
                z[3 * idx + 1] = cy
                z[3 * idx + 2] = ring2_r
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        z0_relaxed = _force_relax(z0, max_iter=500)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))

        # Try both objectives for each initial guess
        for obj_func, label in [(objective, 'hard'), (smooth_obj, 'smooth')]:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(obj_func, z0_relaxed, method='SLSQP',
                               bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

                z_opt = res.x
                valid, val = check_validity(z_opt)

                if valid and val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
                elif not valid:
                    z_repair = z_opt.copy()
                    r_idx = np.arange(2, len(z_repair) - 1, 3)
                    z_repair[r_idx] *= 0.95
                    valid, val = check_validity(z_repair)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_repair.copy()

            except Exception:
                continue

        # Also try with a second smooth objective at different beta
        if time.time() > deadline - 12:
            break
        try:
            def smooth_obj_beta100(z):
                return _smooth_objective(z, beta=100.0)

            res2 = minimize(smooth_obj_beta100, z0_relaxed, method='SLSQP',
                            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                            options={'maxiter': maxiter // 2, 'ftol': 1e-12, 'maxls': 60})
            z_opt2 = res2.x
            valid, val = check_validity(z_opt2)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt2.copy()
        except Exception:
            pass

    # --- Phase 2: Polish ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Epigraph radius growth (new)
        if best_result is not None and time.time() < deadline - 18:
            epigraph = _epigraph_radius_growth(best_result, deadline - (time.time() - start_time) - 10)
            if epigraph is not None:
                valid, val = check_validity(epigraph)
                if valid and val > best_value:
                    best_value = val
                    best_result = epigraph

        # 3. Voronoi-based expansion
        if best_result is not None and time.time() < deadline - 12:
            expanded = _voronoi_expansion_polish(best_result, deadline - (time.time() - start_time) - 6)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Second epigraph pass after constraint refinement
        if best_result is not None and time.time() < deadline - 6:
            epigraph2 = _epigraph_radius_growth(best_result, deadline - (time.time() - start_time) - 3)
            if epigraph2 is not None:
                valid, val = check_validity(epigraph2)
                if valid and val > best_value:
                    best_value = val
                    best_result = epigraph2

        # 6. Final quick polish
        if best_result is not None and time.time() < deadline - 3:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

        # 7. High-precision final polish
        if best_result is not None and time.time() < deadline - 1:
            final_polished = _high_precision_polish(best_result, deadline - (time.time() - start_time))
            if final_polished is not None:
                valid, val = check_validity(final_polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = final_polished

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
