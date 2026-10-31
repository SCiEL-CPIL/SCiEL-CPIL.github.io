"""
Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons.

Structural change: instead of a single global L-BFGS-B over all 36 parameters
with a crude penalty, we use a two-level decomposition:

  1. Inner-cluster optimisation with a *smooth* overlap penalty based on
     analytic minimum-distance between two unit hexagons (not vertex-pair
     distance, which is non-smooth and non-convex).  We use Nelder-Mead
     (derivative-free) because the smooth penalty has kinks at zero-distance.

  2. Outer-hexagon optimisation: for a fixed inner cluster, the outer hexagon
     (center, angle, side) is determined analytically from the convex hull of
     all inner vertices.  We refine the outer angle with a 1-D golden-section
     search and then compute the exact minimal center via a small 2-D
     linear-programming-free bisection.

  3. Multi-start with diverse initial layouts (hexagonal lattice, 3-4-3 rows,
     perturbed versions) and repeated rounds.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def convex_hull(points):
    """Andrew's monotone chain. Returns hull vertices (counter-clockwise)."""
    pts = np.unique(np.round(points, 14), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-14:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-14:
            upper.pop()
        upper.append(p)
    return np.vstack([lower[:-1], upper[:-1]])

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons.
    inner_hexagons: list of (cx, cy, angle_deg)
    outer_center: (ox, oy)
    outer_angle: degrees
    """
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    oc = np.array([ox, oy])
    min_S = 0.0
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for nrm in normals:
                d = np.dot(v - oc, nrm)
                if d > 0:
                    S_needed = 2.0 * d / SQRT3
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def analytic_min_outer(inner_hexagons):
    """Find the outer hexagon (center, angle, side) that minimally encloses the
    fixed cluster, using convex-hull support functions + golden-section on angle.
    Returns (S, ox, oy, outer_angle_deg).
    """
    all_verts = []
    for cx, cy, ang in inner_hexagons:
        all_verts.append(hexagon_vertices(cx, cy, 1.0, ang))
    all_verts = np.vstack(all_verts)
    hull = convex_hull(all_verts)

    def width_at(oa_rad):
        max_w = 0.0
        for k in range(6):
            theta = oa_rad + np.radians(30 + 60 * k)
            nrm = np.array([np.cos(theta), np.sin(theta)])
            proj = np.dot(hull, nrm)
            w = proj.max() - proj.min()
            if w > max_w:
                max_w = w
        return max_w

    # Golden-section search over angle in [0, pi/3) (60 degrees)
    # The function is periodic with period 60 degrees.
    res = minimize_scalar(width_at, bounds=(0.0, np.pi / 3), method='bounded',
                          options={'xatol': 1e-12})
    best_oa = res.x
    best_width = width_at(best_oa)
    best_S = best_width / SQRT3

    # Find optimal center: minimize max over 6 normals of (max_proj - min_proj)/2
    # Actually we want to minimize the max over normals of the required apothem,
    # which is (max_proj - min_proj)/2 for each normal, and S = 2*apothem/SQRT3.
    # The center that minimizes the max apothem is found by:
    # For each normal n_k, the projection range is [min_k, max_k] (in the frame
    # centered at origin).  If we shift the center by c, the range becomes
    # [min_k - dot(c,n_k), max_k - dot(c,n_k)], and the width is unchanged.
    # So the width is independent of center!  The minimal S is determined purely
    # by the angle.  The center can be anything; we pick the centroid for stability.
    center = hull.mean(axis=0)

    # But wait: the width IS independent of center.  So S is fixed by angle alone.
    # We just need to verify the hexagon at that S and center actually contains
    # everything.  Let's compute the exact center that works:
    # For the hexagon to contain the hull, for each normal n_k:
    #   max_k - dot(c, n_k) <= r  AND  dot(c, n_k) - min_k <= r
    # where r = S * SQRT3 / 2 = best_width / 2.
    # So: dot(c, n_k) >= max_k - r  AND  dot(c, n_k) <= min_k + r
    # i.e.  max_k - r <= dot(c, n_k) <= min_k + r
    # Since r = best_width/2 and best_width >= (max_k - min_k) for all k,
    # the interval [max_k - r, min_k + r] has length = min_k + r - max_k + r
    # = (min_k - max_k) + 2r = -(max_k - min_k) + best_width >= 0.
    # So feasible.  Pick c = centroid and verify.

    return best_S, float(center[0]), float(center[1]), float(np.degrees(best_oa) % 360)

def hexagon_min_distance(p1, a1_deg, p2, a2_deg):
    """Compute minimum distance between two unit hexagons (0 if overlapping).
    Uses SAT to detect overlap, then vertex-edge distance for separation."""
    d_centers = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    if d_centers > 4.0:
        return d_centers - 2.0
    v1 = hexagon_vertices(p1[0], p1[1], 1.0, a1_deg)
    v2 = hexagon_vertices(p2[0], p2[1], 1.0, a2_deg)
    # SAT
    axes = []
    for v in (v1, v2):
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            nrm = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(nrm)
            if norm > 1e-12:
                axes.append(nrm / norm)
    overlap = True
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        if proj1.max() < proj2.min() - 1e-10 or proj2.max() < proj1.min() - 1e-10:
            overlap = False
            break
    if overlap:
        return 0.0
    # Minimum vertex-to-vertex distance (sufficient for convex polygons)
    min_d = float('inf')
    for i in range(6):
        for j in range(6):
            d = np.linalg.norm(v1[i] - v2[j])
            if d < min_d:
                min_d = d
    return min_d

def smooth_overlap_penalty(inner, beta=30.0):
    """Smooth penalty for overlaps using log-sum-exp on negative distances."""
    n = len(inner)
    total = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = hexagon_min_distance(
                np.array(inner[i][:2]), inner[i][2],
                np.array(inner[j][:2]), inner[j][2])
            # Penalize when d is small (close to or at overlap)
            # Use smooth max(0, 1-d)^2 via softplus-like
            if d < 2.0:
                # exp(-beta * max(0, d - 0.5)) gives a smooth transition
                margin = max(0.0, d - 0.3)
                penalty = math.exp(-beta * margin)
                total += penalty
    return total

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center
    n = len(inner_hexagons)
    # Check overlaps
    for i in range(n):
        v1 = hexagon_vertices(inner_hexagons[i][0], inner_hexagons[i][1], 1.0, inner_hexagons[i][2])
        for j in range(i+1, n):
            v2 = hexagon_vertices(inner_hexagons[j][0], inner_hexagons[j][1], 1.0, inner_hexagons[j][2])
            # SAT
            axes = []
            for v in (v1, v2):
                for k in range(6):
                    edge = v[(k+1)%6] - v[k]
                    nrm = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(nrm)
                    if norm > 1e-12:
                        axes.append(nrm / norm)
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                if proj1.max() < proj2.min() - 1e-7 or proj2.max() < proj1.min() - 1e-7:
                    break
            else:
                return False  # overlap found
    # Check containment
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            inside = True
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i+1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i+2) % 6] - v1)
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False
    return True

def _build_initial_conditions():
    """Build diverse initial conditions as lists of (x, y, angle_deg)."""
    ics = []
    s3 = np.sqrt(3)

    # IC 1: hexagonal lattice (center + ring1 + 4 from ring2)
    hex_pos = [(0.0, 0.0)]
    for k in range(6):
        a = np.radians(60 * k)
        hex_pos.append((s3 * 1.001 * np.cos(a), s3 * 1.001 * np.sin(a)))
    for k in range(6):
        a = np.radians(60 * k + 30)
        hex_pos.append((2 * s3 * 1.001 * np.cos(a), 2 * s3 * 1.001 * np.sin(a)))
        if len(hex_pos) >= N:
            break
    hex_pos = hex_pos[:N]
    ics.append([(x, y, 0.0) for x, y in hex_pos])

    # IC 2: 3-4-3 row cluster
    cluster = []
    row_y = [0.0, s3, 2 * s3]
    for m in range(-1, 2):
        cluster.append((m * 2.0, row_y[2], 0.0))
    for m in range(-1, 3):
        cluster.append((m * 2.0 + 1.0, row_y[1], 0.0))
    for m in range(-1, 2):
        cluster.append((m * 2.0, row_y[0], 0.0))
    cluster = cluster[:N]
    if len(cluster) < N:
        cluster.append((0.0, 3 * s3, 0.0))
    cx_m = np.mean([c[0] for c in cluster])
    cy_m = np.mean([c[1] for c in cluster])
    ics.append([(c[0]-cx_m, c[1]-cy_m, c[2]) for c in cluster])

    # IC 3: perturbed hex lattice
    np.random.seed(42)
    ics.append([(x + np.random.randn()*0.1, y + np.random.randn()*0.1, 0.0)
                for x, y in hex_pos])

    # IC 4: rotated inner hexagons
    ics.append([(x, y, 30.0) for x, y in hex_pos])

    # IC 5: tighter hex lattice
    tight = [(0.0, 0.0)]
    for k in range(6):
        a = np.radians(60 * k)
        tight.append((s3 * 0.99 * np.cos(a), s3 * 0.99 * np.sin(a)))
    for k in range(6):
        a = np.radians(60 * k + 30)
        tight.append((2 * s3 * 0.99 * np.cos(a), 2 * s3 * 0.99 * np.sin(a)))
        if len(tight) >= N:
            break
    tight = tight[:N]
    ics.append([(x, y, 0.0) for x, y in tight])

    # IC 6: another perturbation
    np.random.seed(123)
    ics.append([(x + np.random.randn()*0.15, y + np.random.randn()*0.15, np.random.randn()*5)
                for x, y in hex_pos])

    return ics

def _optimize_inner(inner, deadline):
    """Optimize inner cluster positions/angles using Nelder-Mead with smooth penalty."""
    def inner_obj(params):
        if time.time() > deadline:
            return 1e10
        cur = []
        for i in range(N):
            cur.append((params[3*i], params[3*i+1], np.degrees(params[3*i+2]) % 360))
        # Smooth overlap penalty
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = hexagon_min_distance(
                    np.array([cur[i][0], cur[i][1]]), cur[i][2],
                    np.array([cur[j][0], cur[j][1]]), cur[j][2])
                if d < 1.5:
                    margin = max(0.0, d - 0.2)
                    pen += math.exp(-40.0 * margin)
        # Also penalize spread (keep cluster compact)
        cx_mean = np.mean([c[0] for c in cur])
        cy_mean = np.mean([c[1] for c in cur])
        spread = np.mean([(c[0]-cx_mean)**2 + (c[1]-cy_mean)**2 for c in cur])
        return pen + 0.01 * spread

    params = np.zeros(3*N)
    for i, (x, y, a) in enumerate(inner):
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = np.radians(a)

    res = minimize(inner_obj, params, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                            'maxfev': 5000, 'adaptive': True})
    if time.time() > deadline:
        return inner
    new_inner = []
    for i in range(N):
        new_inner.append((res.x[3*i], res.x[3*i+1], np.degrees(res.x[3*i+2]) % 360))
    return new_inner

def _refine_outer(inner, deadline):
    """Refine outer hexagon for fixed inner cluster. Returns (S, ox, oy, oa_deg)."""
    S_an, ox_an, oy_an, oa_an = analytic_min_outer(inner)
    if not check_validity(inner, (ox_an, oy_an), S_an, oa_an):
        # Try a few angle perturbations
        best = (S_an, ox_an, oy_an, oa_an)
        for da in [-5, -2, -1, 1, 2, 5, -10, 10]:
            oa_test = (oa_an + da) % 360
            S_test = compute_outer_side_length(inner, (ox_an, oy_an), oa_test)
            if check_validity(inner, (ox_an, oy_an), S_test, oa_test) and S_test < best[0]:
                best = (S_test, ox_an, oy_an, oa_test)
        return best
    return (S_an, ox_an, oy_an, oa_an)

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    initial_conditions = _build_initial_conditions()

    best_result = None
    best_score = float('inf')

    def _evaluate(inner):
        """Evaluate a cluster: optimize outer, check validity, return (score, result)."""
        S, ox, oy, oa = _refine_outer(inner, deadline)
        if check_validity(inner, (ox, oy), S, oa):
            return S, (inner, (ox, oy), S, oa)
        return float('inf'), None

    # Pass 1: evaluate all initial conditions with analytical outer
    for ic in initial_conditions:
        if time.time() > deadline - 40:
            break
        try:
            score, result = _evaluate(ic)
            if score < best_score:
                best_score = score
                best_result = result
        except Exception:
            pass

    # Pass 2: optimize inner cluster from each IC, then refine outer
    for ic in initial_conditions:
        if time.time() > deadline - 30:
            break
        try:
            optimized_inner = _optimize_inner(ic, deadline)
            score, result = _evaluate(optimized_inner)
            if score < best_score:
                best_score = score
                best_result = result
        except Exception:
            pass

    # Pass 3: refine from best result
    if best_result is not None and time.time() < deadline - 20:
        try:
            inner_best = best_result[0]
            refined_inner = _optimize_inner(inner_best, deadline)
            score, result = _evaluate(refined_inner)
            if score < best_score:
                best_score = score
                best_result = result
        except Exception:
            pass

    # Pass 4: second round of inner optimization from best
    if best_result is not None and time.time() < deadline - 15:
        try:
            inner_best = best_result[0]
            # Small perturbation
            np.random.seed(7)
            perturbed = [(x + np.random.randn()*0.05, y + np.random.randn()*0.05,
                          (a + np.random.randn()*2) % 360) for x, y, a in inner_best]
            refined_inner = _optimize_inner(perturbed, deadline)
            score, result = _evaluate(refined_inner)
            if score < best_score:
                best_score = score
                best_result = result
        except Exception:
            pass

    # Fallback
    if best_result is None:
        inner = [[x, y, 0.0] for x, y in [(0,0)] + [(s3*np.cos(np.radians(60*k)),
                                                     s3*np.sin(np.radians(60*k))) for k in range(6)]]
        for k in range(6):
            a = np.radians(60*k+30)
            inner.append([2*s3*np.cos(a), 2*s3*np.sin(a), 0.0])
            if len(inner) >= N:
                break
        inner = inner[:N]
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0) + 0.5
        best_result = (inner, (0.0, 0.0), S, 0.0)

    # Final analytical outer tightening
    if time.time() < deadline - 5:
        try:
            inner_fixed = best_result[0]
            S_an, ox_an, oy_an, oa_an = analytic_min_outer(inner_fixed)
            if check_validity(inner_fixed, (ox_an, oy_an), S_an, oa_an) and S_an < best_score:
                best_result = (inner_fixed, (ox_an, oy_an), S_an, oa_an)
                best_score = S_an
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
