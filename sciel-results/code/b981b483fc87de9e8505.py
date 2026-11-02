"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
_H = np.sqrt(3.0) / 2.0
_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _H]])
_AREA_TRI = _H / 2.0  # area of the unit equilateral triangle

# Precompute all triplets of indices
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIP = len(_TRIPLETS)


def _bary_to_cart(bary):
    """Convert barycentric coordinates (n,3) to cartesian (n,2)."""
    # bary columns: weights for vertices v0, v1, v2
    return bary @ _VERTICES


def _cart_to_bary(pts):
    """Convert cartesian (n,2) to barycentric (n,3) relative to the triangle."""
    # Solve: p = b0*v0 + b1*v1 + b2*v2, b0+b1+b2=1
    # Using the standard formula for equilateral triangle with vertices (0,0),(1,0),(0.5,H)
    x = pts[:, 0]
    y = pts[:, 1]
    b1 = y / _H
    b2 = x - 0.5 * b1
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _min_area(pts):
    """Compute the minimum triangle area among all triplets."""
    a, b, c = _TRIPLETS[:, 0], _TRIPLETS[:, 1], _TRIPLETS[:, 2]
    pa, pb, pc = pts[a], pts[b], pts[c]
    areas = 0.5 * np.abs(pa[:, 0] * (pb[:, 1] - pc[:, 1]) +
                         pb[:, 0] * (pc[:, 1] - pa[:, 1]) +
                         pc[:, 0] * (pa[:, 1] - pb[:, 1]))
    return np.min(areas)


def _soft_min(areas, alpha):
    """Smooth approximation of min via log-sum-exp."""
    m = np.min(areas)
    # Use: -log(sum(exp(-alpha*(a_i - m))))/alpha + m
    # To avoid overflow, shift by min
    shifted = -alpha * (areas - m)
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / alpha


def _objective_and_grad(x, alpha):
    """
    Objective: minimize -soft_min(areas) to maximize min area.
    x: barycentric coordinates flattened, shape (3*N,)
    Returns (value, gradient).
    """
    bary = x.reshape(N, 3)
    pts = _bary_to_cart(bary)

    a, b, c = _TRIPLETS[:, 0], _TRIPLETS[:, 1], _TRIPLETS[:, 2]
    pa, pb, pc = pts[a], pts[b], pts[c]

    # Signed double area (without 0.5 factor for simplicity, sign matters)
    # d = pa_x*(pb_y - pc_y) + pb_x*(pc_y - pa_y) + pc_x*(pa_y - pb_y)
    d = (pa[:, 0] * (pb[:, 1] - pc[:, 1]) +
         pb[:, 0] * (pc[:, 1] - pa[:, 1]) +
         pc[:, 0] * (pa[:, 1] - pb[:, 1]))
    areas = 0.5 * np.abs(d)

    # Soft min
    m = np.min(areas)
    shifted = -alpha * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    exp_vals = np.exp(shifted)
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = m - log_sum / alpha

    # Objective is -soft_min_val (we minimize negative of min area)
    obj = -soft_min_val

    # Gradient computation
    # d(soft_min)/d(area_i) = -exp(shifted_i) / sum(exp(shifted))
    # d(areas)/d(pts) then d(pts)/d(bary) then d(bary)/d(x)
    # areas = 0.5 * |d|, so d(area)/d(pts) = 0.5 * sign(d) * d(d)/d(pts)
    # For the soft min gradient, we use the weighted average

    weights = exp_vals / np.sum(exp_vals)  # shape (_N_TRIP,)

    # d(soft_min)/d(area_i) = -weights[i]
    # d(area_i)/d(pt_j) for each point j in triplet i
    # area_i = 0.5 * |d_i|
    # d(d_i)/d(pa) = (0, pb_y - pc_y, pc_x - pb_x) ... wait, let me be more careful

    # d_i = pa_x*(pb_y - pc_y) + pb_x*(pc_y - pa_y) + pc_x*(pa_y - pb_y)
    # d(d_i)/d(pa_x) = pb_y - pc_y
    # d(d_i)/d(pa_y) = pc_x - pb_x
    # d(d_i)/d(pb_x) = pc_y - pa_y
    # d(d_i)/d(pb_y) = pa_x - pc_x
    # d(d_i)/d(pc_x) = pa_y - pb_y
    # d(d_i)/d(pc_y) = pb_x - pa_x

    # d(area_i)/d(pts) = 0.5 * sign(d_i) * d(d_i)/d(pts)
    sign_d = np.sign(d)
    # For soft min, the gradient w.r.t. area is -weights
    # So d(soft_min)/d(pts_k) = sum_i weights[i] * d(area_i)/d(pts_k)
    # = -sum_i weights[i] * 0.5 * sign(d_i) * d(d_i)/d(pts_k)

    # Initialize gradient w.r.t. points
    grad_pts = np.zeros_like(pts)  # (N, 2)

    for i in range(_N_TRIP):
        w = -weights[i] * 0.5 * sign_d[i]
        if w == 0:
            continue
        ai, bi, ci = a[i], b[i], c[i]
        # d(d_i)/d(pa) = (pb_y - pc_y, pc_x - pb_x)
        grad_pts[ai, 0] += w * (pts[bi, 1] - pts[ci, 1])
        grad_pts[ai, 1] += w * (pts[ci, 0] - pts[bi, 0])
        # d(d_i)/d(pb) = (pc_y - pa_y, pa_x - pc_x)
        grad_pts[bi, 0] += w * (pts[ci, 1] - pts[ai, 1])
        grad_pts[bi, 1] += w * (pts[ai, 0] - pts[ci, 0])
        # d(d_i)/d(pc) = (pa_y - pb_y, pb_x - pa_x)
        grad_pts[ci, 0] += w * (pts[ai, 1] - pts[bi, 1])
        grad_pts[ci, 1] += w * (pts[bi, 0] - pts[ai, 0])

    # Chain rule: pts = bary @ VERTICES, so d(pts)/d(bary) = VERTICES^T
    # grad_bary = grad_pts @ VERTICES
    grad_bary = grad_pts @ _VERTICES  # (N, 3)
    grad_x = grad_bary.flatten()

    # Objective is -soft_min, so gradient is -grad(soft_min)
    # We computed grad(soft_min) w.r.t. x as grad_x above
    # So grad(obj) = -grad_x
    return obj, -grad_x


def _objective(x, alpha):
    val, _ = _objective_and_grad(x, alpha)
    return val


def _make_bounds():
    """Bounds for barycentric coordinates: each in [0, 1], and sum=1 is handled by structure."""
    # Actually, we need b0+b1+b2=1. Let's use only 2 free barycentric coords per point.
    # Let's reformulate: use b1, b2 as free, b0 = 1 - b1 - b2
    # Constraints: b1 >= 0, b2 >= 0, b1 + b2 <= 1
    # For L-BFGS-B, we can use bounds on b1, b2 in [0, 1] and add a constraint
    # But simpler: use bounds [0,1] for b1 and b2, and clip b0, then renormalize?
    # Actually, let's just use 2D cartesian coordinates with triangle constraint.
    # But that's harder. Let's stick with barycentric but use only 2 components.

    # Alternative: use b1, b2 in [0, 1] with the constraint b1 + b2 <= 1
    # For L-BFGS-B, we can't easily do inequality constraints.
    # Let's use a different parameterization: let u, v be free, and b1 = u, b2 = v, b0 = 1-u-v
    # with u, v in [0, 1] and u + v in [0, 1]
    # We can use bounds u in [0,1], v in [0,1] and add a penalty if u+v > 1

    # Actually, let's just use cartesian coordinates and check validity.
    # Or use barycentric with 2 free params and trust the optimizer stays in bounds.

    # Let's go with: x has 2*N entries, representing (b1, b2) for each point.
    # b0 = 1 - b1 - b2
    # Bounds: b1 in [0, 1], b2 in [0, 1]
    # We'll add a large penalty if b0 < 0 (i.e., b1 + b2 > 1)
    bounds = [(0.0, 1.0)] * (2 * N)
    return bounds


def _bary2_to_cart(b12):
    """Convert (b1, b2) pairs to cartesian. b0 = 1 - b1 - b2."""
    b1 = b12[0::2]
    b2 = b12[1::2]
    b0 = 1.0 - b1 - b2
    bary = np.column_stack([b0, b1, b2])
    return _bary_to_cart(bary)


def _objective2(x, alpha):
    """Objective with penalty for going outside triangle."""
    pts = _bary2_to_cart(x)
    b1 = x[0::2]
    b2 = x[1::2]
    b0 = 1.0 - b1 - b2

    # Penalty for b0 < 0
    pen = 0.0
    if np.any(b0 < 0):
        pen = 100.0 * np.sum(np.clip(-b0, 0, None) ** 2)

    a, b, c = _TRIPLETS[:, 0], _TRIPLETS[:, 1], _TRIPLETS[:, 2]
    pa, pb, pc = pts[a], pts[b], pts[c]
    d = (pa[:, 0] * (pb[:, 1] - pc[:, 1]) +
         pb[:, 0] * (pc[:, 1] - pa[:, 1]) +
         pc[:, 0] * (pa[:, 1] - pb[:, 1]))
    areas = 0.5 * np.abs(d)

    m = np.min(areas)
    shifted = -alpha * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / alpha

    return -soft_min_val + pen


def _objective2_grad(x, alpha):
    pts = _bary2_to_cart(x)
    b1 = x[0::2]
    b2 = x[1::2]
    b0 = 1.0 - b1 - b2

    a, b, c = _TRIPLETS[:, 0], _TRIPLETS[:, 1], _TRIPLETS[:, 2]
    pa, pb, pc = pts[a], pts[b], pts[c]
    d = (pa[:, 0] * (pb[:, 1] - pc[:, 1]) +
         pb[:, 0] * (pc[:, 1] - pa[:, 1]) +
         pc[:, 0] * (pa[:, 1] - pb[:, 1]))
    areas = 0.5 * np.abs(d)

    m = np.min(areas)
    shifted = -alpha * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    exp_vals = np.exp(shifted)
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = m - log_sum / alpha
    weights = exp_vals / np.sum(exp_vals)

    sign_d = np.sign(d)
    grad_pts = np.zeros_like(pts)

    for i in range(_N_TRIP):
        w = -weights[i] * 0.5 * sign_d[i]
        if w == 0:
            continue
        ai, bi, ci = a[i], b[i], c[i]
        grad_pts[ai, 0] += w * (pts[bi, 1] - pts[ci, 1])
        grad_pts[ai, 1] += w * (pts[ci, 0] - pts[bi, 0])
        grad_pts[bi, 0] += w * (pts[ci, 1] - pts[ai, 1])
        grad_pts[bi, 1] += w * (pts[ai, 0] - pts[ci, 0])
        grad_pts[ci, 0] += w * (pts[ai, 1] - pts[bi, 1])
        grad_pts[ci, 1] += w * (pts[bi, 0] - pts[ai, 0])

    # Chain rule: pts = bary @ VERTICES, bary = [1-b1-b2, b1, b2]
    # d(pts)/d(b1) = VERTICES[1] - VERTICES[0]
    # d(pts)/d(b2) = VERTICES[2] - VERTICES[0]
    # So grad_b1 = grad_pts @ (VERTICES[1] - VERTICES[0])
    #    grad_b2 = grad_pts @ (VERTICES[2] - VERTICES[0])
    db1 = _VERTICES[1] - _VERTICES[0]
    db2 = _VERTICES[2] - _VERTICES[0]
    grad_b1 = grad_pts @ db1
    grad_b2 = grad_pts @ db2

    # Penalty gradient
    pen_grad_b1 = np.zeros(N)
    pen_grad_b2 = np.zeros(N)
    mask = b0 < 0
    if np.any(mask):
        pen_grad_b1[mask] = -200.0 * (-b0[mask])
        pen_grad_b2[mask] = -200.0 * (-b0[mask])

    grad = np.zeros(2 * N)
    grad[0::2] = -grad_b1 - pen_grad_b1  # objective is -soft_min + pen
    grad[1::2] = -grad_b2 - pen_grad_b2

    obj = -soft_min_val + (100.0 * np.sum(np.clip(-b0, 0, None) ** 2) if np.any(b0 < 0) else 0.0)
    return obj, grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_score = 0.0
    best_points = None

    # Initial points: use a reasonable starting configuration
    # Place points in a grid-like pattern inside the triangle
    def _initial_points(seed):
        rng = np.random.RandomState(seed)
        # Start with points spread out
        pts = []
        # Add some fixed good positions and random ones
        # Center
        pts.append([0.5, _H / 3.0])
        # Midpoints of sides
        pts.append([0.5, 0.0])
        pts.append([0.25, _H / 2.0])
        pts.append([0.75, _H / 2.0])
        # Corners inset
        pts.append([0.1, 0.05])
        pts.append([0.9, 0.05])
        pts.append([0.5, _H * 0.8])
        # Random
        for _ in range(N - len(pts)):
            for _ in range(100):
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    pts.append([u + 0.5 * v, _H * v])
                    break
        return np.array(pts[:N])

    # Try multiple restarts with different seeds and alpha values
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    alphas = [50.0, 100.0, 200.0]

    for seed in seeds:
        if time.time() > deadline:
            break
        for alpha in alphas:
            if time.time() > deadline:
                break

            pts0 = _initial_points(seed)
            # Convert to barycentric (b1, b2)
            bary = _cart_to_bary(pts0)
            x0 = np.zeros(2 * N)
            x0[0::2] = bary[:, 1]
            x0[1::2] = bary[:, 2]

            try:
                res = minimize(
                    _objective2,
                    x0,
                    jac=_objective2_grad,
                    method='L-BFGS-B',
                    bounds=_make_bounds(),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
            except Exception:
                continue

            x_opt = res.x
            pts_opt = _bary2_to_cart(x_opt)

            # Check if points are inside triangle
            bary_opt = _cart_to_bary(pts_opt)
            if np.any(bary_opt < -1e-6):
                # Clip to boundary
                bary_opt = np.clip(bary_opt, 0, None)
                # Renormalize
                sums = bary_opt.sum(axis=1, keepdims=True)
                bary_opt = bary_opt / sums
                pts_opt = _bary_to_cart(bary_opt)

            score = _min_area(pts_opt) / _AREA_TRI
            if score > best_score:
                best_score = score
                best_points = pts_opt.copy()

    if best_points is None:
        # Fallback to seed program
        import random
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, _H * v])
        best_points = np.array(points)
        best_score = _min_area(best_points) / _AREA_TRI

    return best_points
# EVOLVE-BLOCK-END
