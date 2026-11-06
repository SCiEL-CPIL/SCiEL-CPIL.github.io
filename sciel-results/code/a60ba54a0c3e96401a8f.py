"""Cover the unit sphere with 50 equal caps.

Strategy:
  1. Build diverse initial configurations (Fibonacci, icosahedral-based, random).
  2. Run Voronoi-based Lloyd iteration for fast convergence.
  3. Geodesic gradient refinement targeting worst facets with tangent projection.
  4. Random perturbation escape when stuck in local minima.
  5. Micro-polish with very small steps.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 50
DEADLINE_SECONDS = 115  # safety margin below 150s hard limit


def _covering_radius_deg(pts):
    """Compute covering radius in degrees using convex hull circumcenters."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: dense sampling
        n_samples = 15000
        i_s = np.arange(n_samples) + 0.5
        z_s = 1.0 - 2.0 * i_s / n_samples
        phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
        r_s = np.sqrt(np.maximum(0, 1.0 - z_s * z_s))
        samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
        max_dist = 0.0
        chunk = 5000
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            s = samples[start:end]
            dots = s @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            min_dist = np.arccos(dots).min(axis=1)
            max_dist = max(max_dist, min_dist.max())
        return math.degrees(max_dist), None, None

    max_circ_r = 0.0
    worst_simplex = None
    worst_cc = None

    for simplex in hull.simplices:
        tri = pts[simplex]
        a, b, c = tri[0], tri[1], tri[2]

        # Spherical circumcenter via cross products
        ab = np.cross(a, b)
        bc = np.cross(b, c)
        ca = np.cross(c, a)

        cc_vec = ab + bc + ca
        cc_norm = np.linalg.norm(cc_vec)

        if cc_norm < 1e-12:
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            cc_vec = n / n_norm
            cc_norm = 1.0

        cc = cc_vec / cc_norm

        centroid = (a + b + c) / 3.0
        if np.dot(cc, centroid) < 0:
            cc = -cc

        dot_a = np.clip(np.dot(cc, a), -1.0, 1.0)
        circ_r = np.arccos(dot_a)

        if circ_r > max_circ_r:
            max_circ_r = circ_r
            worst_simplex = simplex
            worst_cc = cc

    return math.degrees(max_circ_r), worst_simplex, worst_cc


def _geodesic_step(pts, indices, target, step_size):
    """Move points at indices toward target along geodesic (tangent projection)."""
    candidate = pts.copy()
    for i in indices:
        p = pts[i]
        # Project direction onto tangent plane at p
        direction = target - np.dot(target, p) * p
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-12:
            continue
        direction = direction / norm_dir
        # Spherical rotation by step_size
        new_p = np.cos(step_size) * p + np.sin(step_size) * direction
        candidate[i] = new_p / np.linalg.norm(new_p)
    return candidate


def _fast_lloyd(pts, n_iters=25, deadline=None):
    """Voronoi-based Lloyd iteration."""
    pts = pts.copy()
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    n_samples = 6000
    i_s = np.arange(n_samples) + 0.5
    z_s = 1.0 - 2.0 * i_s / n_samples
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
    r_s = np.sqrt(np.maximum(0, 1.0 - z_s * z_s))
    samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])

    best_pts = pts.copy()
    best_cr, _, _ = _covering_radius_deg(pts)

    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break

        dots = samples @ pts.T
        nearest = np.argmax(dots, axis=1)

        new_pts = np.zeros_like(pts)
        for k in range(N):
            mask = (nearest == k)
            if mask.sum() > 0:
                new_pts[k] = samples[mask].mean(axis=0)
            else:
                new_pts[k] = pts[k]

        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        new_pts = new_pts / norms

        # Geodesic blend
        alpha = 0.4
        candidate = np.zeros_like(pts)
        for k in range(N):
            # Spherical interpolation
            dot_val = np.clip(np.dot(pts[k], new_pts[k]), -1.0, 1.0)
            theta = np.arccos(dot_val)
            if theta < 1e-10:
                candidate[k] = pts[k]
            else:
                sin_theta = np.sin(theta)
                candidate[k] = (np.sin((1 - alpha) * theta) / sin_theta) * pts[k] + \
                               (np.sin(alpha * theta) / sin_theta) * new_pts[k]
                candidate[k] = candidate[k] / np.linalg.norm(candidate[k])

        cr, _, _ = _covering_radius_deg(candidate)
        if cr < best_cr:
            best_cr = cr
            best_pts = candidate.copy()
            pts = candidate

    return best_pts, best_cr


def _gradient_refine(pts, deadline, max_iters=300):
    """Geodesic gradient refinement with perturbation escape."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr, _, _ = _covering_radius_deg(pts)

    step = 0.04
    stuck_count = 0
    rng = np.random.RandomState(42)

    for it in range(max_iters):
        if time.time() > deadline:
            break

        cr, worst_simplex, worst_cc = _covering_radius_deg(pts)
        if worst_simplex is None or worst_cc is None:
            break

        if cr < best_cr - 1e-12:
            best_cr = cr
            best_pts = pts.copy()
            stuck_count = 0
        else:
            stuck_count += 1

        # Perturbation escape when stuck
        if stuck_count > 20 and it % 5 == 0:
            candidate = pts.copy()
            for i in worst_simplex:
                noise = rng.randn(3) * 0.02
                candidate[i] = pts[i] + noise
                candidate[i] = candidate[i] / np.linalg.norm(candidate[i])
            cr_new, _, _ = _covering_radius_deg(candidate)
            if cr_new < best_cr - 1e-12:
                pts = candidate
                best_cr = cr_new
                best_pts = candidate.copy()
                stuck_count = 0
                step = max(step, 0.02)
                continue

        improved = False
        for trial_step in [step, step * 0.5, step * 0.25, step * 0.125]:
            candidate = _geodesic_step(pts, worst_simplex, worst_cc, trial_step)
            cr_new, _, _ = _covering_radius_deg(candidate)
            if cr_new < best_cr - 1e-13:
                best_cr = cr_new
                best_pts = candidate.copy()
                pts = candidate
                improved = True
                stuck_count = 0
                break

        if not improved:
            step *= 0.85
            if step < 1e-6:
                break

    return best_pts, best_cr


def _micro_polish(pts, deadline, max_iters=400):
    """Very fine geodesic polish."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr, _, _ = _covering_radius_deg(pts)

    step = 0.004
    for it in range(max_iters):
        if time.time() > deadline:
            break

        cr, worst_simplex, worst_cc = _covering_radius_deg(pts)
        if worst_simplex is None or worst_cc is None:
            break

        improved = False
        for trial_step in [step, step * 0.5, step * 0.25]:
            candidate = _geodesic_step(pts, worst_simplex, worst_cc, trial_step)
            cr_new, _, _ = _covering_radius_deg(candidate)
            if cr_new < best_cr - 1e-14:
                best_cr = cr_new
                best_pts = candidate.copy()
                pts = candidate
                improved = True
                break

        if not improved:
            step *= 0.7
            if step < 1e-8:
                break

    return best_pts, best_cr


def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedral_base(n):
    """Generate points based on icosahedral symmetry with subdivision."""
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    # Subdivide each face
    faces = [
        [0, 1, 8], [0, 8, 4], [0, 4, 5], [0, 5, 9], [0, 9, 1],
        [1, 9, 2], [1, 2, 6], [1, 6, 8],
        [2, 9, 7], [2, 7, 10], [2, 10, 6],
        [3, 4, 11], [3, 11, 7], [3, 7, 5], [3, 5, 4],
        [4, 5, 11], [5, 7, 11], [6, 8, 10], [7, 10, 11],
        [8, 6, 10], [9, 1, 2], [4, 8, 6], [5, 4, 3], [7, 9, 2]
    ]

    # Generate midpoints of edges
    midpoints = []
    seen = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                mid = (verts[a] + verts[b]) / 2.0
                mid = mid / np.linalg.norm(mid)
                midpoints.append(mid)

    all_pts = np.vstack([verts, np.array(midpoints)])

    # If we need more, add more subdivisions
    while len(all_pts) < n:
        # Add more midpoints between existing points
        new_pts = []
        for i in range(len(all_pts)):
            for j in range(i + 1, len(all_pts)):
                if np.linalg.norm(all_pts[i] - all_pts[j]) > 0.3:
                    mid = (all_pts[i] + all_pts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    new_pts.append(mid)
                if len(all_pts) + len(new_pts) >= n:
                    break
            if len(all_pts) + len(new_pts) >= n:
                break
        if not new_pts:
            break
        all_pts = np.vstack([all_pts, np.array(new_pts)])

    return all_pts[:n]


def _random_init(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _antipodal_init(n, seed):
    rng = np.random.RandomState(seed)
    half = n // 2
    pts_half = rng.randn(half, 3)
    pts_half = pts_half / np.linalg.norm(pts_half, axis=1, keepdims=True)
    pts = np.vstack([pts_half, -pts_half])
    if n % 2 == 1:
        extra = rng.randn(1, 3)
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        pts = np.vstack([pts, extra])
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    best_pts = None
    best_cr = float('inf')

    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_spiral(N))
    inits.append(_icosahedral_base(N))
    for seed in [1, 2, 3, 7, 13, 42, 99, 123]:
        inits.append(_random_init(N, seed))
    for seed in [10, 20, 30, 40]:
        inits.append(_antipodal_init(N, seed))

    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20:
            break

        # Quick validity check
        norms = np.linalg.norm(init_pts, axis=1)
        if np.any(norms < 0.01):
            continue

        # Check if origin is inside convex hull (not in closed hemisphere)
        try:
            hull = ConvexHull(init_pts)
        except Exception:
            continue

        # Phase 1: Fast Lloyd
        try:
            lloyd_pts, lloyd_cr = _fast_lloyd(init_pts, n_iters=25, deadline=deadline - 15)
        except Exception:
            continue

        if lloyd_cr < best_cr:
            best_cr = lloyd_cr
            best_pts = lloyd_pts.copy()

        # Phase 2: Gradient refinement
        if time.time() < deadline - 10:
            try:
                grad_pts, grad_cr = _gradient_refine(lloyd_pts, deadline - 5, max_iters=250)
                if grad_cr < best_cr:
                    best_cr = grad_cr
                    best_pts = grad_pts.copy()
            except Exception:
                pass

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_cr, _, _ = _covering_radius_deg(best_pts)

    # Phase 3: Micro polish
    if time.time() < deadline - 3:
        try:
            polished_pts, polished_cr = _micro_polish(best_pts, deadline - 1, max_iters=400)
            if polished_cr < best_cr:
                best_cr = polished_cr
                best_pts = polished_pts
        except Exception:
            pass

    # Final normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)

    return best_pts
