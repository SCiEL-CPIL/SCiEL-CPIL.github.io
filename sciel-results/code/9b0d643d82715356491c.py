import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _build_two_ring(n_out, r_inner, angle_offset, radial_pert=None, angular_pert=None):
    """
    Two-ring configuration:
      Outer ring: n_out points at radius 1.0, angles 0, 2pi/n_out, ...
      Inner ring: n_in points at radius r_inner, angles angle_offset, angle_offset + 2pi/n_in, ...
    Optional per-point radial/angular perturbations (arrays of length n_out / n_in).
    """
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)

    r_out = np.ones(n_out)
    r_in = np.full(n_in, r_inner)

    if radial_pert is not None:
        rp_out, rp_in = radial_pert[:n_out], radial_pert[n_out:]
        r_out = r_out * (1.0 + rp_out)
        r_in = r_in * (1.0 + rp_in)
    if angular_pert is not None:
        ap_out, ap_in = angular_pert[:n_out], angular_pert[n_out:]
        a_out = a_out + ap_out
        a_in = a_in + ap_in

    outer = np.column_stack([r_out * np.cos(a_out), r_out * np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([outer, inner])


def _build_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _build_regular_perturbed(base, radial_pert, angular_pert):
    """Base regular N-gon with per-point radial and angular perturbations."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + angular_pert
    radii = 1.0 + radial_pert
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _optimize_from_points(points, deadline, max_iters_nm=500):
    """Polish a point configuration using Nelder-Mead on the true objective."""
    best = points.copy()
    best_s = _score(best)

    x0 = best.flatten()

    # Nelder-Mead fine polish
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                x0,
                method='Nelder-Mead',
                options={'maxiter': max_iters_nm, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin under 150s kill

    best_points = None
    best_score = 0.0

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # 1. Structured Starts
    # Regular 13-gon
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    _update(np.column_stack([np.cos(ang), np.sin(ang)]))
    
    # Two-ring candidates (quick evaluation without heavy optimization first)
    # We will polish these later
    candidates = []
    for n_out in [5, 6, 7]:
        n_in = N - n_out
        for r_in in [0.5, 0.7]:
            for a_off in [0.0, np.pi / n_out]:
                pts = _build_two_ring(n_out, r_in, a_off)
                candidates.append(pts)
                
    # Three-ring candidates
    for r2, r3 in [(0.6, 0.3), (0.7, 0.4)]:
        for o2, o3 in [(0.0, 0.0), (np.pi/6, 0.0)]:
             # 5-4-4 split
             pts = _build_three_ring(5, 4, 4, r2, r3, o2, o3)
             candidates.append(pts)

    # 2. Random Starts
    # Generate random points in the unit disk
    np.random.seed(42) # For reproducibility in this block, though evaluator might vary
    n_random_starts = 20
    for _ in range(n_random_starts):
        # Random polar coordinates
        radii = np.sqrt(np.random.rand(N))
        angles = np.random.rand(N) * 2 * np.pi
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)

    # 3. Polish all candidates
    # Sort candidates by initial score to prioritize good ones? 
    # Or just iterate and check time.
    
    # Evaluate initial scores to sort
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    for s, pts in scored_candidates:
        if time.time() > deadline - 5:
            break
        # If the candidate is already very good, skip heavy polish?
        # No, always polish.
        try:
            polished_pts, polished_s = _optimize_from_points(pts, deadline, max_iters_nm=300)
            _update(polished_pts)
        except Exception:
            pass

    # 4. Final Polish on the best found
    if best_points is not None and time.time() < deadline - 2:
        try:
            # Use a more aggressive polish
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(ang), np.sin(ang)])

    return best_points
