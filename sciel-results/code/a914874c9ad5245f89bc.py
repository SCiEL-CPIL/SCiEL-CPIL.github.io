"""Improved program: place 14 points in 3D to minimize (max/min pairwise distance)^2.

Combines the coordinate-space L-BFGS-B refinement from the current program with
the log-ratio soft-max/soft-min objective and progressive tau schedule from the
inspiration.  Also adds structured initializations (icosahedron + random extras,
Fibonacci sphere, random sphere, two-ring) and a short Gram-matrix polish pass.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3
DEADLINE = 145.0  # safety margin before 150s kill

def _pairwise_dists_sq(pts):
    """Return all pairwise squared distances (upper triangle), shape (n_pairs,)."""
    n = len(pts)
    idx_i, idx_j = np.triu_indices(n, k=1)
    diffs = pts[idx_i] - pts[idx_j]
    return np.sum(diffs * diffs, axis=1)

def _score(pts):
    """Compute (max_dist / min_dist)^2."""
    d2 = _pairwise_dists_sq(pts)
    dmin = np.min(d2)
    dmax = np.max(d2)
    if dmin < 1e-30:
        return 1e12
    return (dmax / dmin) ** 2

def _objective_log_ratio(pts_flat, tau=50.0):
    """
    Smooth log-ratio objective: soft_max(log d^2) - soft_min(log d^2).
    Centering and normalizing remove translation/scale invariance.
    """
    pts = pts_flat.reshape(-1, D)
    # Center
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    d2 = _pairwise_dists_sq(pts)
    mean_d2 = np.mean(d2)
    if mean_d2 < 1e-12:
        return 1e10
    # Normalize so mean squared distance = 1
    pts = pts / np.sqrt(mean_d2)
    d2 = _pairwise_dists_sq(pts)
    d2 = np.maximum(d2, 1e-15)
    log_d2 = np.log(d2)

    # Soft max
    m_max = np.max(log_d2)
    exp_max = np.exp(tau * (log_d2 - m_max))
    soft_max = m_max + np.log(np.sum(exp_max)) / tau

    # Soft min
    m_min = np.min(log_d2)
    exp_min = np.exp(-tau * (log_d2 - m_min))
    soft_min = m_min - np.log(np.sum(exp_min)) / tau

    return soft_max - soft_min

def _objective_ratio(pts_flat):
    """Direct (max/min)^2 objective with distinctness penalty."""
    pts = pts_flat.reshape(-1, D)
    d2 = _pairwise_dists_sq(pts)
    dmin = np.min(d2)
    dmax = np.max(d2)
    if dmin < 1e-20:
        return 1e12
    ratio_sq = (dmax / dmin) ** 2
    penalty = 0.0
    for v in d2:
        if v < 1e-6:
            penalty += (1e-6 - v) ** 2 * 1e6
    return ratio_sq + penalty

def _optimize(pts, objective, args, max_iter=2000, tol=1e-10):
    """Run L-BFGS-B from pts using given objective."""
    x0 = pts.flatten()
    try:
        res = minimize(objective, x0, args=args, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        return res.x.reshape(-1, D)
    except Exception:
        return pts

def _icosahedron_init():
    """12 icosahedron vertices + 2 random sphere points."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 points on sphere
    rng = np.random.default_rng(123)
    extra = rng.normal(size=(2, D))
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    pts = np.vstack([ico, extra])
    # Random rotation
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def _fibonacci_sphere(n):
    """Fibonacci sphere with n points."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    pts = np.zeros((n, 3))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(1 - z * z)
        theta = golden_angle * i
        pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return pts

def _random_sphere(n, rng):
    """n random points on unit sphere."""
    pts = rng.normal(size=(n, D))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _two_ring_init():
    """Two rings of 7 points at different z levels."""
    ring1_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ring2_angles = ring1_angles + np.pi / 7
    ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.zeros(7)])
    ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, 0.5)])
    pts = np.vstack([ring1, ring2])
    rng = np.random.default_rng(456)
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def _tetrahedron_init():
    """Tetrahedron vertices + edge midpoints + face centers = 14 points."""
    v0 = np.array([1, 1, 1], dtype=float)
    v1 = np.array([1, -1, -1], dtype=float)
    v2 = np.array([-1, 1, -1], dtype=float)
    v3 = np.array([-1, -1, 1], dtype=float)
    scale = 1.0 / (2.0 * np.sqrt(2.0))
    verts = np.array([v0, v1, v2, v3]) * scale
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    mids = np.array([(verts[i] + verts[j]) / 2.0 for i, j in edges])
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    face_centers = np.array([(verts[i] + verts[j] + verts[k]) / 3.0 for i, j, k in faces])
    return np.vstack([verts, mids, face_centers])

def construct():
    t_start = time.time()
    best_pts = None
    best_score = 1e12

    def _update(pts):
        nonlocal best_pts, best_score
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        d2 = _pairwise_dists_sq(pts)
        if np.min(d2) < 1e-10:
            return
        sc = _score(pts)
        if sc < best_score:
            best_score = sc
            best_pts = pts.copy()

    rng = np.random.default_rng(42)

    # Generate diverse initial candidates
    candidates = []

    # 1. Icosahedron + 2 random
    candidates.append(_icosahedron_init())

    # 2. Fibonacci sphere
    candidates.append(_fibonacci_sphere(N))

    # 3. Random sphere points (a few)
    for _ in range(4):
        candidates.append(_random_sphere(N, rng))

    # 4. Two-ring
    candidates.append(_two_ring_init())

    # 5. Tetrahedron-based
    candidates.append(_tetrahedron_init())

    # 6. Uniform random in cube
    for _ in range(3):
        candidates.append(rng.uniform(-1, 1, size=(N, D)))

    # Progressive tau schedule
    tau_schedule = [5.0, 15.0, 50.0, 150.0, 500.0]

    # Phase 1: Quick optimization with low tau on all candidates
    for i, start in enumerate(candidates):
        if time.time() - t_start > DEADLINE - 30:
            break
        try:
            opt = _optimize(start, _objective_log_ratio, (tau_schedule[0],),
                            max_iter=100, tol=1e-8)
            _update(opt)
            candidates[i] = opt
        except Exception:
            pass

    # Phase 2: Progressive tau increase
    for tau_idx in range(1, len(tau_schedule)):
        if time.time() - t_start > DEADLINE - 15:
            break
        tau = tau_schedule[tau_idx]
        max_iter = 200 + tau_idx * 100
        for i in range(len(candidates)):
            if time.time() - t_start > DEADLINE - 15:
                break
            try:
                opt = _optimize(candidates[i], _objective_log_ratio, (tau,),
                                max_iter=max_iter, tol=1e-10)
                _update(opt)
                candidates[i] = opt
            except Exception:
                pass

    # Phase 3: Polish with direct ratio objective from best candidates
    if best_pts is not None and time.time() - t_start < DEADLINE - 10:
        try:
            polished = _optimize(best_pts, _objective_ratio, (),
                                 max_iter=3000, tol=1e-12)
            _update(polished)
        except Exception:
            pass

    # Phase 4: One more polish pass if time allows
    if best_pts is not None and time.time() - t_start < DEADLINE - 5:
        try:
            polished2 = _optimize(best_pts, _objective_log_ratio, (500.0,),
                                  max_iter=2000, tol=1e-12)
            _update(polished2)
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _tetrahedron_init()

    # Ensure distinctness
    d2 = _pairwise_dists_sq(best_pts)
    if np.min(d2) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.sum((best_pts[i] - best_pts[j]) ** 2) < 1e-10:
                    best_pts[j] += rng.normal(size=D) * 1e-5

    return best_pts
