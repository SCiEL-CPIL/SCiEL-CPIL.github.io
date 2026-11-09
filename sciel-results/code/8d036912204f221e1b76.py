"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """Feasible common radius: min of boundary clearance and half min pairwise distance."""
    if points is None or len(points) < 2:
        return 0.0
    norms = np.linalg.norm(points, axis=1)
    min_boundary = float(1.0 - np.max(norms))
    if min_boundary <= 0:
        return 0.0
    # Use KDTree for efficient nearest-neighbor distance
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = float(np.min(dists[:, 1])) / 2.0
    return min(min_boundary, min_pair)


def _hexagonal_lattice(spacing, margin=None):
    """Hexagonal lattice points inside the unit disk (with margin), closest to centre."""
    if margin is None:
        margin = spacing / 2.0
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - margin:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _concentric_rings(n_rings):
    """Concentric rings packing; returns points or None."""
    if n_rings <= 0:
        return None
    if n_rings == 1:
        ring_counts = [N]
    else:
        remaining = N - 1
        base = remaining // (n_rings - 1)
        extra = remaining % (n_rings - 1)
        ring_counts = [1] + [base + (1 if i < extra else 0) for i in range(n_rings - 1)]
    pts = []
    if ring_counts[0] > 0:
        pts.append([0.0, 0.0])
    for ring_idx in range(1, n_rings):
        count = ring_counts[ring_idx]
        if count <= 0:
            continue
        ring_radius = ring_idx / n_rings * 0.95
        for i in range(count):
            angle = 2 * np.pi * i / count
            pts.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle)])
    if len(pts) != N:
        return None
    return np.array(pts)


def _square_grid(spacing):
    """Square grid points inside the unit disk, closest to centre."""
    k = int(1.0 / spacing) + 3
    pts = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing
            y = j * spacing
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _refine_local(points, deadline, max_iters=3000, base_step=0.03):
    """Hill-climbing local refinement with adaptive step size; respects the deadline."""
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _compute_radius(points)
    rng = np.random.default_rng(12345)
    step = base_step
    no_improve = 0
    for it in range(max_iters):
        if time.time() > deadline:
            break
        idx = rng.integers(0, len(points))
        original = points[idx].copy()
        best_local = original
        best_local_r = best_r
        # Try more perturbations with decreasing step sizes
        for trial in range(8):
            if time.time() > deadline:
                break
            s = step * (1.0 - trial * 0.1)
            if s < 0.001:
                break
            dx = rng.uniform(-s, s)
            dy = rng.uniform(-s, s)
            points[idx] = original + np.array([dx, dy])
            r = _compute_radius(points)
            if r > best_local_r:
                best_local_r = r
                best_local = points[idx].copy()
        points[idx] = best_local
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
            no_improve = 0
            step = max(base_step * 0.95, 0.005)
        else:
            no_improve += 1
            if no_improve > 50:
                step = max(step * 0.9, 0.002)
                no_improve = 0
    return best_points, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s wall-clock limit

    candidates = []

    # Hexagonal lattice: try a range of spacings with different margins
    for trial in range(50):
        spacing = 0.055 + trial * 0.003
        # Try with standard margin and a slightly smaller margin
        for margin_frac in [0.5, 0.45, 0.4]:
            margin = spacing * margin_frac
            pts = _hexagonal_lattice(spacing, margin=margin)
            if pts is not None:
                r = _compute_radius(pts)
                if r > 0:
                    candidates.append((pts, r))

    # Concentric rings: a few ring counts
    for n_rings in range(4, 25):
        pts = _concentric_rings(n_rings)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    # Square grid: a range of spacings
    for trial in range(20):
        spacing = 0.055 + trial * 0.004
        pts = _square_grid(spacing)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    if not candidates:
        # Fallback: single ring of N points.
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]

    # Refine the best candidates while time permits
    refined_count = 0
    for cand_idx in range(min(5, len(candidates))):
        if time.time() > deadline - 10.0:
            break
        cand_pts, cand_r = candidates[cand_idx]
        refined_pts, refined_r = _refine_local(cand_pts, deadline,
                                               max_iters=2000, base_step=0.03)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r
            refined_count += 1
        # Stop refining if we've made good progress
        if refined_count >= 2 and time.time() > deadline - 30.0:
            break

    # Final refinement pass on the best result if time allows
    if time.time() < deadline - 5.0:
        final_pts, final_r = _refine_local(best_points, deadline,
                                           max_iters=1000, base_step=0.015)
        if final_pts is not None and final_r > best_r:
            best_points, best_r = final_pts, final_r

    return best_points

# EVOLVE-BLOCK-END
