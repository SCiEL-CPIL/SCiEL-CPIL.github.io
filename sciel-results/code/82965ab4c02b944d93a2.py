"""Optimized 200-circle packing via hex warm-start + projected gradient on min distance."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _hex_lattice(n, offset=(0.0, 0.0), angle=0.0):
    """Generate n points on a hexagonal lattice inside [-0.5, 0.5]^2, rotated by angle."""
    # Hex lattice spacing chosen so we get roughly n points in unit square
    # Area per hex point = (sqrt(3)/2) * s^2 where s is spacing
    # We want n points in area 1 => s^2 = 1/(n * sqrt(3)/2)
    s = np.sqrt(2.0 / (n * np.sqrt(3)))
    # Generate lattice points in a region larger than the square, then clip
    extent = 1.2
    nx = int(np.ceil(2 * extent / s)) + 2
    ny = int(np.ceil(2 * extent / (s * np.sqrt(3) / 2))) + 2
    pts = []
    for j in range(ny):
        for i in range(nx):
            x = i * s + (j % 2) * (s / 2)
            y = j * s * np.sqrt(3) / 2
            pts.append((x, y))
    pts = np.array(pts)
    # Center and rotate
    pts = pts - pts.mean(axis=0)
    if angle != 0:
        c, sn = np.cos(angle), np.sin(angle)
        R = np.array([[c, -sn], [sn, c]])
        pts = pts @ R.T
    pts = pts + np.array(offset)
    # Keep only points inside [-0.5, 0.5]^2
    mask = (pts[:, 0] >= -0.5) & (pts[:, 0] <= 0.5) & (pts[:, 1] >= -0.5) & (pts[:, 1] <= 0.5)
    pts = pts[mask]
    # If we have more than n, take the n closest to center (denser in middle)
    if len(pts) > n:
        d = np.linalg.norm(pts, axis=1)
        idx = np.argsort(d)[:n]
        pts = pts[idx]
    elif len(pts) < n:
        # Pad with random points inside square
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(pts), 2))
        pts = np.vstack([pts, extra])
    return pts


def _project_into_square(pts):
    """Project points into [-0.5, 0.5]^2."""
    pts = np.clip(pts, -0.5, 0.5)
    return pts


def _min_pairwise_dist(pts):
    """Compute minimum pairwise distance."""
    n = len(pts)
    min_d = np.inf
    for i in range(n):
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        if len(d) > 0:
            min_d = min(min_d, d.min())
    return min_d


def _score(pts):
    """Compute the packing score: min(min_pairwise/2, min_wall_margin)."""
    min_pair = _min_pairwise_dist(pts)
    wall_margins = np.min(np.minimum(0.5 - np.abs(pts), axis=1))
    return min(min_pair / 2.0, wall_margins)


def _gradient_ascent(pts, n_iter=2000, step0=0.01):
    """Projected gradient ascent on min(min_pairwise/2, wall_margin)."""
    pts = pts.copy()
    n = len(pts)
    step = step0
    best_pts = pts.copy()
    best_score = _score(pts)

    for it in range(n_iter):
        # Compute min pairwise distance and which pairs achieve it
        min_pair = np.inf
        min_pairs = []
        for i in range(n):
            d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
            if len(d) > 0:
                local_min = d.min()
                if local_min < min_pair:
                    min_pair = local_min
                    min_pairs = [(i, i + 1 + np.argmin(d))]
                elif local_min == min_pair:
                    idx = np.argmin(d)
                    min_pairs.append((i, i + 1 + idx))

        # Wall margins
        wall_margins = np.minimum(0.5 - np.abs(pts), axis=1)
        min_wall = wall_margins.min()

        current_score = min(min_pair / 2.0, min_wall)

        # Gradient: move points to increase the bottleneck
        grad = np.zeros_like(pts)
        if min_pair / 2.0 <= min_wall:
            # Bottleneck is pairwise: move each point away from its nearest neighbor(s)
            for (i, j) in min_pairs:
                diff = pts[i] - pts[j]
                norm = np.linalg.norm(diff)
                if norm > 1e-12:
                    unit = diff / norm
                    grad[i] += unit
                    grad[j] -= unit
        else:
            # Bottleneck is wall: move points away from the nearest wall
            for i in range(n):
                if wall_margins[i] <= min_wall + 1e-12:
                    # Determine which wall is closest
                    if pts[i][0] < 0:
                        grad[i, 0] += 1.0
                    elif pts[i][0] > 0:
                        grad[i, 0] -= 1.0
                    if pts[i][1] < 0:
                        grad[i, 1] += 1.0
                    elif pts[i][1] > 0:
                        grad[i, 1] -= 1.0

        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm

        # Adaptive step: decrease if score worsened
        new_pts = pts + step * grad
        new_pts = _project_into_square(new_pts)
        new_score = _score(new_pts)

        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()

        if new_score >= current_score - 1e-12:
            pts = new_pts
        else:
            step *= 0.7

        # Slowly decay step
        step = max(step * 0.999, 1e-5)

    return best_pts, best_score


def _local_refine(pts, n_iter=3000):
    """Fine-grained local refinement with very small steps."""
    pts = pts.copy()
    n = len(pts)
    step = 0.002
    best_pts = pts.copy()
    best_score = _score(pts)

    for it in range(n_iter):
        min_pair = np.inf
        min_pairs = []
        for i in range(n):
            d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
            if len(d) > 0:
                local_min = d.min()
                if local_min < min_pair:
                    min_pair = local_min
                    min_pairs = [(i, i + 1 + np.argmin(d))]
                elif local_min == min_pair:
                    idx = np.argmin(d)
                    min_pairs.append((i, i + 1 + idx))

        wall_margins = np.minimum(0.5 - np.abs(pts), axis=1)
        min_wall = wall_margins.min()
        current_score = min(min_pair / 2.0, min_wall)

        grad = np.zeros_like(pts)
        if min_pair / 2.0 <= min_wall:
            for (i, j) in min_pairs:
                diff = pts[i] - pts[j]
                norm = np.linalg.norm(diff)
                if norm > 1e-12:
                    unit = diff / norm
                    grad[i] += unit
                    grad[j] -= unit
        else:
            for i in range(n):
                if wall_margins[i] <= min_wall + 1e-12:
                    if pts[i][0] < 0:
                        grad[i, 0] += 1.0
                    elif pts[i][0] > 0:
                        grad[i, 0] -= 1.0
                    if pts[i][1] < 0:
                        grad[i, 1] += 1.0
                    elif pts[i][1] > 0:
                        grad[i, 1] -= 1.0

        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm

        new_pts = pts + step * grad
        new_pts = _project_into_square(new_pts)
        new_score = _score(new_pts)

        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()

        if new_score >= current_score - 1e-12:
            pts = new_pts
        else:
            step *= 0.8

        step = max(step * 0.9995, 1e-6)

    return best_pts, best_score


def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    np.random.seed(42)

    best_pts = None
    best_score = -np.inf

    # Try multiple hex lattice configurations with different rotations and offsets
    configs = [
        (0.0, (0.0, 0.0)),
        (np.pi / 6, (0.0, 0.0)),
        (np.pi / 12, (0.01, 0.01)),
        (np.pi / 4, (0.0, 0.0)),
        (0.0, (0.02, -0.01)),
        (np.pi / 8, (-0.01, 0.02)),
        (np.pi / 3, (0.0, 0.0)),
        (np.pi / 5, (0.015, 0.015)),
    ]

    for angle, offset in configs:
        pts = _hex_lattice(N, offset=offset, angle=angle)
        # Coarse gradient ascent
        pts, sc = _gradient_ascent(pts, n_iter=1500, step0=0.008)
        # Local refinement
        pts, sc = _local_refine(pts, n_iter=2000)

        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()

    # Final polish on the best configuration
    best_pts, best_score = _local_refine(best_pts, n_iter=3000)

    # Verify validity
    assert best_pts.shape == (N, 2), f"Wrong shape: {best_pts.shape}"
    assert np.all(np.isfinite(best_pts)), "Non-finite values"
    assert np.all(np.abs(best_pts) <= 0.5), "Points outside square"

    return best_pts
# EVOLVE-BLOCK-END
