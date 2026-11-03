"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START

def _radius(pts):
    """Compute packing radius for N points in unit circle."""
    if pts is None or len(pts) < 2:
        return 0.0
    # Boundary term
    r_centers = np.linalg.norm(pts, axis=1)
    min_boundary = 1.0 - np.max(r_centers)
    if min_boundary < 0:
        return 0.0
    # Pairwise min distance
    # Use full matrix for simplicity; 300x300 = 90k, fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair_d = np.sqrt(np.min(d2))
    return min(min_pair_d / 2.0, min_boundary)


def _init_hex(n):
    """Generate a hexagonal lattice of points, then trim to n closest to origin."""
    spacing = 0.112
    pts = []
    k = int(1.0 / spacing) + 4
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j & 1)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < n:
        # fallback: increase coverage
        spacing = 0.105
        pts = []
        k = int(1.0 / spacing) + 4
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j & 1)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                    pts.append((x, y))
        pts = np.array(pts)
    # Trim to n closest to origin
    d2 = np.sum(pts ** 2, axis=1)
    idx = np.argsort(d2)[:n]
    return pts[idx].copy()


def _neighborhood_indices(pts, k=12):
    """For each point, return indices of its k nearest neighbors (excluding self)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    # top-k
    k = min(k, n - 1)
    idx = np.argpartition(d2, k, axis=1)[:, :k]
    # sort each row by distance for locality (optional)
    # return idx
    return idx


def _local_optimize(pts, max_iter=200, step_scale=0.002):
    """Coordinate descent: try to move each point to improve min radius."""
    pts = pts.copy()
    n = len(pts)
    for it in range(max_iter):
        improved = False
        for i in range(n):
            orig = pts[i].copy()
            # Try small random perturbations
            for _ in range(8):
                delta = np.random.randn(2) * step_scale
                cand = pts[i] + delta
                if np.linalg.norm(cand) > 0.999:
                    continue
                pts[i] = cand
                r = _radius(pts)
                pts[i] = orig
                # Compare to current
            # Instead, do a proper local search: keep if improves
        # Better: do sequential coordinate ascent
    return pts


def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    pts = _init_hex(N)
    best_pts = pts.copy()
    best_r = _radius(best_pts)

    # Build neighbor structure once
    neigh = _neighborhood_indices(best_pts, k=16)

    # Simulated annealing with local acceptance
    T = 0.003
    T_min = 1e-6
    alpha = 0.9998
    step = 0.03

    # Track for early stopping
    no_improve = 0
    max_no_improve = 3000

    while time.time() < deadline and T > T_min:
        i = np.random.randint(N)
        neighbors = neigh[i]
        # Propose move
        delta = np.random.randn(2) * step
        cand = best_pts[i].copy()
        cand += delta
        if np.linalg.norm(cand) > 0.999:
            continue
        best_pts[i] = cand
        r_new = _radius(best_pts)
        delta_r = r_new - best_r
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            best_r = r_new
            no_improve = 0
            # Rebuild neighbors occasionally
        else:
            best_pts[i] -= delta
            no_improve += 1

        T *= alpha
        if T < T_min:
            break

        # Periodically rebuild neighbor list
        if no_improve % 500 == 0 and no_improve > 0:
            if time.time() < deadline - 5:
                neigh = _neighborhood_indices(best_pts, k=16)

        if no_improve > max_no_improve:
            # Restart with higher T
            T = max(T, 0.002)
            no_improve = 0

    # Final local polish: coordinate ascent on closest pairs
    if time.time() < deadline - 10:
        # Find closest pair and push apart
        for _ in range(5000):
            if time.time() > deadline - 10:
                break
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(d2, np.inf)
            idx = np.unravel_index(np.argmin(d2), d2.shape)
            i, j = idx
            d = np.sqrt(d2[i, j])
            if d < 1e-12:
                break
            direction = (best_pts[i] - best_pts[j]) / d
            moved = False
            for s in [0.001, 0.0005, 0.002]:
                ni = best_pts[i] + direction * s
                if np.linalg.norm(ni) < 0.999:
                    best_pts[i] = ni
                    if _radius(best_pts) > best_r:
                        best_r = _radius(best_pts)
                        moved = True
                        break
                    else:
                        best_pts[i] -= direction * s
                nj = best_pts[j] - direction * s
                if np.linalg.norm(nj) < 0.999:
                    best_pts[j] = nj
                    if _radius(best_pts) > best_r:
                        best_r = _radius(best_pts)
                        moved = True
                        break
                    else:
                        best_pts[j] += direction * s
            if not moved:
                break

    return best_pts

# EVOLVE-BLOCK-END
