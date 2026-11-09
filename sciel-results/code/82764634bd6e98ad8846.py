"""Improved packer: hex-lattice init + force-based local refinement with deadline guard."""
import numpy as np
import time

N = 50
DEADLINE = 135.0  # seconds; leaves 15s margin under 150s kill


def _score(pts):
    """Compute the packing radius r for centres pts (N,2)."""
    n = len(pts)
    if n == 0:
        return 0.0
    # distance to boundary
    d = 1.0 - np.linalg.norm(pts, axis=1)
    r_bound = np.min(d)
    # pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # zero out diagonal
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_bound, r_pair)


def _force_refine(pts, max_iter=20000, tol=1e-12):
    """
    Force-based local refinement.
    Each circle 'pushes' on others when too close, and is pulled inward when too close to boundary.
    We use a simple iterative relaxation.
    """
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    r = _score(pts)

    for it in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break
        forces = np.zeros_like(pts)
        # pairwise repulsion: if distance < 2r, push apart
        # Use a soft force model: F = k * (2r - d) * direction if d < 2r
        k = 10.0
        for i in range(n):
            for j in range(i + 1, n):
                dvec = pts[i] - pts[j]
                d = np.linalg.norm(dvec)
                if d < 1e-12:
                    # coincident; nudge randomly
                    dvec = np.random.randn(2) * 1e-6
                    d = np.linalg.norm(dvec)
                if d < 2.0 * r:
                    f = k * (2.0 * r - d) / d
                    forces[i] += f * dvec
                    forces[j] -= f * dvec
        # boundary constraint: if |p| > 1 - r, push inward
        for i in range(n):
            norm = np.linalg.norm(pts[i])
            if norm > 1.0 - r:
                # push toward origin
                forces[i] -= k * (norm - (1.0 - r)) / norm * pts[i]
        # update
        step = 0.01
        pts += step * forces
        # clip to ensure inside
        norms = np.linalg.norm(pts, axis=1)
        max_norm = 1.0 - r * 0.5  # keep some margin
        mask = norms > max_norm
        if np.any(mask):
            pts[mask] *= max_norm / norms[mask]

        new_r = _score(pts)
        if new_r < r - 1e-10:
            # revert if worse
            pts -= step * forces
            break
        if abs(new_r - r) < tol:
            r = new_r
            break
        r = new_r

    return pts


def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, pick best N."""
    # Try several spacings
    best_pts = None
    best_r = 0.0
    for spacing in np.linspace(0.1, 0.25, 30):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # pick the N closest to center
            idx = np.argsort(np.hypot(points[:, 0], points[:, 1]), kind="stable")[:N]
            pts = points[idx]
            r = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    return best_pts, best_r


def construct():
    t0 = time.time()
    # Start with hex lattice
    pts, r = _hex_lattice_init()
    if pts is None:
        # Fallback: ring placement
        pts = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            pts[i] = [0.9 * np.cos(angle), 0.9 * np.sin(angle)]
        r = _score(pts)

    # Force-based refinement
    if time.time() < DEADLINE - 10.0:
        pts = _force_refine(pts, max_iter=50000, tol=1e-13)

    # Second pass with tighter tolerance
    if time.time() < DEADLINE - 5.0:
        pts = _force_refine(pts, max_iter=20000, tol=1e-14)

    # Ensure validity: clip any point outside
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        pts[mask] /= norms[mask][:, np.newaxis] * 1.0001

    return pts
