"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    def compute_r(pts):
        """Compute the packing radius for a set of points."""
        pts = np.asarray(pts, dtype=np.float64)
        # Pairwise distances
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        # Boundary margin
        radii = np.sqrt(np.sum(pts * pts, axis=1))
        min_bound = 1.0 - np.max(radii)
        return min(min_pair, min_bound)

    def objective(x):
        pts = x.reshape(-1, 2)
        # Return negative of the packing radius (to minimize)
        return -compute_r(pts)

    def hex_initial():
        """Generate a hexagonal lattice configuration."""
        pts = []
        # Use a hexagonal grid with spacing ~0.25
        spacing = 0.25
        for row in range(-10, 11):
            for col in range(-10, 11):
                x = col * spacing + (0.5 * spacing if row % 2 else 0.0)
                y = row * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) < 0.85:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) < N:
            # Fallback: add random points
            while len(pts) < N:
                x, y = np.random.uniform(-0.8, 0.8, 2)
                if np.hypot(x, y) < 0.85:
                    pts = np.vstack([pts, [[x, y]]])
        # Sort by distance from center and take N closest
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists)[:N]
        return pts[idx]

    def random_initial():
        """Generate a random initial configuration."""
        pts = np.random.uniform(-0.85, 0.85, (N, 2))
        # Ensure all points are inside
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 0.85
        if np.any(mask):
            pts[mask] *= 0.85 / radii[mask, None]
        return pts

    best_r = 0.0
    best_points = None

    # Try hexagonal initial configuration
    x0 = hex_initial().flatten()
    r0 = -objective(x0)
    if r0 > best_r:
        best_r = r0
        best_points = x0.reshape(-1, 2)

    # Optimize from hexagonal start
    try:
        result = minimize(objective, x0, method='BFGS',
                          options={'maxiter': 2000, 'gtol': 1e-10, 'disp': False})
        if result.fun < objective(x0):
            pts = result.x.reshape(-1, 2)
            r = -result.fun
            if r > best_r:
                best_r = r
                best_points = pts
    except Exception:
        pass

    # Try random restarts
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        x0 = random_initial().flatten()
        r0 = -objective(x0)
        if r0 > best_r:
            best_r = r0
            best_points = x0.reshape(-1, 2)

        try:
            result = minimize(objective, x0, method='BFGS',
                              options={'maxiter': 1000, 'gtol': 1e-10, 'disp': False})
            if result.fun < objective(x0):
                pts = result.x.reshape(-1, 2)
                r = -result.fun
                if r > best_r:
                    best_r = r
                    best_points = pts
        except Exception:
            pass

    if best_points is None:
        best_points = hex_initial()

    return best_points
# EVOLVE-BLOCK-END
