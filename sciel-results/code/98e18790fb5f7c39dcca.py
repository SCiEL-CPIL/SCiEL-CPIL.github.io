"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit

    n = 8

    def compute_log_product(points):
        """Compute log of product of pairwise distances after scaling to diameter 2."""
        # points: (n, 2) array
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))

        # Find diameter
        max_dist = np.max(dists)
        if max_dist < 1e-10:
            return -np.inf

        # Scale to diameter 2
        scale = 2.0 / max_dist
        scaled_dists = dists * scale

        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = scaled_dists[mask]

        # Avoid log(0)
        off_diag_dists = np.maximum(off_diag_dists, 1e-15)

        log_prod = np.sum(np.log(off_diag_dists))
        return log_prod

    def objective(x):
        """Objective function to minimize: negative log product of distances."""
        points = x.reshape(n, 2)
        log_prod = compute_log_product(points)
        return -log_prod

    # Regular octagon as baseline
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    reg_log_prod = compute_log_product(reg_oct)

    best_score = reg_log_prod
    best_points = reg_oct.copy()

    # Generate diverse initial guesses
    inits = [reg_oct.flatten()]

    np.random.seed(42)

    # Perturbed regular octagon
    for _ in range(8):
        perturb = np.random.normal(0, 0.05, (n, 2))
        inits.append((reg_oct + perturb).flatten())

    # Two squares with varying relative rotation
    for rot_offset in [0.05, 0.1, 0.15, 0.2, 0.3]:
        a1 = 2 * np.pi * np.arange(4) / 4
        a2 = 2 * np.pi * (np.arange(4) / 4 + rot_offset)
        sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
        sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
        inits.append(np.vstack([sq1, sq2]).flatten())

    # Random points on unit circle
    for _ in range(6):
        angles_rand = np.sort(np.random.uniform(0, 2 * np.pi, n))
        pts = np.column_stack([np.cos(angles_rand), np.sin(angles_rand)])
        inits.append(pts.flatten())

    # Random 2D points
    for _ in range(4):
        pts = np.random.uniform(-1, 1, (n, 2))
        inits.append(pts.flatten())

    # Optimize each initial guess using BFGS
    for x0 in inits:
        if time.time() - start_time > timeout - 5:
            break

        try:
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 300, 'gtol': 1e-10})
            if res.fun < -np.inf:
                continue
            current_log_prod = -res.fun
            if current_log_prod > best_score + 1e-10:
                best_score = current_log_prod
                best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Final verification: ensure distinct points and valid coordinates
    # Check if best_points are all distinct
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        # Points collapsed, fallback to regular octagon
        best_points = reg_oct
        best_score = reg_log_prod

    # Ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    # Convert to list of lists of floats
    result = [[float(p[0]), float(p[1])] for p in best_points]

    return result
# EVOLVE-BLOCK-END
