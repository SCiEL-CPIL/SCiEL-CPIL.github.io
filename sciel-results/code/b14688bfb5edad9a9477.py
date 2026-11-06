"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def compute_radius(c):
        """Compute the minimum radius for a given set of centres."""
        c = np.asarray(c, dtype=np.float64)
        if c.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(c)):
            return 0.0
        # Check bounds with small tolerance
        if np.any(np.abs(c) > 0.5 + 1e-12):
            return 0.0
        # Compute pairwise distances
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        # Distance to walls
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist / 2.0, min_wall_dist)

    def negative_radius(c):
        """Negative of the radius for minimization."""
        return -compute_radius(c)

    def make_grid_config(m, n, p):
        """Create a grid configuration with m x n x p points."""
        # Distribute N points in a grid
        configs = []
        # Try to find good grid dimensions
        for m in range(1, 8):
            for n in range(1, 8):
                for p in range(1, 8):
                    if m * n * p >= N:
                        xs = (np.arange(m) + 0.5) / m - 0.5
                        ys = (np.arange(n) + 0.5) / n - 0.5
                        zs = (np.arange(p) + 0.5) / p - 0.5
                        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
                        configs.append((m * n * p, grid[:N].copy()))
        if not configs:
            return None
        # Pick the configuration closest to cubic
        best = min(configs, key=lambda x: abs(x[0] - N))
        return best[1]

    def perturbed_grid(base, sigma):
        """Add Gaussian perturbation to a base configuration."""
        return base + np.random.normal(0, sigma, size=base.shape)

    def fcc_like_config():
        """Create an FCC-like configuration."""
        # FCC lattice points in a cube
        a = 0.5  # lattice parameter
        points = []
        # Corner and face-centered positions
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    points.append((i * a, j * a, k * a))
                    points.append((i * a + a/2, j * a + a/2, k * a))
                    points.append((i * a, j * a + a/2, k * a + a/2))
                    points.append((i * a + a/2, j * a, k * a + a/2))
        points = np.array(points)
        # Center and scale to fit in [-0.5, 0.5]^3
        points = points - np.mean(points, axis=0)
        scale = 0.5 / np.max(np.abs(points))
        points = points * scale
        # Take first N points
        return points[:N].copy()

    def bcc_like_config():
        """Create a BCC-like configuration."""
        a = 0.5
        points = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    points.append((i * a, j * a, k * a))
                    points.append((i * a + a/2, j * a + a/2, k * a + a/2))
        points = np.array(points)
        points = points - np.mean(points, axis=0)
        scale = 0.5 / np.max(np.abs(points))
        points = points * scale
        return points[:N].copy()

    # Generate diverse initial configurations
    np.random.seed(42)
    
    initial_configs = []
    
    # Grid-based
    grid_base = make_grid_config(4, 4, 4)
    if grid_base is not None:
        initial_configs.append(("grid444", grid_base))
    
    # Perturbed grids
    for sigma in [0.02, 0.05, 0.1]:
        if grid_base is not None:
            initial_configs.append((f"grid_sigma{sigma}", perturbed_grid(grid_base, sigma)))
    
    # FCC-like
    fcc = fcc_like_config()
    initial_configs.append(("fcc", fcc))
    initial_configs.append(("fcc_pert", perturbed_grid(fcc, 0.03)))
    
    # BCC-like
    bcc = bcc_like_config()
    initial_configs.append(("bcc", bcc))
    initial_configs.append(("bcc_pert", perturbed_grid(bcc, 0.03)))
    
    # Random configurations
    for i in range(5):
        initial_configs.append((f"random_{i}", np.random.uniform(-0.5, 0.5, size=(N, 3))))
    
    best_c = None
    best_r = -np.inf

    # Phase 1: Coarse optimization using Powell (good for non-smooth objectives)
    for name, c_init in initial_configs:
        if time.time() > deadline - 30.0:
            break
        
        c_init = np.clip(c_init, -0.5, 0.5)
        
        try:
            result = minimize(
                negative_radius,
                c_init,
                method='Powell',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            r = -result.fun
            if r > best_r:
                best_r = r
                best_c = result.x.copy()
        except Exception:
            continue

    # Phase 2: Fine-tuning the best configuration with Nelder-Mead
    if best_c is not None and time.time() < deadline - 15.0:
        try:
            result = minimize(
                negative_radius,
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            r = -result.fun
            if r > best_r:
                best_r = r
                best_c = result.x.copy()
        except Exception:
            pass

    # Phase 3: Try a few more local improvements from the best solution
    if best_c is not None and time.time() < deadline - 5.0:
        for i in range(3):
            if time.time() > deadline - 5.0:
                break
            c_pert = best_c + np.random.normal(0, 0.01, size=(N, 3))
            c_pert = np.clip(c_pert, -0.5, 0.5)
            try:
                result = minimize(
                    negative_radius,
                    c_pert,
                    method='Powell',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 50, 'xtol': 1e-10, 'ftol': 1e-10}
                )
                r = -result.fun
                if r > best_r:
                    best_r = r
                    best_c = result.x.copy()
            except Exception:
                continue

    # Final validation and clipping
    if best_c is None:
        # Fallback to grid
        return grid_base if grid_base is not None else np.random.uniform(-0.5, 0.5, size=(N, 3))
    
    best_c = np.clip(best_c, -0.5, 0.5)
    final_r = compute_radius(best_c)
    
    if final_r <= 0:
        return grid_base if grid_base is not None else np.random.uniform(-0.5, 0.5, size=(N, 3))
    
    return best_c
# EVOLVE-BLOCK-END
