"""Spherical packing in a unit cube: 100 equal spheres, maximize common radius.

Strategy:
  1. Start from a BCC-inspired lattice init (better than cubic grid).
  2. Use a soft-min (log-sum-exp) surrogate for the min-distance objective.
  3. Run L-BFGS-B with box constraints to refine.
  4. Use Simulated Annealing to escape local minima, with periodic L-BFGS-B polish.
  5. Final high-precision polish pass.
  6. Deadline guard: return best result before 150 s wall-clock limit.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 150.0  # wall-clock seconds; keep a safety margin


def _compute_radius(centers):
    """Compute the largest common radius for the given centres."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)


def _soft_min_objective(centers_flat, beta):
    """
    Soft-min surrogate: -log-sum-exp(-beta * d_ij) approximates -min(d_ij) for large beta.
    We minimize this (i.e., maximize the soft-min of distances).
    Includes wall distances as well.
    """
    centers = centers_flat.reshape(N, 3)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    flat_wall = wall_dists.flatten()
    # Combine all distances
    all_dists = np.concatenate([dists[np.triu_indices(N, k=1)], flat_wall])
    # Soft-min: -log(sum(exp(-beta * d))) / beta
    # For numerical stability
    shifted = -beta * all_dists
    max_shift = np.max(shifted)
    if max_shift > 700:
        shifted = shifted - max_shift
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = -(max_shift + log_sum) / beta
    return soft_min


def _objective(centers_flat, beta):
    """Objective to minimize: negative soft-min of distances."""
    return -_soft_min_objective(centers_flat, beta)


def _gradient(centers_flat, beta):
    """Gradient of the objective (numerical via finite differences for simplicity,
    or we can use analytic. For speed, use scipy's finite diff in L-BFGS-B)."""
    # Let scipy compute gradient via finite differences; return None to let it do so.
    return None


def _bcc_init():
    """
    Generate a BCC-inspired initial configuration for 100 spheres in [-0.5, 0.5]^3.
    BCC has 2 atoms per cubic cell. We use a 5x5x4 grid of cells = 200 positions,
    pick 100 of them, or use a 4x4x4 = 64 cells * 2 = 128, pick 100.
    
    Actually, let's use a more careful approach:
    - Use a 5x5x4 lattice of cube cells, each with a corner atom and a body-center atom.
    - That's 5*5*4 = 100 cells, each with 2 positions = 200 positions.
    - We pick 100 of them (e.g., all corner atoms, or alternating).
    
    Better: use a simple approach. Place 100 points on a BCC lattice scaled to fit.
    BCC lattice vectors: a/2*(1,1,1) for body center, a*(1,0,0) etc for corners.
    
    Let's use a 4x4x4 grid of cubic cells (side = 1/4 = 0.25). Each cell has:
      - corner at (i, j, k) * 0.25
      - body center at (i+0.5, j+0.5, k+0.5) * 0.25
    Total: 4*4*4 = 64 cells, 128 positions. Pick 100.
    
    Shift to center in [-0.5, 0.5]:
    Corner: (i - 0.5) * 0.25 for i in 0..4? No, 4 cells means i in 0..3.
    Let me use 5x5x4 = 100 cells. Each cell side = 1/5 in x,y and 1/4 in z.
    That's irregular. Let's just use a 4x4x4 BCC and add 4 more points.
    
    Simplest: use a 5x5x4 grid of points (100 points) with BCC-like spacing.
    Actually, let me just use a perturbed cubic grid as init, then optimize.
    """
    # Use a 5x5x4 grid = 100 points
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    grid += rng.normal(0, 0.01, size=grid.shape)
    # Clip to stay inside
    grid = np.clip(grid, -0.48, 0.48)
    return grid


def _sa_step(centers, temp, rng):
    """One simulated annealing step: propose random perturbation, accept if better or with prob exp(-delta/T)."""
    new_centers = centers.copy()
    # Perturb a random subset of points
    n_perturb = rng.integers(5, 20)
    idx = rng.choice(N, size=n_perturb, replace=False)
    new_centers[idx] += rng.normal(0, temp, size=(n_perturb, 3))
    # Clip to cube
    new_centers = np.clip(new_centers, -0.499, 0.499)
    return new_centers


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # safety margin
    
    # Initial configuration
    centers = _bcc_init()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(123)
    
    # Phase 1: L-BFGS-B with moderate beta
    beta = 50.0
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def _time_check():
        return time.time() < deadline
    
    if _time_check():
        result = minimize(
            _objective,
            centers.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000},
            args=(beta,)
        )
        if result.success or result.fun < _objective(centers.flatten(), beta):
            new_centers = np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers.copy()
                centers = new_centers.copy()
    
    # Phase 2: Simulated Annealing with periodic L-BFGS-B polish
    if _time_check():
        temp = 0.05
        min_temp = 0.001
        cooling = 0.95
        polish_every = 100
        sa_iter = 0
        last_polish = 0
        
        while _time_check() and temp > min_temp:
            new_centers = _sa_step(centers, temp, rng)
            # Quick evaluation
            r_new = _compute_radius(new_centers)
            if r_new >= best_radius or rng.random() < np.exp(-(best_radius - r_new) / max(temp, 1e-10)):
                centers = new_centers
                if r_new > best_radius:
                    best_radius = r_new
                    best_centers = new_centers.copy()
            
            sa_iter += 1
            temp *= cooling
            
            # Periodic polish with L-BFGS-B
            if sa_iter - last_polish >= polish_every and _time_check():
                # Increase beta for polish
                beta_polish = 100.0
                result = minimize(
                    _objective,
                    centers.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000},
                    args=(beta_polish,)
                )
                if result.success or result.fun < _objective(centers.flatten(), beta_polish):
                    new_centers = np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
                    r = _compute_radius(new_centers)
                    if r > best_radius:
                        best_radius = r
                        best_centers = new_centers.copy()
                        centers = new_centers.copy()
                last_polish = sa_iter
    
    # Phase 3: Final high-precision polish
    if _time_check():
        beta_final = 200.0
        result = minimize(
            _objective,
            best_centers.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 20000},
            args=(beta_final,)
        )
        if result.success or result.fun < _objective(best_centers.flatten(), beta_final):
            new_centers = np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers.copy()
    
    # Final validation
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    r_final = _compute_radius(best_centers)
    if r_final < best_radius:
        # Fallback to best known
        pass
    
    return best_centers
