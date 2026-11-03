"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(positions):
    """Compute the packing score (max radius) for given positions."""
    if positions.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    # Check bounds
    if np.any(positions < -0.5) or np.any(positions > 0.5):
        return 0.0
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls
    wall_dists = np.min([
        np.min(positions[:, k] + 0.5) for k in range(3)
    ] + [
        np.min(0.5 - positions[:, k]) for k in range(3)
    ])
    r = min(min_pair_dist / 2.0, wall_dists)
    return max(0.0, r)

def _energy_and_grad(positions, r_target):
    """
    Energy: sum of violations. If all pairs are >= 2*r_target apart and all within walls,
    energy is 0. Otherwise positive.
    We use a smooth approximation: for each pair i<j, penalty = max(0, 2*r_target - d_ij)^2
    For walls: penalty = max(0, |c_ik| + r_target - 0.5)^2
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            d = np.linalg.norm(diff)
            if d < 2 * r_target and d > 1e-12:
                violation = 2 * r_target - d
                # gradient of max(0, 2r - d)^2 w.r.t. positions[i]
                # d/dx_i (2r - d)^2 = 2(2r-d) * d(2r-d)/dx_i = 2(2r-d) * (diff/d)
                coeff = 2 * violation * (diff / d)
                grad[i] += coeff
                grad[j] -= coeff
                energy += violation ** 2
            elif d <= 1e-12:
                # Coincident points: push apart randomly
                energy += (2 * r_target) ** 2
                grad[i] += np.array([1.0, 0.0, 0.0]) * 2 * r_target
                grad[j] -= np.array([1.0, 0.0, 0.0]) * 2 * r_target
    
    # Wall constraints
    for i in range(n):
        for k in range(3):
            # Constraint: |c_ik| + r_target <= 0.5
            val = abs(positions[i, k]) + r_target - 0.5
            if val > 0:
                energy += val ** 2
                if positions[i, k] > 0:
                    grad[i, k] += 2 * val
                else:
                    grad[i, k] -= 2 * val
    
    return energy, grad

def _score_and_grad(positions, r_target):
    """Returns (score_proxy, gradient) where we want to MINIMIZE energy."""
    e, g = _energy_and_grad(positions, r_target)
    return e, g

def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin
    
    # Start with a reasonable initial configuration
    # Use a body-centered cubic-like arrangement or simple grid
    # Let's start with a 5x5x4 grid (100 points)
    m1, m2, m3 = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, m1)
    ys = np.linspace(-0.4, 0.4, m2)
    zs = np.linspace(-0.4, 0.4, m3)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    positions = grid[:N].copy()
    
    best_positions = positions.copy()
    best_score = _score(best_positions)
    
    # Try to increase radius iteratively
    r_target = best_score
    # Initial radius estimate
    r_target = min(r_target, 0.1)
    
    # SGLD-style stochastic local search with energy minimization
    np.random.seed(42)
    
    for iteration in range(500):
        if time.time() > deadline - 20.0:
            break
        
        # Try to increase radius slightly
        r_target_new = r_target * 1.005 + 0.0001
        
        # Use L-BFGS-B to minimize energy at the new target radius
        try:
            result = minimize(
                lambda x: _score_and_grad(x, r_target_new)[0],
                positions,
                jac=lambda x: _score_and_grad(x, r_target_new)[1],
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * 3 * N,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            positions = result.x
            energy = _score_and_grad(positions, r_target_new)[0]
            
            if energy < 1e-6:
                # Success: we can achieve this radius
                r_target = r_target_new
                new_score = _score(positions)
                if new_score > best_score:
                    best_score = new_score
                    best_positions = positions.copy()
            else:
                # Failed: revert and try smaller increase
                r_target_new = r_target * 1.002
                result = minimize(
                    lambda x: _score_and_grad(x, r_target_new)[0],
                    positions,
                    jac=lambda x: _score_and_grad(x, r_target_new)[1],
                    method='L-BFGS-B',
                    bounds=[(-0.499, 0.499)] * 3 * N,
                    options={'maxiter': 50, 'ftol': 1e-10}
                )
                positions = result.x
                energy = _score_and_grad(positions, r_target_new)[0]
                if energy < 1e-6:
                    r_target = r_target_new
                    new_score = _score(positions)
                    if new_score > best_score:
                        best_score = new_score
                        best_positions = positions.copy()
                else:
                    # Add noise to escape local minimum
                    noise = np.random.normal(0, 0.01, positions.shape)
                    positions = np.clip(positions + noise, -0.499, 0.499)
        except Exception:
            # Add noise
            noise = np.random.normal(0, 0.01, positions.shape)
            positions = np.clip(positions + noise, -0.499, 0.499)
    
    # Final polish: try to maximize radius with a few more iterations
    for _ in range(10):
        if time.time() > deadline - 5.0:
            break
        r_try = r_target * 1.001
        try:
            result = minimize(
                lambda x: _score_and_grad(x, r_try)[0],
                best_positions,
                jac=lambda x: _score_and_grad(x, r_try)[1],
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * 3 * N,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            pos = result.x
            e = _score_and_grad(pos, r_try)[0]
            if e < 1e-8:
                s = _score(pos)
                if s > best_score:
                    best_score = s
                    best_positions = pos.copy()
                    r_target = r_try
            else:
                break
        except Exception:
            break
    
    return best_positions
# EVOLVE-BLOCK-END
