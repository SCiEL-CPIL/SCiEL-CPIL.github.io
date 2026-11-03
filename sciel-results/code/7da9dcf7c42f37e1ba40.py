"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _score(centers):
    """Compute the maximum common radius for given centers."""
    # Centers must be in [-0.5, 0.5]^3
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Pairwise distances
    # Use broadcasting for efficiency
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_wall, min_pair / 2.0)
    return r

def _neg_score(centers_flat):
    """Negative score for minimization."""
    centers = centers_flat.reshape(N, 3)
    s = _score(centers)
    return -s

def _initial_lattice():
    """Create a reasonable initial lattice configuration."""
    # Use a body-centered cubic-like or face-centered arrangement
    # For 50 spheres, try a 4x4x4 grid with some removals and adjustments
    # Better: use a known good starting point
    
    # Simple approach: 5x5x2 grid with center column removed to get ~50
    # 5*5*2 = 50 exactly
    positions = []
    for z in range(2):
        for y in range(5):
            for x in range(5):
                positions.append([x, y, z])
    
    positions = np.array(positions, dtype=float)
    # Normalize to fit in cube
    # Scale so that the grid fits nicely
    # The grid spans 0-4 in x,y and 0-1 in z
    # We want to map this to [-0.5, 0.5] with some margin
    # Let's use a scaling factor
    scale = 0.8  # Leave some room for optimization
    centers = (positions - np.array([2, 2, 0.5])) * scale / 2.0
    # Add small random perturbation
    np.random.seed(42)
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    return centers

def _optimize_from(initial_centers, time_budget):
    """Optimize centers using differential evolution or L-BFGS."""
    start_time = time.time()
    best_centers = initial_centers.copy()
    best_score = _score(initial_centers)
    
    # Try multiple optimization strategies
    # Strategy 1: Differential Evolution (global)
    if time.time() - start_time < time_budget * 0.4:
        try:
            bounds = [(-0.5, 0.5)] * (N * 3)
            # Use a smaller population for speed
            de_result = differential_evolution(
                _neg_score,
                bounds,
                seed=42,
                maxiter=100,
                popsize=10,
                tol=1e-8,
                callback=None
            )
            if de_result.fun < -best_score:
                best_score = -de_result.fun
                best_centers = de_result.x.reshape(N, 3)
        except Exception:
            pass
    
    # Strategy 2: Local optimization with L-BFGS-B from best point
    if time.time() - start_time < time_budget * 0.9:
        try:
            res = minimize(
                _neg_score,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_centers = res.x.reshape(N, 3)
        except Exception:
            pass
    
    return best_centers, best_score

def construct():
    """Construct the optimal arrangement of 50 spheres in a unit cube."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Generate multiple initial configurations
    np.random.seed(12345)
    
    # Initial configuration 1: Lattice-based
    init1 = _initial_lattice()
    
    # Initial configuration 2: Random
    init2 = np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    # Initial configuration 3: Another lattice variant
    # 5x5x2 but with different scaling
    positions = []
    for z in range(2):
        for y in range(5):
            for x in range(5):
                positions.append([x, y, z])
    positions = np.array(positions, dtype=float)
    scale = 0.9
    init3 = (positions - np.array([2, 2, 0.5])) * scale / 2.0
    init3 += np.random.uniform(-0.02, 0.02, size=init3.shape)
    
    inits = [init1, init2, init3]
    
    best_centers = None
    best_score = 0.0
    
    for i, init in enumerate(inits):
        remaining = deadline - (time.time() - start_time)
        if remaining < 10:
            break
        
        # Allocate time for this optimization
        time_budget = min(remaining / (len(inits) - i), 40.0)
        
        centers, score = _optimize_from(init, time_budget)
        
        if score > best_score:
            best_score = score
            best_centers = centers
    
    # Final polish with high-precision L-BFGS if time permits
    if best_centers is not None and time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                _neg_score,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_centers = res.x.reshape(N, 3)
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback: return initial lattice
        return init1
    
    return best_centers
# EVOLVE-BLOCK-END
