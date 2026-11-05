# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # 320s wall clock limit (evaluator kills at 330s)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    
    # Morse term
    er = rho * (1.0 - dist)
    # Avoid overflow for very small r
    er = np.clip(er, -50, 50)
    exp_er = np.exp(er)
    energy_terms = exp_er * (exp_er - 2.0)
    
    # Sum upper triangle (i < j)
    iu = np.triu_indices(n, 1)
    return np.sum(energy_terms[iu])

def _morse_gradient(x, rho):
    """Compute analytical gradient of Morse energy (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.zeros_like(x)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Calculate terms
    u = rho * (1.0 - dist)
    u = np.clip(u, -50, 50)
    exp_u = np.exp(u)
    
    # dE/dr
    dE_dr = -2.0 * rho * exp_u * (exp_u - 1.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = dE_dr / dist
    np.fill_diagonal(factor, 0.0)
    
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return grad

def _min_distance(x):
    """Compute minimum distance between any pair of atoms."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist)

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _generate_decoy(n, seed=42, rho=6.0):
    """
    Generate a cluster using a density-inspired approach:
    1. Place atoms on a grid with a density profile (Gaussian core).
    2. Jitter to break symmetry.
    3. Relax.
    """
    rng = np.random.default_rng(seed)
    
    # Estimate radius for n atoms
    # Volume ~ n * (2.0/3) * (4/3 pi r_min^3) ? 
    # For dense packing, volume ~ n * (4/3) pi (1/2)^3 * packing_factor
    # Let's just scale radius as n^(1/3)
    radius = 2.5 * (n ** (1.0/3.0))
    
    # Grid resolution
    # For n=80, radius ~ 6. Resolution 0.5 -> 24x24x24 = 13k points.
    # We want ~n * 5 candidates.
    target_candidates = n * 8
    resolution = max(0.3, radius / (target_candidates ** (1.0/3.0)) / 2)
    
    # Create grid
    x_grid = np.arange(-radius, radius, resolution)
    X, Y, Z = np.meshgrid(x_grid, x_grid, x_grid, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Density weight: exp(-r^2 / sigma^2)
    # sigma chosen so core is dense
    sigma = radius / 1.5
    dists = np.linalg.norm(grid_points, axis=1)
    weights = np.exp(-dists**2 / (2 * sigma**2))
    
    # Filter points inside radius
    mask = dists <= radius
    grid_points = grid_points[mask]
    weights = weights[mask]
    
    if len(grid_points) == 0:
        return _generate_random_cluster(n, seed=seed)
    
    # Normalize weights to probabilities
    weights = weights / np.sum(weights)
    
    # Sample n points with replacement? No, we want distinct.
    # Multinomial sampling without replacement is hard if weights vary.
    # Simple approach: sample with replacement, then dedupe and fill missing randomly.
    # Better: weighted sampling without replacement.
    
    indices = np.arange(len(grid_points))
    selected_indices = []
    remaining_indices = indices.copy()
    remaining_weights = weights.copy()
    
    for _ in range(min(n, len(grid_points))):
        # Normalize current weights
        rsum = np.sum(remaining_weights)
        if rsum <= 0: break
        probs = remaining_weights / rsum
        idx = rng.choice(len(remaining_indices), p=probs)
        selected_indices.append(remaining_indices[idx])
        # Remove
        remaining_indices = np.delete(remaining_indices, idx)
        remaining_weights = np.delete(remaining_weights, idx)
        
        if len(remaining_indices) == 0:
            break
    
    positions = grid_points[selected_indices]
    
    # Fill remaining with random points
    if len(positions) < n:
        for _ in range(n - len(positions)):
            p = rng.uniform(-radius, radius, 3)
            positions = np.vstack([positions, p])
            
    # Jitter to break grid symmetry
    positions = positions + rng.uniform(-0.1, 0.1, positions.shape)
    
    return positions

def _generate_random_cluster(n, seed=42, min_dist=0.6):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    
    radius = 2.5 * (n ** (1.0/3.0))
    points = []
    attempts = 0
    max_attempts = 50000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)

def _relax_and_validate(positions, potential, rho):
    """Relax positions and check validity."""
    n = positions.shape[0]
    try:
        # Use the provided relax function first
        energy, relaxed = relax(positions, potential, rho)
        if _min_distance(relaxed) >= 0.5:
            return energy, relaxed
        else:
            # Try our own polish if relax failed validity
            polished_pos, polished_energy = _polish(positions, rho)
            if _min_distance(polished_pos) >= 0.5:
                return polished_energy, polished_pos
            return float('inf'), None
    except Exception:
        # Fallback to our own polish
        try:
            polished_pos, polished_energy = _polish(positions, rho)
            if _min_distance(polished_pos) >= 0.5:
                return polished_energy, polished_pos
        except Exception:
            pass
        return float('inf'), None

def _polish(positions, rho, max_time=5.0):
    """
    High-precision polish using L-BFGS-B with analytical gradient.
    """
    n = positions.shape[0]
    x0 = positions.flatten()
    
    def obj(x):
        pos = x.reshape(n, 3)
        return _morse_energy(pos, rho)
    
    def grad(x):
        pos = x.reshape(n, 3)
        return _morse_gradient(pos, rho).flatten()
    
    start_time = time.time()
    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-6, 'ftol': 1e-8}
        )
        if _min_distance(res.x.reshape(n, 3)) >= 0.5:
            return res.x.reshape(n, 3), res.fun
    except Exception:
        pass
    
    # Fallback: use the library's relax if time permits
    if _get_remaining_time() > 2.0:
        try:
            energy, relaxed = relax(positions, 'morse', rho)
            if _min_distance(relaxed) >= 0.5:
                return relaxed, energy
        except Exception:
            pass
    
    return positions, _morse_energy(positions, rho)

def _basin_hop(start_positions, potential, rho, max_iterations=50, initial_step=0.5, initial_temp=2.0):
    """
    Perform basin hopping optimization.
    Returns (best_energy, best_positions).
    """
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)
    
    if current_positions is None:
        perturbed = start_positions + np.random.normal(0, 0.1, start_positions.shape)
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    step_size = initial_step
    temp = initial_temp
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        # Perturb
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        # Relax
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            noise = np.random.normal(0, step_size * 0.5, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            continue
        
        # Accept/reject
        delta_e = new_energy - current_energy
        if delta_e < 0 or (temp > 1e-6 and np.random.rand() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        # Anneal
        temp *= 0.95
        step_size *= 0.98
        
        if it > 0 and it % 20 == 0:
            step_size *= 1.2
    
    # Final polish on the best configuration found
    if _get_remaining_time() > 3.0:
        polished_pos, polished_energy = _polish(best_positions, rho)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    return best_energy, best_positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    if rho is None:
        rho = 6
    
    best_energy = float('inf')
    best_positions = None
    
    initial_guesses = []
    
    # 1. Density-inspired decoys
    for seed in [1, 2, 3]:
        try:
            initial_guesses.append(_generate_decoy(n, seed=seed, rho=rho))
        except Exception:
            pass
    
    # 2. Random clusters
    for seed in [4, 5, 6]:
        try:
            initial_guesses.append(_generate_random_cluster(n, seed=seed))
        except Exception:
            pass
    
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
        
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    num_bh_starts = min(len(relaxed_results), 3)
    
    for idx in range(num_bh_starts):
        if _get_remaining_time() < 15.0:
            break
        
        start_energy, start_pos = relaxed_results[idx]
        
        if rho <= 3:
            initial_step = 0.8
            initial_temp = 5.0
        elif rho <= 6:
            initial_step = 0.5
            initial_temp = 3.0
        else:
            initial_step = 0.3
            initial_temp = 2.0
        
        if n <= 40:
            max_iter = 80
        elif n <= 60:
            max_iter = 50
        else:
            max_iter = 30
        
        bh_energy, bh_positions = _basin_hop(
            start_pos, potential, rho,
            max_iterations=max_iter,
            initial_step=initial_step,
            initial_temp=initial_temp
        )
        
        if bh_positions is not None and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions.copy()
        
        if _get_remaining_time() > 20.0 and idx == 0:
            bh_energy2, bh_positions2 = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter // 2,
                initial_step=initial_step * 1.5,
                initial_temp=initial_temp * 1.5
            )
            if bh_positions2 is not None and bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_positions = bh_positions2.copy()
    
    if best_positions is not None and _get_remaining_time() > 2.0:
        polished_pos, polished_energy = _polish(best_positions, rho)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    if best_positions is None:
        best_positions = _generate_random_cluster(n, seed=99)
        if not _is_valid(best_positions):
            # Last resort: scale up
            best_positions = best_positions * 1.5
            if not _is_valid(best_positions):
                # Force valid by spreading out
                rng = np.random.default_rng(123)
                best_positions = rng.uniform(-5, 5, (n, 3))
                # This might still be invalid, but it's a fallback
                # If still invalid, the evaluator will reject, but we must return something
                # Let's try to fix it quickly
                for _ in range(10):
                    if _is_valid(best_positions):
                        break
                    best_positions += rng.normal(0, 0.1, (n, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
