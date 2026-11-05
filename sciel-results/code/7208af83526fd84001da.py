# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # 320s wall clock limit to be safe

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy_gradient(x, rho):
    """
    Wrapper for morse_energy_gradient that reshapes input/output for scipy.
    x: (n*3,) array
    Returns: (energy, gradient) where gradient is (n*3,)
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, grad = morse_energy_gradient(positions, rho)
        return energy, grad.flatten()
    except Exception:
        return 1e10, np.zeros_like(x)

def _generate_random_cluster(n, seed, radius_factor=1.5):
    """Generate a random cluster with minimum distance constraints."""
    rng = np.random.default_rng(seed)
    # Estimate radius: n points in volume V ~ n * (1/2)^3 * (4/3) pi ? 
    # Density of equilibrium is roughly 1 point per unit volume (since r_eq=1).
    # Volume of sphere R is 4/3 pi R^3. 
    # If we pack n points, R ~ (3n/4pi)^(1/3) * packing_factor.
    # Let's use a slightly larger radius to allow flexibility.
    radius = radius_factor * (n ** (1.0/3.0))
    
    points = []
    max_attempts = 10000
    attempts = 0
    min_dist = 0.51 # Slightly larger than 0.5 to avoid boundary issues
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check min distance
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
            attempts += 1
            
    # If we didn't get enough points, relax the constraint or add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _generate_crystalline_template(n, seed, lattice_type='fcc'):
    """Generate a crystalline template."""
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0))
    lattice_points = []
    
    # Generate lattice points in a box
    r = int(math.ceil(radius)) + 1
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            for k in range(-r, r+1):
                if lattice_type == 'fcc':
                    basis = [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]
                elif lattice_type == 'bcc':
                    basis = [(0,0,0), (0.5,0.5,0.5)]
                else: # hcp approximation
                    basis = [(0,0,0), (0, 1/math.sqrt(3), 0.5)]
                
                for dx, dy, dz in basis:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
    
    if len(lattice_points) >= n:
        # Sort by distance to center
        lattice_points.sort(key=lambda p: np.linalg.norm(p))
        # Add small random jitter to break symmetry and avoid exact degeneracy
        jitter = rng.uniform(-0.01, 0.01, (n, 3))
        return np.array(lattice_points[:n]) + jitter
    else:
        # Fill with random points
        remaining = n - len(lattice_points)
        for _ in range(remaining):
            p = rng.uniform(-radius, radius, 3)
            lattice_points.append(p)
        return np.array(lattice_points)

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # Strategy: Multi-start with diverse initializations
    # 1. Crystalline templates (FCC, BCC)
    # 2. Random clusters with different densities
    # 3. Jittered crystals
    
    initializations = []
    
    # Add crystalline templates
    for lattice in ['fcc', 'bcc']:
        for seed in [1, 2, 3]:
            try:
                initializations.append(_generate_crystalline_template(n, seed, lattice))
            except:
                pass
                
    # Add random clusters
    for seed in range(10):
        try:
            initializations.append(_generate_random_cluster(n, seed, radius_factor=1.5))
        except:
            pass
            
    # Shuffle initializations to mix types
    np.random.shuffle(initializations)
    
    # Iterate through initializations, relaxing each
    # Use a time budget per attempt to ensure we try many
    # Total time ~ 300s. If we have 20 initializations, ~15s each.
    # But relax might be fast. Let's just loop until time is up.
    
    for i, init_pos in enumerate(initializations):
        if _get_remaining_time() < 5.0:
            break
            
        # Try to relax this initialization
        # The 'relax' function in the suite is likely a robust local optimizer
        # We can also try L-BFGS-B directly for speed if relax is slow
        
        try:
            # Option 1: Use the provided relax function
            # This is robust but might be slow.
            # Let's try it first.
            energy, positions = relax(init_pos, potential, rho)
            
            # Verify validity
            min_dist = np.inf
            for j in range(n):
                for k in range(j+1, n):
                    d = np.linalg.norm(positions[j] - positions[k])
                    if d < min_dist:
                        min_dist = d
                    if d < 0.5:
                        min_dist = 0.0
                        break
                if min_dist < 0.5:
                    break
                    
            if min_dist >= 0.5 and energy < best_energy:
                best_energy = energy
                best_positions = positions
                
        except Exception:
            # If relax fails, try L-BFGS-B
            try:
                x0 = init_pos.flatten()
                res = minimize(_morse_energy_gradient, x0, args=(rho,), method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30})
                
                if res.success or res.fun < best_energy:
                    pos = res.x.reshape(n, 3)
                    # Check validity
                    min_dist = np.inf
                    for j in range(n):
                        for k in range(j+1, n):
                            d = np.linalg.norm(pos[j] - pos[k])
                            if d < min_dist:
                                min_dist = d
                            if d < 0.5:
                                min_dist = 0.0
                                break
                        if min_dist < 0.5:
                            break
                            
                    if min_dist >= 0.5 and res.fun < best_energy:
                        best_energy = res.fun
                        best_positions = pos
            except Exception:
                pass

    # If no valid positions found, return a random valid one as fallback
    if best_positions is None:
        best_positions = _generate_random_cluster(n, 999, radius_factor=2.0)
        
    return best_positions
# EVOLVE-BLOCK-END
