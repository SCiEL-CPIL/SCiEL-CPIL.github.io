"""Morse cluster global minima: Core-Shell and Basin Hopping search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _fibonacci_sphere(n, radius=1.0, seed=0):
    """Generate n points on a sphere using the Fibonacci lattice."""
    rng = np.random.default_rng(seed)
    # Add slight randomization to break symmetry
    offset = rng.uniform(0, 2 * np.pi, 1)[0]
    
    points = np.zeros((n, 3))
    # Golden angle
    phi = np.pi * (3.0 - np.sqrt(5.0))
    
    for i in range(n):
        z = 1.0 - (i / float(n - 1)) * 2.0  # z goes from 1 to -1
        radius_xy = np.sqrt(1.0 - z * z)
        theta = phi * i + offset
        
        points[i, 0] = radius_xy * np.cos(theta) * radius
        points[i, 1] = radius_xy * np.sin(theta) * radius
        points[i, 2] = z * radius
        
    return points


def _random_ball(n, radius, seed=0, min_dist=0.0):
    """Generate n random points in a ball of given radius."""
    rng = np.random.default_rng(seed)
    points = []
    attempts = 0
    max_attempts = n * 100
    
    while len(points) < n and attempts < max_attempts:
        # Generate point in cube, check if in ball
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if min_dist > 0:
                # Check min distance
                ok = True
                for q in points:
                    if np.linalg.norm(p - q) < min_dist:
                        ok = False
                        break
                if not ok:
                    attempts += 1
                    continue
            points.append(p)
        attempts += 1
        
    # If we didn't get enough (e.g. min_dist too high), fill rest randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
        else:
            # Scale to fit
            if np.linalg.norm(p) > 1e-6:
                p = p / np.linalg.norm(p) * (radius * rng.uniform(0.5, 1.0))
            else:
                p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)


def _core_shell_seed(n, seed=0, core_fraction=0.4):
    """
    Generate a seed with a dense core and a spherical shell.
    This mimics the core-shell structure of many cluster minima.
    """
    n_core = max(1, int(n * core_fraction))
    n_shell = n - n_core
    
    # Estimate radii
    # Total volume ~ n * v_atom. Let v_atom ~ 1.
    # R_total ~ n^(1/3)
    R_total = n ** (1.0/3.0) * 0.9
    R_core = R_total * (core_fraction ** (1.0/3.0))
    R_shell = R_total
    
    core = _random_ball(n_core, R_core, seed=seed, min_dist=0.8)
    shell = _fibonacci_sphere(n_shell, radius=R_shell, seed=seed+1)
    
    # Shift shell slightly if needed? No, centered is fine.
    # Combine
    points = np.vstack([core, shell])
    
    # Shuffle to mix
    idx = np.random.default_rng(seed+2).permutation(n)
    points = points[idx]
    
    return points


def _random_compact(n, spacing=1.0, seed=0):
    """Generate n random points in a compact sphere with minimum distance constraint."""
    # This is similar to _random_ball but with min_dist
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.6 * spacing
    return _random_ball(n, radius, seed=seed, min_dist=min_dist)


def _relax_and_kick(seed, potential, rho, kick_scale, max_kicks=5):
    """
    Perform basin hopping: Relax, Kick, Relax, Kick...
    Returns the best position and energy found.
    """
    best_pos = seed.copy()
    best_energy = np.inf
    
    current_pos = seed.copy()
    
    for k in range(max_kicks):
        if _time_left() < 5.0:
            break
            
        try:
            energy, pos = relax(current_pos, potential, rho)
        except Exception:
            break
            
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
        # Kick
        # Use a deterministic seed based on kick index to ensure reproducibility if needed,
        # but here we just want diversity.
        rng = np.random.default_rng(k + 1000)
        noise = rng.normal(0, kick_scale, pos.shape)
        current_pos = pos + noise
        
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations via Core-Shell and Basin Hopping."""
    _instance_call_count[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Estimate per-instance time budget
    # Total 280s for 6 instances. ~46s each.
    per_instance_budget = 44.0
    if _instance_call_count[0] <= 2:
        per_instance_budget = 46.0
    elif _instance_call_count[0] >= 5:
        per_instance_budget = 40.0
        
    instance_deadline = time.time() + per_instance_budget
    
    # Define seeds
    seeds = []
    
    # 1. Core-Shell seeds (various core fractions)
    for cf in [0.2, 0.4, 0.6]:
        seeds.append(_core_shell_seed(size, seed=100+int(cf*10), core_fraction=cf))
        
    # 2. Random Compact seeds
    for seed_val in [0, 1, 2]:
        seeds.append(_random_compact(size, seed=seed_val))
        
    # 3. Pure Shell / Loose Random
    seeds.append(_random_ball(size, size**(1/3)*1.2, seed=99))
    
    best_energy = np.inf
    best_positions = None
    
    # Determine kick scale
    kick_scale = 0.15
    
    for i, seed in enumerate(seeds):
        if time.time() > instance_deadline - 5.0:
            break
            
        # Adjust max_kicks based on time left and seed index
        max_kicks = 5
        if size > 60:
            max_kicks = 4 # Larger clusters take longer to relax
        
        pos, energy = _relax_and_kick(seed, potential, parameter, kick_scale, max_kicks=max_kicks)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
            
    # If no valid position found (shouldn't happen), fallback
    if best_positions is None:
        best_positions = _random_compact(size, seed=123)
        
    # Final refinement: If we have time, do a few more kicks on the BEST result
    if best_positions is not None and time.time() < instance_deadline - 2.0:
        # Use a smaller kick for fine-tuning
        fine_kick_scale = 0.05
        # Try 2-3 more kicks
        current = best_positions.copy()
        for _ in range(3):
            if time.time() > instance_deadline - 1.0:
                break
            try:
                noise = np.random.default_rng(42).normal(0, fine_kick_scale, current.shape)
                perturbed = current + noise
                energy, pos = relax(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
                    current = pos
            except Exception:
                break

    # Safety check: minimum distance
    n = len(best_positions)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < 0.5:
                mid = 0.5 * (best_positions[i] + best_positions[j])
                direction = best_positions[i] - best_positions[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = np.array([1.0, 0.0, 0.0])
                    norm = 1.0
                direction /= norm
                offset = 0.26 * direction
                best_positions[i] = mid + offset
                best_positions[j] = mid - offset

    return best_positions
# EVOLVE-BLOCK-END
