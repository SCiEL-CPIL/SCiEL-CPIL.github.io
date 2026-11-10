"""
Strategy: Custom Basin Hopping Global Optimization
- Uses icosahedral and random compact seeds to start.
- Implements a Basin Hopping loop: Perturb -> Relax -> Accept/Reject.
- This allows escaping local minima that the current single-shot relax() misses.
- Deadline-guarded to ensure valid output within 330s.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # Safety margin below 330s

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_vertices():
    """Generate vertices of a regular icosahedron normalized to unit radius."""
    phi = (1 + 5**0.5) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _generate_icosahedral_seed(n):
    """
    Generate a seed based on Mackay icosahedra.
    For n=13, it's a perfect icosahedron.
    For n>13, we start with 13 and add atoms on the surface (approximate).
    For n<13, we take a subset (less ideal, but rare in this task).
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Get base 13-atom icosahedron
    v12 = _icosahedron_vertices()
    base13 = np.vstack([np.array([[0.0, 0.0, 0.0]]), v12])
    
    if n <= 13:
        # Just take the first n atoms (center + some vertices)
        # This is a rough approximation for non-magic numbers < 13
        return base13[:n]
        
    # For n > 13, we have 13 atoms. Need n-13 more.
    # We will place them on the surface of the convex hull (approximated by sphere of radius 1.2)
    # To make it more structured, we can try to place them near the vertices/edges.
    # Simple heuristic: random directions scaled to 1.2 radius.
    rng = np.random.default_rng(12345)
    extra = []
    for _ in range(n - 13):
        dir = rng.normal(size=3)
        dir /= np.linalg.norm(dir)
        # Place slightly outside the 12 vertices (radius 1.0)
        p = dir * 1.2
        extra.append(p)
        
    if extra:
        base13 = np.vstack([base13, np.array(extra)])
        
    return base13

def _generate_random_compact(n, rng):
    """
    Generate a random compact cluster using rejection sampling.
    """
    points = []
    # Estimate radius based on volume
    # Density ~ 1 atom / (0.7^3) volume roughly
    radius = (n / 2.0) ** (1.0 / 3.0) * 0.8 
    min_dist = 0.6
    
    # Try to fill a sphere
    attempts = 0
    while len(points) < n and attempts < n * 100:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        # Check min distance
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < min_dist:
                continue
        points.append(p)
        
    if len(points) < n:
        # Fallback: just random points in a box if rejection fails
        points = []
        for _ in range(n):
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)

def _basin_hopping(start_pos, potential, kind, rng, max_iter=50, sigma_start=0.1, sigma_end=0.01):
    """
    Perform Basin Hopping starting from start_pos.
    Returns (best_energy, best_pos).
    """
    current_pos = start_pos.copy()
    try:
        current_energy, current_pos = relax(current_pos, potential, kind)
    except:
        return float('inf'), start_pos
        
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    n = len(current_pos)
    
    for i in range(max_iter):
        if _time_left() < 5.0:
            break
            
        # Anneal sigma
        progress = i / max_iter
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        # Perturb
        perturbed = current_pos + rng.normal(0, sigma, current_pos.shape)
        
        try:
            new_energy, new_pos = relax(perturbed, potential, kind)
        except:
            continue
            
        # Metropolis acceptance (simplified: always accept if better, or random if worse to escape)
        # For global minima, we want to explore. 
        # If new_energy < current_energy, accept.
        # If new_energy > current_energy, accept with probability exp(-dE/T).
        # But T is tricky. Let's just accept if it's better, or with a small prob if it's worse but not too bad.
        # Actually, standard BH accepts any move that leads to a local min, but keeps the best so far.
        # The "acceptance" in BH is usually just: move to the new local min. 
        # To avoid getting stuck, we rely on the perturbation.
        # A common variant: Accept if new_energy < current_energy OR with prob p.
        # Let's use a simple rule: Accept if new_energy < current_energy + 0.1 (small tolerance)
        # Or just always accept the relaxed state as the new current, but track the global best.
        # This is "steepest descent" in the space of local minima.
        
        # To ensure we don't just drift, let's use a Metropolis-like acceptance with a fixed 'temperature'
        # that decays.
        T = 0.5 * (1.0 - progress) + 0.01
        dE = new_energy - current_energy
        if dE < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-dE / T)
            
        if accept:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
                
    return best_energy, best_pos

def construct(instance):
    """
    Strategy:
    1. Generate diverse seeds (Icosahedral, Random Compact).
    2. Run Basin Hopping from each seed.
    3. Keep the best result.
    """
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    best_energy = float('inf')
    best_pos = None
    
    # Define seeds
    seeds = []
    
    # 1. Icosahedral seed
    try:
        ico_seed = _generate_icosahedral_seed(n)
        if len(ico_seed) == n:
            seeds.append(ico_seed)
    except:
        pass
        
    # 2. Random compact seeds (3 variations)
    rng = np.random.default_rng(999)
    for i in range(3):
        try:
            rand_seed = _generate_random_compact(n, rng)
            if len(rand_seed) == n:
                seeds.append(rand_seed)
        except:
            pass
            
    # 3. If n is close to a magic number, maybe try a specific geometry?
    # For now, the ico seed covers 13, 55 roughly.
    # For 38, 59, 75, 79, 80, ico+random is a good mix.
    
    # If no seeds generated (shouldn't happen), create a dummy
    if not seeds:
        seeds.append(np.random.default_rng(1).normal(0, 1, (n, 3)))

    # Run BH on each seed
    bh_rng = np.random.default_rng(54321)
    
    for i, seed in enumerate(seeds):
        if _time_left() < 10.0:
            break
            
        # Allocate time based on number of seeds
        # If 4 seeds, give each ~25% of time?
        # BH max_iter is a proxy for time.
        # Let's just run BH with a fixed max_iter and let the deadline guard stop it.
        
        e, pos = _basin_hopping(seed, potential, kind, bh_rng, max_iter=80)
        
        if e < best_energy:
            best_energy = e
            best_pos = pos
            
    # If we still have time, try more random seeds or refine the best one
    if best_pos is not None and _time_left() > 20.0:
        # Refine the best one with more BH iterations
        e, pos = _basin_hopping(best_pos, potential, kind, bh_rng, max_iter=100)
        if e < best_energy:
            best_energy = e
            best_pos = pos
            
    # Fallback
    if best_pos is None:
        # Return a valid but high energy structure
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    return best_pos
# EVOLVE-BLOCK-END
