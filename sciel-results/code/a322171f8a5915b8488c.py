"""
Strategy: Multi-start Basin Hopping with structural diversity.
Instead of fixed templates (FCC/HCP/BCC) as the primary driver, we use:
1. A diverse set of initial configurations: FCC fragments, HCP fragments,
   random compact clusters, and "liquid-like" random starts.
2. An adaptive basin-hopping loop that runs as long as time permits.
   - Step size starts moderate and adapts: increases on rejection, decreases on acceptance.
   - We track the global best and also keep a "current" state for local hopping.
   - We periodically "kick" the current state to escape deep local minima.
3. Parallelization is not used (max 4 threads allowed, but GIL limits CPU-bound Python).
   We rely on efficient single-threaded search with tight time management.
4. We use the provided relax() function which wraps L-BFGS-B.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 295.0  # Safety margin before 300s hard kill
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_fragment(n, spacing=0.72):
    """Generate n atoms from an FCC lattice fragment centered at origin."""
    # FCC lattice: points (x,y,z) where x,y,z are integers and x+y+z is even, scaled by spacing/2
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 3
    points = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if (x + y + z) % 2 == 0:
                    points.append([x * 0.5 * a, y * 0.5 * a, z * 0.5 * a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _hcp_fragment(n, spacing=0.72):
    """Generate n atoms from an HCP lattice fragment centered at origin."""
    a = spacing
    c = 1.633 * a
    points = []
    max_r = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            x = a * (i + 0.5 * j)
            y = a * (np.sqrt(3) / 2 * j)
            for k in range(-max_r, max_r + 1):
                points.append([x, y, k * c])
                points.append([x + 2*a/3, y + a*np.sqrt(3)/3, (k + 0.5) * c])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _random_compact(n, spacing=0.72, seed=0):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    min_dist = 0.55 * spacing
    attempts = 0
    max_attempts = n * 150
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: add random points, relaxer will fix overlaps
        remaining = n - len(points)
        extra = rng.uniform(-radius, radius, (remaining, 3))
        points.extend(extra.tolist())
    return np.array(points)

def _liquid_random(n, spacing=0.72, seed=0):
    """Generate a random liquid-like cluster (no min distance constraint)."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    return np.array(points)

def _min_distance(positions):
    """Compute minimum pairwise distance."""
    n = len(positions)
    min_d = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_d:
                min_d = d
    return min_d

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    spacing = 0.72
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse starting points
    starts = []
    
    # 1. FCC fragment
    fcc = _fcc_fragment(n, spacing)
    if fcc is not None:
        starts.append(fcc)
        
    # 2. HCP fragment
    hcp = _hcp_fragment(n, spacing)
    if hcp is not None:
        starts.append(hcp)
        
    # 3. Random compact (multiple seeds)
    for seed in range(6):
        rand = _random_compact(n, spacing, seed)
        starts.append(rand)
        
    # 4. Liquid random (multiple seeds)
    for seed in range(4):
        liq = _liquid_random(n, spacing, seed)
        starts.append(liq)
        
    # Relax each start
    for start in starts:
        if _time_left() < 3:
            break
        try:
            energy, positions = relax(start, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                min_d = _min_distance(positions)
                if min_d >= 0.35:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
    
    # Basin hopping on the best found
    if best_positions is not None and _time_left() > 20:
        current_best = best_positions.copy()
        current_energy = best_energy
        step_size = 0.15
        rng = np.random.default_rng(42)
        
        # Adaptive basin hopping
        max_hops = 200
        for hop in range(max_hops):
            if _time_left() < 3:
                break
            
            # Perturb
            noise = rng.normal(0, step_size, (n, 3))
            perturbed = current_best + noise
            
            try:
                energy, positions = relax(perturbed, potential, kind)
                if np.isfinite(energy):
                    min_d = _min_distance(positions)
                    if min_d < 0.35:
                        # Invalid, skip
                        continue
                    
                    if energy < current_energy:
                        # Accept
                        current_energy = energy
                        current_best = positions
                        step_size *= 0.92  # Decrease step size
                    else:
                        # Reject, increase step size
                        step_size *= 1.08
                        # Cap step size
                        step_size = min(step_size, 0.5)
                else:
                    step_size *= 1.1
            except Exception:
                continue
            
            # Periodically try a larger kick
            if hop % 10 == 0 and _time_left() > 10:
                kick_noise = rng.normal(0, step_size * 2, (n, 3))
                kicked = current_best + kick_noise
                try:
                    energy, positions = relax(kicked, potential, kind)
                    if np.isfinite(energy):
                        min_d = _min_distance(positions)
                        if min_d >= 0.35 and energy < current_energy:
                            current_energy = energy
                            current_best = positions
                except Exception:
                    pass
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_best
    
    # Final safety check
    if best_positions is not None:
        min_d = _min_distance(best_positions)
        if min_d < 0.35:
            # Try to find any valid configuration from starts
            for start in starts:
                try:
                    energy, positions = relax(start, potential, kind)
                    if np.isfinite(energy):
                        md = _min_distance(positions)
                        if md >= 0.35:
                            best_positions = positions
                            best_energy = energy
                            break
                except Exception:
                    continue
            # If still invalid, return a safe fallback
            if best_positions is not None and _min_distance(best_positions) < 0.35:
                # Generate a simple valid configuration
                best_positions = _random_compact(n, spacing, seed=999)
                # Ensure validity
                md = _min_distance(best_positions)
                if md < 0.35:
                    # Spread out more
                    best_positions = best_positions * 1.5
                    md = _min_distance(best_positions)
                    if md < 0.35:
                        best_positions = best_positions * 2.0
    
    return best_positions
# EVOLVE-BLOCK-END
