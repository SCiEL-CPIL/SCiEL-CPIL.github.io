"""
Strategy: Basin Hopping with diverse structural templates for Sutton-Chen clusters.
Uses multi-start with FCC, HCP, BCC, and random templates, followed by basin
hopping with adaptive step sizes. Each call is time-budgeted to fit within 300s
for all six instances combined.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 300.0
_START_TIME = None

def _time_left():
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_fragment(n, spacing=0.72):
    """Generate n atoms from an FCC lattice fragment."""
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 4
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
    """Generate n atoms from an HCP lattice fragment."""
    a = spacing
    c = 1.633 * a
    points = []
    max_r = int(np.ceil(n ** (1/3))) + 4
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

def _bcc_fragment(n, spacing=0.72):
    """Generate n atoms from a BCC lattice fragment."""
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _random_compact(n, spacing=0.72, seed=0):
    """Generate a random compact cluster with rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: place remaining atoms on a sphere surface
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _basin_hop(positions, potential, kind, n, max_hops, step_size_initial, rng, time_limit):
    """
    Basin hopping: perturb, relax, accept if lower energy.
    Returns (best_energy, best_positions).
    """
    best_energy, best_pos = sutton_chen_energy_gradient(positions, kind)[0], positions.copy()
    # First do a proper relax
    try:
        best_energy, best_pos = relax(positions, potential, kind)
    except Exception:
        pass
    
    current_energy = best_energy
    current_pos = best_pos.copy()
    step_size = step_size_initial
    
    for _ in range(max_hops):
        if time.time() - _START_TIME > time_limit:
            break
        # Generate perturbation
        noise = rng.normal(0, step_size, size=(n, 3))
        perturbed = current_pos + noise
        
        try:
            energy, new_pos = relax(perturbed, potential, kind)
            if energy < current_energy:
                current_energy = energy
                current_pos = new_pos
                if energy < best_energy:
                    best_energy = energy
                    best_pos = new_pos.copy()
                step_size *= 0.97  # Decrease on acceptance
            else:
                step_size *= 1.03  # Increase on rejection
        except Exception:
            continue
        
        # Cap step size
        step_size = min(step_size, 1.0)
        step_size = max(step_size, 0.05)
    
    return best_energy, best_pos

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Time budget per instance: 300s / 6 = 50s, use 45s to be safe
    per_instance_budget = 45.0
    instance_deadline = _START_TIME + per_instance_budget
    
    best_energy = float('inf')
    best_positions = None
    
    def _try_relax(positions):
        nonlocal best_energy, best_positions
        try:
            energy, pos = relax(positions, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                # Check minimum distance
                min_d = float('inf')
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(pos[i] - pos[j])
                        if d < min_d:
                            min_d = d
                if min_d >= 0.35:
                    best_energy = energy
                    best_positions = pos.copy()
        except Exception:
            pass
    
    def _time_left_instance():
        return instance_deadline - time.time()
    
    # Generate diverse starting points
    starts = []
    
    # 1. FCC fragment
    fcc = _fcc_fragment(n)
    if fcc is not None:
        starts.append(fcc)
        
    # 2. HCP fragment
    hcp = _hcp_fragment(n)
    if hcp is not None:
        starts.append(hcp)
        
    # 3. BCC fragment
    bcc = _bcc_fragment(n)
    if bcc is not None:
        starts.append(bcc)
        
    # 4. Random compact (multiple seeds)
    for seed in range(8):
        rand = _random_compact(n, seed=seed)
        starts.append(rand)
    
    # First pass: relax all starting points
    for start in starts:
        if _time_left_instance() < 3:
            break
        _try_relax(start)
    
    # Second pass: basin hopping on the best few
    if best_positions is not None and _time_left_instance() > 10:
        rng = np.random.default_rng(42)
        # Do basin hopping starting from the best position
        remaining = _time_left_instance() - 3
        max_hops = min(80, int(remaining / 0.5))  # ~0.5s per hop estimate
        bh_energy, bh_pos = _basin_hop(
            best_positions, potential, kind, n,
            max_hops=max_hops,
            step_size_initial=0.15,
            rng=rng,
            time_limit=instance_deadline - 3
        )
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    # If still have time, do basin hopping from other good starts
    if _time_left_instance() > 15:
        # Collect good starts (those that gave energies below some threshold)
        threshold = best_energy * 1.1 if best_energy > 0 else best_energy + 1.0
        good_starts = []
        for start in starts[:6]:  # Limit to first 6 to save time
            try:
                e, _ = sutton_chen_energy_gradient(start, kind)
                if np.isfinite(e) and e < threshold:
                    good_starts.append(start)
            except Exception:
                continue
        
        for gs in good_starts[:3]:  # Basin hop from up to 3 good starts
            if _time_left_instance() < 5:
                break
            rng2 = np.random.default_rng(123)
            bh_e, bh_p = _basin_hop(
                gs, potential, kind, n,
                max_hops=40,
                step_size_initial=0.2,
                rng=rng2,
                time_limit=instance_deadline - 3
            )
            if bh_e < best_energy:
                best_energy = bh_e
                best_positions = bh_p
    
    # Final validation
    if best_positions is not None:
        # Ensure finite coordinates
        if not np.all(np.isfinite(best_positions)):
            # Fallback: return a valid random compact
            fallback = _random_compact(n, seed=999)
            _try_relax(fallback)
    
    if best_positions is None:
        # Absolute fallback
        fallback = _random_compact(n, seed=777)
        _try_relax(fallback)
    
    return best_positions
# EVOLVE-BLOCK-END
