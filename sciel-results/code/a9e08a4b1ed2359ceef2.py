"""
Morse cluster global minima via Deterministic Multi-Start Basin Hopping.
Strategy:
1. Generate a rich library of structurally diverse seeds:
   - Optimized Spherical Shells (Magic numbers)
   - FCC and BCC lattice fragments
   - Greedy dense packing
   - Random compact clusters
2. Pre-relax all seeds to local minima.
3. Perform a single, global best-tracking basin hopping loop.
   - Each step perturbs the current global best.
   - If the perturbation leads to a lower energy (after relaxation), it becomes the new best.
   - If rejected, we kick the step size and try again.
   - This avoids the overhead of managing multiple chains and allows focusing all computational budget on refining the best candidate.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
_DEADLINE = _START_TIME + 285.0  # 15s safety margin before 300s limit

def _time_remaining():
    return _DEADLINE - time.time()

def _morse_energy(coords, rho):
    """Compute Morse energy; return inf if any pair < 0.5 apart."""
    n = len(coords)
    if n < 2:
        return 0.0
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        return np.inf
    iu = np.triu_indices(n, 1)
    r = dists[iu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)

def _ensure_valid(coords, rho, rng, n):
    """Ensure coordinates are valid (min distance >= 0.5)."""
    if len(coords) != n:
        return _build_random_compact(n, rng)
    # Check distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        # Relax to fix
        try:
            energy, relaxed = relax(coords, 'morse', rho)
            if np.isfinite(energy):
                # Check again
                diff = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=-1))
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) >= 0.5:
                    return relaxed
        except:
            pass
        # Fallback: random
        return _build_random_compact(n, rng)
    return coords

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like compact cluster."""
    points = []
    a = spacing * 0.9
    max_idx = int(np.ceil((n / 4) ** (1/3))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k]) * a
                p3 = np.array([i + 0.5, j, k + 0.5]) * a
                p4 = np.array([i, j + 0.5, k + 0.5]) * a
                for p in [p1, p2, p3, p4]:
                    if len(points) < n:
                        points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = np.array(points[0]) + np.random.uniform(-0.5, 0.5, 3)
            points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_bcc_cluster(n, spacing=1.0):
    """Build a BCC-like compact cluster."""
    points = []
    a = spacing * 1.0
    max_idx = int(np.ceil((n / 2) ** (1/3))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                for p in [p1, p2]:
                    if len(points) < n:
                        points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = np.array(points[0]) + np.random.uniform(-0.5, 0.5, 3)
            points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_icosahedral_shells(n, spacing=1.0):
    """Build icosahedral shells."""
    points = []
    phi = (1 + np.sqrt(5)) / 2
    r = spacing * 1.0
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    for v in verts:
        v = np.array(v)
        v = v / np.linalg.norm(v) * r
        points.append(v)
    
    if n > 12:
        # Add 30 points at radius r*1.5 (approximating the second shell)
        for i in range(n - 12):
            angle1 = np.random.uniform(0, 2*np.pi)
            angle2 = np.random.uniform(0, np.pi)
            r2 = r * 1.5
            x = r2 * np.sin(angle2) * np.cos(angle1)
            y = r2 * np.sin(angle2) * np.sin(angle1)
            z = r2 * np.cos(angle2)
            points.append([x, y, z])
            
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_spherical_shells(n, spacing=1.0):
    """Build spherical shells."""
    points = []
    shells = [1, 12, 30, 55, 91]
    radii = [0, 1.0, 1.5, 2.0, 2.5]
    
    points.append([0,0,0])
    
    if n > 1:
        phi = (1 + np.sqrt(5)) / 2
        verts = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ]
        for v in verts[:min(12, n-1)]:
            v = np.array(v)
            v = v / np.linalg.norm(v) * radii[1]
            points.append(v)
            
    while len(points) < n:
        shell_idx = min(len(radii)-1, int((len(points)**0.5)**0.5))
        r = radii[shell_idx]
        angle1 = np.random.uniform(0, 2*np.pi)
        angle2 = np.random.uniform(0, np.pi)
        x = r * np.sin(angle2) * np.cos(angle1)
        y = r * np.sin(angle2) * np.sin(angle1)
        z = r * np.cos(angle2)
        points.append([x, y, z])
        
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_random_compact(n, rng, spacing=1.0):
    """Generate a compact random cluster."""
    points = []
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_dist = 0.55 * spacing
    max_attempts = 100000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        ok = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                ok = False
                break
        if ok:
            points.append(p)
        attempts += 1
    while len(points) < n:
        base = points[rng.integers(len(points))] if points else np.zeros(3)
        p = base + rng.uniform(-0.1, 0.1, 3)
        points.append(p)
    return np.array(points)

def _relax_and_check(coords, rho):
    """Relax coordinates and check validity."""
    try:
        energy, relaxed = relax(coords, 'morse', rho)
        if np.isfinite(energy):
            diff = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                return energy, relaxed
    except:
        pass
    return np.inf, coords

def construct(instance):
    """Find global minimum of Morse cluster via Deterministic Multi-Start Basin Hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Allocate time based on cluster size
    if n <= 40:
        time_limit = 40.0
    elif n <= 60:
        time_limit = 50.0
    else:
        time_limit = 60.0
        
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse seeds
    seeds = []
    seeds.append(_build_fcc_cluster(n))
    seeds.append(_build_bcc_cluster(n))
    seeds.append(_build_icosahedral_shells(n))
    seeds.append(_build_spherical_shells(n))
    seeds.append(_build_random_compact(n, rng))
    
    # Filter valid seeds
    valid_seeds = []
    for seed in seeds:
        if len(seed) == n:
            valid_seeds.append(seed)
            
    if not valid_seeds:
        valid_seeds = [_build_random_compact(n, rng)]
        
    # Pre-relax all seeds and find the best starting point
    best_energy = np.inf
    best_coords = None
    for seed in valid_seeds:
        seed = _ensure_valid(seed, rho, rng, n)
        energy, relaxed = _relax_and_check(seed, rho)
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_coords = relaxed.copy()
            
    if best_coords is None:
        # Fallback
        coords = _build_random_compact(n, rng)
        best_energy, best_coords = _relax_and_check(coords, rho)
        
    # Basin Hopping Loop
    start_time = time.time()
    step_size = 0.2
    min_step = 0.05
    max_step = 0.5
    stagnation_count = 0
    max_stagnation = 50
    
    while _time_remaining() > 5.0:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Perturb the best structure
        perturbed = best_coords + rng.normal(0, step_size, (n, 3))
        
        # Relax
        new_energy, new_coords = _relax_and_check(perturbed, rho)
        
        if np.isfinite(new_energy):
            if new_energy < best_energy:
                # Accept
                best_energy = new_energy
                best_coords = new_coords.copy()
                step_size *= 0.95  # Shrink step on success
                stagnation_count = 0
            else:
                # Reject
                step_size = min(max_step, step_size * 1.1)  # Grow step on failure
                stagnation_count += 1
                
                # If stuck for a long time, try a bigger kick or reset
                if stagnation_count > max_stagnation:
                    # Random restart from one of the seeds
                    seed = valid_seeds[rng.integers(len(valid_seeds))]
                    seed = _ensure_valid(seed, rho, rng, n)
                    energy, relaxed = _relax_and_check(seed, rho)
                    if np.isfinite(energy):
                        best_coords = relaxed.copy()
                        best_energy = energy
                    step_size = 0.3
                    stagnation_count = 0
        else:
            # Invalid, increase step
            step_size = min(max_step, step_size * 1.2)
            stagnation_count += 1
            if stagnation_count > max_stagnation:
                step_size = 0.3
                stagnation_count = 0
                
    # Final check
    if best_coords is None or not np.isfinite(best_energy):
        coords = _build_random_compact(n, rng)
        best_energy, best_coords = _relax_and_check(coords, rho)
        
    # Ensure final validity
    best_coords = _ensure_valid(best_coords, rho, rng, n)
    
    return best_coords
# EVOLVE-BLOCK-END
