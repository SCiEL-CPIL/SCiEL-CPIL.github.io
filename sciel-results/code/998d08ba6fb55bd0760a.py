"""Sutton-Chen transition-metal cluster global minima: Structure prediction via hierarchical growth and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Time management
_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 10.0

def _time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

# --- Structural initialization helpers ---

def _generate_lattice_points(lattice_type, n, spacing=0.72):
    """Generate points from a lattice structure filled into a sphere."""
    if n == 0:
        return np.zeros((0, 3))
    
    a = spacing
    if lattice_type == 'fcc':
        atoms_per_vol = 4.0 / (a**3)
    elif lattice_type == 'bcc':
        atoms_per_vol = 2.0 / (a**3)
    else:
        atoms_per_vol = 3.0 / (a**3)
    
    volume_needed = n / atoms_per_vol
    R = (3.0 * volume_needed / (4.0 * np.pi)) ** (1.0/3.0)
    
    max_coord = int(np.ceil(R / a)) + 2
    candidates = []
    
    if lattice_type == 'fcc':
        for ix in range(-max_coord, max_coord + 1):
            for iy in range(-max_coord, max_coord + 1):
                for iz in range(-max_coord, max_coord + 1):
                    candidates.append(np.array([ix * a, iy * a, iz * a]))
                    candidates.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, iz * a]))
                    candidates.append(np.array([(ix + 0.5) * a, iy * a, (iz + 0.5) * a]))
                    candidates.append(np.array([ix * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    elif lattice_type == 'bcc':
        for ix in range(-max_coord, max_coord + 1):
            for iy in range(-max_coord, max_coord + 1):
                for iz in range(-max_coord, max_coord + 1):
                    candidates.append(np.array([ix * a, iy * a, iz * a]))
                    candidates.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    else:
        for ix in range(-max_coord, max_coord + 1):
            for iy in range(-max_coord, max_coord + 1):
                for iz in range(-max_coord, max_coord + 1):
                    candidates.append(np.array([ix * a, iy * a, iz * a]))
    
    candidates = np.array(candidates)
    if len(candidates) == 0:
        return np.zeros((n, 3))
        
    dists = np.linalg.norm(candidates, axis=1)
    inside = candidates[dists <= R]
    
    if len(inside) >= n:
        dists_inside = np.linalg.norm(inside, axis=1)
        order = np.argsort(dists_inside)
        return inside[order[:n]]
    else:
        points = list(inside)
        while len(points) < n:
            p = np.random.uniform(-R, R, 3)
            points.append(p)
        return np.array(points)

def _random_compact(n, spacing=0.72, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    max_attempts = n * 1000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.4 * spacing:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    
    return np.array(points)

def _icosahedron(n, spacing=0.72):
    """Generate an icosahedral cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]
    
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(v)
    
    if len(positions) < n:
        octa = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ])
        octa = octa / np.linalg.norm(octa, axis=1, keepdims=True) * spacing * 1.6
        for v in octa:
            if len(positions) >= n:
                break
            positions.append(v)
            
    if len(positions) < n:
        cube = np.array(list(product([-1, 1], repeat=3)))
        cube = cube / np.linalg.norm(cube, axis=1, keepdims=True) * spacing * 2.2
        for v in cube:
            if len(positions) >= n:
                break
            positions.append(v)

    while len(positions) < n:
        p = np.random.uniform(-spacing * 2.5, spacing * 2.5, 3)
        positions.append(p)
    
    return np.array(positions[:n])

def _validate(positions):
    """Check if positions are valid."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i_idx] - positions[j_idx])
    if np.any(d < 0.35):
        return False
    return True

def _relax_one(start, kind):
    """Relax a single starting configuration. Returns (energy, positions) or (inf, None)."""
    try:
        if not _validate(start):
            return (np.inf, None)
        energy, positions = relax(start, 'sutton_chen', kind)
        if not np.all(np.isfinite(positions)):
            return (np.inf, None)
        if not _validate(positions):
            return (np.inf, None)
        return (energy, positions)
    except Exception:
        return (np.inf, None)

def _grow_structure(core, target_n, kind, rng):
    """
    Hierarchical growth: Add atoms to core one by one, relaxing after each addition.
    This helps find structures that are built upon stable smaller cores.
    """
    current = core.copy()
    if len(current) >= target_n:
        return current[:target_n]
    
    spacing = 0.72
    # Estimate current radius
    R_core = np.max(np.linalg.norm(current, axis=1)) if len(current) > 0 else 0
    
    while len(current) < target_n:
        # Generate candidate positions for the new atom
        # We try to place it on the "surface" or in interstitial sites
        best_candidate = None
        best_energy = np.inf
        
        # Generate a few random candidates
        for _ in range(10):
            # Random point on a sphere slightly larger than current radius
            r = R_core + spacing * rng.uniform(0.5, 1.5)
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            
            p = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            
            # Check distance to existing atoms
            if len(current) > 0:
                dists = np.linalg.norm(current - p, axis=1)
                if np.min(dists) < 0.4 * spacing:
                    continue
            
            # Add atom and relax
            candidate = np.vstack([current, p])
            if not _validate(candidate):
                continue
                
            try:
                e, pos = relax(candidate, 'sutton_chen', kind)
                if np.all(np.isfinite(pos)) and _validate(pos):
                    if e < best_energy:
                        best_energy = e
                        best_candidate = pos
            except:
                pass
            
            if _time_left() < 5.0:
                break
        
        if best_candidate is not None:
            current = best_candidate
        else:
            # Fallback: add a random point and relax
            p = rng.uniform(-R_core - spacing, R_core + spacing, 3)
            candidate = np.vstack([current, p])
            try:
                e, pos = relax(candidate, 'sutton_chen', kind)
                if np.all(np.isfinite(pos)) and _validate(pos):
                    current = pos
                else:
                    current = candidate # Keep unrelaxed if relaxation fails, but this is risky
            except:
                current = candidate
        
        # Update radius
        R_core = np.max(np.linalg.norm(current, axis=1))
        
        if _time_left() < 5.0:
            break
            
    return current[:target_n]

def _basin_hopping(start, kind, rng, n_steps=10):
    """
    Simple Basin Hopping:
    1. Relax start
    2. Perturb
    3. Relax
    4. Accept if lower energy
    """
    current_energy, current_pos = _relax_one(start, kind)
    if current_pos is None:
        return current_energy, start
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for _ in range(n_steps):
        if _time_left() < 5.0:
            break
            
        # Perturb: add random noise
        # Scale noise with size to keep cluster compact
        noise_scale = 0.1 * np.sqrt(len(current_pos))
        perturbed = current_pos + rng.normal(0, noise_scale, current_pos.shape)
        
        # Relax perturbed
        new_energy, new_pos = _relax_one(perturbed, kind)
        
        if new_pos is not None:
            if new_energy < current_energy:
                # Accept
                current_energy = new_energy
                current_pos = new_pos
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
            else:
                # With small probability, accept higher energy to escape basin (Metropolis-like)
                # But for time-limited task, greedy is often safer unless we have lots of time
                # Let's use a very small acceptance probability for higher energy
                if rng.random() < 0.01:
                    current_energy = new_energy
                    current_pos = new_pos
        else:
            # If relaxation failed, try a smaller perturbation
            current_pos = current_pos + rng.normal(0, noise_scale * 0.1, current_pos.shape)
            
    return best_energy, best_pos

def construct(instance):
    """Find near-global minimum for Sutton-Chen transition-metal clusters."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 67890)
    
    best_energy = np.inf
    best_positions = _random_compact(n, 0.72, rng) # Fallback
    
    # 1. Generate diverse initial configurations
    starts = []
    
    # Lattice structures
    for lat in ['fcc', 'bcc']:
        try:
            s = _generate_lattice_points(lat, n, 0.72)
            if len(s) == n and _validate(s):
                starts.append(s)
        except:
            pass
            
    # Icosahedral
    try:
        s = _icosahedron(n, 0.72)
        if len(s) == n and _validate(s):
            starts.append(s)
    except:
        pass
        
    # Random compact
    for _ in range(5):
        try:
            s = _random_compact(n, 0.72, rng)
            if len(s) == n and _validate(s):
                starts.append(s)
        except:
            pass

    # 2. Hierarchical Growth
    # Try to grow from a smaller core (e.g., n-5, n-10)
    # We assume smaller clusters are easier to solve or have known structures
    core_sizes = [max(1, n - 10), max(1, n - 5)]
    for cs in core_sizes:
        if cs >= n:
            continue
        # Generate a core
        # Use icosahedron for small cores, lattice for larger
        if cs <= 13:
            core = _icosahedron(cs, 0.72)
        else:
            core = _generate_lattice_points('fcc', cs, 0.72)
            
        if len(core) == cs and _validate(core):
            # Grow
            grown = _grow_structure(core, n, kind, rng)
            if len(grown) == n and _validate(grown):
                starts.append(grown)
                
        if _time_left() < 20.0:
            break

    if not starts:
        starts = [best_positions]
    
    # 3. Optimization Loop
    # We will process starts in parallel where possible, but Basin Hopping is sequential per start.
    # We can run multiple Basin Hopping loops in parallel threads.
    
    max_workers = 4
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for s in starts:
            if _time_left() < 10.0:
                break
            # Run Basin Hopping on this start
            # Limit steps based on time
            steps = 5 if n > 50 else 10
            futures.append(executor.submit(_basin_hopping, s, kind, rng, steps))
            
        for future in as_completed(futures):
            if _time_left() < 5.0:
                break
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                pass

    # 4. Final Polish
    # Re-relax the best position found
    if np.isfinite(best_energy) and _time_left() > 5.0:
        try:
            energy, positions = relax(best_positions, 'sutton_chen', kind)
            if np.all(np.isfinite(positions)) and _validate(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
        except:
            pass

    # Final validation
    if not _validate(best_positions):
        # Force valid
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = best_positions[i] - best_positions[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0, 0])
                    best_positions[j] = best_positions[i] + direction * 0.36
                    
    return best_positions
# EVOLVE-BLOCK-END
