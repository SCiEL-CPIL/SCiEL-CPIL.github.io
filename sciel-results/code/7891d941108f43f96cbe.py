# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_HARD_LIMIT = 290.0  # Safety margin: 300s limit, keep 10s buffer
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_INDEX = {n: i for i, n in enumerate(_INSTANCE_ORDER)}

# Dzugutov potential parameters
A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = np.linalg.norm(positions[i] - positions[j])
            if rij < 1e-10:
                continue
            if rij < a_DZ:
                v = A_DZ * (rij ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (rij - a_DZ))
            elif rij < b_DZ:
                v = B_DZ * np.exp(d_DZ / (rij - b_DZ))
            else:
                v = 0.0
            total += v
    return total


def _generate_mackay_icosahedron(n_atoms):
    """
    Generate an icosahedral cluster structure for n_atoms atoms.
    Uses concentric icosahedral shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    positions = []
    positions.append(np.zeros(3))
    
    r1 = 1.0
    positions.extend([v * r1 for v in ico_verts])
    
    remaining = n_atoms - len(positions)
    
    if remaining > 0:
        # Generate second shell points (edges and faces)
        ico_edges = []
        dists = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                dists.append((d, i, j))
        dists.sort()
        min_dist = dists[0][0]
        for d, i, j in dists:
            if d < min_dist * 1.01:
                ico_edges.append((i, j))
        
        edge_mids = []
        for i, j in ico_edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2
            edge_mids.append(mid)
        edge_mids = np.array(edge_mids)
        
        connected = set()
        for i, j in ico_edges:
            connected.add((i, j))
            connected.add((j, i))
        
        face_centers = []
        for i in range(12):
            for j in range(i+1, 12):
                if (i, j) in connected:
                    for k in range(j+1, 12):
                        if (j, k) in connected and (i, k) in connected:
                            center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                            face_centers.append(center)
        face_centers = np.array(face_centers)
        
        r2 = r1 * np.sqrt(2)
        for mid in edge_mids:
            if remaining <= 0:
                break
            positions.append(mid * r2)
            remaining -= 1
        
        if remaining > 0:
            for fc in face_centers:
                if remaining <= 0:
                    break
                positions.append(fc * r2)
                remaining -= 1
    
    if remaining > 0:
        r3 = r1 * np.sqrt(3)
        rng = np.random.default_rng(42)
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = r3 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            positions.append(p)
    
    return np.array(positions[:n_atoms])


def _langevin_seed(n_atoms, seed, deadline):
    """
    Generate a seed via physics-driven self-assembly:
    sequentially add atoms at random positions near the growing cluster,
    then perform short Langevin-like relaxation steps.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a small nucleus (center + a few atoms)
    positions = [np.zeros(3)]
    
    # Add first shell atoms around the center
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Add up to 12 atoms from icosahedral vertices with small random perturbation
    n_added = 0
    for v in ico_verts:
        if len(positions) >= n_atoms:
            break
        if time.time() > deadline:
            break
        p = v * 1.0 + rng.normal(0, 0.05, 3)
        # Ensure minimum distance from existing atoms
        valid = True
        for q in positions:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            positions.append(p)
            n_added += 1
    
    # Now sequentially add remaining atoms via self-assembly
    while len(positions) < n_atoms:
        if time.time() > deadline:
            break
        
        # Compute current cluster centroid and radius
        pos_arr = np.array(positions)
        centroid = pos_arr.mean(axis=0)
        if len(pos_arr) > 1:
            radius = np.max(np.linalg.norm(pos_arr - centroid, axis=1))
        else:
            radius = 1.0
        
        # Try to add an atom at a random position on the surface
        for attempt in range(50):
            # Random direction
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            direction = np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            
            # Place at radius + small offset
            r_new = radius + rng.uniform(-0.2, 0.3)
            p_new = centroid + direction * r_new
            
            # Check minimum distance
            valid = True
            for q in positions:
                if np.linalg.norm(p_new - q) < 0.5:
                    valid = False
                    break
            if valid:
                positions.append(p_new)
                break
        
        # If we couldn't add an atom after 50 attempts, place it anyway with random position
        if len(positions) < n_atoms and attempt >= 49:
            p_new = centroid + rng.normal(0, 0.5, 3)
            positions.append(p_new)
    
    positions = np.array(positions[:n_atoms])
    
    # Perform a few relaxation steps to settle the structure
    if time.time() < deadline:
        try:
            _, positions = relax(positions.copy(), 'dzugutov', None)
        except Exception:
            pass
    
    return positions


def _generate_fcc_cluster(n_atoms):
    """Generate an FCC-like cluster."""
    positions = []
    a = 1.0
    max_coord = int(np.ceil(n_atoms ** 0.5)) + 2
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    candidates.append(np.array([i, j, k]) * a / np.sqrt(2))
                candidates.append(np.array([i, j, k + 0.5]) * a / np.sqrt(2))
                candidates.append(np.array([i, j + 0.5, k]) * a / np.sqrt(2))
                candidates.append(np.array([i + 0.5, j, k]) * a / np.sqrt(2))
    
    candidates = np.array(candidates)
    centers = np.zeros(3)
    dists = np.linalg.norm(candidates - centers, axis=1)
    order = np.argsort(dists)
    
    selected = []
    for idx in order:
        p = candidates[idx]
        if all(np.linalg.norm(p - q) >= 0.5 for q in selected):
            selected.append(p)
        if len(selected) == n_atoms:
            break
    
    rng = np.random.default_rng(n_atoms)
    while len(selected) < n_atoms:
        p = rng.uniform(-3, 3, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in selected):
            selected.append(p)
    
    return np.array(selected)


def _generate_random_compact(n_atoms, seed):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    spacing = 1.1
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.75
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < n_atoms:
        rng2 = np.random.default_rng(seed + 1)
        radius2 = spacing * n_atoms ** (1.0 / 3.0) * 0.6
        points2 = []
        attempts = 0
        while len(points2) < n_atoms and attempts < max_attempts:
            p = rng2.uniform(-radius2, radius2, 3)
            if np.linalg.norm(p) <= radius2:
                if all(np.linalg.norm(p - q) >= 0.5 for q in points2):
                    points2.append(p)
            attempts += 1
        if len(points2) >= n_atoms:
            points = points2
    
    while len(points) < n_atoms:
        p = np.array([rng.uniform(-radius, radius) for _ in range(3)])
        points.append(p)
    
    return np.array(points[:n_atoms])


def _basin_hopping(positions, potential, parameter, rng, max_iters, kick_scale_start, kick_scale_end, deadline):
    """
    Basin hopping: random kick + local relaxation.
    Returns (best_energy, best_positions).
    """
    best_energy, best_positions = relax(positions.copy(), potential, parameter)
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        t = it / max(1, max_iters - 1)
        kick_scale = kick_scale_start + (kick_scale_end - kick_scale_start) * t
        
        kick = rng.normal(0, kick_scale, size=current_positions.shape)
        new_positions = current_positions + kick
        
        try:
            new_energy, new_relaxed = relax(new_positions, potential, parameter)
        except Exception:
            continue
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_positions = new_relaxed
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed.copy()
        else:
            delta_e = new_energy - current_energy
            if delta_e < 0.1:
                accept_prob = np.exp(-delta_e / 0.05)
                if rng.random() < accept_prob:
                    current_energy = new_energy
                    current_positions = new_relaxed
    
    return best_energy, best_positions


def _search_chain(seed_positions, potential, parameter, n_atoms, deadline, chain_id):
    """
    Run a basin hopping search chain starting from seed_positions.
    Returns (best_energy, best_positions).
    """
    rng = np.random.default_rng(chain_id * 1000 + n_atoms)
    
    # Reduced iterations to fit in time budget
    max_iters = 12
    kick_scale_start = 0.4
    kick_scale_end = 0.15
    
    best_energy, best_positions = _basin_hopping(
        seed_positions, potential, parameter, rng,
        max_iters, kick_scale_start, kick_scale_end, deadline
    )
    
    return best_energy, best_positions


def construct(instance):
    """
    Find low-energy Dzugutov cluster structures using basin hopping
    with diverse polytetrahedral seeds.
    """
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Calculate time budget
    elapsed = time.time() - _START_TIME
    remaining_total = max(0, _HARD_LIMIT - elapsed)
    
    # Determine how many instances are left, including this one
    idx = _INSTANCE_INDEX.get(size, 0)
    n_rem = len(_INSTANCE_ORDER) - idx
    
    # Allocate time for this call: share remaining time among remaining instances
    # Leave a small buffer for the last instance to be safe
    if n_rem > 0:
        call_budget = remaining_total / n_rem
    else:
        call_budget = remaining_total
    
    # Ensure we don't allocate negative or zero time
    call_budget = max(call_budget, 1.0)
    
    call_deadline = time.time() + call_budget
    
    # Generate diverse seed structures
    seeds = []
    
    # Primary seed: icosahedral
    try:
        ico = _generate_mackay_icosahedron(size)
        seeds.append(ico)
    except Exception:
        pass
    
    # Physics-driven self-assembly seeds
    n_langevin_seeds = 3
    for ls in range(n_langevin_seeds):
        try:
            lang = _langevin_seed(size, seed=ls + 100, deadline=call_deadline)
            seeds.append(lang)
        except Exception:
            pass
    
    # FCC seed
    try:
        fcc = _generate_fcc_cluster(size)
        seeds.append(fcc)
    except Exception:
        pass
    
    # Add random seeds based on available time
    n_random_seeds = 2
    for seed_val in range(n_random_seeds):
        try:
            rand = _generate_random_compact(size, seed_val + 1)
            seeds.append(rand)
        except Exception:
            pass
    
    if not seeds:
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        seeds.append(np.array(points))
    
    best_energy = float('inf')
    best_positions = None
    
    # If time is very tight, just relax the best seed (icosahedron)
    if call_budget < 5.0:
        try:
            _, best_positions = relax(seeds[0].copy(), potential, parameter)
        except Exception:
            best_positions = seeds[0]
        return best_positions
    
    n_threads = min(4, len(seeds))
    
    if n_threads == 1:
        energy, positions = _search_chain(seeds[0], potential, parameter, size, call_deadline, 0)
        best_energy = energy
        best_positions = positions
    else:
        results = [None] * len(seeds)
        
        def run_chain(i):
            if time.time() > call_deadline:
                return (float('inf'), None)
            return _search_chain(seeds[i], potential, parameter, size, call_deadline, i)
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {executor.submit(run_chain, i): i for i in range(len(seeds))}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    # Timeout should be slightly less than remaining time to allow handling
                    timeout = max(0.1, call_deadline - time.time())
                    energy, positions = future.result(timeout=timeout)
                    if positions is not None and energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                except Exception:
                    continue
    
    if best_positions is None:
        try:
            _, best_positions = relax(seeds[0].copy(), potential, parameter)
        except Exception:
            best_positions = seeds[0]
    
    return best_positions
# EVOLVE-BLOCK-END
