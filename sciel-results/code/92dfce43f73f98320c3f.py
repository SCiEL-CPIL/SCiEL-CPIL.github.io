"""Structure-motif + adaptive basin-hopping search for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


# ---- Motif generators (all return (n,3) arrays in reduced units) ----

def _icosa_shells(n):
    """Mackay icosahedral shells. Returns (n,3) array."""
    phi = (1 + 5**0.5) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Shell counts: 1 (center), 12, 42, 120, ...
    shell_counts = [1, 12, 42, 120, 220, 330, 460]
    # Nearest neighbor distance in Mackay icosahedron ~ 1.0
    # Shell radii: r_k = sqrt(k*(k+1)/3) * a where a is NN distance
    # Actually for Mackay icosahedra: r_k = a * sqrt(k*(k+1)/3) for shell k
    # Let's use approximate radii
    radii = [0.0, 1.0, 1.63, 2.38, 3.16, 3.95, 4.77]
    
    points = []
    for k in range(len(shell_counts)):
        count = shell_counts[k]
        radius = radii[k]
        if k == 0:
            points.append(np.zeros(3))
        else:
            # Place count points on sphere of radius k
            # Use Fibonacci sphere for uniform distribution
            if k == 1:
                pts = verts * radius
            else:
                # Fibonacci sphere
                idx = np.arange(count)
                golden = (1 + 5**0.5) / 2
                z = 1 - 2 * (idx + 0.5) / count
                r = np.sqrt(np.maximum(0, 1 - z**2))
                theta = 2 * np.pi * idx / golden
                pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z]) * radius
            points.extend(pts.tolist())
    
    points = np.array(points[:n])
    if len(points) < n:
        # Pad with points near the surface
        while len(points) < n:
            p = points[np.random.randint(len(points))] + np.random.normal(0, 0.1, 3)
            points = np.vstack([points, p.reshape(1, 3)])
    return points[:n]


def _fcc_fragment(n):
    """Generate an FCC-like cluster by taking atoms from a large FCC lattice."""
    # Generate FCC lattice points
    a = 1.0  # lattice constant
    points = []
    max_coord = int(np.ceil(n**(1/3) * 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: corners + face centers
                base = np.array([i, j, k]) * a
                # Corner
                points.append(base)
                # Face centers
                if j < max_coord:
                    points.append(base + np.array([0, 0.5, 0]) * a)
                if k < max_coord:
                    points.append(base + np.array([0, 0, 0.5]) * a)
                if i < max_coord:
                    points.append(base + np.array([0.5, 0, 0]) * a)
    
    points = np.array(points)
    # Center
    centroid = points.mean(axis=0)
    points -= centroid
    # Sort by distance from center, take n closest
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _hcp_fragment(n):
    """Generate an HCP-like cluster."""
    a = 1.0
    c_over_a = np.sqrt(8/3)  # ideal HCP
    points = []
    max_coord = int(np.ceil(n**(1/3) * 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # HCP: two atoms per unit cell
                x = i * a + (0.5 if j % 2 else 0) * a
                y = j * a * np.sqrt(3) / 2
                z = k * a * c_over_a
                points.append(np.array([x, y, z]))
                # Second atom in cell
                points.append(np.array([x + a/2, y + a*np.sqrt(3)/6, z + a*c_over_a/2]))
    
    points = np.array(points)
    centroid = points.mean(axis=0)
    points -= centroid
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _bcc_fragment(n):
    """Generate a BCC-like cluster."""
    a = 1.0
    points = []
    max_coord = int(np.ceil(n**(1/3) * 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                points.append(base)
                points.append(base + np.array([0.5, 0.5, 0.5]) * a)
    
    points = np.array(points)
    centroid = points.mean(axis=0)
    points -= centroid
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _random_compact(n, seed=42):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    spacing = 0.72
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 100000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fill remaining with random points
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)


def _liquid_like(n, seed=42):
    """Generate a liquid-like configuration with short-range order."""
    rng = np.random.default_rng(seed)
    spacing = 0.72
    radius = spacing * n ** (1.0 / 3.0) * 0.7
    points = []
    attempts = 0
    while len(points) < n and attempts < 50000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Allow closer packing than random but not too close
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)


def _generate_starts(n, kind, seed=42):
    """Generate diverse starting configurations."""
    starts = []
    # Crystal motifs
    try:
        starts.append(_icosa_shells(n))
    except:
        pass
    try:
        starts.append(_fcc_fragment(n))
    except:
        pass
    try:
        starts.append(_hcp_fragment(n))
    except:
        pass
    try:
        starts.append(_bcc_fragment(n))
    except:
        pass
    # Random and liquid-like with different seeds
    for s in range(8):
        try:
            starts.append(_random_compact(n, seed=seed + s))
        except:
            pass
    for s in range(4):
        try:
            starts.append(_liquid_like(n, seed=seed + 100 + s))
        except:
            pass
    return starts


def _validity_check(positions):
    """Check if positions are valid: finite and min distance > 0.35."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                min_dist = d
    return min_dist > 0.35


def _basin_hopping(positions, kind, max_iter=20, initial_kick=0.3, max_kick=0.8, seed=42):
    """Adaptive basin hopping with kick restarts."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = relax(positions, 'sutton_chen', kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    kick = initial_kick
    for it in range(max_iter):
        # Perturb
        perturbation = rng.normal(0, kick, size=current_pos.shape)
        new_pos = current_pos + perturbation
        new_energy, new_pos = relax(new_pos, 'sutton_chen', kind)
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            kick *= 0.9  # Shrink kick on success
        else:
            kick = min(kick * 1.15, max_kick)  # Grow kick on failure
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Structure-motif + basin-hopping search for Sutton-Chen cluster global minima."""
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin before 330s kill
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Generate diverse starting configurations
    starts = _generate_starts(n, kind, seed=n)
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Relax all starting configurations in parallel
    def relax_one(start_pos):
        energy, pos = relax(start_pos, 'sutton_chen', kind)
        return energy, pos
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(relax_one, s): i for i, s in enumerate(starts)}
        results = []
        for future in as_completed(futures):
            if time.time() > deadline - 60:
                break
            try:
                energy, pos = future.result()
                if _validity_check(pos):
                    results.append((energy, pos))
            except:
                pass
    
    # Sort by energy
    results.sort(key=lambda x: x[0])
    
    if results:
        best_energy = results[0][0]
        best_pos = results[0][1].copy()
    
    # Phase 2: Basin hopping from the best few configurations
    # Use the top 5 results for basin hopping
    top_results = results[:5]
    
    for i, (e0, p0) in enumerate(top_results):
        if time.time() > deadline - 30:
            break
        
        # Adaptive basin hopping with varying seeds
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            try:
                bh_energy, bh_pos = _basin_hopping(
                    p0, kind, 
                    max_iter=15,
                    initial_kick=0.25 + 0.1 * i,
                    max_kick=0.7,
                    seed=n * 100 + i * 10 + seed
                )
                if _validity_check(bh_pos) and bh_energy < best_energy:
                    best_energy = bh_energy
                    best_pos = bh_pos.copy()
            except:
                pass
    
    # Phase 3: If time remains, try more random restarts with basin hopping
    if time.time() < deadline - 40:
        rng = np.random.default_rng(n + 999)
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            try:
                random_start = _random_compact(n, seed=n + 1000 + attempt)
                bh_energy, bh_pos = _basin_hopping(
                    random_start, kind,
                    max_iter=12,
                    initial_kick=0.3,
                    max_kick=0.8,
                    seed=n * 200 + attempt
                )
                if _validity_check(bh_pos) and bh_energy < best_energy:
                    best_energy = bh_energy
                    best_pos = bh_pos.copy()
            except:
                pass
    
    # Final validity check
    if best_pos is None or not _validity_check(best_pos):
        # Fallback: use the best valid result from phase 1
        for e, p in results:
            if _validity_check(p):
                best_pos = p
                break
        if best_pos is None:
            # Last resort: simple linear arrangement
            best_pos = np.zeros((n, 3))
            for i in range(n):
                best_pos[i] = [i * 0.72, 0, 0]
    
    return best_pos
# EVOLVE-BLOCK-END
