"""Morse cluster global minima: structured lattice seeds + basin hopping with adaptive cooling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy_positions(pos, rho):
    """Compute Morse energy for positions."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    # Avoid division by zero / invalid distances
    mask = dist > 0.5
    dist_masked = np.where(mask, dist, 1e10)
    x = rho * (1.0 - dist_masked)
    e = np.exp(x)
    pair_e = e * (e - 2.0)
    # Only count pairs once
    iu = np.triu_indices(n, k=1)
    total = np.sum(pair_e[iu[0], iu[1]])
    return total


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice centered at origin."""
    a = spacing * 0.7071
    r_est = spacing * n ** (1.0 / 3.0) * 1.2
    
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                p = np.array([i, j, k]) * a * np.sqrt(2)
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
                if i < extent and j < extent:
                    p2 = np.array([i + 0.5, j + 0.5, k]) * a * np.sqrt(2)
                    if np.linalg.norm(p2) <= r_est:
                        points.append(p2)
                if i < extent and k < extent:
                    p3 = np.array([i + 0.5, j, k + 0.5]) * a * np.sqrt(2)
                    if np.linalg.norm(p3) <= r_est:
                        points.append(p3)
                if j < extent and k < extent:
                    p4 = np.array([i, j + 0.5, k + 0.5]) * a * np.sqrt(2)
                    if np.linalg.norm(p4) <= r_est:
                        points.append(p4)
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        return rng.uniform(-r_est, r_est, (n, 3))
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    if len(points) >= n:
        return points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
        return np.vstack([points, extra])


def _bcp_points(n, spacing=1.0):
    """Generate n points from a BCC lattice centered at origin."""
    a = 2.0 * spacing / np.sqrt(3.0)
    r_est = spacing * (n ** (1.0 / 3.0) + 2) * 0.8
    
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        return rng.uniform(-r_est, r_est, (n, 3))
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    if len(points) >= n:
        return points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
        return np.vstack([points, extra])


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice centered at origin."""
    a = spacing
    r_est = spacing * (n ** (1.0 / 3.0) + 2) * 0.8
    
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        return rng.uniform(-r_est, r_est, (n, 3))
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    if len(points) >= n:
        return points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
        return np.vstack([points, extra])


def _icosahedral(n, spacing=1.0):
    """Generate atoms in icosahedral shells (like icosahedral quasicrystal)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0]) * spacing
    pts = [np.zeros(3)]
    if n <= 1:
        return np.array(pts[:n])
    pts.extend(verts.tolist())
    
    # Second shell: icosahedron of larger radius
    r2 = 2.0 * spacing
    verts2 = verts / spacing * r2
    for v in verts2:
        if len(pts) < n:
            pts.append(v.tolist())
    
    # Third shell
    r3 = 2.5 * spacing
    verts3 = verts / spacing * r3
    for v in verts3:
        if len(pts) < n:
            pts.append(v.tolist())
    
    # Fourth shell
    r4 = 3.0 * spacing
    verts4 = verts / spacing * r4
    for v in verts4:
        if len(pts) < n:
            pts.append(v.tolist())
    
    # Fifth shell
    r5 = 3.5 * spacing
    verts5 = verts / spacing * r5
    for v in verts5:
        if len(pts) < n:
            pts.append(v.tolist())
    
    # Fill remaining with random
    rng = np.random.default_rng(42)
    while len(pts) < n:
        p = rng.uniform(-r5, r5, 3)
        pts.append(p.tolist())
    return np.array(pts[:n])


def _random_compact(n, spacing=1.0, seed=0):
    """Generate n random points in a compact sphere with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.55 * spacing
    
    points = []
    attempts = 0
    max_attempts = n * 200
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    return np.array(points)


def _basin_hopping(pos, rho, max_iter=50, kick_scale=0.15, cool_rate=0.95):
    """Perform basin hopping with adaptive cooling."""
    current_pos = pos.copy()
    current_energy, current_pos = relax(current_pos, 'morse', rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(hash(tuple(pos.ravel())) % 2**32)
    
    for it in range(max_iter):
        if _time_left() < 2.0:
            break
        
        # Generate kick with adaptive scale
        kick = rng.uniform(-kick_scale, kick_scale, pos.shape)
        proposed_pos = current_pos + kick
        
        # Relax the proposed position
        proposed_energy, proposed_pos = relax(proposed_pos, 'morse', rho)
        
        # Metropolis acceptance
        if proposed_energy < current_energy:
            current_pos = proposed_pos
            current_energy = proposed_energy
            if current_energy < best_energy:
                best_pos = proposed_pos.copy()
                best_energy = current_energy
        else:
            # Accept with probability exp(-(E_new - E_old)/T)
            # Use a simple temperature that decreases
            T = 0.1 * (1.0 - 0.95**it) + 0.01
            delta = proposed_energy - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_pos = proposed_pos
                current_energy = proposed_energy
        
        # Cool the kick scale
        kick_scale *= cool_rate
        kick_scale = max(kick_scale, 0.02)
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster configurations via multi-restart lattice-based search with basin hopping."""
    _instance_call_count[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Per-instance time budget
    per_instance_budget = 45.0
    if _instance_call_count[0] <= 2:
        per_instance_budget = 50.0
    elif _instance_call_count[0] >= 5:
        per_instance_budget = 40.0
    
    instance_deadline = time.time() + per_instance_budget
    
    # Generate multiple starting configurations from different lattices
    generators = [
        lambda: _fcc_points(size, 1.0),
        lambda: _bcp_points(size, 1.0),
        lambda: _sc_points(size, 1.0),
        lambda: _icosahedral(size, 1.0),
        lambda: _random_compact(size, 1.0, seed=42),
        lambda: _random_compact(size, 1.0, seed=123),
        lambda: _random_compact(size, 1.0, seed=456),
    ]
    
    best_energy = np.inf
    best_pos = None
    
    for i, gen in enumerate(generators):
        if time.time() > instance_deadline or _time_left() < 5.0:
            break
        
        try:
            init_pos = gen()
            # Ensure minimum distance constraint
            # If not, perturb
            n = len(init_pos)
            valid = True
            for a in range(n):
                for b in range(a + 1, n):
                    if np.linalg.norm(init_pos[a] - init_pos[b]) < 0.5:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                # Add small random perturbation to resolve overlaps
                rng = np.random.default_rng(i + 1000)
                init_pos = init_pos + rng.uniform(-0.1, 0.1, init_pos.shape)
            
            # First do a quick relaxation to get a local minimum
            energy, pos = relax(init_pos, potential, parameter)
            
            # Then do basin hopping from this local minimum
            max_iter = 30 if size > 50 else 40
            bh_energy, bh_pos = _basin_hopping(pos, parameter, max_iter=max_iter, kick_scale=0.12, cool_rate=0.96)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
                
        except Exception as e:
            continue
    
    # If we still don't have a result, do a final simple relaxation
    if best_pos is None:
        init_pos = _random_compact(size, 1.0, seed=0)
        best_energy, best_pos = relax(init_pos, potential, parameter)
    
    # Final validation: ensure minimum distance
    n = len(best_pos)
    for a in range(n):
        for b in range(a + 1, n):
            d = np.linalg.norm(best_pos[a] - best_pos[b])
            if d < 0.5:
                # Push apart
                mid = (best_pos[a] + best_pos[b]) / 2
                direction = best_pos[a] - best_pos[b]
                if np.linalg.norm(direction) < 1e-10:
                    direction = np.array([1.0, 0.0, 0.0])
                direction = direction / np.linalg.norm(direction)
                shift = (0.5 - d) / 2.0 + 0.01
                best_pos[a] += direction * shift
                best_pos[b] -= direction * shift
    
    return best_pos
# EVOLVE-BLOCK-END
