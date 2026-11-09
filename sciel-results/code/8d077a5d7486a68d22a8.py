"""Morse cluster global minima via structured initialization and adaptive basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
# Safety margin: return well before the 330s kill switch
_DEADLINE = _START_TIME + 285.0 

def _time_remaining():
    return _DEADLINE - time.time()

def _morse_energy(coords, rho):
    """Compute Morse energy; return inf if any pair < 0.5 apart."""
    n = len(coords)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        return np.inf
    # Upper triangle
    iu = np.triu_indices(n, 1)
    r = dists[iu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like compact cluster."""
    points = []
    # Generate FCC lattice points in a cube
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # We scale so nearest neighbor distance is 1.
    # In FCC, a=1/sqrt(2) if nn=1.
    a = 1.0 / np.sqrt(2.0)
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    max_range = 10
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                for b in basis:
                    p = (np.array([i, j, k]) + b) * a
                    points.append(p)
    
    points = np.array(points)
    # Keep closest n points to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _build_bcc_cluster(n, spacing=1.0):
    """Build a BCC-like compact cluster."""
    points = []
    # BCC basis: (0,0,0), (0.5, 0.5, 0.5)
    # Nearest neighbor distance in BCC is a*sqrt(3)/2.
    # We want nn=1, so a = 2/sqrt(3).
    a = 2.0 / np.sqrt(3.0)
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ])
    
    max_range = 10
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                for b in basis:
                    p = (np.array([i, j, k]) + b) * a
                    points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _build_icosahedral_cluster(n, spacing=1.0):
    """Build an icosahedral shell cluster."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices (12)
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico12 /= np.linalg.norm(ico12[0])
    
    # Dodecahedron vertices (20) - derived from icosahedron dual
    # Or simply use a second shell of icosahedron scaled up
    # For simplicity, use concentric shells of icosahedra
    
    points = [np.zeros(3)] # Center
    
    # Shell 1: 12 vertices
    # Distance from center to vertex in regular icosahedron with edge 1 is 1/sqrt(2+phi) * something?
    # Let's just scale to approximate packing.
    # Nearest neighbor in icosahedron is 1. Center-to-vertex is ~1.051.
    # Let's use radius 1.0 for first shell to be close to equilibrium.
    r1 = 1.0 
    for v in ico12:
        points.append(v * r1)
    
    # Shell 2: 20 vertices (dodecahedron-like)
    # We can approximate by taking face centers of icosahedron or just scale up ico12 and add more points
    # Better: Use a known 20-point cluster (dodecahedron)
    # Dodecahedron vertices:
    dodeca20 = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ], dtype=float)
    dodeca20 /= np.linalg.norm(dodeca20[0])
    
    r2 = 1.8 # Approximate next shell radius
    for v in dodeca20:
        points.append(v * r2)
        
    # If we need more, add another icosahedral shell
    if len(points) < n:
        r3 = 2.8
        for v in ico12:
            points.append(v * r3)
            if len(points) >= n:
                break
        # If still not enough, add random points on a larger sphere
        while len(points) < n:
            rng = np.random.default_rng(len(points))
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * 3.5
            points.append(p)

    return np.array(points[:n])

def _build_random_compact(n, spacing=1.0, seed=42):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    max_attempts = 5000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
            points.append(p)
    
    # If not enough, fill remaining randomly (might violate spacing, relax will fix)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _basin_hopping(n, rho, rng, time_limit, initial):
    """Run basin hopping for the given time limit. Returns best (energy, coords)."""
    start = time.time()
    
    best_coords = initial.copy()
    best_energy, _ = relax(best_coords, 'morse', rho)
    if not np.isfinite(best_energy):
        best_energy = _morse_energy(best_coords, rho)
        if not np.isfinite(best_energy):
            # If initial is invalid, try to spread out
            best_coords = _build_random_compact(n, 1.0, seed=int(time.time()*1000))
            best_energy, _ = relax(best_coords, 'morse', rho)
            if not np.isfinite(best_energy):
                return np.inf, best_coords

    current_coords = best_coords.copy()
    current_energy = best_energy
    
    # Perturbation parameters
    base_step = 0.2
    step_scale = 1.0
    
    iteration = 0
    while _time_remaining() > 5.0:
        elapsed = time.time() - start
        if elapsed > time_limit:
            break
        
        # Perturb
        step = base_step * step_scale
        perturbed = current_coords + rng.normal(0, step, (n, 3))
        
        # Relax
        energy, relaxed = relax(perturbed, 'morse', rho)
        
        if not np.isfinite(energy):
            energy = _morse_energy(relaxed, rho)
        
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_coords = relaxed.copy()
            current_coords = relaxed.copy()
            current_energy = energy
            # Reduce step size on improvement
            step_scale *= 0.9
        else:
            # Accept with Metropolis criterion
            if np.isfinite(energy):
                delta_e = energy - current_energy
                # Temperature decreases slightly over time
                T = 0.2 * (1.0 - elapsed / (time_limit * 1.5))
                if T < 0.01: T = 0.01
                if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                    current_coords = relaxed.copy()
                    current_energy = energy
                    if energy < best_energy:
                        best_energy = energy
                        best_coords = relaxed.copy()
        
        iteration += 1
        
        # Occasionally reset step scale to explore larger moves
        if iteration % 30 == 0:
            step_scale = max(0.5, step_scale * 1.1)
    
    return best_energy, best_coords

def construct(instance):
    """Find global minimum of Morse cluster via structured init + basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Check if we have enough time to do anything meaningful
    if _time_remaining() < 10.0:
        # Emergency: just relax a random config
        pos = _build_random_compact(n, 1.0, seed=n)
        try:
            _, pos = relax(pos, 'morse', rho)
        except Exception:
            pass
        return pos

    # Generate diverse starting configurations
    start_configs = []
    
    # 1. FCC-like
    try:
        start_configs.append(_build_fcc_cluster(n, 1.0))
    except Exception:
        pass
        
    # 2. BCC-like
    try:
        start_configs.append(_build_bcc_cluster(n, 1.0))
    except Exception:
        pass
        
    # 3. Icosahedral-like
    try:
        start_configs.append(_build_icosahedral_cluster(n, 1.0))
    except Exception:
        pass
        
    # 4. Random Compact
    try:
        start_configs.append(_build_random_compact(n, 1.0, seed=n))
    except Exception:
        pass
        
    # 5. Another Random with different seed
    try:
        start_configs.append(_build_random_compact(n, 1.0, seed=n+100))
    except Exception:
        pass

    if not start_configs:
        return np.zeros((n, 3))

    # Determine time budget per configuration
    # We want to try all of them, but give more time to the ones that look promising?
    # Or just split time evenly?
    # Relaxation cost scales with N^2. Basin hopping adds overhead.
    # Let's estimate cost per relax.
    # For N=80, relax might take ~0.5s. For N=38, ~0.1s.
    # Let's just split the available time among the start_configs.
    
    tl = _time_remaining() - 5.0 # Leave 5s safety
    n_configs = len(start_configs)
    time_per_config = tl / n_configs
    
    best_energy = np.inf
    best_coords = start_configs[0]
    
    rng = np.random.default_rng(42)
    
    for i, config in enumerate(start_configs):
        if _time_remaining() < 10.0:
            break
            
        # Ensure config is valid (min dist >= 0.5)
        # If not, relax it first to get a valid starting point
        # Actually, relax handles invalid inputs by moving atoms apart, but let's check energy
        # We run basin hopping which starts with a relax
        
        energy, coords = _basin_hopping(n, rho, rng, time_per_config, config)
        
        if energy < best_energy:
            best_energy = energy
            best_coords = coords
            
        # If we found a very good result (close to theoretical lower bound?), we could stop early
        # But we don't know the bound. Let's just use the time budget.
        
        # Adaptive: if this config was much better than previous, maybe give next one less time?
        # Or if it was worse, give next one more?
        # For simplicity, keep it even for now, but ensure we don't exceed deadline.
        
    # Final polish: if we have time left, do a few more perturbations on the best found
    if best_coords is not None and _time_remaining() > 10.0:
        # Use remaining time for a final basin hopping run on the best result
        # This effectively adds more iterations to the best candidate
        final_time = _time_remaining() - 5.0
        energy, coords = _basin_hopping(n, rho, rng, final_time, best_coords)
        if energy < best_energy:
            best_energy = energy
            best_coords = coords

    return best_coords
# EVOLVE-BLOCK-END
