"""Morse cluster global minima search with basin hopping and structured initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline shared across all six construct() calls
_START_TIME = time.time()
_TOTAL_BUDGET = 320.0  # 10s safety margin before 330s kill

def _time_left():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _morse_energy(pos, rho):
    """Compute total Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)))

def _build_fcc_cluster(n, spacing=1.0):
    """Build FCC-like cluster by selecting closest n lattice points to origin."""
    a = spacing * np.sqrt(2.0)  # lattice constant for nearest neighbor = spacing
    max_coord = int(np.ceil(n / 2.5)) + 3
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a
                    if np.linalg.norm(p) <= spacing * np.sqrt(n) * 1.5:
                        points.append(p)
    
    if not points:
        return np.zeros((n, 3))
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _build_hcp_cluster(n, spacing=1.0):
    """Build HCP-like cluster."""
    points = []
    a = spacing
    layer_spacing = a * np.sqrt(8.0 / 3.0)
    offset = a / 2.0
    
    def hex_points(radius_layers):
        pts = []
        for q in range(-radius_layers, radius_layers + 1):
            for r in range(-radius_layers, radius_layers + 1):
                if abs(q) + abs(r) <= radius_layers and abs(q + r) <= radius_layers:
                    x = a * (q + 0.5 * r)
                    y = a * np.sqrt(3) / 2 * r
                    pts.append([x, y])
        return np.array(pts) if pts else np.zeros((0, 2))
    
    layer_count = 0
    while len(points) < n:
        if layer_count == 0:
            layer_pts = np.array([[0.0, 0.0]])
        else:
            needed = n - len(points)
            r_est = int(np.ceil(np.sqrt(needed / 3.0))) + 1
            layer_pts = hex_points(r_est)
            if layer_count % 2 == 1:
                layer_pts[:, 0] += offset
        
        z = layer_count * layer_spacing
        for pt in layer_pts:
            if len(points) >= n:
                break
            points.append([pt[0], pt[1], z])
        
        layer_count += 1
    
    return np.array(points[:n])

def _build_icosahedral_cluster(n, spacing=1.0):
    """Build icosahedral-like cluster."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico12 /= np.linalg.norm(ico12[0])
    
    cubo12 = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ], dtype=float)
    cubo12 /= np.linalg.norm(cubo12[0])
    
    points = [np.zeros(3)]
    for v in ico12:
        points.append(v * 1.0)
    for v in cubo12:
        points.append(v * 1.7)
        
    if len(points) < n:
        # Fill with random points on a sphere shell
        rng = np.random.default_rng(42)
        shell_radius = 2.5
        attempts = 0
        while len(points) < n and attempts < 5000:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * shell_radius
            min_dist = min(np.linalg.norm(p - q) for q in points)
            if min_dist >= 0.6:
                points.append(p)
            attempts += 1
            if attempts % 500 == 0:
                shell_radius += 0.2
    
    return np.array(points[:n])

def _build_random_compact(n, spacing=1.0, seed=42):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * (n ** (1.0/3.0)) * 1.2
    points = [np.zeros(3)]
    
    max_attempts = 5000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        # Check min distance
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
    
    # Fill remaining if needed
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points[:n])

def _ensure_min_distance(pos, min_d=0.5):
    """Ensure all pairs are at least min_d apart."""
    n = len(pos)
    pos = pos.copy()
    for _ in range(20):
        violated = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    violated = True
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                    diff /= np.linalg.norm(diff)
                    push = (min_d - d) * 0.5 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not violated:
            break
    return pos

def _relax_with_timeout(pos, potential, param, time_limit):
    """Relax a configuration with a time limit check."""
    start_time = time.time()
    try:
        energy, relaxed = relax(pos, potential, param)
        return energy, relaxed
    except Exception:
        return float('inf'), pos

def construct(instance):
    """Find low-energy Morse cluster configuration using basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", 10))
    
    # Check time budget
    tl = _time_left()
    if tl < 5.0:
        pos = _build_fcc_cluster(size)
        pos = _ensure_min_distance(pos)
        return pos

    # Generate diverse starting configurations
    start_configs = []
    try: start_configs.append(_build_fcc_cluster(size))
    except: pass
    try: start_configs.append(_build_hcp_cluster(size))
    except: pass
    try: start_configs.append(_build_icosahedral_cluster(size))
    except: pass
    try: start_configs.append(_build_random_compact(size, seed=size))
    except: pass
    
    if not start_configs:
        pos = np.zeros((size, 3))
        pos = _ensure_min_distance(pos)
        return pos

    best_energy = float('inf')
    best_pos = start_configs[0].copy()
    current_pos = start_configs[0].copy()
    current_energy = float('inf')
    
    # Initial relaxation of starting configs
    for config in start_configs:
        tl = _time_left()
        if tl < 10.0:
            break
        energy, relaxed = _relax_with_timeout(config, potential, parameter, tl - 5.0)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
            current_pos = relaxed.copy()
            current_energy = energy

    # Basin Hopping
    # Estimate time per step
    # Relaxation is the bottleneck. For N=80, it might take ~1-2s.
    # We will run as many steps as possible.
    step_size = 0.5
    rng = np.random.default_rng(12345)
    
    while _time_left() > 10.0:
        # Perturb
        noise = rng.normal(0, step_size, current_pos.shape)
        perturbed_pos = current_pos + noise
        
        # Relax
        energy, relaxed = _relax_with_timeout(perturbed_pos, potential, parameter, _time_left() - 5.0)
        
        # Accept if lower energy (greedy) or with Metropolis (for exploration)
        # Since we want global minima, greedy is usually fine if step_size is large enough
        # or we use Metropolis to escape local traps. Let's use Metropolis.
        if energy < current_energy:
            current_energy = energy
            current_pos = relaxed.copy()
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
                # Reset step size if we found a new best? Or keep it.
                # Maybe reduce step size slightly to refine? No, keep it to explore.
        else:
            # Metropolis acceptance
            # T is not defined, but we can use a simple heuristic or fixed T
            # For cluster problems, T ~ 0.1-0.5 is common in reduced units
            T = 0.5 
            delta_E = energy - current_energy
            if np.random.rand() < np.exp(-delta_E / T):
                current_energy = energy
                current_pos = relaxed.copy()
        
        # Adaptive step size? 
        # If we haven't improved in a while, increase step size to escape.
        # If we are improving, decrease step size to refine.
        # Simple heuristic:
        if energy < current_energy:
            step_size = max(0.2, step_size * 0.9)
        else:
            step_size = min(1.5, step_size * 1.1)

    # Final: ensure minimum distance constraint
    pos = _ensure_min_distance(best_pos, min_d=0.5)
    
    return pos
# EVOLVE-BLOCK-END
