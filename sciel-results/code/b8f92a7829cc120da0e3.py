"""Morse cluster global minima search with direct structural initialization and gradient-based refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline shared across all six construct() calls
_START_TIME = time.time()
_TOTAL_BUDGET = 315.0  # 15s safety margin before 330s kill

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

def _build_close_packing(n, spacing=1.0):
    """Build a cluster using close-packing lattice points (FCC-like)."""
    points = []
    a = spacing * np.sqrt(2.0)
    max_coord = int(np.ceil(n / 3.0)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a
                    if np.linalg.norm(p) <= spacing * np.sqrt(n) * 1.5:
                        points.append(p)
    
    if len(points) == 0:
        return np.zeros((n, 3))
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _build_hcp_like(n, spacing=1.0):
    """Build a cluster using HCP-like layer stacking."""
    points = []
    a = spacing
    def hex_points(radius_layers):
        pts = []
        for q in range(-radius_layers, radius_layers + 1):
            for r in range(-radius_layers, radius_layers + 1):
                if abs(q) + abs(r) <= radius_layers and abs(q + r) <= radius_layers:
                    x = a * (q + 0.5 * r)
                    y = a * np.sqrt(3) / 2 * r
                    pts.append([x, y])
        return np.array(pts) if pts else np.zeros((0, 2))
    
    layer_spacing = a * np.sqrt(8.0 / 3.0)
    offset = a / 2.0
    
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

def _build_icosahedral(n, spacing=1.0):
    """Build a cluster based on icosahedral symmetry with shells."""
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
    
    shell3 = []
    for perm in [(0, 1, phi), (1, phi, 0), (phi, 0, 1)]:
        for signs in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                      (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
            p = np.array([perm[0] * signs[0], perm[1] * signs[1], perm[2] * signs[2]])
            norm = np.linalg.norm(p)
            if norm > 0:
                shell3.append(p / norm)
    shell3 = np.array(shell3)
    
    points = [np.zeros(3)]
    for v in ico12:
        points.append(v * 1.0)
    for v in cubo12:
        points.append(v * 1.7)
    for v in shell3:
        points.append(v * 2.5)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        shell_radius = 3.2
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

def _build_random_dense(n, spacing=1.0, seed=42):
    """Build a random dense cluster using rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = spacing * (n ** (1.0/3.0)) * 1.2
    points = [np.zeros(3)]
    
    max_attempts = 10000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        min_dist = min(np.linalg.norm(p - q) for q in points)
        if min_dist >= 0.7:
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _build_fibonacci_sphere(n, spacing=1.0):
    """Build a cluster using Fibonacci sphere for uniform distribution."""
    if n <= 1:
        return np.zeros((n, 3))
    points = [np.zeros(3)]
    remaining = n - 1
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    radius = spacing * (remaining ** (1.0/3.0)) * 0.8
    
    for i in range(remaining):
        t = i / remaining
        z = 1.0 - 2.0 * t
        radius_circle = np.sqrt(max(0, 1.0 - z * z))
        theta = golden_angle * i
        x = np.cos(theta) * radius_circle
        y = np.sin(theta) * radius_circle
        p = np.array([x, y, z]) * radius
        points.append(p)
    
    return np.array(points[:n])

def _relax_single(pos, potential, param):
    """Relax a single configuration."""
    try:
        energy, relaxed = relax(pos, potential, param)
        return energy, relaxed
    except Exception:
        return float('inf'), pos

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

def construct(instance):
    """Find low-energy Morse cluster configuration using basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", 10))
    
    tl = _time_left()
    if tl < 5.0:
        pos = _build_close_packing(size)
        pos = _ensure_min_distance(pos)
        return pos

    start_configs = []
    try: start_configs.append(_build_close_packing(size))
    except: pass
    try: start_configs.append(_build_hcp_like(size))
    except: pass
    try: start_configs.append(_build_icosahedral(size))
    except: pass
    try: start_configs.append(_build_random_dense(size, seed=size))
    except: pass
    try: start_configs.append(_build_fibonacci_sphere(size))
    except: pass
    
    if not start_configs:
        pos = np.zeros((size, 3))
        pos = _ensure_min_distance(pos)
        return pos

    best_energy = float('inf')
    best_pos = start_configs[0].copy()
    current_pos = start_configs[0].copy()
    current_energy = float('inf')
    
    for config in start_configs:
        tl = _time_left()
        if tl < 10.0:
            break
        energy, relaxed = _relax_single(config, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
            current_pos = relaxed.copy()
            current_energy = energy

    step_size = 0.5
    rng = np.random.default_rng(12345)
    
    while _time_left() > 10.0:
        noise = rng.normal(0, step_size, current_pos.shape)
        perturbed_pos = current_pos + noise
        
        energy, relaxed = _relax_single(perturbed_pos, potential, parameter)
        
        if energy < current_energy:
            current_energy = energy
            current_pos = relaxed.copy()
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        else:
            T = 0.5
            delta_E = energy - current_energy
            if np.random.rand() < np.exp(-delta_E / T):
                current_energy = energy
                current_pos = relaxed.copy()
        
        if energy < current_energy:
            step_size = max(0.2, step_size * 0.9)
        else:
            step_size = min(1.5, step_size * 1.1)

    pos = _ensure_min_distance(best_pos, min_d=0.5)
    return pos
# EVOLVE-BLOCK-END
