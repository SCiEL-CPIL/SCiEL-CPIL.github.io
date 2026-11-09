"""Morse cluster global minima search with direct structural initialization and gradient-based refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize

# Global deadline shared across all six construct() calls
_START_TIME = time.time()
_TOTAL_BUDGET = 310.0  # 20s safety margin before 330s kill

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
    """Build a cluster using close-packing lattice points (FCC/HCP-like)."""
    points = []
    # FCC lattice generation: points where i+j+k is even, scaled so nearest neighbor = 1
    # In FCC, nearest neighbor distance = a*sqrt(2)/2 where a is lattice constant
    # We want nearest neighbor = spacing, so a = spacing * sqrt(2)
    a = spacing * np.sqrt(2.0)
    
    # Generate a large enough grid
    max_coord = int(np.ceil(n / 3.0)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a
                    # Keep only points within a reasonable radius
                    if np.linalg.norm(p) <= spacing * np.sqrt(n) * 1.5:
                        points.append(p)
    
    if len(points) == 0:
        return np.zeros((n, 3))
    
    points = np.array(points)
    # Keep closest n points to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _build_hcp_like(n, spacing=1.0):
    """Build a cluster using HCP-like layer stacking."""
    points = []
    # HCP: layers of hexagonally packed points
    a = spacing  # nearest neighbor distance
    
    # Generate hexagonal lattice points in 2D
    def hex_points(radius_layers):
        pts = []
        for q in range(-radius_layers, radius_layers + 1):
            for r in range(-radius_layers, radius_layers + 1):
                # Only include if within hexagonal radius
                if abs(q) + abs(r) <= radius_layers and abs(q + r) <= radius_layers:
                    x = a * (q + 0.5 * r)
                    y = a * np.sqrt(3) / 2 * r
                    pts.append([x, y])
        return np.array(pts) if pts else np.zeros((0, 2))
    
    # Layer spacing for HCP: c/a = sqrt(8/3) ≈ 1.633
    layer_spacing = a * np.sqrt(8.0 / 3.0)
    offset = a / 2.0  # Offset for alternating layers
    
    current_radius = 0
    layer_count = 0
    while len(points) < n:
        # Determine how many points in this layer
        if layer_count == 0:
            layer_pts = np.array([[0.0, 0.0]])
        else:
            # Estimate needed radius
            needed = n - len(points)
            # Hex number: ~3*r^2 points for radius r
            r_est = int(np.ceil(np.sqrt(needed / 3.0))) + 1
            layer_pts = hex_points(r_est)
            # Offset alternating layers
            if layer_count % 2 == 1:
                layer_pts[:, 0] += offset
        
        z = layer_count * layer_spacing
        for pt in layer_pts:
            if len(points) >= n:
                break
            p = np.array([pt[0], pt[1], z])
            points.append(p)
        
        layer_count += 1
        # Increase search radius
        current_radius += 1
    
    return np.array(points[:n])

def _build_icosahedral(n, spacing=1.0):
    """Build a cluster based on icosahedral symmetry with shells."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices (12 points)
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico12 /= np.linalg.norm(ico12[0])
    
    # Cuboctahedron vertices (12 points) - for second shell
    cubo12 = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ], dtype=float)
    cubo12 /= np.linalg.norm(cubo12[0])
    
    # Icosidodecahedron-like third shell (30 points)
    # Use permutations of (0, 1, phi) and (1, phi, 2*phi) etc.
    shell3 = []
    for perm in [(0, 1, phi), (1, phi, 0), (phi, 0, 1)]:
        for signs in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                      (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
            p = np.array([perm[0] * signs[0], perm[1] * signs[1], perm[2] * signs[2]])
            norm = np.linalg.norm(p)
            if norm > 0:
                shell3.append(p / norm)
    shell3 = np.array(shell3)
    
    points = [np.zeros(3)]  # Center
    
    # First shell: 12 icosahedron vertices at radius ~1.0
    for v in ico12:
        points.append(v * 1.0)
    
    # Second shell: 12 cuboctahedron vertices at radius ~1.7
    for v in cubo12:
        points.append(v * 1.7)
    
    # Third shell: 30 points at radius ~2.5
    for v in shell3:
        points.append(v * 2.5)
    
    # If we need more, add fourth shell
    if len(points) < n:
        # Use more densely packed spherical shell
        rng = np.random.default_rng(42)
        shell_radius = 3.2
        attempts = 0
        while len(points) < n and attempts < 5000:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * shell_radius
            # Check minimum distance
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
    
    # Fill remaining with random points if needed
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _build_fibonacci_sphere(n, spacing=1.0):
    """Build a cluster using Fibonacci sphere for uniform distribution."""
    if n <= 1:
        return np.zeros((n, 3))
    
    points = [np.zeros(3)]  # Center
    remaining = n - 1
    
    # Fibonacci sphere
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    radius = spacing * (remaining ** (1.0/3.0)) * 0.8
    
    for i in range(remaining):
        t = i / remaining
        z = 1.0 - 2.0 * t  # z goes from 1 to -1
        radius_circle = np.sqrt(max(0, 1.0 - z * z))
        theta = golden_angle * i
        x = np.cos(theta) * radius_circle
        y = np.sin(theta) * radius_circle
        p = np.array([x, y, z]) * radius
        points.append(p)
    
    return np.array(points[:n])

def _relax_scipy(pos, rho, time_limit=5.0):
    """Relax a configuration using L-BFGS-B with analytical gradient."""
    x0 = pos.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
        
    def gradient(x):
        pos = x.reshape(-1, 3)
        grad = morse_energy_gradient(pos, rho)
        return grad.flatten()
        
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000})
        if res.success:
            return res.fun, res.x.reshape(-1, 3)
        else:
            # Return the best point found even if not fully converged
            return res.fun, res.x.reshape(-1, 3)
    except Exception:
        return _morse_energy(pos, rho), pos

def _ensure_min_distance(pos, min_d=0.5):
    """Ensure all pairs are at least min_d apart."""
    n = len(pos)
    pos = pos.copy()
    for _ in range(50):
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
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", 10))
    
    # Check time budget
    tl = _time_left()
    if tl < 5.0:
        # Emergency: return a quick valid result
        pos = _build_close_packing(size)
        pos = _ensure_min_distance(pos)
        return pos
    
    # Generate diverse starting configurations
    start_configs = []
    
    # 1. Close packing (FCC-like)
    try:
        start_configs.append(_build_close_packing(size))
    except Exception:
        pass
    
    # 2. HCP-like
    try:
        start_configs.append(_build_hcp_like(size))
    except Exception:
        pass
    
    # 3. Icosahedral
    try:
        start_configs.append(_build_icosahedral(size))
    except Exception:
        pass
    
    # 4. Random dense
    try:
        start_configs.append(_build_random_dense(size, seed=size))
    except Exception:
        pass
    
    # 5. Fibonacci sphere
    try:
        start_configs.append(_build_fibonacci_sphere(size))
    except Exception:
        pass
    
    # 6. Another random with different seed
    try:
        start_configs.append(_build_random_dense(size, seed=size + 7))
    except Exception:
        pass
    
    if not start_configs:
        pos = np.zeros((size, 3))
        pos = _ensure_min_distance(pos)
        return pos
    
    # Determine number of attempts based on time budget
    tl = _time_left()
    # Estimate time per relaxation: L-BFGS-B is faster, assume ~2-4s
    est_per_relax = 2.5 if size <= 50 else 4.0
    n_attempts = max(1, min(len(start_configs), int(tl / (est_per_relax * 1.5))))
    
    best_energy = float('inf')
    best_pos = start_configs[0].copy()
    
    for i in range(n_attempts):
        tl = _time_left()
        if tl < 8.0:
            break
        
        config = start_configs[i]
        
        # Try multiple perturbations for this config
        rng = np.random.default_rng(i * 1234 + size)
        n_perturb = 3 if tl > 30 else 1
        pert_time = min(tl * 0.5 / n_perturb, 10.0)
        
        for p in range(n_perturb):
            tl = _time_left()
            if tl < 5.0:
                break
            
            if p == 0:
                pos_try = config.copy()
            else:
                # Small perturbation
                noise_scale = 0.1 * (1.0 + 0.5 * p)
                pos_try = config + rng.normal(0, noise_scale, config.shape)
            
            # Set a time limit for this relaxation
            start_t = time.time()
            energy, relaxed = _relax_scipy(pos_try, parameter, time_limit=pert_time)
            elapsed = time.time() - start_t
            
            # Post-process: enforce constraints
            relaxed = _ensure_min_distance(relaxed, min_d=0.5)
            # Re-evaluate energy after constraint enforcement
            final_energy = _morse_energy(relaxed, parameter)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_pos = relaxed.copy()
            
            # If we used too much time, break
            if time.time() - start_t > pert_time:
                break
    
    # Final: ensure minimum distance constraint
    pos = _ensure_min_distance(best_pos, min_d=0.5)
    
    return pos
# EVOLVE-BLOCK-END
