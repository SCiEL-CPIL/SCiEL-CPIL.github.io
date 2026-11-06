"""Morse cluster global minima search with symmetry-adapted seeds and multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy(pos, rho):
    """Compute Morse potential energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)

def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient."""
    try:
        return morse_energy_gradient(pos, rho)
    except Exception:
        n = len(pos)
        e = _morse_energy(pos, rho)
        grad = np.zeros_like(pos)
        eps = 1e-6
        for i in range(n):
            for d in range(3):
                orig = pos[i, d]
                pos[i, d] = orig + eps
                ep = _morse_energy(pos, rho)
                pos[i, d] = orig - eps
                em = _morse_energy(pos, rho)
                pos[i, d] = orig
                grad[i, d] = (ep - em) / (2.0 * eps)
        return e, grad

def _ensure_min_dist(pos, min_d=0.51, max_iter=100):
    """Separate atoms that are too close."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        min_dist = np.inf
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist >= min_d:
            break
        i, j = min_pair
        rij = pos[i] - pos[j]
        r = np.linalg.norm(rij)
        if r < 1e-10:
            rij = np.array([1.0, 0, 0])
            r = 1.0
        push = (min_d - r + 0.01) / 2.0
        pos[i] += push * rij / r
        pos[j] -= push * rij / r
    return pos

def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points, keeping n closest to origin."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _hcp_points(n, spacing=1.0):
    """Generate HCP lattice points, keeping n closest to origin."""
    pts = []
    a = spacing
    c_over_a = np.sqrt(8.0/3.0)
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1/3.0, j + 2/3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points, keeping n closest to origin."""
    pts = []
    a = spacing * np.sqrt(3.0)/2.0
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _icosahedral_cluster(n, spacing=1.0):
    """Generate icosahedral cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    edge_len = 2.0
    scale = spacing / edge_len
    verts = verts * scale
    
    pts = [np.zeros(3)]
    
    if n > 1:
        for v in verts:
            pts.append(v.copy())
    
    if n > 13:
        faces = [
            [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
            [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
            [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
            [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
            [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
        ]
        for f in faces:
            center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            r = np.linalg.norm(center)
            if r > 1e-10:
                center = center / r * r * 1.15
            pts.append(center)
    
    if n > 33:
        edges = set()
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.5 * scale:
                    mid = (verts[i] + verts[j]) / 2.0
                    r = np.linalg.norm(mid)
                    if r > 1e-10:
                        mid = mid / r * r * 1.3
                    pts.append(mid)
    
    if n > 63:
        verts2 = verts * 2.0
        for v in verts2:
            pts.append(v.copy())
        if n > 75:
            faces = [
                [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
                [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
                [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
                [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
                [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
            ]
            for f in faces:
                center = (verts2[f[0]] + verts2[f[1]] + verts2[f[2]]) / 3.0
                pts.append(center)
    
    pts = np.array(pts[:n])
    return pts

def _lattice_init(n, rng, spacing=1.0):
    """Generate lattice-based initial configuration."""
    shell = int(np.ceil(n ** (1.0 / 3.0))) + 2
    points = []
    for ix in range(shell):
        for iy in range(shell):
            for iz in range(shell):
                if len(points) < n:
                    points.append(np.array([ix, iy, iz]) * spacing)
    
    if len(points) < n:
        for _ in range(n - len(points)):
            p = rng.uniform(-0.5, shell * spacing + 0.5, 3)
            points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    points += rng.normal(0, 0.05, points.shape)
    return points

def _random_packed_init(n, rng, spacing=1.0):
    """Generate random packed initial configuration."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.6 * spacing
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points

def _fibonacci_sphere_init(n, rng, radius=1.0):
    """Generate Fibonacci sphere initial configuration."""
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * i / max(n - 1, 1)
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
        points.append(p)
    
    points = np.array(points)
    points -= points.mean(axis=0)
    points += rng.normal(0, 0.02, points.shape)
    return points

def _symmetry_adapted_seed(n, rho, seed_idx=0):
    """Generate a symmetry-adapted seed with radial shell structure."""
    rng = np.random.default_rng(seed_idx * 1000 + 7)
    
    # Estimate cluster radius
    R_est = 1.2 * (n / (4/3 * np.pi)) ** (1/3)
    
    # Number of shells
    n_shells = max(2, int(np.floor(n / 8)))
    
    # Generate shells with decreasing density
    points = [np.zeros(3)]  # center atom
    
    for shell in range(1, n_shells):
        r_shell = R_est * shell / n_shells
        # Fibonacci sphere for this shell
        n_in_shell = min(n - len(points), max(1, int(4 * np.pi * r_shell**2 / 0.5)))
        
        for i in range(n_in_shell):
            z = 1 - 2 * i / max(n_in_shell - 1, 1)
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / ((1 + np.sqrt(5)) / 2)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r_shell
            # Add small perturbation to break exact symmetry
            p += rng.normal(0, 0.05, 3)
            points.append(p)
        
        if len(points) >= n:
            break
    
    # Fill remaining with random points in the volume
    while len(points) < n:
        r = R_est * (0.5 + 0.5 * rng.rand())
        theta = rng.rand() * 2 * np.pi
        phi = np.arccos(2 * rng.rand() - 1)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 45.0  # 50s per instance, leave 5s margin for 6 instances in 300s
    
    seeds = []
    
    # 1. FCC
    try:
        s = _fcc_points(size)
        if s is not None:
            seeds.append(s)
    except Exception:
        pass
        
    # 2. HCP
    try:
        s = _hcp_points(size)
        if s is not None:
            seeds.append(s)
    except Exception:
        pass
        
    # 3. BCC
    try:
        s = _bcc_points(size)
        if s is not None:
            seeds.append(s)
    except Exception:
        pass
        
    # 4. Icosahedral
    try:
        s = _icosahedral_cluster(size)
        if s is not None:
            seeds.append(s)
    except Exception:
        pass
        
    # 5. Symmetry-adapted seed
    try:
        s = _symmetry_adapted_seed(size, parameter, seed_idx=0)
        seeds.append(s)
    except Exception:
        pass
        
    # 6. Random Dense 1
    try:
        rng = np.random.default_rng(1)
        s = _random_packed_init(size, rng)
        seeds.append(s)
    except Exception:
        pass
        
    # 7. Random Dense 2
    try:
        rng = np.random.default_rng(42)
        s = _random_packed_init(size, rng)
        seeds.append(s)
    except Exception:
        pass
        
    # 8. Fibonacci sphere
    try:
        rng = np.random.default_rng(123)
        s = _fibonacci_sphere_init(size, rng, radius=1.5 * size ** (1/3))
        seeds.append(s)
    except Exception:
        pass

    best_energy = float('inf')
    best_positions = None
    
    # Phase 1: Relax all seeds
    for seed in seeds:
        if time.time() > deadline - 10.0:  # Reserve 10s for hopping
            break
        try:
            energy, positions = relax(seed, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
            
    if best_positions is None:
        # Fallback
        if seeds:
            return seeds[0]
        return np.array([np.array([i, 0, 0]) for i in range(size)])

    # Phase 2: Basin Hopping
    # Try to escape local minima from the best configuration
    current_pos = best_positions.copy()
    current_energy = best_energy
    hop_scale = 0.5
    rng = np.random.default_rng(123)
    
    while time.time() < deadline - 2.0:
        # Perturb
        noise = rng.normal(0, hop_scale, current_pos.shape)
        perturbed = current_pos + noise
        
        try:
            energy, positions = relax(perturbed, potential, parameter)
            if energy < current_energy:
                current_energy = energy
                current_pos = positions
                # Reduce hop scale on success
                hop_scale *= 0.9
            else:
                # Accept with some probability if not too worse
                # For global min search, usually we accept if it's better
                # Simple basin hopping: only accept if better
                hop_scale *= 0.95
        except Exception:
            break
            
    return current_pos
# EVOLVE-BLOCK-END
