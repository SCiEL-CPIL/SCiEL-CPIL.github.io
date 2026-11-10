"""Dzugutov cluster global minima: portfolio of structural heuristics + sequential relaxation with basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 315.0  # safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _compute_energy(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute Dzugutov energy for a cluster."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 0.5:
                return float('inf')
            if r < a:
                energy += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def _generate_mackay_icosahedron(n, rng):
    """Generate a Mackay icosahedral cluster of n atoms with correct geometry."""
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron (unit radius)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    # Scale to Dzugutov equilibrium distance ~1.9
    scale = 1.9
    
    positions = [np.zeros(3)]  # center
    
    if n <= 1:
        return np.array(positions[:n])
    
    # Shell 1: 12 icosahedron vertices
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(v * scale)
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Find edges (pairs with distance < 1.2 * scale)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.2:
                edges.append((i, j))
    
    # Shell 2: 30 edge midpoints
    for i, j in edges:
        if len(positions) >= n:
            break
        mid = (verts[i] + verts[j]) / 2
        positions.append(mid * scale)
    
    if n <= 43:
        return np.array(positions[:n])
    
    # Shell 3: 20 face centers (for n up to 63)
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if d1 < 1.2 and d2 < 1.2 and d3 < 1.2:
                    faces.append((i, j, k))
    
    for i, j, k in faces:
        if len(positions) >= n:
            break
        centroid = (verts[i] + verts[j] + verts[k]) / 3
        centroid = centroid / np.linalg.norm(centroid)
        positions.append(centroid * scale)
    
    # If still not enough, add atoms near surface
    while len(positions) < n:
        idx = rng.integers(0, len(positions))
        p = positions[idx] + rng.normal(0, 0.3, size=3)
        positions.append(p)
    
    return np.array(positions[:n])


def _generate_fcc(n, rng):
    """Generate an FCC lattice chunk with Dzugutov-appropriate spacing."""
    a = 1.87  # closer to Dzugutov equilibrium
    
    side = int((n * a**3 / 4) ** (1/3)) + 3
    radius = side * a / 2
    
    points = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    if len(points) >= n:
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
    else:
        points = np.array(points)
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points = np.vstack([points, p])
        return points[:n]


def _generate_bcc(n, rng):
    """Generate a BCC lattice chunk."""
    a = 1.87
    
    side = int((n * a**3 / 2) ** (1/3)) + 3
    radius = side * a / 2
    
    points = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    if len(points) >= n:
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
    else:
        points = np.array(points)
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points = np.vstack([points, p])
        return points[:n]


def _generate_random_cluster(n, rng, spacing=1.2):
    """Generate a random compact cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 20000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.55:
                    points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _generate_quasicrystal_like(n, rng):
    """Generate a quasicrystal-like structure using icosahedral tiling approximation."""
    phi = (1 + np.sqrt(5)) / 2
    scale = 1.9
    
    # Start with icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    positions = [np.zeros(3)]
    
    # Add vertices at multiple scales for quasicrystal-like density
    for shell_scale in [1.0, 0.7, 0.45]:
        for v in verts:
            if len(positions) >= n:
                break
            p = v * scale * shell_scale
            # Check minimum distance
            if len(positions) > 0:
                dists = np.linalg.norm(np.array(positions) - p, axis=1)
                if np.min(dists) < 0.55:
                    continue
            positions.append(p)
        if len(positions) >= n:
            break
    
    # Fill remaining with random points
    while len(positions) < n:
        p = rng.normal(0, scale * 0.8, size=3)
        if len(positions) > 0:
            dists = np.linalg.norm(np.array(positions) - p, axis=1)
            if np.min(dists) < 0.55:
                continue
        positions.append(p)
    
    return np.array(positions[:n])


def _basin_hopping(positions, rng, potential, parameter, max_iter=50, step_size=0.5):
    """Simple basin hopping: perturb, relax, accept if lower energy."""
    current = positions.copy()
    current_energy = _compute_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    for i in range(max_iter):
        if _get_time_remaining() < 30:
            break
        
        # Perturb
        perturbed = current + rng.normal(0, step_size, size=current.shape)
        
        # Relax
        try:
            energy, relaxed = relax(perturbed, potential, parameter)
        except Exception:
            energy = float('inf')
            relaxed = perturbed
        
        # Accept if better
        if energy < best_energy:
            best = relaxed.copy()
            best_energy = energy
        
        # Always move to relaxed position (Metropolis-like with T=0)
        current = relaxed.copy()
        current_energy = energy
        
        # Reduce step size gradually
        step_size *= 0.95
    
    return best, best_energy


def construct(instance):
    """Portfolio of structural heuristics with sequential relaxation and basin hopping."""
    global _START_TIME
    _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance
    
    rng = np.random.default_rng(size)
    
    # Generate diverse starting structures
    generators = [
        _generate_mackay_icosahedron,
        _generate_fcc,
        _generate_bcc,
        _generate_quasicrystal_like,
        _generate_random_cluster,
    ]
    
    initial_structures = []
    for gen in generators:
        try:
            pos = gen(size, rng)
            if len(pos) == size and np.all(np.linalg.norm(pos, axis=1) > 0):
                # Check minimum distance
                if size > 1:
                    min_d = float('inf')
                    for i in range(size):
                        for j in range(i+1, size):
                            d = np.linalg.norm(pos[i] - pos[j])
                            if d < min_d:
                                min_d = d
                    if min_d < 0.4:
                        continue
                initial_structures.append(pos)
        except Exception:
            pass
    
    # Add more random variants
    for _ in range(3):
        try:
            pos = _generate_random_cluster(size, rng)
            if len(pos) == size:
                initial_structures.append(pos)
        except Exception:
            pass
    
    if not initial_structures:
        pos = _generate_random_cluster(size, rng)
        initial_structures = [pos]
    
    # Phase 1: Relax all initial structures, keep best
    best_energy = float('inf')
    best_positions = initial_structures[0]
    
    for pos in initial_structures:
        if _get_time_remaining() < 40:
            break
        try:
            energy, relaxed = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            pass
    
    # Phase 2: Basin hopping from best structure
    if _get_time_remaining() > 60:
        best_positions, best_energy = _basin_hopping(
            best_positions, rng, potential, parameter, max_iter=80, step_size=0.6
        )
    
    # Phase 3: More basin hopping from other good structures if time allows
    if _get_time_remaining() > 90 and len(initial_structures) > 1:
        for pos in initial_structures[1:4]:
            if _get_time_remaining() < 50:
                break
            try:
                energy, relaxed = relax(pos, potential, parameter)
                if energy < best_energy + 5:  # within 5 units of best
                    bp, be = _basin_hopping(
                        relaxed, rng, potential, parameter, max_iter=40, step_size=0.5
                    )
                    if be < best_energy:
                        best_energy = be
                        best_positions = bp
            except Exception:
                pass
    
    # Phase 4: Final polish - high-precision relaxation
    if _get_time_remaining() > 20:
        try:
            # Try relaxing with smaller perturbations
            for amp in [0.1, 0.05, 0.02]:
                if _get_time_remaining() < 15:
                    break
                perturbed = best_positions + rng.normal(0, amp, size=best_positions.shape)
                energy, relaxed = relax(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
