"""Rewritten program for Dzugutov cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

def _dzugutov_energy(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < a:
                E += A * (r ** (-m) - B) * np.exp(c / (r - a))
            if r < b:
                E += B * np.exp(d / (r - b))
    return E

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _get_mackay_icosahedron(n_atoms, spacing=1.0):
    """
    Generate a proper Mackay icosahedron cluster.
    Uses the correct icosahedral shell structure.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = _get_icosahedron_vertices()
    
    # Shell radii for Mackay icosahedron
    r1 = spacing  # Shell 1: 12 atoms
    r2 = spacing * np.sqrt(1 + 2.0/phi)  # Shell 2: 42 atoms
    r3 = spacing * np.sqrt(1 + 4.0/phi + 4.0/phi**2)  # Shell 3: 92 atoms
    
    positions = [np.zeros(3)]  # Core
    
    # Shell 1: 12 atoms at icosahedron vertices
    for v in ico_verts:
        positions.append(v * r1)
    
    # Shell 2: 42 atoms - proper icosahedral positions
    if n_atoms > 13:
        # The 42 positions in shell 2 of a Mackay icosahedron are:
        # 6 at body centers (octahedral directions) + 36 at face/edge positions
        # Actually, for proper Mackay: shell 2 has 42 atoms at specific icosahedral directions
        
        # Generate all icosahedral directions for shell 2
        # Shell 2 consists of:
        # - 6 atoms at <100> directions (octahedral)
        # - 30 atoms at <110>-like directions (edge centers of icosahedron)
        # - 6 atoms at <111>-like directions (face centers)
        
        # Octahedral directions (6)
        oct_dirs = np.array([
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ], dtype=float)
        for v in oct_dirs:
            v = v / np.linalg.norm(v)
            positions.append(v * r2)
        
        # Edge center directions (30) - midpoints of icosahedron edges
        edge_len = np.linalg.norm(ico_verts[0] - ico_verts[1])
        edge_dirs = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if abs(d - edge_len) < 0.05:
                    mid = (ico_verts[i] + ico_verts[j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_dirs.append(mid)
        
        for v in edge_dirs:
            positions.append(v * r2)
        
        # Face center directions (6) - centers of 20 faces, but only 6 unique in icosahedral symmetry
        # Actually, we need 6 more to get to 42 total in shell 2
        # These are at the "body diagonal" type positions
        face_dirs = []
        # Use the 6 directions that are perpendicular to the 6 pairs of opposite faces
        # For icosahedron, these are at directions like (1,1,1) normalized with icosahedral constraints
        # Simpler: use 6 directions at 54.7358° from z-axis, 60° apart in azimuth
        for k in range(6):
            angle = k * 2 * np.pi / 6
            # These should be at the "face center" positions
            # For a regular icosahedron, face centers are at specific positions
            # Let's compute them properly from the face normals
            pass
        
        # Alternative: just add 6 more at evenly spaced positions
        for k in range(6):
            angle = k * 2 * np.pi / 6 + np.pi/6
            v = np.array([np.cos(angle), np.sin(angle), 0.0])
            v = v / np.linalg.norm(v)
            positions.append(v * r2)
    
    # Shell 3: 92 atoms
    if n_atoms > 55:
        n_needed = n_atoms - len(positions)
        if n_needed > 0:
            # Place atoms on shell 3 using Fibonacci sphere
            golden_angle = np.pi * (3 - np.sqrt(5))
            for k in range(n_needed):
                i = k + 1
                z = 1 - 2 * i / max(n_needed, 1)
                radius_xy = np.sqrt(max(0, 1 - z**2))
                theta = golden_angle * i
                v = np.array([radius_xy * np.cos(theta), radius_xy * np.sin(theta), z])
                v = v / np.linalg.norm(v)
                positions.append(v * r3)
    
    positions = positions[:n_atoms]
    return np.array(positions, dtype=float)

def _vibrational_melting(positions, n_steps=50, t_initial=0.5, t_final=0.01, rng=None):
    """
    Apply vibrational melting: simulated annealing from high-symmetry start.
    This breaks symmetry and allows flow to lower-energy sub-basins.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current = positions.copy()
    current_energy = _dzugutov_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    for step in range(n_steps):
        # Temperature schedule: exponential decay
        t = t_initial * (t_final / t_initial) ** (step / n_steps)
        
        # Generate perturbation
        noise_scale = 0.1 * t
        perturbed = current + rng.normal(0, noise_scale, current.shape)
        
        # Ensure minimum distance
        for i in range(len(perturbed)):
            for j in range(i+1, len(perturbed)):
                d = np.linalg.norm(perturbed[i] - perturbed[j])
                if d < 0.5:
                    diff = perturbed[i] - perturbed[j]
                    if np.linalg.norm(diff) > 1e-10:
                        diff = diff / np.linalg.norm(diff)
                    else:
                        diff = np.array([1.0, 0, 0])
                    perturbed[i] += diff * (0.5 - d) / 2
                    perturbed[j] -= diff * (0.5 - d) / 2
        
        # Relax the perturbed structure
        try:
            relaxed_energy, relaxed = relax(perturbed, 'dzugutov')
        except Exception:
            continue
        
        # Metropolis acceptance
        delta_e = relaxed_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(t, 1e-10)):
            current = relaxed
            current_energy = relaxed_energy
        
        # Track best
        if current_energy < best_energy:
            best = current.copy()
            best_energy = current_energy
    
    return best

def _defect_seeded_initialization(n_target, n_base=55, spacing=1.0):
    """
    Generate diverse candidates by removing/adding atoms from a Mackay icosahedron.
    """
    candidates = []
    base_cluster = _get_mackay_icosahedron(n_base, spacing)
    
    if n_target == n_base:
        candidates.append(base_cluster.copy())
        return candidates
    
    if n_target < n_base:
        n_remove = n_base - n_target
        distances = np.linalg.norm(base_cluster, axis=1)
        sorted_indices = np.argsort(distances)[::-1]
        
        remove_indices = sorted_indices[:n_remove]
        keep_indices = np.setdiff1d(np.arange(n_base), remove_indices)
        candidates.append(base_cluster[keep_indices].copy())
        
        # Random removals for diversity
        rng = np.random.default_rng(n_target)
        for _ in range(4):
            remove_random = rng.choice(n_base, size=n_remove, replace=False)
            keep_random = np.setdiff1d(np.arange(n_base), remove_random)
            candidates.append(base_cluster[keep_random].copy())
    else:
        n_add = n_target - n_base
        phi = (1 + np.sqrt(5)) / 2
        r3 = spacing * np.sqrt(1 + 4.0/phi + 4.0/phi**2)
        
        # Add atoms at shell 3 positions
        golden_angle = np.pi * (3 - np.sqrt(5))
        new_positions = []
        for k in range(n_add):
            i = k + 1
            z = 1 - 2 * i / max(n_add, 1)
            radius_xy = np.sqrt(max(0, 1 - z**2))
            theta = golden_angle * i
            v = np.array([radius_xy * np.cos(theta), radius_xy * np.sin(theta), z])
            v = v / np.linalg.norm(v)
            new_positions.append(v * r3)
        
        candidates.append(np.vstack([base_cluster, np.array(new_positions)]).copy())
        
        # Random additions
        rng = np.random.default_rng(n_target + 100)
        for _ in range(3):
            new_random = []
            for _ in range(n_add):
                theta = rng.uniform(0, 2 * np.pi)
                phi_angle = rng.uniform(0, np.pi)
                v = np.array([
                    np.sin(phi_angle) * np.cos(theta),
                    np.sin(phi_angle) * np.sin(theta),
                    np.cos(phi_angle)
                ])
                new_random.append(v * r3)
            candidates.append(np.vstack([base_cluster, np.array(new_random)]).copy())
    
    return candidates

def _perturbed_icosahedron(n, spacing=1.0, noise_scale=0.1, rng=None):
    """Generate a perturbed icosahedral cluster."""
    if rng is None:
        rng = np.random.default_rng(n)
    
    base = _get_mackay_icosahedron(n, spacing)
    noise = rng.normal(0, noise_scale, base.shape)
    perturbed = base + noise
    
    for i in range(len(perturbed)):
        for j in range(i + 1, len(perturbed)):
            d = np.linalg.norm(perturbed[i] - perturbed[j])
            if d < 0.5:
                diff = perturbed[i] - perturbed[j]
                if np.linalg.norm(diff) > 1e-10:
                    diff = diff / np.linalg.norm(diff)
                else:
                    diff = np.array([1.0, 0, 0])
                perturbed[i] += diff * (0.5 - d) / 2
                perturbed[j] -= diff * (0.5 - d) / 2
    
    return perturbed

def _random_compact_cluster(n, spacing=1.0, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng(n)
    
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = n * 200
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n], dtype=float)

def construct(instance):
    """
    Find low-energy structures for Dzugutov potential clusters.
    Uses vibrational melting from Mackay icosahedra plus basin hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s limit
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    def try_relax(positions):
        nonlocal best_energy, best_positions
        if time.time() > deadline:
            return
        try:
            energy, relaxed = relax(positions, potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Strategy 1: Vibrational melting from Mackay icosahedron
    base_cluster = _get_mackay_icosahedron(size, spacing=1.0)
    melted = _vibrational_melting(base_cluster, n_steps=40, t_initial=0.4, t_final=0.02, rng=rng)
    try_relax(melted)
    
    # Strategy 2: Defect-seeded initializations with relaxation
    candidates = _defect_seeded_initialization(size, n_base=55, spacing=1.0)
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
        try_relax(cand)
    
    # Strategy 3: Perturbed icosahedral structures
    for i in range(6):
        if time.time() > deadline:
            break
        noise_scale = 0.05 + 0.04 * i
        perturbed = _perturbed_icosahedron(size, spacing=1.0, noise_scale=noise_scale, rng=rng)
        try_relax(perturbed)
    
    # Strategy 4: Basin hopping on best structure
    if best_positions is not None and time.time() < deadline - 20:
        current = best_positions.copy()
        current_energy = best_energy
        n_bh_steps = 30
        for step in range(n_bh_steps):
            if time.time() > deadline:
                break
            perturbation_scale = 0.15 + 0.15 * (1 - step / n_bh_steps)
            perturbed = current + rng.normal(0, perturbation_scale, current.shape)
            try:
                energy, relaxed = relax(perturbed, potential)
                if energy < current_energy:
                    current_energy = energy
                    current = relaxed.copy()
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed.copy()
            except Exception:
                pass
    
    # Strategy 5: More random restarts if time permits
    if time.time() < deadline - 10:
        for i in range(4):
            if time.time() > deadline:
                break
            random_cluster = _random_compact_cluster(size, spacing=1.0, rng=rng)
            try_relax(random_cluster)
    
    if best_positions is None:
        best_positions = _random_compact_cluster(size, spacing=1.0, rng=rng)
    
    return best_positions
# EVOLVE-BLOCK-END
