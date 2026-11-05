"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16


def _dzugutov_energy(pos):
    """Compute Dzugutov energy for a cluster (vectorized)."""
    n = len(pos)
    if n < 2:
        return 0.0
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < a:
                energy += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def _mackay_icosahedron(n, shell_spacing=1.0):
    """Generate Mackay icosahedral cluster with up to n atoms.
    
    Mackay icosahedra: 1 (center) + 12 (1st shell) + 42 (2nd shell) + 92 (3rd shell)
    For Dzugutov, these are the natural polytetrahedral global minima.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Generate icosahedron vertices (12 vertices)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_verts, axis=1)
    ico_verts = ico_verts / norms[:, np.newaxis]
    
    atoms = [np.zeros(3)]  # Center
    
    # Shell 1: 12 atoms at icosahedron vertices
    if n > 1:
        shell1 = ico_verts * shell_spacing
        atoms.extend([shell1[i].copy() for i in range(min(12, n - 1))])
    
    if len(atoms) >= n:
        return np.array(atoms[:n])
    
    # Shell 2: 42 atoms (icosahedron faces subdivided)
    # Generate by taking all midpoints of edges and face centers of the icosahedron
    # More systematically: use the icosahedral symmetry
    # Shell 2 positions: 20 face centers + 30 edge midpoints = 50, but only 42 are in Mackay
    # Actually Mackay 2nd shell has 42 atoms at specific positions
    
    # Generate icosahedron faces (20 triangles)
    # The icosahedron has 20 faces. We'll generate shell 2 by projecting
    # tetrahedral sub-structures onto the icosahedron surface.
    
    # Alternative: generate points on icosahedron surface using quaternions
    # For shell 2, use the 42 positions of the Mackay icosahedron
    # These are at radius sqrt(5)/2 * shell_spacing from center (for unit ico)
    
    # Actually, let's use a simpler approach: generate shell 2 as
    # all points that are vertices of tetrahedra inscribed in the icosahedron
    # Shell 2 radius for Mackay: r2 = r1 * sqrt(5)/2 ≈ 1.118 * r1
    # But let's just use the known structure
    
    # For simplicity, generate additional shells by scaling icosahedron vertices
    # and adding intermediate points
    
    # Shell 2: 42 atoms
    # Use the fact that shell 2 of Mackay icosahedron consists of:
    # - 20 atoms at face centers of the icosahedron (radius = r1 * 2/phi ≈ 1.236*r1)
    # - 30 atoms at edge midpoints (radius = r1 * sqrt(1 + 1/phi^2) ≈ 1.118*r1)
    # Wait, that's 50 not 42. Let me reconsider.
    
    # Actually Mackay icosahedron shells: 1, 12, 42, 92, 120, 140, 168...
    # 42 = 20 (face centers) + 22? No...
    # 42 = 12 (vertex positions of outer icosahedron) + 30 (edge-related)
    # Let me just use a parametric approach
    
    # Generate shell 2 using icosahedral harmonics
    # Simpler: just place 42 points on a sphere of appropriate radius
    # with icosahedral symmetry
    
    # For now, use a simpler approach: generate points on concentric shells
    # with icosahedral symmetry
    
    if len(atoms) < n:
        # Shell 2: use icosahedron vertices at larger radius + face centers
        shell2_radius = shell_spacing * 1.618  # golden ratio scaling
        shell2 = ico_verts * shell2_radius
        
        # Add face centers of the icosahedron (20 faces)
        # Icosahedron faces: need to define them
        # For a regular icosahedron, the 20 faces can be enumerated
        # Let's compute face centers from the vertex list
        
        # Actually, let's use a different approach for shell 2:
        # Generate all points at the 42 Mackay positions
        # These form a truncated icosahedron-like structure
        
        # Simpler: use the icosahedron vertices at 2 different radii
        # and add edge midpoints
        edge_midpoints = []
        # Icosahedron edges: each vertex connects to 5 others
        # For 12 vertices, there are 30 edges
        # Compute all pairs and find edges (distance = 2*r1 for unit ico)
        r1 = shell_spacing
        edge_dist = 2 * r1  # edge length of icosahedron with circumradius r1
        
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j]) * r1
                if abs(d - edge_dist) < 0.1 * r1:
                    mid = (shell1[i] + shell1[j]) / 2
                    edge_midpoints.append(mid)
        
        # Deduplicate edge midpoints
        unique_mids = []
        for mid in edge_midpoints:
            if not any(np.linalg.norm(mid - u) < 0.01 * r1 for u in unique_mids):
                unique_mids.append(mid)
        
        # Add shell 2 atoms: icosahedron vertices at larger radius + edge midpoints
        for v in shell2:
            if len(atoms) < n:
                atoms.append(v.copy())
        for mid in unique_mids:
            if len(atoms) < n:
                atoms.append(mid.copy())
    
    if len(atoms) >= n:
        return np.array(atoms[:n])
    
    # Shell 3: 92 atoms (for n > 55)
    if len(atoms) < n:
        shell3_radius = shell_spacing * 2.236  # sqrt(5) * r1
        # Generate 92 points using icosahedral symmetry
        # Use icosahedron vertices at shell3 radius + additional points
        shell3 = ico_verts * shell3_radius
        for v in shell3:
            if len(atoms) < n:
                atoms.append(v.copy())
        
        # Add more points to fill up to n
        # Use a dense icosahedral distribution
        if len(atoms) < n:
            # Generate additional points on a sphere
            rng = np.random.default_rng(42)
            while len(atoms) < n:
                theta = rng.uniform(0, 2 * np.pi)
                cos_phi = rng.uniform(-1, 1)
                sin_phi = np.sqrt(1 - cos_phi**2)
                p = shell3_radius * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
                # Check minimum distance
                if all(np.linalg.norm(p - q) >= 0.4 * shell_spacing for q in atoms):
                    atoms.append(p)
    
    return np.array(atoms[:n])


def _parametric_polytetrahedral(n, seed=0):
    """Generate a polytetrahedral structure using tetrahedral stacking.
    
    Dzugutov's potential favors local tetrahedral coordination (CN=4).
    Build by stacking tetrahedra in a compact arrangement.
    """
    rng = np.random.default_rng(seed)
    spacing = 1.0
    
    # Start with a central atom
    atoms = [np.zeros(3)]
    
    # Add tetrahedral neighbors iteratively
    tet_dirs = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float) / np.sqrt(3)
    
    attempts = 0
    max_attempts = n * 50
    
    while len(atoms) < n and attempts < max_attempts:
        attempts += 1
        # Pick a random existing atom
        base_idx = rng.integers(0, len(atoms))
        base = atoms[base_idx]
        
        # Try to add a tetrahedral neighbor
        for _ in range(4):
            if len(atoms) >= n:
                break
            # Random rotation of tetrahedral direction
            angle = rng.uniform(0, 2 * np.pi)
            axis = rng.normal(size=3)
            axis /= (np.linalg.norm(axis) + 1e-10)
            
            # Apply rotation
            dir_idx = rng.integers(0, 4)
            d = tet_dirs[dir_idx].copy()
            
            # Rotate around axis
            c, s = np.cos(angle), np.sin(angle)
            d = c * d + s * np.cross(axis, d) + (1 - c) * np.dot(axis, d) * axis
            d /= (np.linalg.norm(d) + 1e-10)
            
            candidate = base + d * spacing
            # Check minimum distance
            min_dist = min(np.linalg.norm(candidate - q) for q in atoms)
            if min_dist >= 0.45 * spacing:
                atoms.append(candidate)
    
    # If we didn't get enough atoms, add more randomly
    while len(atoms) < n:
        base_idx = rng.integers(0, len(atoms))
        base = atoms[base_idx]
        offset = rng.normal(scale=0.8 * spacing, size=3)
        candidate = base + offset
        if all(np.linalg.norm(candidate - q) >= 0.45 * spacing for q in atoms):
            atoms.append(candidate)
        else:
            # Just add it (relax will fix)
            atoms.append(candidate)
    
    return np.array(atoms[:n])


def _basin_hopping(pos, max_iters=150, kick_start=0.3, kick_min=0.05, 
                   kick_decay=0.98, time_limit=60.0, start_time=None):
    """Basin hopping: local minimize -> kick -> repeat.
    
    Uses the provided relax function for local minimization.
    """
    if start_time is None:
        start_time = time.time()
    
    current_pos = pos.copy()
    current_energy = _dzugutov_energy(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    kick_size = kick_start
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Local minimization using provided relax
        try:
            energy, relaxed_pos = relax(current_pos, 'dzugutov', None)
            if energy < current_energy:
                current_energy = energy
                current_pos = relaxed_pos
        except Exception:
            pass
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
        
        # Kick: random perturbation
        n = len(current_pos)
        kick = np.random.normal(0, kick_size, size=(n, 3))
        new_pos = current_pos + kick
        
        # Ensure minimum distance constraint
        # Simple check: if any pair too close, reduce kick
        diff = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 0.5:
            # Scale down kick
            scale = 0.5 / (np.min(dist) + 1e-10) * 0.9
            new_pos = current_pos + kick * scale
        
        current_pos = new_pos
        current_energy = _dzugutov_energy(current_pos)
        
        # Decay kick size
        kick_size = max(kick_min, kick_size * kick_decay)
    
    return best_pos, best_energy


def _parallel_basin_hopping(pos, n_workers=4, time_limit=250.0):
    """Run basin hopping from multiple perturbed starting points in parallel."""
    start_time = time.time()
    results = []
    
    # Generate diverse perturbations
    rng = np.random.default_rng(12345)
    perturbations = []
    for i in range(n_workers * 3):
        kick = rng.normal(0, 0.5 + 0.3 * i / (n_workers * 3), size=(len(pos), 3))
        perturbations.append(pos + kick)
    
    # Process in batches to respect time limit
    batch_size = n_workers
    for batch_start in range(0, len(perturbations), batch_size):
        if time.time() - start_time > time_limit:
            break
        
        batch = perturbations[batch_start:batch_start + batch_size]
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for p in batch:
                remaining = time_limit - (time.time() - start_time)
                if remaining < 10:
                    break
                futures.append(executor.submit(
                    _basin_hopping, p, max_iters=80, 
                    kick_start=0.25, kick_min=0.03,
                    kick_decay=0.97, time_limit=remaining,
                    start_time=start_time
                ))
            
            for f in as_completed(futures):
                try:
                    pos_result, energy_result = f.result(timeout=5)
                    results.append((energy_result, pos_result))
                except Exception:
                    continue
    
    if not results:
        # Fallback: just relax the original
        try:
            energy, relaxed = relax(pos, 'dzugutov', None)
            return relaxed, energy
        except Exception:
            return pos, _dzugutov_energy(pos)
    
    results.sort(key=lambda x: x[0])
    return results[0][1], results[0][0]


def _dense_random_cluster(n, spacing=1.0, seed=0):
    """Generate a dense random cluster with icosahedral bias."""
    rng = np.random.default_rng(seed)
    radius = spacing * n**(1.0/3.0) * 0.7
    
    points = [np.zeros(3)]  # Center
    
    attempts = 0
    while len(points) < n and attempts < n * 100:
        attempts += 1
        # Bias toward icosahedral directions
        if rng.random() < 0.5:
            # Icosahedral direction
            phi = (1 + np.sqrt(5)) / 2
            ico_verts = np.array([
                [0, 1, phi], [0, -1, phi],
                [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0],
                [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [phi, 0, -1],
                [-phi, 0, 1], [-phi, 0, -1]
            ], dtype=float)
            v = ico_verts[rng.integers(0, 12)]
            v /= np.linalg.norm(v)
            r = rng.uniform(0.3, radius)
            p = v * r
        else:
            # Random direction
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = radius * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            r = rng.uniform(0.3, 1.0)
            p *= r
        
        # Check minimum distance
        if all(np.linalg.norm(p - q) >= 0.45 * spacing for q in points):
            points.append(p)
    
    while len(points) < n:
        points.append(rng.normal(scale=0.5 * radius, size=3))
    
    return np.array(points[:n])


def construct(instance):
    """Find low-energy structures for Dzugutov clusters using basin hopping
    with Mackay icosahedral seeding and parallel multi-start exploration."""
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    time_limit = 290.0  # Safety margin (330s total, 40s buffer)
    
    # Generate diverse starting structures
    starting_structures = []
    
    # 1. Mackay icosahedral structure (primary seed)
    try:
        mackay = _mackay_icosahedron(size, shell_spacing=1.0)
        starting_structures.append(mackay)
    except Exception:
        pass
    
    # 2. Parametric polytetrahedral structures
    for seed in [size, size + 100, size + 200]:
        try:
            pt = _parametric_polytetrahedral(size, seed=seed)
            starting_structures.append(pt)
        except Exception:
            pass
    
    # 3. Dense random with icosahedral bias
    for seed in [size + 50, size + 150]:
        try:
            dr = _dense_random_cluster(size, spacing=1.0, seed=seed)
            starting_structures.append(dr)
        except Exception:
            pass
    
    # Ensure we have at least one structure
    if not starting_structures:
        starting_structures.append(_dense_random_cluster(size, spacing=1.0, seed=0))
    
    best_energy = np.inf
    best_pos = None
    
    # First pass: quick relaxation of each starting structure
    for i, pos in enumerate(starting_structures):
        if time.time() - start_time > time_limit - 30:
            break
        
        try:
            energy, relaxed = relax(pos, potential, None)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    # If no valid result yet, use first structure
    if best_pos is None:
        best_pos = starting_structures[0].copy()
        best_energy = _dzugutov_energy(best_pos)
    
    # Second pass: basin hopping from the best structure
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 20:
        try:
            bh_pos, bh_energy = _parallel_basin_hopping(
                best_pos, n_workers=4, time_limit=remaining_time - 10
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            pass
    
    # Third pass: basin hopping from other starting structures if time remains
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 30:
        for i, pos in enumerate(starting_structures[1:4]):
            if time.time() - start_time > time_limit - 15:
                break
            
            remaining = time_limit - (time.time() - start_time) - 5
            if remaining < 15:
                break
            
            try:
                bh_pos, bh_energy = _basin_hopping(
                    pos, max_iters=60, kick_start=0.2, kick_min=0.02,
                    kick_decay=0.96, time_limit=remaining, start_time=start_time
                )
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_pos = bh_pos.copy()
            except Exception:
                continue
    
    # Final relaxation of the best result
    if time.time() - start_time < time_limit - 5:
        try:
            energy, relaxed = relax(best_pos, potential, None)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Ensure minimum distance constraint
    # Check and fix any violations
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 0.5:
        # Scale up positions to fix
        scale = 0.5 / (min_dist + 1e-10) * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
