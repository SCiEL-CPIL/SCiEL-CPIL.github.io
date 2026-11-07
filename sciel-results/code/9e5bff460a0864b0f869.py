"""Sutton-Chen transition-metal cluster global minima - growth-based construction with basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n, spacing=1.0):
    """Generate positions for a truncated FCC cluster with n atoms."""
    # FCC lattice: atoms at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cells
    # Nearest neighbor distance = spacing
    # For FCC, a = spacing * sqrt(2)
    a = spacing * np.sqrt(2)
    
    # Generate candidate positions in a sphere
    max_r = spacing * (n / 4.0) ** (1.0 / 3.0) * 2.0
    n_side = int(2 * max_r / a) + 3
    
    candidates = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                for offset in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + offset[0]) * a
                    y = (j + offset[1]) * a
                    z = (k + offset[2]) * a
                    r = np.sqrt(x*x + y*y + z*z)
                    if r <= max_r:
                        candidates.append((r, np.array([x, y, z])))
    
    candidates.sort(key=lambda t: t[0])
    positions = [c[1] for c in candidates[:n]]
    return np.array(positions) if positions else np.zeros((n, 3))


def _icosahedral_positions(n, spacing=1.0):
    """Generate positions for an icosahedral cluster with n atoms."""
    # Icosahedral shells: 1, 12, 42, 120, ...
    # For small n, use icosahedral geometry
    phi = (1 + np.sqrt(5)) / 2
    
    # Core atom
    positions = [np.array([0.0, 0.0, 0.0])]
    
    # First shell: 12 atoms (icosahedron vertices)
    if n > 1:
        r1 = spacing
        verts = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        for v in verts[:min(12, n - 1)]:
            v = np.array(v, dtype=float)
            v = v / np.linalg.norm(v) * r1
            positions.append(v)
    
    # Second shell: 42 atoms (octahedron + additional)
    if n > 13:
        r2 = spacing * np.sqrt(2)
        # Octahedron vertices
        octa = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ]
        # Edge midpoints of icosahedron
        icos_verts = [
            np.array([0, 1, phi]), np.array([0, -1, phi]), np.array([0, 1, -phi]),
            np.array([0, -1, -phi]), np.array([1, phi, 0]), np.array([-1, phi, 0]),
            np.array([1, -phi, 0]), np.array([-1, -phi, 0]), np.array([phi, 0, 1]),
            np.array([phi, 0, -1]), np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
        ]
        icos_verts = [v / np.linalg.norm(v) for v in icos_verts]
        
        # Find edge midpoints
        edge_mids = []
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(icos_verts[i] - icos_verts[j])
                if dist < 1.5:  # adjacent vertices
                    mid = (icos_verts[i] + icos_verts[j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_mids.append(mid)
        
        # Add octahedron vertices and edge midpoints
        shell2 = octa + edge_mids
        for v in shell2[:min(42, n - len(positions))]:
            v = np.array(v, dtype=float)
            v = v / np.linalg.norm(v) * r2
            positions.append(v)
    
    # Fill remaining with random positions near surface
    while len(positions) < n:
        # Add atoms at random positions on a sphere of radius ~ (n)^(1/3) * spacing
        r = spacing * (len(positions) / 4.0) ** (1.0/3.0) * 1.5
        theta = np.random.uniform(0, 2 * np.pi)
        phi_angle = np.random.uniform(0, np.pi)
        p = r * np.array([
            np.sin(phi_angle) * np.cos(theta),
            np.sin(phi_angle) * np.sin(theta),
            np.cos(phi_angle)
        ])
        positions.append(p)
    
    return np.array(positions[:n])


def _random_compact(n, spacing=0.72, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points], default=np.inf)
            if min_dist >= 0.7 * spacing:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, fill with random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _growth_construct(n, kind, rng, deadline):
    """Grow a cluster by adding atoms to relaxed cores."""
    start_time = time.time()
    
    # Start with a small relaxed cluster and grow
    # Use FCC-like initial guess for the core
    core_size = max(3, min(10, n // 4))
    
    # Generate initial core using FCC positions
    core = _fcc_positions(core_size, spacing=1.0)
    
    # Relax the core
    energy, core = relax(core, 'sutton_chen', kind)
    
    # Grow by adding atoms
    current = core.copy()
    current_energy = energy
    
    while len(current) < n:
        if time.time() > deadline - 10:
            break
        
        # Try adding atoms at various positions around the cluster
        best_add = None
        best_add_energy = np.inf
        
        # Generate candidate positions for new atom
        centroid = np.mean(current, axis=0)
        max_dist = np.max(np.linalg.norm(current - centroid, axis=1))
        
        n_candidates = 50
        for _ in range(n_candidates):
            # Try positions near existing atoms (interstitial sites)
            idx = rng.integers(0, len(current))
            neighbor = current[idx]
            # Random direction
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            
            # Try different distances
            for dist in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
                candidate = neighbor + direction * dist
                test_cluster = np.vstack([current, candidate])
                
                # Check minimum distance
                if np.min(np.linalg.norm(test_cluster[:, np.newaxis] - test_cluster[np.newaxis, :], axis=2).diagonal(offset=1)) < 0.35:
                    continue
                
                # Quick energy estimate
                e, _ = sutton_chen_energy_gradient(test_cluster, kind)
                if e < best_add_energy:
                    best_add_energy = e
                    best_add = candidate
        
        if best_add is None:
            # Fallback: add at random position
            best_add = centroid + rng.normal(scale=max_dist * 0.5, size=3)
        
        current = np.vstack([current, best_add])
        
        # Relax periodically
        if len(current) % 5 == 0 or len(current) == n:
            if time.time() > deadline - 15:
                break
            energy, current = relax(current, 'sutton_chen', kind)
            current_energy = energy
    
    return current, current_energy


def construct(instance):
    """Construct a low-energy Sutton-Chen cluster using growth and basin hopping."""
    size = instance["n"]
    kind = instance["kind"]
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    rng = np.random.default_rng(size * 1000 + hash(kind) % 10000)
    
    best_positions = None
    best_energy = np.inf
    
    # Strategy 1: Growth algorithm
    if time.time() < deadline - 60:
        try:
            grown, grown_energy = _growth_construct(size, kind, rng, deadline)
            if grown_energy < best_energy:
                best_energy = grown_energy
                best_positions = grown.copy()
        except Exception:
            pass
    
    # Strategy 2: Multiple random starts with relaxation
    n_attempts = 6
    for i in range(n_attempts):
        if time.time() > deadline - 30:
            break
        
        # Different initial geometries
        if i == 0:
            # FCC-based
            init = _fcc_positions(size, spacing=1.0)
        elif i == 1:
            # Icosahedral-based
            init = _icosahedral_positions(size, spacing=1.0)
        elif i == 2:
            # Random compact
            init = _random_compact(size, spacing=0.72, rng=rng)
        elif i == 3:
            # Spherical shell
            r = 1.0 * size ** (1.0/3.0) * 0.8
            theta = rng.uniform(0, 2*np.pi, size)
            phi_angle = rng.arccos(2*rng.uniform(size)-1)
            init = r * np.column_stack([
                np.sin(phi_angle)*np.cos(theta),
                np.sin(phi_angle)*np.sin(theta),
                np.cos(phi_angle)
            ])
        else:
            # More random
            init = _random_compact(size, spacing=0.65, rng=rng)
        
        try:
            energy, positions = relax(init, 'sutton_chen', kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    # Strategy 3: Basin hopping on the best solution found
    if best_positions is not None and time.time() < deadline - 20:
        try:
            current_pos = best_positions.copy()
            current_energy = best_energy
            
            # Simple basin hopping
            n_bh_steps = 20
            step_size = 0.5
            
            for step in range(n_bh_steps):
                if time.time() > deadline - 10:
                    break
                
                # Perturb
                noise = rng.normal(scale=step_size, size=current_pos.shape)
                perturbed = current_pos + noise
                
                # Relax
                e, relaxed = relax(perturbed, 'sutton_chen', kind)
                
                if e < current_energy:
                    current_energy = e
                    current_pos = relaxed.copy()
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_positions = current_pos.copy()
                else:
                    # Metropolis acceptance (simplified)
                    if rng.random() < np.exp(-(e - current_energy) / 0.1):
                        current_pos = relaxed.copy()
                        current_energy = e
            
            # Final relaxation
            if time.time() < deadline - 5:
                final_e, final_pos = relax(best_positions, 'sutton_chen', kind)
                if final_e < best_energy:
                    best_energy = final_e
                    best_positions = final_pos.copy()
        except Exception:
            pass
    
    # Final validation and return
    if best_positions is None:
        # Fallback: simple random
        best_positions = _random_compact(size, spacing=0.72, rng=rng)
        _, best_positions = relax(best_positions, 'sutton_chen', kind)
    
    # Ensure no atoms are too close
    min_dists = []
    for i in range(len(best_positions)):
        for j in range(i+1, len(best_positions)):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            min_dists.append(d)
    
    if min(min_dists) < 0.35:
        # Nudge apart
        for i in range(len(best_positions)):
            for j in range(i+1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = (best_positions[i] - best_positions[j]) / (d + 1e-10)
                    shift = (0.35 - d) / 2 + 0.01
                    best_positions[i] += direction * shift
                    best_positions[j] -= direction * shift
    
    return best_positions
# EVOLVE-BLOCK-END
