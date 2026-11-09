"""
Dzugutov cluster global minima (polytetrahedral model metal).
Strategy: Physics-informed seeding using Mackay supericosahedra and icosahedral shells,
combined with parallel local relaxation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax

# Constants for Dzugutov potential (for reference, though relax handles them)
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _generate_mackay_55():
    """
    Generate the 55-atom Mackay supericosahedron.
    Structure: 1 center, 12 shell 1 (radius 1), 30 shell 2 (radius sqrt(2)), 12 shell 3 (radius sqrt(5)).
    Note: The radii in the Dzugutov potential are scaled such that the equilibrium distance is ~1.87.
    However, the `relax` function will handle the scaling/optimization. We need a good geometric start.
    Standard Mackay radii relative to first shell:
    R1 = 1
    R2 = sqrt(2)
    R3 = sqrt(5)
    We scale these to a reasonable starting radius, say R1 = 1.8 (close to potential minimum).
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    verts1 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts1 = verts1 / np.linalg.norm(verts1[0])
    
    # Dodecahedron vertices (20) - for the second shell (30 atoms in Mackay, but let's use the standard icosahedral symmetry points)
    # The 30 atoms of the second shell in Mackay 55 are located at the midpoints of the edges of the first icosahedron? No.
    # Mackay 55:
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 30 atoms (dodecahedron vertices? No, it's the 30 points of the icosidodecahedron? No.)
    # Actually, the 30 atoms in the second shell of the Mackay supericosahedron are at the vertices of a dodecahedron? No, a dodecahedron has 20 vertices.
    # The 30 points are the vertices of the *icosidodecahedron*? No, that has 30 vertices. Yes.
    # Wait, the standard Mackay 55 structure consists of:
    # 1 center
    # 12 atoms at icosahedron vertices (R1)
    # 30 atoms at dodecahedron vertices? No, dodecahedron has 20.
    # The 30 atoms are at the positions corresponding to the *edges* of the icosahedron? 
    # Let's use the standard coordinates for Mackay 55.
    
    # Shell 2 (30 atoms): Vertices of the Icosidodecahedron
    # Icosidodecahedron vertices are permutations of (0, ±1, ±phi)
    verts2 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [0, phi, 1], [0, phi, -1], [0, -phi, 1], [0, -phi, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 1, 0], [phi, -1, 0], [-phi, 1, 0], [-phi, -1, 0],
        [1, 0, phi], [-1, 0, phi], [1, 0, -phi], [-1, 0, -phi],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [0, 1, phi], [0, -1, phi], # Duplicate check? No, let's generate properly.
    ])
    # Better way to generate Icosidodecahedron vertices (30 points):
    # All permutations of (0, ±1, ±phi)
    verts2_list = []
    for x in [0, 1, -1]:
        for y in [0, 1, -1]:
            for z in [0, 1, -1]:
                # This is getting messy. Let's use the known fact that the 30 points are the midpoints of the edges of the cuboctahedron? No.
                # Let's use the standard icosahedral symmetry.
                pass
    
    # Let's use a simpler, robust generator for icosahedral shells.
    # For the 55-atom Mackay:
    # 1 center
    # 12 at R1 (Icosahedron)
    # 30 at R2 (Dodecahedron? No, the 30 points are the vertices of the *dodecahedron*? No, 20 vertices.
    # The 30 points are the vertices of the *icosidodecahedron*? Yes, 30 vertices.
    # The 12 points of the 3rd shell are the vertices of the *dodecahedron*? No, 12 vertices. Yes.
    
    # Icosidodecahedron vertices: permutations of (0, ±1, ±phi)
    verts2 = []
    for perm in [(0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
                 (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
                 (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1),
                 (0, phi, 1), (0, phi, -1), (0, -phi, 1), (0, -phi, -1),
                 (1, 0, phi), (-1, 0, phi), (1, 0, -phi), (-1, 0, -phi),
                 (phi, 1, 0), (phi, -1, 0), (-phi, 1, 0), (-phi, -1, 0),
                 (1, phi, 0), (-1, phi, 0), # Wait, I need 30 unique points.
                 ]:
        verts2.append(perm)
    # The set of 30 vertices of the icosidodecahedron is:
    # All even permutations of (0, ±1, ±phi) and (±1, ±phi, 0) and (±phi, 0, ±1)?
    # Actually, the icosidodecahedron has 30 vertices.
    # Let's just use the 12 icosahedron vertices, 30 "edge midpoint" like points, and 12 "face center" like points?
    # No, Mackay 55 is:
    # 1 (0,0,0)
    # 12 (Icosahedron vertices)
    # 30 (Dodecahedron vertices? No, Dodecahedron has 20. The 30 points are the *vertices* of the *icosidodecahedron*)
    # 12 (Icosahedron vertices of a larger shell? No, the 12 points of the 3rd shell are the *vertices* of the *dodecahedron*)
    
    # Let's use a reliable source of coordinates for Mackay 55.
    # R1 = 1, R2 = sqrt(2), R3 = sqrt(5)
    # Shell 1: 12 Icosahedron vertices
    # Shell 2: 30 Icosidodecahedron vertices
    # Shell 3: 12 Dodecahedron vertices
    
    # Icosahedron (12)
    v_ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    v_ico = v_ico / np.linalg.norm(v_ico[0])
    
    # Icosidodecahedron (30)
    # Vertices are permutations of (0, ±1, ±phi)
    v_icod = []
    for x in [0, 1, -1]:
        for y in [0, 1, -1]:
            for z in [0, 1, -1]:
                # This generates 27 points, not 30.
                pass
    # Correct generation of icosidodecahedron vertices:
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    v_icod_list = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            v_icod_list.append([0, s1, s2*phi])
            v_icod_list.append([s1, s2*phi, 0])
            v_icod_list.append([s1*phi, 0, s2])
            v_icod_list.append([0, s1*phi, s2]) # Wait, is this correct?
            # Let's check symmetry.
            # The icosidodecahedron has 30 vertices.
            # The set is:
            # (0, ±1, ±phi) -> 4 points
            # (±1, ±phi, 0) -> 4 points
            # (±phi, 0, ±1) -> 4 points
            # (0, ±phi, ±1) -> 4 points
            # (±phi, ±1, 0) -> 4 points
            # (±1, 0, ±phi) -> 4 points
            # Total 24? No.
            # Let's use the dual of the rhombic triacontahedron?
            
    # Alternative: Use the 12 icosahedron vertices, then add 30 points which are the midpoints of the edges of the icosahedron?
    # The midpoints of the 30 edges of an icosahedron form an icosidodecahedron? No, they form a dodecahedron? No.
    # The midpoints of the edges of an icosahedron are the vertices of a *dodecahedron*? No, a dodecahedron has 20 vertices.
    # The midpoints of the 30 edges of an icosahedron are the vertices of an *icosidodecahedron*? Yes!
    
    # So, for Shell 2, we take the midpoints of the edges of the icosahedron.
    edges = []
    # Find edges of icosahedron (distance between vertices)
    min_dist = np.min([np.linalg.norm(v_ico[i] - v_ico[j]) for i in range(12) for j in range(i+1, 12)])
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(v_ico[i] - v_ico[j]) < min_dist * 1.1:
                mid = (v_ico[i] + v_ico[j]) / 2.0
                edges.append(mid)
    v_icod = np.array(edges)
    # Normalize
    v_icod = v_icod / np.linalg.norm(v_icod[0])
    
    # Shell 3: 12 Dodecahedron vertices
    # Dodecahedron vertices are permutations of (±1, ±1, ±1) and (0, ±1/phi, ±phi)
    v_dod = []
    for x in [1, -1]:
        for y in [1, -1]:
            for z in [1, -1]:
                v_dod.append([x, y, z])
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            v_dod.append([0, s1/phi, s2*phi])
            v_dod.append([s1/phi, s2*phi, 0])
            v_dod.append([s1*phi, 0, s2/phi])
    v_dod = np.array(v_dod)
    v_dod = v_dod / np.linalg.norm(v_dod[0])
    
    # Combine
    # Scale to a reasonable starting radius. The potential minimum is around 1.87.
    # In Mackay 55, the nearest neighbor distance is R1.
    # So we set R1 = 1.87.
    scale = 1.87
    
    points = [np.zeros(3)] # Center
    points.extend((v_ico * scale).tolist()) # 12
    points.extend((v_icod * scale * np.sqrt(2)).tolist()) # 30
    points.extend((v_dod * scale * np.sqrt(5)).tolist()) # 12
    
    return np.array(points)

def _generate_icosahedron_shells(n):
    """
    Generate a cluster using icosahedral shells.
    1 center + 12 (R1) + 30 (R2) + 12 (R3) + ...
    For n <= 13: Center + 12
    For 14 <= n <= 43: Center + 12 + partial 30
    For 44 <= n <= 55: Center + 12 + 30 + partial 12
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    v_ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    v_ico = v_ico / np.linalg.norm(v_ico[0])
    
    # Icosidodecahedron vertices (midpoints of icosahedron edges)
    min_dist = np.min([np.linalg.norm(v_ico[i] - v_ico[j]) for i in range(12) for j in range(i+1, 12)])
    v_icod = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(v_ico[i] - v_ico[j]) < min_dist * 1.1:
                mid = (v_ico[i] + v_ico[j]) / 2.0
                v_icod.append(mid)
    v_icod = np.array(v_icod)
    v_icod = v_icod / np.linalg.norm(v_icod[0])
    
    # Dodecahedron vertices
    v_dod = []
    for x in [1, -1]:
        for y in [1, -1]:
            for z in [1, -1]:
                v_dod.append([x, y, z])
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            v_dod.append([0, s1/phi, s2*phi])
            v_dod.append([s1/phi, s2*phi, 0])
            v_dod.append([s1*phi, 0, s2/phi])
    v_dod = np.array(v_dod)
    v_dod = v_dod / np.linalg.norm(v_dod[0])
    
    scale = 1.87
    points = [np.zeros(3)]
    
    # Add shells
    if n <= 1:
        return np.array(points[:n])
    
    # Shell 1
    num_ico = min(12, n - 1)
    points.extend((v_ico[:num_ico] * scale).tolist())
    if len(points) >= n:
        return np.array(points[:n])
        
    # Shell 2
    remaining = n - len(points)
    num_icod = min(30, remaining)
    points.extend((v_icod[:num_icod] * scale * np.sqrt(2)).tolist())
    if len(points) >= n:
        return np.array(points[:n])
        
    # Shell 3
    remaining = n - len(points)
    num_dod = min(12, remaining)
    points.extend((v_dod[:num_dod] * scale * np.sqrt(5)).tolist())
    
    return np.array(points[:n])

def _generate_mackay_variant(n, seed=42):
    """
    Generate a structure by modifying the 55-atom Mackay structure.
    If n < 55, remove atoms from the outermost shell.
    If n > 55, add atoms to the outermost shell.
    """
    p55 = _generate_mackay_55()
    rng = np.random.default_rng(seed)
    
    if n == 55:
        return p55
        
    if n < 55:
        # Remove atoms from the outermost shell (last 12)
        # Keep the first 55 - (55 - n) = n atoms
        # This might break symmetry, but relax will fix it.
        # To be safer, we can randomly select atoms to remove from the outer shell.
        outer_indices = list(range(43, 55)) # Indices of the 12 outer atoms
        to_remove = rng.choice(outer_indices, size=55-n, replace=False)
        keep_indices = [i for i in range(55) if i not in to_remove]
        return p55[keep_indices]
        
    else:
        # Add atoms
        # Add to the outer shell or a new shell
        # For simplicity, add random points in a shell around the cluster
        current_points = p55
        while len(current_points) < n:
            # Add a point in a shell of radius ~ 2.5 * scale
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            r = 1.87 * np.sqrt(5) + rng.uniform(0.5, 1.0)
            p = v * r
            current_points = np.vstack([current_points, p])
        return current_points[:n]

def construct(instance):
    """
    Parallel relaxation of diverse structural templates.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 320.0 
    
    candidates = []
    
    # 1. Physics-informed templates
    try:
        if size == 55:
            candidates.append(_generate_mackay_55())
        else:
            # Generate Mackay variant
            candidates.append(_generate_mackay_variant(size, seed=size))
            # Generate Icosahedral shells
            candidates.append(_generate_icosahedron_shells(size))
    except Exception as e:
        pass
        
    # 2. Random perturbations of the best template
    if candidates:
        base = candidates[0]
        rng = np.random.default_rng(size + 100)
        for i in range(3):
            noise = rng.normal(0, 0.1, size=base.shape)
            candidates.append(base + noise)
            
    # 3. Random packing (diversity)
    try:
        rng = np.random.default_rng(size + 200)
        radius = 1.5 * size ** (1.0 / 3.0)
        points = []
        min_dist = 0.8
        attempts = 0
        max_attempts = size * 50
        while len(points) < size and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
        if len(points) == size:
            candidates.append(np.array(points))
    except Exception:
        pass

    best_energy = np.inf
    best_pos = None
    
    def _relax_task(pos):
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            return energy, relaxed_pos
        except Exception:
            return np.inf, pos

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_task, pos): i for i, pos in enumerate(candidates)}
        
        for future in as_completed(futures):
            if time.time() > deadline - 10:
                break
            try:
                energy, pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                continue

    # 4. Refinement (Basin Hopping)
    if best_pos is not None:
        rng = np.random.default_rng(size + 300)
        current_energy = best_energy
        current_pos = best_pos.copy()
        
        max_refine_rounds = 10
        perturbations_per_round = 4
        
        for round_num in range(max_refine_rounds):
            if time.time() > deadline - 5:
                break
                
            sigma = 0.2 * np.exp(-round_num * 0.1)
            perturbations = []
            for _ in range(perturbations_per_round):
                noise = rng.normal(0, sigma, size=current_pos.shape)
                perturbations.append(current_pos + noise)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_relax_task, p): i for i, p in enumerate(perturbations)}
                for future in as_completed(futures):
                    if time.time() > deadline - 2:
                        break
                    try:
                        energy, pos = future.result()
                        if energy < current_energy:
                            current_energy = energy
                            current_pos = pos
                            best_energy = energy
                            best_pos = pos
                    except Exception:
                        continue
                        
    if best_pos is None:
        if candidates:
            return candidates[0]
        else:
            return np.array([np.array([i, 0, 0]) for i in range(size)])

    return best_pos
# EVOLVE-BLOCK-END
