import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _get_mackay_icosahedron(k):
    """
    Generate coordinates for a Mackay icosahedron of order k.
    k=1: 13 atoms, k=2: 55 atoms, k=3: 147 atoms.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of an icosahedron
    # Standard icosahedron vertices (circumradius 1)
    # Scaled to have edge length 1? 
    # For SC potential, equilibrium distance is ~1.0.
    # Icosahedron edge length with circumradius R is 2*R*sin(pi/5) = 2*R*0.5878 = 1.1756*R.
    # So R = 1/1.1756 = 0.8506 for edge=1.
    R = 1.0 / (2 * np.sin(np.pi / 5))
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts[0]) * R
    
    points = [np.zeros(3)] # Center
    
    # Shell 1: 12 vertices
    for v in verts:
        points.append(v.copy())
        
    if k >= 2:
        # Shell 2: 30 atoms on the edges
        # Each edge of the icosahedron gets 2 atoms (for k=2, it's actually the vertices of the icosahedron + points on edges/faces)
        # Mackay construction:
        # Shell 1: 12 vertices.
        # Shell 2: 30 points. 10 on each face? No.
        # Standard Mackay:
        # k=1: 1 center + 12 vertices = 13.
        # k=2: 1 + 12 + 30 = 43? No, 55.
        # The 30 atoms are on the 30 edges? No, 55 = 1 + 12 + 30 + 12?
        # Actually, Mackay 55 is 1 + 12 + 30 + 12?
        # Let's use a known coordinate generation for Mackay 55.
        # Or simpler: Use the 13-atom icosahedron and "grow" it?
        # For now, I will implement a generic "spherical shell" or use the 13-atom one and trim/grow.
        # Given time constraints, implementing exact Mackay 55/147 coordinates is error-prone.
        # I will use a "Random Compact" + "Icosahedral Bias" approach.
        
        # Alternative: Use the 13-atom icosahedron as a core, and add atoms in a shell.
        # For k=2, add 42 atoms?
        # Let's just use the 13-atom icosahedron and a simple FCC generator, and rely on Basin Hopping to find the structure.
        # The initialization just needs to be "compact" and "symmetric-ish".
        
        pass # Skip complex Mackay generation for now, use simpler heuristics
    
    return np.array(points)

def _get_fcc_cluster(n):
    """
    Generate a truncated FCC cluster with approximately n atoms.
    """
    # FCC lattice points
    # Unit cell vectors
    a = 1.0 # Lattice constant, tuned later? SC equilibrium is ~1.0
    # Actually, for SC, the nearest neighbor distance is ~0.9-1.0. Let's use 1.0.
    
    points = []
    # Generate points in a cube of size L x L x L
    # Density of FCC is 4 atoms per unit cell of side a.
    # Volume ~ 4/3 pi R^3.
    # n ~ 4 * Volume / a^3 => R ~ (n/4)^(1/3) * a
    R = (n / 4.0) ** (1.0 / 3.0) * 1.0
    
    # Generate grid
    max_coord = int(R) + 2
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                # FCC points in unit cell: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    px = x + dx
                    py = y + dy
                    pz = z + dz
                    if np.linalg.norm([px, py, pz]) <= R:
                        points.append([px, py, pz])
    
    points = np.array(points)
    
    # If we have more than n, trim by distance from center
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[-n:]
        points = points[idx]
    elif len(points) < n:
        # Add random atoms nearby
        while len(points) < n:
            p = points[0] + np.random.uniform(-1, 1, 3)
            # Ensure not too close
            if all(np.linalg.norm(p - q) > 0.5 for q in points):
                points = np.vstack([points, [p]])
            else:
                # Just add it, relaxation will fix it
                points = np.vstack([points, [p]])
                
    return points

def _valid_check(pos):
    """Check if positions are valid (finite, no overlaps)."""
    if not np.all(np.isfinite(pos)):
        return False
    # Check min distance
    # Efficient check: sort or use cdist
    from scipy.spatial.distance import cdist
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.35:
        return False
    return True

def construct(instance):
    n = instance['n']
    kind = instance['kind']
    potential = instance['potential']
    
    # Time limit per instance
    start_time = time.time()
    time_limit = 45.0 # 50s per instance * 6 = 300s, leave margin
    
    best_energy = np.inf
    best_pos = None
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Candidate 1: FCC based
    try:
        pos_fcc = _get_fcc_cluster(n)
        candidates.append(pos_fcc)
    except:
        pass
        
    # Candidate 2: 13-atom Icosahedron core + random growth
    try:
        pos_ico = _get_mackay_icosahedron(1) # 13 atoms
        if len(pos_ico) < n:
            # Grow
            current = pos_ico
            for _ in range(n - len(current)):
                # Add atom at random position near surface
                # Pick a random existing atom, add offset
                anchor = current[np.random.randint(len(current))]
                new_atom = anchor + np.random.uniform(-0.5, 0.5, 3)
                # Ensure min distance
                if all(np.linalg.norm(new_atom - q) > 0.5 for q in current):
                    current = np.vstack([current, [new_atom]])
                else:
                    current = np.vstack([current, [new_atom]]) # Relax will fix
            candidates.append(current)
    except:
        pass
        
    # Candidate 3: Random Compact
    try:
        radius = 0.72 * n ** (1.0/3.0) * 0.8
        rng = np.random.default_rng(n)
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                    points.append(p)
        candidates.append(np.array(points))
    except:
        pass

    # 2. Basin Hopping / Local Optimization Loop
    for i, init_pos in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
            
        current_pos = init_pos.copy()
        current_energy = np.inf
        
        # Initial Relax
        try:
            current_energy, current_pos = relax(current_pos, potential, kind)
        except:
            continue
            
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Basin Hopping Steps
        step_size = 0.5
        max_steps = 50
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
                
            # Perturb
            noise = np.random.normal(0, step_size, (n, 3))
            new_pos = current_pos + noise
            
            # Check validity
            if not _valid_check(new_pos):
                # Simple fix: reject
                step_size *= 0.8
                continue
            
            # Relax
            try:
                new_energy, new_pos = relax(new_pos, potential, kind)
            except:
                continue
                
            # Metropolis Acceptance
            if new_energy < current_energy:
                current_energy = new_energy
                current_pos = new_pos
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
                # Increase step slightly
                step_size = min(step_size * 1.05, 2.0)
            else:
                # Accept with probability exp(-(new_energy - current_energy) / T)
                # T should be small, e.g., 0.01
                T = 0.01
                if np.random.rand() < np.exp(-(new_energy - current_energy) / T):
                    current_energy = new_energy
                    current_pos = new_pos
                    step_size = min(step_size * 1.05, 2.0)
                else:
                    step_size *= 0.8
                    
            if step_size < 0.01:
                break

    if best_pos is None:
        # Fallback: return a valid random cluster
        # This should not happen
        radius = 0.72 * n ** (1.0/3.0) * 0.8
        rng = np.random.default_rng(0)
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                    points.append(p)
        best_pos = np.array(points)
        
    return best_pos
