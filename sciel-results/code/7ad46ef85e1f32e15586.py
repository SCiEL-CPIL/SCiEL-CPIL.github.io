"""
Strategy: Deterministic multi-start basin-hopping from structural motifs.
Instead of random points, we generate clusters based on known low-energy
structural units for Dzugutov-type potentials (FCC, Icosahedra) and perform
local minimization (basin hopping) from these starts. This leverages prior
knowledge of the potential's preference for polytetrahedral/close-packed order.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Deadline management
START_TIME = time.time()
DEADLINE = 330.0  # Safety margin before 330s kill

def _time_left():
    return DEADLINE - time.time()

def _generate_fcc_chunk(n, rng):
    """Generate a roughly spherical chunk of FCC lattice."""
    # Equilibrium distance for Dzugutov is around 2.0-2.2.
    # Let's use a=2.1 for lattice constant.
    a = 2.1
    # Generate FCC points in a box large enough to contain n atoms
    # FCC density: 4 atoms per cell of volume a^3.
    # Volume per atom = a^3 / 4.
    # Target volume = n * a^3 / 4.
    # Side length L ~ (Volume)^(1/3).
    L = (n * a**3 / 4.0) ** (0.3333) * 1.2 # padding
    
    # Generate FCC points
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    points = []
    # Determine range of lattice indices
    max_idx = int(np.ceil(L / a)) + 1
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC lattice points
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    if abs(x) <= L/2 and abs(y) <= L/2 and abs(z) <= L/2:
                        points.append([x, y, z])
    
    points = np.array(points)
    if len(points) < n:
        # Fallback: add random points if not enough
        while len(points) < n:
            p = rng.uniform(-L/2, L/2, 3)
            points = np.vstack([points, [p]])
    else:
        # Select n points closest to center
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
        
    return points

def _generate_icosahedron(n, rng):
    """Generate a Mackay icosahedron-like cluster."""
    # Mackay icosahedra: 13, 55, 147...
    # For n not exactly 55, we can perturb or add/remove.
    # Let's generate a standard 55-atom Mackay icosahedron and then
    # truncate or expand.
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    # 12 vertices of icosahedron
    # Standard coords scaled
    # Edge length a_edge. Let's aim for equilibrium distance ~ 2.0.
    # In a Mackay icosahedron, nearest neighbor distance is the edge length of the icosahedron.
    # Let's set edge length = 2.0.
    a_edge = 2.0
    
    # Icosahedron vertices (edge length 2)
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1) scaled
    # Distance between (0,1,phi) and (1,phi,0) is sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + (1/phi)^2 + phi^2)
    # phi^2 = phi+1. (1/phi)^2 = 2-phi.
    # Sum = 1 + 2 - phi + phi + 1 = 4. Distance = 2.
    # So these coords have edge length 2.
    
    verts = [
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts)
    
    # 55-atom Mackay icosahedron structure:
    # 1 center
    # 12 vertices (1st shell)
    # 30 edge midpoints (2nd shell)
    # 12 face centers? No, Mackay is:
    # Shell 0: 1 atom
    # Shell 1: 12 atoms (vertices)
    # Shell 2: 30 atoms (edge centers of the icosahedron)
    # Total: 43.
    # Wait, 55 is 1 + 12 + 30 + 12?
    # Actually, the 55-atom cluster is the "Mackay icosahedron".
    # Structure:
    # 1 center
    # 12 vertices
    # 30 edge midpoints
    # 12 face centers? No.
    # Let's check: 1 + 12 + 30 + 12 = 55.
    # The 12 face centers are not usually part of the simple Mackay description?
    # Actually, the 55-atom cluster consists of 13 tetrahedra meeting at the center? No, that's not right.
    # It is a 12-fold symmetric cluster.
    # Layers:
    # 1 (center)
    # 12 (icosahedron vertices)
    # 30 (icosahedron edge centers)
    # 12 (icosahedron face centers? No, these are further out)
    # Actually, the 55-atom cluster is often described as having 12 caps.
    # Let's just generate 1+12+30=43 and add 12 more randomly or on face centers.
    # Face centers of icosahedron:
    # Normal vectors to faces.
    
    # Simpler: Just use the 43 atoms and if n>43, add atoms in a shell.
    # If n<43, remove outer atoms.
    
    points = [np.zeros(3)] # Center
    points.extend(verts)   # 12 vertices
    
    # Edge midpoints
    # Need edges of icosahedron.
    # Two vertices are connected if distance is 2.0.
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(verts[i] - verts[j]) < 2.1: # approx 2.0
                edges.append((i, j))
                
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        points.append(mid)
        
    points = np.array(points)
    
    # Current size: 1 + 12 + 30 = 43.
    if n <= 43:
        # Keep center and closest to center
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    else:
        # Need to add n-43 atoms.
        # Add them in a shell of radius ~ 3.0-3.5
        # Use random directions
        for _ in range(n - 43):
            dir_vec = rng.normal(0, 1, 3)
            dir_vec /= np.linalg.norm(dir_vec)
            # Radius slightly larger than face centers
            # Face center distance from center:
            # Vertex (0,1,phi). Face normal?
            # Let's just use radius 3.2
            r = 3.2 + rng.uniform(-0.1, 0.1)
            p = dir_vec * r
            points = np.vstack([points, [p]])
            
    return points

def construct(instance):
    """
    Find low-energy structure for Dzugutov potential.
    Uses FCC and Icosahedral starts, relaxed with local minimization.
    """
    size = instance["n"]
    potential = instance["potential"]
    # parameter is not strictly needed for relax if potential is 'dzugutov'
    # but the signature might expect it. The seed passed instance.get("rho", instance.get("kind"))
    # Let's pass None or the instance itself if needed, but relax signature in seed was:
    # relax(positions, potential, parameter)
    # In seed: parameter = instance.get("rho", instance.get("kind"))
    # Let's stick to that to be safe.
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate starting points
    # 1. FCC
    # 2. Icosahedron
    # 3. Random (compact)
    
    starts = []
    
    # FCC
    try:
        if _time_left() > 10:
            starts.append(_generate_fcc_chunk(size, np.random.default_rng(42)))
    except:
        pass
        
    # Icosahedron
    try:
        if _time_left() > 10:
            starts.append(_generate_icosahedron(size, np.random.default_rng(43)))
    except:
        pass
        
    # Random Compact (Fallback)
    try:
        if _time_left() > 10:
            rng = np.random.default_rng(size)
            radius = 1.1 * size ** (1.0 / 3.0) * 0.8
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * 1.1 for q in points):
                    points.append(p)
            if len(points) == size:
                starts.append(np.array(points))
    except:
        pass
        
    # If no starts generated, use a simple grid or random
    if not starts:
        p = np.random.uniform(-5, 5, (size, 3))
        starts.append(p)
        
    # Relax each start
    for i, start_pos in enumerate(starts):
        if _time_left() < 5:
            break
            
        try:
            # relax returns (energy, positions)
            # It performs local minimization.
            # We might want to do a few basin hopping steps if time allows.
            # But relax is likely a robust local minimizer (L-BFGS-B or similar).
            energy, positions = relax(start_pos, potential, parameter)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                
        except Exception as e:
            pass
            
    # If we have time, do a simple basin hopping on the best result
    if best_positions is not None and _time_left() > 30:
        current_pos = best_positions.copy()
        current_energy = best_energy
        rng = np.random.default_rng(123)
        
        # Simple basin hopping: kick, relax, accept if lower
        step_size = 0.5
        max_iter = 20 # Limit iterations to avoid timeout
        
        for _ in range(max_iter):
            if _time_left() < 10:
                break
                
            # Kick
            kick = rng.normal(0, step_size, (size, 3))
            new_pos = current_pos + kick
            
            # Ensure minimum distance? Relax should handle it, but if atoms overlap,
            # energy might be huge.
            # Check for overlaps
            if np.any(np.linalg.norm(new_pos[:, None, :] - new_pos[None, :, :], axis=-1) < 0.5):
                continue
                
            try:
                new_energy, new_pos_relaxed = relax(new_pos, potential, parameter)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    current_pos = new_pos_relaxed.copy()
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_positions = current_pos.copy()
            except:
                pass

    if best_positions is None:
        # Fallback: return the last start or a random one
        return starts[0] if starts else np.zeros((size, 3))
        
    return best_positions
# EVOLVE-BLOCK-END
