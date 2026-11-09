"""
Optimized Morse Cluster Minimizer using Structured Initializers and Basin Hopping.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Global time management
TOTAL_TIME_BUDGET = 295.0 # Leave 5s margin for overhead
INSTANCE_TIME_BUDGET = TOTAL_TIME_BUDGET / 6.0
start_time_global = time.time()

def generate_fcc_shells(n, spacing=1.0):
    """Generate n atoms in FCC shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Generate a large FCC lattice
    # FCC basis vectors
    # Simple cubic grid with FCC centers
    max_coord = int(np.ceil(n ** (1/3.0)) * 2) + 5
    coords = []
    
    # Generate points in a cubic box
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC positions: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of lattice constant a
                # To get spacing 1.0, a = sqrt(2)
                a = np.sqrt(2.0)
                base = np.array([i, j, k]) * a
                
                # 4 FCC points
                p1 = base
                p2 = base + a * np.array([0, 0.5, 0.5])
                p3 = base + a * np.array([0.5, 0, 0.5])
                p4 = base + a * np.array([0.5, 0.5, 0])
                
                coords.extend([p1, p2, p3, p4])
                
    coords = np.array(coords)
    
    # Center of mass
    com = coords.mean(axis=0)
    coords = coords - com
    
    # Sort by distance to center
    dists = np.linalg.norm(coords, axis=1)
    indices = np.argsort(dists)
    
    # Take n closest
    selected = coords[indices[:n]]
    
    # Ensure minimum distance? FCC spacing is 1.0. 
    # If n is small, this is fine. If n is large, we take a chunk.
    return selected

def generate_hcp_shells(n, spacing=1.0):
    """Generate n atoms in HCP shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
        
    # HCP lattice
    a = 1.0 # lattice constant in xy plane
    c = np.sqrt(8.0/3.0) * a # ideal c/a ratio
    
    max_coord = int(np.ceil(n ** (1/3.0)) * 2) + 5
    coords = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # HCP layers
                # Layer A: (i*a, j*a, k*c)
                # Layer B: (i*a + a/2, j*a + a/(2*sqrt(3)), (k+0.5)*c) -- simplified projection
                # Let's use a simpler hexagonal grid
                # x = a * (i + 0.5 * (j % 2))
                # y = a * sqrt(3)/2 * j
                # z = c * k (with shift for odd layers)
                
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3)/2 * j
                z = c * k
                
                # Add the "B" layer atoms (shifted)
                # Actually, HCP is ABAB...
                # Atom 1 in layer k: (x, y, z)
                # Atom 2 in layer k (if k is odd? no, usually defined by basis)
                # Standard HCP basis: (0,0,0) and (1/3, 2/3, 1/2) in hex coords.
                # Let's just generate a hexagonal grid and add the second basis atom.
                
                p1 = np.array([x, y, z])
                
                # Second atom in the unit cell
                # Shift by (a/2, a/(2*sqrt(3)), c/2) ? 
                # In hex coordinates: (1/3, 2/3, 1/2)
                # Cart: 
                # x2 = a * (i + 0.5 * (j % 2) + 1/3) ? No, simpler to just shift the whole layer.
                # Let's use the standard: 
                # Layer A: (0,0,0)
                # Layer B: (a/2, a/(2*sqrt(3)), c/2) relative to A layer origin?
                # Actually, for HCP, the second atom is directly above the center of the triangle formed by 3 atoms in the layer below.
                # For simplicity, let's just use a distorted FCC or a simple hexagonal close pack approximation.
                # Given time constraints, FCC is often sufficient for global minima of Morse clusters (which are often icosahedral or distorted FCC).
                # Let's stick to FCC and maybe a deformed Icosahedron.
                
                coords.append(p1)
                # Add second basis atom shifted by (a/2, a/(2*sqrt(3)), c/2)
                p2 = p1 + np.array([a/2, a/(2*np.sqrt(3)), c/2])
                coords.append(p2)
                
    coords = np.array(coords)
    com = coords.mean(axis=0)
    coords = coords - com
    dists = np.linalg.norm(coords, axis=1)
    indices = np.argsort(dists)
    selected = coords[indices[:n]]
    return selected

def generate_icosahedral(n):
    """Generate an icosahedral structure."""
    # 1 center + 12 vertices + 30 edge midpoints + 20 face centers + ...
    # This is complex to generate exactly for arbitrary n.
    # Instead, we will use a "spherical shell" packing which approximates icosahedral symmetry for small N.
    # For N > 20, FCC is usually better.
    # We will skip explicit icosahedral generation and rely on FCC/HCP + Basin Hopping.
    pass

def construct(instance):
    global start_time_global
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Time check
    elapsed = time.time() - start_time_global
    if elapsed > TOTAL_TIME_BUDGET - 5.0:
        # Emergency: return something valid quickly
        # Just return a simple line of atoms (valid but high energy)
        positions = np.array([np.array([i, 0, 0]) for i in range(size)], dtype=float)
        # Relax once to ensure validity
        try:
            _, positions = relax(positions, potential, rho)
        except:
            pass
        return positions

    # Determine time budget for this instance
    remaining_total = TOTAL_TIME_BUDGET - elapsed
    # Heuristic: assume we have 1/6 of remaining time left if we are at instance k?
    # Better: just use a fixed fraction of remaining time, e.g., 30% of remaining total, 
    # but capped by INSTANCE_TIME_BUDGET.
    # Actually, if we just use INSTANCE_TIME_BUDGET for each, we might run out.
    # Let's track remaining instances.
    # Since we don't know the order, let's just use a dynamic budget:
    # time_per_instance = remaining_total / max(1, 6 - current_instance_index)
    # But we don't have current_instance_index.
    # Let's assume equal distribution of remaining time.
    # If this is the first call, use 1/6 of total.
    # If it's later, we can't know.
    # Safe bet: Use 45 seconds per instance. If we run out, we return best found.
    inst_budget = min(INSTANCE_TIME_BUDGET, remaining_total * 0.4)
    
    start_time_inst = time.time()
    
    rng = np.random.default_rng(size)
    
    # 1. Generate Initial Structures
    candidates = []
    
    # FCC
    try:
        fcc_pos = generate_fcc_shells(size)
        candidates.append(fcc_pos)
    except Exception as e:
        pass
        
    # HCP
    try:
        hcp_pos = generate_hcp_shells(size)
        candidates.append(hcp_pos)
    except Exception as e:
        pass
        
    # Random Dense Pack (Fallback)
    radius = size ** (1.0 / 3.0) * 0.8
    points = []
    for _ in range(20 * size): # Try many times
        if len(points) >= size:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance
            if all(np.linalg.norm(p - q) >= 0.9 for q in points):
                points.append(p)
    if len(points) == size:
        candidates.append(np.array(points))
        
    # If no candidates (should not happen), use a line
    if not candidates:
        candidates.append(np.array([np.array([i, 0, 0]) for i in range(size)], dtype=float))

    # 2. Relax all candidates
    best_energy = np.inf
    best_pos = None
    current_pos = None
    current_energy = np.inf
    
    for pos in candidates:
        try:
            e, relaxed_pos = relax(pos, potential, rho)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed_pos
            if e < current_energy:
                current_energy = e
                current_pos = relaxed_pos
        except:
            pass
            
    if best_pos is None:
        return candidates[0]

    # 3. Basin Hopping / Global Kick
    # We will perform perturbations from the current best local minimum.
    # If a perturbation leads to a lower energy, we accept it as the new current.
    # We always track the global best.
    
    # Determine number of steps based on time
    # Relaxation takes ~0.1s for N=80? Let's say 0.05s.
    # Budget 45s -> ~900 steps.
    # Let's run until time is up.
    
    noise_scale = 0.5 # Initial noise
    no_improve_counter = 0
    max_no_improve = 20
    
    while time.time() - start_time_inst < inst_budget:
        # Check global time limit
        if time.time() - start_time_global > TOTAL_TIME_BUDGET - 2.0:
            break
            
        # Perturb current position
        noise = rng.normal(0, noise_scale, size=(size, 3))
        perturbed_pos = current_pos + noise
        
        # Relax
        try:
            e_new, pos_new = relax(perturbed_pos, potential, rho)
        except:
            # If relaxation fails, try smaller noise
            noise_scale *= 0.9
            continue
            
        if e_new < current_energy:
            # Accept as new current state
            current_energy = e_new
            current_pos = pos_new
            no_improve_counter = 0
            
            # Update global best
            if e_new < best_energy:
                best_energy = e_new
                best_pos = pos_new
                # Increase noise slightly to explore more? Or keep same?
                # Keeping same is safer.
        else:
            no_improve_counter += 1
            if no_improve_counter > max_no_improve:
                # Reset to best found so far to escape local basin
                current_pos = best_pos.copy()
                current_energy = best_energy
                no_improve_counter = 0
                # Increase noise to jump further
                noise_scale = min(noise_scale * 1.1, 1.0)
            else:
                # Maybe decrease noise if stuck?
                # noise_scale = max(noise_scale * 0.99, 0.1)
                pass

    return best_pos

# EVOLVE-BLOCK-END
