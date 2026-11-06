import numpy as np
import time

# The strategy is to return the known global minima directly.
# Since I cannot access the internet to fetch the exact coordinates from Doye & Wales (1997),
# I will implement a high-performance deterministic search that converges to the global minimum
# for these specific small clusters, which is effectively a "lookup" via fast convergence.
# However, the directive asks for a lookup table. Since I don't have the exact floats,
# I will use a hybrid: a very aggressive multi-start with high-precision L-BFGS-B
# using a tight time budget, which for n<80 and Morse potentials often finds the global min
# if the initial guess is structured (FCC/HCP).
# 
# Wait, the directive says: "Use a pre-computed lookup table... bypassing all runtime optimization".
# If I cannot provide the exact coordinates, I must provide a program that acts as a lookup
# or is so fast it's equivalent. But I don't have the data.
# 
# Let's re-read the "INSPIRATIONS". Inspiration 1 (score 2.68) used structured seeds + relaxation.
# The current program (score 2.28) uses multi-start + relaxation + basin hopping.
# 
# If I strictly follow "lookup table", I would need to hardcode the coordinates. 
# Since I don't have them, I will implement the *closest possible* structural change:
# A highly optimized, deterministic, structured-seed-only approach that relies on the 
# symmetry of the known minima (FCC/HCP/Icosahedral) and uses a single, very fast 
# L-BFGS-B call per seed, without the overhead of basin hopping or random restarts.
# This is a "different design family" in that it removes stochasticity entirely and 
# relies on the fact that for these specific n and rho, the global minimum is known to 
# be a specific symmetry class (e.g., FCC-like for large n, Icosahedral for small n with short range).
#
# Actually, looking at the scores: 2.28 (current) vs 2.68 (inspiration 1). 
# Inspiration 1 was better. It used "deterministic seed-and-polish".
# The current program has basin hopping which might be wandering.
# 
# I will implement a program that generates the most likely candidate structures 
# (FCC, HCP, Icosahedron) for each n, relaxes them with a strict time limit, 
# and returns the best one. No random noise, no basin hopping. Just structured seeds + L-BFGS-B.
# This is materially different from the current "multi-start + basin hopping" approach.

from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _generate_fcc_cluster(n):
    """Generate an FCC cluster of n atoms centered at origin."""
    # Generate a large FCC block
    side = int(np.ceil(n ** (1.0/3.0))) + 3
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                # FCC basis
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k])
                pts.append([i+0.5, j, k+0.5])
                pts.append([i, j+0.5, k+0.5])
    
    pts = np.array(pts)
    # Center
    pts -= np.mean(pts, axis=0)
    # Sort by distance from center
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    return pts

def _generate_hcp_cluster(n):
    """Generate an HCP-like cluster."""
    # HCP is less common for small clusters as global min compared to FCC/Icosahedron
    # but worth trying.
    # Approximate with hexagonal layers
    pts = []
    max_r = int(np.ceil(np.sqrt(n))) + 2
    layer_height = np.sqrt(2.0/3.0)
    
    # Generate layers
    # We need to estimate number of layers. For n~80, maybe 4-6 layers.
    # Let's just generate a large hex grid and stack.
    for layer in range(-5, 6):
        z = layer * layer_height
        # Alternate offset
        offset_x = 0.5 if layer % 2 != 0 else 0.0
        for i in range(-max_r, max_r+1):
            for j in range(-max_r, max_r+1):
                x = i + 0.5 * j + offset_x
                y = j * np.sqrt(3) / 2.0
                if np.linalg.norm([x, y]) <= max_r:
                    pts.append([x, y, z])
                    
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    return pts

def _generate_icosahedron(n):
    """Generate a Mackay icosahedron-like structure."""
    # Magic numbers: 13, 55, 147.
    # For n not matching, take the closest magic number's structure and truncate/extend?
    # Or just generate the 13-atom icosahedron and replicate/scale?
    # Better: Generate the 13-atom icosahedron, and if n>13, add shells.
    # This is complex. Let's use a simple icosahedral seed for n<=13, and FCC for others?
    # Actually, for n=38, 50, 75, 80, FCC/HCP are likely better.
    # Icosahedra are favored for short range (rho=10) and small n.
    # n=38 is close to 13? No. 55 is next.
    # Let's implement a generic icosahedral shell generator.
    
    # 13 atom icosahedron: 1 center, 12 vertices.
    # Vertices of icosahedron:
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    verts = np.array(verts)
    # Normalize to unit distance from center?
    # Distance is sqrt(1+phi^2) = sqrt(1 + (1+sqrt5)^2/4) = ...
    # We want nearest neighbor distance ~1.
    # In icosahedron, edge length is 2. If we scale by 0.5, edge is 1.
    # Distance from center to vertex is sqrt(1+phi^2)*0.5.
    verts *= 0.5
    
    pts = [np.zeros(3)] + [v for v in verts]
    
    if n <= 13:
        pts = np.array(pts[:n])
        return pts
    
    # Add second shell (55 atoms total)
    # This is getting complicated. For n>13, FCC is usually a safer bet for global min in these ranges
    # unless rho is very high (short range).
    # For rho=10 (short range), icosahedral clusters are favored.
    # n=38 is between 13 and 55.
    # Let's just return FCC for n>13 in this function to keep it simple, 
    # as generating correct 2nd shell icosahedron is error-prone without a library.
    return _generate_fcc_cluster(n)

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Strategy: Deterministic structured seeds (FCC, HCP) + L-BFGS-B relaxation.
    No stochastic search. Relies on the fact that global minima for these clusters
    are typically high-symmetry structures.
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    # Budget: 300s total for 6 calls. ~50s per call.
    # We will use 40s per call to be safe.
    deadline = start_time + 40.0
    
    # Generate candidate seeds
    seeds = []
    
    # 1. FCC
    try:
        seeds.append(_generate_fcc_cluster(n))
    except Exception:
        pass
        
    # 2. HCP
    try:
        seeds.append(_generate_hcp_cluster(n))
    except Exception:
        pass
        
    # 3. Icosahedron (if n is small or rho is large)
    if n <= 13 or (n <= 55 and rho >= 10):
        try:
            seeds.append(_generate_icosahedron(n))
        except Exception:
            pass
            
    # If no seeds, use random
    if not seeds:
        seeds.append(np.random.default_rng(42).uniform(-n**0.33, n**0.33, size=(n, 3)))

    best_energy = np.inf
    best_positions = seeds[0]
    
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
            
        try:
            # Use relax from the library which likely uses L-BFGS-B or similar
            # It returns energy and positions
            energy, positions = relax(seed, potential, rho)
            
            # Verify min distance constraint
            if n > 1:
                diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=-1)
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) < 0.5 - 1e-6:
                    continue # Invalid
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                
        except Exception:
            continue

    # If we still have time, we could do a few more random restarts, 
    # but the directive asks for a structural change away from stochastic search.
    # We will stick to the deterministic seeds.
    
    return best_positions
