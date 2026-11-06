# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _get_icosahedron_vertices():
    """Generate 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _generate_mackay_55():
    """
    Generate the 55-atom Mackay icosahedron.
    Structure: 1 center + 12 inner shell + 12 outer vertices + 30 outer edge midpoints.
    """
    v = _get_icosahedron_vertices()
    
    # Inner shell (radius ~1.0)
    inner = v * 1.05
    
    # Outer shell (radius ~1.8)
    outer_verts = v * 1.8
    
    # Find edges of the icosahedron to generate midpoints
    dists = np.linalg.norm(v[:, np.newaxis] - v[np.newaxis, :], axis=2)
    min_dist = np.min(dists[dists > 1e-6])
    
    outer_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if dists[i, j] < 1.1 * min_dist:
                mid = (outer_verts[i] + outer_verts[j]) / 2.0
                outer_mids.append(mid)
                
    points = [np.zeros(3)]
    points.extend(list(inner))
    points.extend(list(outer_verts))
    points.extend(outer_mids)
    
    return np.array(points)

def _generate_dodecahedron_20():
    """
    Generate a 20-atom Dodecahedron cluster (approximation).
    1 center + 12 icosahedron vertices (inner) + 7 dodecahedron vertices?
    Actually, a common quasicrystalline motif is the Dodecahedron.
    Let's use a simpler approach: Random compact + icosahedral bias.
    For now, we will generate a "Fibonacci sphere" distribution for diversity.
    """
    # This function is a placeholder for a specific quasicrystalline seed if needed,
    # but for now we will rely on the Mackay template and Random Compact seeds.
    pass

def _generate_seeds(n, rng):
    """
    Generate a diverse pool of initial configurations (seeds) for basin hopping.
    1. Mackay-derived (for n close to 55, or scaled)
    2. Random Compact (Fibonacci sphere or uniform sphere)
    """
    seeds = []
    
    # Seed 1: Mackay Icosahedron based
    # If n == 55, use perfect Mackay.
    # If n != 55, we can't easily use the 55 template directly without complex addition/removal.
    # So for n != 55, we generate a scaled icosahedron-like structure.
    if n == 55:
        seeds.append(_generate_mackay_55())
    else:
        # Generate an icosahedron-like structure for n atoms.
        # We can use the 12 vertices of an icosahedron and add random points inside.
        # Or simply use a scaled random compact.
        pass

    # Seed 2: Random Compact (Uniform Sphere)
    # Generate points in a sphere of radius R such that density is reasonable.
    # R ~ (n / (4/3 pi rho))^(1/3). Let's assume rho ~ 1.
    R = 1.5 * n**(1/3)
    for _ in range(3):
        pts = rng.uniform(-R, R, (n, 3))
        # Center it
        pts -= np.mean(pts, axis=0)
        # Ensure no overlaps are too tight (relax will fix, but start valid)
        # Simple check: if min dist < 0.5, retry? Or just let relax handle it.
        # Relax usually handles overlaps by pushing apart.
        seeds.append(pts)

    # Seed 3: Fibonacci Sphere (more uniform distribution on surface, good for shells)
    # We will create a few layers.
    # For simplicity, let's just add a "Random Shell" seed.
    R_shell = 1.5 * n**(1/3)
    pts_shell = rng.normal(0, 1, (n, 3))
    norms = np.linalg.norm(pts_shell, axis=1, keepdims=True)
    pts_shell = pts_shell / norms * R_shell
    pts_shell -= np.mean(pts_shell, axis=0)
    seeds.append(pts_shell)

    # Seed 4: If n is close to 55, try to use the 55 template by adding/removing.
    # This is complex. Let's stick to the 4 seeds above for robustness.
    
    return seeds

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin (300s limit, 10s buffer)
    
    rng = np.random.default_rng(42)
    
    # Generate initial seeds
    seeds = _generate_seeds(n, rng)
    
    # If we have the specific 55 case, ensure we have that seed
    if n == 55:
        # _generate_seeds already added it if n==55
        pass
    else:
        # For other n, let's try to create a "Mackay-like" seed by scaling a 55-atom structure?
        # No, that's physically wrong.
        # Let's just rely on the random seeds.
        pass
        
    # Add more random seeds to the pool to ensure diversity
    # We want to explore multiple basins.
    # Let's add 5 more random compact seeds.
    R = 1.5 * n**(1/3)
    for _ in range(5):
        pts = rng.uniform(-R, R, (n, 3))
        pts -= np.mean(pts, axis=0)
        seeds.append(pts)

    best_energy = np.inf
    best_pos = seeds[0].copy()
    
    # Basin Hopping Loop
    # We will iterate through seeds, relax them, and then perform random kicks.
    # Since relax is a local minimizer, we treat it as the "cooling" step.
    
    # Time management:
    # We have ~290 seconds.
    # Relaxation of 50-60 atoms might take 1-5 seconds.
    # We can afford ~50-100 relaxations.
    
    current_pos = None
    
    # We will cycle through seeds.
    # Strategy:
    # 1. Relax seed i. If better, update best.
    # 2. From current best (or current relaxed seed), apply a random kick.
    # 3. Relax the kicked configuration.
    # 4. Accept if better (or with Metropolis criterion? For global min, simple hill climbing on energy is fine if we have many seeds).
    # Actually, simple "best-so-far" hill climbing from multiple seeds is effective.
    
    # Let's implement a simple multi-start basin hopping:
    # For each seed:
    #   - Relax
    #   - If better, update best.
    #   - Perform K local kicks from this relaxed structure, relax each, keep best.
    
    # To maximize diversity, we should process seeds in parallel? No, 4 threads limit.
    # We will just run sequentially but efficiently.
    
    # Let's define a helper to relax and check
    def _relax_and_check(pos):
        nonlocal best_energy, best_pos
        try:
            # relax returns the relaxed positions
            relaxed = relax(pos, potential)
            energy, _ = dzugutov_energy_gradient(relaxed)
            
            # Check constraints
            diff = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 0.5:
                return None, energy # Invalid, but return energy for logging? No, discard.
                
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
            return relaxed, energy
        except Exception:
            return None, np.inf

    # Process seeds
    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
            
        # 1. Relax the seed
        relaxed_seed, e_seed = _relax_and_check(seed)
        
        if relaxed_seed is None:
            continue
            
        # 2. Perform local basin hopping from this relaxed seed
        # We will do a few "kicks" from this local minimum.
        # Kick size: random perturbation.
        # Let's do 5 kicks per seed if time permits.
        
        current_local_best = relaxed_seed
        current_local_energy = e_seed
        
        for kick in range(5):
            if time.time() > deadline:
                break
                
            # Random kick
            # Perturbation magnitude: 0.5-1.0 units
            perturbation = rng.normal(0, 0.8, (n, 3))
            kicked_pos = current_local_best + perturbation
            
            # Relax the kicked position
            relaxed_kick, e_kick = _relax_and_check(kicked_pos)
            
            if relaxed_kick is not None:
                # Accept if better than current local best (Hill Climbing)
                # Or accept if better than global best?
                # Hill climbing within the basin is good to find the bottom.
                if e_kick < current_local_energy:
                    current_local_best = relaxed_kick
                    current_local_energy = e_kick
                # We also update global best inside _relax_and_check

    # Final check
    if best_energy == np.inf:
        # Fallback: return a valid random structure
        R = 1.5 * n**(1/3)
        best_pos = rng.uniform(-R, R, (n, 3))
        best_pos -= np.mean(best_pos, axis=0)
        try:
            best_pos = relax(best_pos, potential)
        except:
            pass

    return best_pos
# EVOLVE-BLOCK-END
