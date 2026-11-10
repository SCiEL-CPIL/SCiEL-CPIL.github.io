"""Dzugutov cluster global minima search using Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax

# Constants
DEADLINE = 320.0  # Safety margin
_start_time = None

def _get_global_start():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _generate_icosahedral_seed(n, rng):
    """
    Generate a seed structure based on icosahedral shells.
    The Dzugutov potential favors icosahedral order.
    Shell 0: 1 atom
    Shell 1: 12 atoms (Icosahedron)
    Shell 2: 42 atoms (Dodecahedron-like / second icosahedral shell)
    Total for full shells: 1, 13, 55.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices (12)
    # Normalized to unit sphere, then scaled
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Dodecahedron vertices (20) - often used as the second shell in icosahedral clusters
    # Actually, the second shell in a perfect icosahedral cluster (Mackay) is 42 atoms.
    # Let's use a simpler approach: place atoms on a sphere with radius r2.
    # For n=38, we need 1+12+25.
    # For n=55, we need 1+12+42.
    
    points = [np.zeros(3)]
    
    # Shell 1: Icosahedron (12 atoms)
    r1 = 1.4 # Equilibrium distance for Dzugutov is around 1.4-1.5
    if n > 1:
        shell1 = ico_verts * r1
        for v in shell1:
            if len(points) >= n: break
            points.append(v)
            
    # Shell 2: Second shell
    if len(points) < n:
        # Generate points on a sphere of radius r2
        r2 = 2.5
        # Use Fibonacci sphere for uniform distribution if we just need points
        # Or use dodecahedron vertices + midpoints of edges?
        # Let's use a Fibonacci sphere for the remaining atoms to fill the shell
        num_needed = n - len(points)
        
        # To be more "icosahedral-like", we can place them near the icosahedron vertices' neighbors
        # But a simple spherical shell with jitter is a good baseline for basin hopping.
        # Let's use a slightly perturbed spherical shell.
        
        # Generate num_needed points on a sphere
        # Using golden spiral
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(num_needed):
            y = 1 - (i / max(num_needed - 1, 1)) * 2
            radius = np.sqrt(1 - y*y)
            theta = golden_angle * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            p = np.array([x, y, z]) * r2
            # Add small random perturbation to break symmetry
            p += rng.normal(0, 0.05, 3)
            points.append(p)
            
    return np.array(points[:n])

def _generate_random_compact(n, rng):
    """Generate a random compact cluster as a diverse starting point."""
    # Estimate radius based on volume
    # Assume density similar to solid
    r_max = 1.8 * (n ** (1.0/3.0))
    points = []
    for _ in range(n):
        # Rejection sampling for uniform sphere
        while True:
            p = rng.uniform(-1, 1, 3)
            if np.linalg.norm(p) <= 1.0:
                break
        p *= r_max
        points.append(p)
    return np.array(points)

def _basin_hopping_worker(args):
    """
    Worker function for parallel basin hopping.
    args: (n, rng_seed, kick_magnitude, max_iter, start_type)
    """
    n, rng_seed, kick_magnitude, max_iter, start_type = args
    rng = np.random.default_rng(rng_seed)
    
    best_energy = np.inf
    best_pos = None
    
    # Initial structure
    if start_type == 'ico':
        pos = _generate_icosahedral_seed(n, rng)
    else:
        pos = _generate_random_compact(n, rng)
        
    # Initial relaxation
    try:
        energy, pos = relax(pos, 'dzugutov', None)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    except Exception:
        return (np.inf, None)
    
    current_energy = energy
    current_pos = pos
    
    for i in range(max_iter):
        # Check time
        if time.time() - _get_global_start() > DEADLINE - 5:
            break
            
        # Kick
        # The magnitude of the kick is crucial. 
        # Dzugutov potential has a soft repulsive core and attractive well.
        # A kick of 0.1-0.5 is typical.
        kick = rng.normal(0, kick_magnitude, (n, 3))
        new_pos = current_pos + kick
        
        # Relax
        try:
            new_energy, new_pos = relax(new_pos, 'dzugutov', None)
        except Exception:
            continue
            
        # Acceptance criterion: Greedy (always accept if better) 
        # This is effectively "Multi-start with local search" but with kicks.
        # Standard Basin Hopping uses Metropolis, but for finding global min in limited time,
        # greedy hill climbing from kicked states is often effective if kicks are large enough to escape basins.
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
                
    return (best_energy, best_pos)

def construct(instance):
    """
    Find the lowest-energy structure for the given cluster size.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    n = instance['n']
    
    # Define search strategies
    # We will run multiple parallel basin hopping chains.
    # Different kick magnitudes and starting structures.
    
    tasks = []
    
    # Strategy 1: Icosahedral seeds with small kicks
    # Icosahedral order is favored by Dzugutov.
    for i in range(4):
        tasks.append((n, 1000 + i, 0.1, 20, 'ico'))
        
    # Strategy 2: Icosahedral seeds with larger kicks
    for i in range(4):
        tasks.append((n, 2000 + i, 0.3, 20, 'ico'))
        
    # Strategy 3: Random seeds with medium kicks
    for i in range(4):
        tasks.append((n, 3000 + i, 0.2, 20, 'rand'))
        
    # Strategy 4: Random seeds with large kicks (more exploration)
    for i in range(2):
        tasks.append((n, 4000 + i, 0.5, 10, 'rand'))
        
    best_energy = np.inf
    best_pos = None
    
    # Use ThreadPoolExecutor. 
    # Note: relax() might release the GIL if it uses C extensions, otherwise threads are limited.
    # Given the constraint "at most 4 threads", we use max_workers=4.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for task in tasks:
            # Check if we have time
            if time.time() - _get_global_start() > DEADLINE - 10:
                break
            futures.append(executor.submit(_basin_hopping_worker, task))
            
        for future in concurrent.futures.as_completed(futures):
            if time.time() - _get_global_start() > DEADLINE - 5:
                break
            try:
                energy, pos = future.result(timeout=1)
                if pos is not None and energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                pass
                
    # If no valid result was found (e.g., relax failed everywhere), fallback to a simple seed
    if best_pos is None:
        rng = np.random.default_rng(42)
        best_pos = _generate_icosahedral_seed(n, rng)
        # Try one last relaxation
        try:
            energy, best_pos = relax(best_pos, 'dzugutov', None)
        except Exception:
            pass
            
    # Ensure minimum distance constraint if relax didn't enforce it strictly
    # The potential has a hard core, so relax should handle it, but let's be safe.
    # However, modifying positions after relax might increase energy.
    # The task says "atoms must be at least 0.5 apart".
    # Dzugutov potential V(r) goes to infinity as r -> 0, so relax should keep them apart.
    # We will assume relax respects the potential's domain.
    
    return best_pos

# EVOLVE-BLOCK-END
