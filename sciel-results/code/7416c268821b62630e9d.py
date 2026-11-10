"""Rewritten program for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax

_T0 = time.time()
_DEADLINE = 320.0  # Safety margin below the 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _generate_fcc_cluster(n, spacing=1.0):
    """Generate an FCC cluster of n atoms."""
    # FCC lattice constant a such that nearest neighbor distance is 'spacing'
    # NN distance = a / sqrt(2)
    a = spacing * np.sqrt(2)
    
    # Estimate radius
    # Density of FCC = 4 atoms / a^3
    rho = 4.0 / (a ** 3)
    volume = n / rho
    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    
    # Generate grid points
    # x, y, z from -radius to radius
    # Step size a
    points = []
    # Use a slightly larger radius to ensure we have enough points
    search_radius = radius * 1.2 + 1.0
    
    # Grid indices
    # x = i * a, y = j * a, z = k * a
    # FCC condition: i+j+k is even?
    # Basis: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # So points are (i*a, j*a, k*a) where i+j+k is even?
    # Let's check: (0,0,0) sum=0 (even). (0, 0.5, 0.5) -> (0, 0.5a, 0.5a). Sum of indices 0+0+0=0? No.
    # The basis vectors are in units of a.
    # Point = u*a*e1 + v*a*e2 + w*a*e3 + basis
    # It's easier to generate all integer grid points (i,j,k) and check if they form an FCC lattice.
    # FCC lattice points in cubic coordinates (x,y,z) with lattice constant a:
    # (x,y,z) such that x/a, y/a, z/a are integers and x/a + y/a + z/a is even.
    
    i_max = int(np.ceil(search_radius / a))
    candidates = []
    for i in range(-i_max, i_max + 1):
        for j in range(-i_max, i_max + 1):
            for k in range(-i_max, i_max + 1):
                if (i + j + k) % 2 == 0:
                    x = i * a
                    y = j * a
                    z = k * a
                    r = np.sqrt(x*x + y*y + z*z)
                    if r <= radius:
                        candidates.append((r, np.array([x, y, z])))
    
    if len(candidates) < n:
        # Expand radius and try again
        return _generate_fcc_cluster(n, spacing) # Recursive fallback, might be slow. 
        # Better: just return a random sphere if this fails.
        # For now, assume radius estimate is good enough or slightly expand.
        # Let's just take what we have and pad with random if needed? No, that breaks symmetry.
        # Let's increase i_max.
        
    # Sort by radius
    candidates.sort(key=lambda x: x[0])
    
    # Take first n
    selected = [c[1] for c in candidates[:n]]
    
    if len(selected) < n:
        # Fallback: random points in sphere
        return _generate_random_cluster(n, radius * 1.2)
        
    return np.array(selected)


def _generate_fibonacci_sphere(n, radius=1.0):
    """Generate points on a sphere using Fibonacci spiral."""
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Golden angle
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = np.linspace(-1.0, 1.0, n)
    radius_per_point = np.sqrt(1.0 - z**2)
    theta = phi * np.arange(n)
    
    x = np.cos(theta) * radius_per_point
    y = np.sin(theta) * radius_per_point
    
    points = np.column_stack((x, y, z)) * radius
    return points


def _generate_random_cluster(n, radius=1.0, min_dist=0.4, attempts=10000):
    """Generate n random points in a sphere with minimum distance constraint."""
    points = []
    rng = np.random.default_rng()
    
    # Start with a central point if n > 0
    if n > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
        
    for _ in range(attempts):
        if len(points) >= n:
            break
        # Sample point in sphere
        # Uniform in sphere: r = R * u^(1/3)
        r = radius * (rng.random() ** (1.0/3.0))
        theta = rng.uniform(0, 2*np.pi)
        z = rng.uniform(-1, 1)
        phi = np.arccos(z)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z_coord = r * np.cos(phi)
        p = np.array([x, y, z_coord])
        
        # Check distance to existing points
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        
        if valid:
            points.append(p)
            
    if len(points) < n:
        # If we couldn't pack enough, relax the constraint or just return what we have?
        # We need exactly n. Let's just add random points without constraint if we fail.
        # This might lead to invalid results, but relax should fix it if possible.
        while len(points) < n:
            r = radius * (rng.random() ** (1.0/3.0))
            theta = rng.uniform(0, 2*np.pi)
            z = rng.uniform(-1, 1)
            phi = np.arccos(z)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z_coord = r * np.cos(phi)
            points.append(np.array([x, y, z_coord]))
            
    return np.array(points)


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # Estimate typical radius for scaling
    # NN distance ~ 1.0
    # Volume ~ N
    # R ~ N^(1/3)
    base_radius = size ** (1.0/3.0) * 1.0
    
    # Generate candidates
    candidates = []
    
    # 1. FCC
    try:
        fcc_points = _generate_fcc_cluster(size, spacing=1.0)
        candidates.append(fcc_points)
    except Exception:
        pass
        
    # 2. Fibonacci Sphere (scaled to typical radius)
    fib_points = _generate_fibonacci_sphere(size, radius=base_radius)
    candidates.append(fib_points)
    
    # 3. Random Compact Clusters
    # Use slightly larger radius for random to allow relaxation to contract
    rand_radius = base_radius * 1.2
    for i in range(5):
        try:
            rand_points = _generate_random_cluster(size, radius=rand_radius, min_dist=0.45)
            candidates.append(rand_points)
        except Exception:
            pass
            
    # Relax candidates
    best_energy = np.inf
    best_positions = None
    
    # Time check
    # We have ~50s per instance. Relax is fast.
    # Let's process all candidates.
    
    for pos in candidates:
        if _time_left() < 5.0:
            break # Safety margin
            
        try:
            energy, relaxed_pos = relax(pos, potential, kind)
            if not np.all(np.isfinite(relaxed_pos)):
                continue
                
            # Check min distance
            # Calculate all pairwise distances
            # For N=80, this is 80*79/2 = 3160 pairs. Fast.
            diff = relaxed_pos[:, np.newaxis, :] - relaxed_pos[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < 0.35:
                continue
                
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
                
        except Exception:
            continue
            
    # If no valid candidate, return the first one (even if invalid? No, must return valid)
    # If best_positions is None, we must return something.
    # Let's try a very loose random start and relax it, hoping it finds a valid min.
    if best_positions is None:
        try:
            loose_pos = _generate_random_cluster(size, radius=base_radius * 2.0, min_dist=0.36)
            energy, relaxed_pos = relax(loose_pos, potential, kind)
            if np.all(np.isfinite(relaxed_pos)):
                best_positions = relaxed_pos
        except Exception:
            # Last resort: return a grid
            # This will likely be invalid, but better than crash?
            # The task says invalid result scores 0.
            # Let's try to construct a valid grid manually.
            side = int(np.ceil(size ** (1/3)))
            grid = []
            x = 0
            y = 0
            z = 0
            while len(grid) < size:
                grid.append([x, y, z])
                x += 1.0
                if x >= side:
                    x = 0
                    y += 1.0
                    if y >= side:
                        y = 0
                        z += 1.0
            best_positions = np.array(grid[:size])
            
    return best_positions
# EVOLVE-BLOCK-END
