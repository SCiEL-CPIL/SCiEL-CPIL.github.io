"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # Safety margin below 330s

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_vertices():
    """Generate vertices of a regular icosahedron."""
    phi = (1 + 5**0.5) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _decahedron_vertices():
    """Generate vertices of a regular decahedron (approximated by FCC stacking)."""
    # A decahedron is a 5-layer structure with 1-4-6-4-1 or similar stacking?
    # Actually, a regular decahedron in cluster science usually refers to the 
    # stacking of 5 triangles in a pyramid, or the FCC 1-4-10-4-1 (20 atoms) etc.
    # Let's use the standard "decahedral" stacking: 1-4-6-4-1 is not right.
    # Common decahedra: 13 (1-4-4-1? no, 1-4-10 is FCC).
    # Let's just use a generic FCC-like compact structure for "deca" fallback if needed, 
    # but icosahedra are the primary target for SC potential.
    # For now, return empty to focus on Icosahedra which are stronger minima for SC.
    return np.array([])

def _generate_icosahedral_cluster(n):
    """
    Generate a cluster of size n based on Mackay icosahedra.
    If n is not a magic number (1, 13, 55, 147...), add atoms to the surface.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Magic numbers: 1, 13, 55, 147
    magic_nums = [1, 13, 55, 147]
    
    # Find the largest magic number <= n
    base_n = 1
    for mn in magic_nums:
        if mn <= n:
            base_n = mn
        else:
            break
            
    # Generate base icosahedron
    if base_n == 1:
        pts = np.array([[0.0, 0.0, 0.0]])
    elif base_n == 13:
        pts = _icosahedron_vertices()
        pts = np.vstack([np.array([[0.0, 0.0, 0.0]]), pts])
    elif base_n == 55:
        # 1 center + 12 vertices + 30 edge midpoints
        v = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])]
        pts.extend([v[i] for i in range(12)])
        # Edges of icosahedron
        # We need to define the edges.
        # Icosahedron edges connect vertices with distance 2/sqrt(10+2*sqrt(5)) * 2?
        # Simpler: just add atoms at midpoints of edges defined by distance.
        # Distance between adjacent vertices in unit icosahedron:
        # v0 = (0, 1, phi). v1 = (1, phi, 0).
        # dist = sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + phi^2 - 2phi + 1 + phi^2) = sqrt(2 + 2phi^2 - 2phi)
        # phi^2 = phi+1. 2+2(phi+1)-2phi = 4. dist = 2.
        # So adjacent vertices are distance 2 apart in this normalized set?
        # Let's check norm of v: sqrt(1+phi^2) = sqrt(phi+2).
        # Actually, let's just compute pairwise distances and find the smallest unique distance > 0.
        d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        
        for i in range(12):
            for j in range(i+1, 12):
                if np.isclose(d[i,j], min_d, atol=1e-6):
                    mid = (v[i] + v[j]) / 2.0
                    pts.append(mid)
        pts = np.array(pts)
        # We should have 1 + 12 + 30 = 43? Wait.
        # Mackay 55: 1 center, 12 vertices, 30 edge midpoints. Total 43? No.
        # 1 + 12 + 30 = 43. Where are the other 12?
        # Ah, the 55-atom icosahedron is 1-12-42? No.
        # Shell 1: 12. Shell 2: 42. Total 55.
        # The 42 atoms in shell 2 are 30 edge midpoints and 12 face centers.
        # Let's add face centers.
        # Faces are triangles.
        # This is getting complex to generate exactly. 
        # Alternative: Use a simpler "spherical shell" approximation if exact geometry is hard.
        # Or just use the 13-atom icosahedron and add atoms randomly on the surface for n > 13.
        # Given the small n values (38, 55, 59, 75, 79, 80), 
        # 38 is close to 13? No.
        # 55 is exactly the 3rd magic number.
        # 59, 75, 79, 80 are larger.
        
        # Let's refine the 55-atom generation.
        # If we just use 13 atoms and add 42 random surface atoms, it might not be optimal.
        # Let's try to generate the 55-atom Mackay icosahedron properly.
        # It consists of 5 icosahedra of 13 atoms sharing a common center? No.
        # It's a 3-shell icosahedron.
        # Shell 0: 1 atom.
        # Shell 1: 12 atoms (vertices of icosahedron).
        # Shell 2: 42 atoms.
        # The 42 atoms are located at the midpoints of the 30 edges and the centers of the 20 faces.
        # 30 + 20 = 50? No, 42.
        # Wait, 12 vertices, 30 edges, 20 faces.
        # If we put atoms at all edge midpoints (30) and face centers (20), that's 50.
        # 1 + 12 + 50 = 63. That's not 55.
        # The Mackay 55-atom cluster is actually composed of 5 interpenetrating 13-atom icosahedra?
        # No, that's the 147-atom?
        
        # Let's look up Mackay 55.
        # It is a 3-shell icosahedron.
        # 1 (center) + 12 (1st shell) + 42 (2nd shell) = 55.
        # The 42 atoms in the 2nd shell are NOT all edge/face centers.
        # They are arranged in a specific pattern.
        # Given the complexity, I will use a heuristic:
        # If n == 55, use a pre-defined or approximate icosahedral structure.
        # If n is not a magic number, start with the largest magic number < n 
        # and add atoms at random positions on the surface of the convex hull, 
        # then relax.
        
        # For now, let's just return the 13-atom icosahedron for n=13, 
        # and for other n, we will use a "grow" strategy in the main loop.
        pass

    # Fallback for generation:
    # If n <= 13, use icosahedron.
    if n <= 13:
        if n == 1: return np.array([[0.0, 0.0, 0.0]])
        if n == 13:
            v = _icosahedron_vertices()
            return np.vstack([np.array([[0.0, 0.0, 0.0]]), v])
        # For n < 13, just take a subset? No, better to use random compact.
        return None # Signal to use random
    
    # For n > 13, we will generate a 13-atom icosahedron and add atoms.
    base = np.vstack([np.array([[0.0, 0.0, 0.0]]), _icosahedron_vertices()])
    num_to_add = n - len(base)
    
    # Add atoms on the surface
    # Sample points on the sphere and project onto the convex hull? 
    # Simpler: sample random points in a shell around the 13-atom cluster.
    rng = np.random.default_rng(42)
    added = []
    for _ in range(num_to_add):
        # Random direction
        dir = rng.normal(size=3)
        dir /= np.linalg.norm(dir)
        # Distance: slightly larger than the radius of the 13-atom cluster
        # Radius of 13-atom icosahedron (unit sphere) is 1.
        # Add atom at r = 1.5
        p = dir * 1.5
        added.append(p)
        
    if added:
        base = np.vstack([base, np.array(added)])
        
    return base

def construct(instance):
    """
    Strategy:
    1. Generate structured seeds (Icosahedra).
    2. Shake and relax each seed.
    3. Keep the best.
    4. If time remains, try random compact seeds.
    """
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    best_energy = float('inf')
    best_pos = None
    
    # 1. Try Icosahedral Seed
    if n <= 13:
        if n == 13:
            seed = _generate_icosahedral_cluster(n)
            # Shake
            rng = np.random.default_rng(1)
            for i in range(5):
                if _time_left() < 5.0: break
                perturbed = seed + rng.normal(0, 0.05, seed.shape)
                try:
                    e, pos = relax(perturbed, potential, kind)
                    if e < best_energy:
                        best_energy = e
                        best_pos = pos
                except:
                    pass
    else:
        seed = _generate_icosahedral_cluster(n)
        if seed is not None and len(seed) == n:
            # Shake and relax
            rng = np.random.default_rng(2)
            for i in range(10): # More shakes for larger clusters
                if _time_left() < 5.0: break
                perturbed = seed + rng.normal(0, 0.1, seed.shape)
                try:
                    e, pos = relax(perturbed, potential, kind)
                    if e < best_energy:
                        best_energy = e
                        best_pos = pos
                except:
                    pass

    # 2. If no valid result yet, or if we have time, try random compact seeds
    # The original code used random compact seeds. We will add them as a fallback.
    if best_pos is None or _time_left() > 10.0:
        # Use a few random seeds
        rng = np.random.default_rng(n)
        spacing = 0.72
        radius = spacing * n ** (1.0 / 3.0) * 0.75
        
        # Try 4 random seeds
        for i in range(4):
            if _time_left() < 5.0: break
            points = []
            # Rejection sampling for compact cluster
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                    points.append(p)
            
            if len(points) == n:
                try:
                    e, pos = relax(np.array(points), potential, kind)
                    if e < best_energy:
                        best_energy = e
                        best_pos = pos
                except:
                    pass

    # 3. If still no result, return a valid dummy (should not happen)
    if best_pos is None:
        # Fallback: linear chain (valid but high energy)
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    return best_pos
# EVOLVE-BLOCK-END
