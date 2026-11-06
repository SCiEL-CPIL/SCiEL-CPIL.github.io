"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global start time for budget enforcement
_START_TIME = time.time()
TIME_LIMIT = 290.0  # Leave 10s safety margin for 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _get_icosahedron_points():
    """
    Returns the 12 vertices, 20 face centers, and 30 edge centers of a regular icosahedron.
    All points are normalized to unit radius.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 Vertices
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    v = v / np.linalg.norm(v[0])
    
    # Define faces (indices of vertices)
    # Standard icosahedron faces
    faces = [
        [0, 1, 8], [0, 8, 4], [0, 4, 5], [0, 5, 10], [0, 10, 1],
        [1, 10, 2], [1, 2, 11], [1, 11, 8], [8, 11, 9], [8, 9, 4],
        [4, 9, 6], [4, 6, 5], [5, 6, 7], [5, 7, 10], [10, 7, 3],
        [10, 3, 2], [2, 3, 12], [2, 12, 11], [11, 12, 9], [9, 12, 6]
    ]
    # Note: The indices above might need verification. 
    # Let's generate face centers and edge centers programmatically to be safe.
    
    # To get faces and edges correctly without hardcoding indices that might be wrong:
    # We can find edges by finding pairs of vertices with minimum distance.
    # And faces by finding triangles of edges.
    
    # Distance matrix
    dists = np.linalg.norm(v[:, np.newaxis, :] - v[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists)
    
    # Edges: pairs with distance == min_dist
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i, j], min_dist):
                edges.append((i, j))
                
    # Face Centers:
    # Faces are triangles of edges.
    # For each edge (i, j), find k such that (i, k) and (j, k) are also edges.
    face_centers = []
    seen_faces = set()
    for i, j in edges:
        # Find k connected to both i and j
        for k in range(12):
            if k == i or k == j:
                continue
            if (i, k) in edges or (k, i) in edges:
                if (j, k) in edges or (k, j) in edges:
                    # Triangle i, j, k
                    face_key = tuple(sorted([i, j, k]))
                    if face_key not in seen_faces:
                        seen_faces.add(face_key)
                        center = (v[i] + v[j] + v[k]) / 3.0
                        center = center / np.linalg.norm(center)
                        face_centers.append(center)
                        
    # Edge Centers:
    edge_centers = []
    for i, j in edges:
        center = (v[i] + v[j]) / 2.0
        center = center / np.linalg.norm(center)
        edge_centers.append(center)
        
    return v, np.array(face_centers), np.array(edge_centers)

# Precompute the special points
V, F, E = _get_icosahedron_points()

def _generate_candidate(n, rng, variant=0):
    """
    Generates a candidate structure.
    variant 0: Icosahedral core + structured outer atoms (F/E centers)
    variant 1: Icosahedral core + random outer atoms
    variant 2: Random compact
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Core: 12 vertices
    # Equilibrium distance for Dzugutov is ~1.1
    r1 = 1.1
    shell1 = V * r1
    points.extend(shell1)
    
    if n <= 13:
        return np.array(points[:n])
    
    needed = n - 13
    
    if variant == 0:
        # Use structured points: Face centers and Edge centers
        # Scale them to a reasonable radius. 
        # In Mackay icosahedron, r2 is larger than r1.
        # Let's try r2 = r1 * phi ~ 1.78
        r2 = r1 * ((1 + np.sqrt(5)) / 2)
        
        # Pool of candidates: F and E
        # We need to pick 'needed' from F (20) and E (30).
        # Shuffle them to get different subsets
        pool = np.vstack([F, E])
        idx = rng.permutation(len(pool))
        selected = pool[idx[:needed]]
        selected = selected * r2
        points.extend(selected)
        
    elif variant == 1:
        # Random points on sphere of radius r2
        r2 = 1.8
        for _ in range(needed):
            # Random direction
            dir = rng.normal(size=3)
            dir = dir / np.linalg.norm(dir)
            p = dir * r2
            points.append(p)
            
    else:
        # Random compact
        radius = 1.1 * (n / (4/3 * np.pi)) ** (1/3) * 1.2
        pts = []
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Check distance
                if all(np.linalg.norm(p - q) >= 0.5 for q in pts):
                    pts.append(p)
        return np.array(pts)
        
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    if potential != "dzugutov":
        # Fallback
        rng = np.random.default_rng(size)
        radius = 1.0 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, parameter)
        return positions

    best_energy = float('inf')
    best_positions = None
    
    # Try multiple variants and seeds
    # We have ~300s. Relaxation of 60 atoms is fast.
    # We can afford ~10-20 relaxations.
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        if not _check_time():
            break
            
        rng = np.random.default_rng(seed)
        
        # Variant 0: Structured Icosahedral
        try:
            pos = _generate_candidate(size, rng, variant=0)
            if len(pos) == size:
                energy, positions = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass
            
        # Variant 1: Random Icosahedral Shell
        try:
            pos = _generate_candidate(size, rng, variant=1)
            if len(pos) == size:
                energy, positions = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass
        
        # Variant 2: Random Compact (less likely to be best, but good for diversity)
        # Only run if time permits and we haven't found a very good solution
        if best_energy > 0 or seed < 3: 
            try:
                pos = _generate_candidate(size, rng, variant=2)
                if len(pos) == size:
                    energy, positions = relax(pos, potential, parameter)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
            except Exception:
                pass

    if best_positions is None:
        # Fallback
        best_positions = np.array([[i*1.2, 0, 0] for i in range(size)])
        try:
             _, best_positions = relax(best_positions, potential, parameter)
        except:
             pass

    return best_positions

# EVOLVE-BLOCK-END
