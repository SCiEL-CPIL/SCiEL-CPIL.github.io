"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

DEADLINE = 300.0  # Safety margin for 330s limit

def _generate_bcc_cluster(n, spacing=1.1):
    """Generate a BCC-like cluster of n atoms."""
    # BCC unit cell points: (0,0,0) and (0.5, 0.5, 0.5) * side_length
    # We use a cubic shell expansion.
    points = []
    # Center atom
    points.append([0.0, 0.0, 0.0])
    
    # Generate lattice points in a cube
    max_k = int(np.ceil(n ** (1/3)))
    candidates = []
    for x in range(-max_k, max_k + 1):
        for y in range(-max_k, max_k + 1):
            for z in range(-max_k, max_k + 1):
                # BCC points: integer coords (even sum? no, just all integer + half-integer shifts)
                # Standard BCC: points at (i, j, k) and (i+0.5, j+0.5, k+0.5)
                # Let's generate a supercell of BCC
                pass
    
    # Simpler approach: Generate a list of BCC lattice points, sort by distance from origin, take first n.
    lattice_points = []
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Corner of cube
                p1 = np.array([i, j, k])
                lattice_points.append(p1)
                # Body center
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                lattice_points.append(p2)
    
    # Sort by distance from origin
    dists = np.linalg.norm(lattice_points, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take first n points
    selected_points = [lattice_points[i] for i in sorted_indices[:n]]
    
    # Scale to match spacing
    # The nearest neighbor distance in BCC with unit cell side a is a * sqrt(3) / 2.
    # If we set a=1, nn_dist = sqrt(3)/2.
    # We want nn_dist = spacing. So scale factor = spacing / (sqrt(3)/2)
    scale = spacing / (np.sqrt(3) / 2.0)
    selected_points = np.array(selected_points) * scale
    
    # Center the cluster
    centroid = np.mean(selected_points, axis=0)
    selected_points -= centroid
    
    return selected_points

def _generate_fcc_cluster(n, spacing=1.1):
    """Generate an FCC-like cluster of n atoms."""
    lattice_points = []
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC points: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
                # In integer grid terms:
                # (i, j, k) if i+j+k is even? No, that's diamond.
                # FCC: (i, j, k) where i,j,k are all integers or all half-integers?
                # Let's use the standard basis vectors.
                # Unit cell side a=1.
                # Points: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                p1 = np.array([i, j, k])
                p2 = np.array([i, j + 0.5, k + 0.5])
                p3 = np.array([i + 0.5, j, k + 0.5])
                p4 = np.array([i + 0.5, j + 0.5, k])
                lattice_points.extend([p1, p2, p3, p4])
                
    dists = np.linalg.norm(lattice_points, axis=1)
    sorted_indices = np.argsort(dists)
    selected_points = [lattice_points[i] for i in sorted_indices[:n]]
    
    # Nearest neighbor in FCC (a=1) is sqrt(2)/2.
    scale = spacing / (np.sqrt(2) / 2.0)
    selected_points = np.array(selected_points) * scale
    
    centroid = np.mean(selected_points, axis=0)
    selected_points -= centroid
    
    return selected_points

def _generate_random_cluster(n, spacing=1.1, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
            points.append(p)
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    
    candidates = []
    
    # 1. BCC start
    try:
        bcc_pts = _generate_bcc_cluster(size)
        candidates.append(bcc_pts)
    except Exception:
        pass
        
    # 2. FCC start
    try:
        fcc_pts = _generate_fcc_cluster(size)
        candidates.append(fcc_pts)
    except Exception:
        pass
        
    # 3. Random starts
    for seed in [size, size + 1, size + 2]:
        try:
            rand_pts = _generate_random_cluster(size, seed=seed)
            candidates.append(rand_pts)
        except Exception:
            pass
            
    best_energy = float('inf')
    best_positions = None
    
    for idx, pts in enumerate(candidates):
        if time.time() > DEADLINE - 30:
            break
            
        try:
            energy, positions = relax(pts, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
            
    if best_positions is None:
        # Fallback: return random points if all relaxations failed
        return _generate_random_cluster(size, seed=123)
        
    return best_positions
# EVOLVE-BLOCK-END
