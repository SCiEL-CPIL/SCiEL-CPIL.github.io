# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 310.0  # Safety margin before 330s kill

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _min_distance(positions):
    """
    Vectorized computation of the minimum pairwise distance.
    """
    n = positions.shape[0]
    if n < 2:
        return float('inf')
    
    # Use KDTree for efficiency if n is large, or vectorized for small n
    if n < 50:
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(n, k=1)
        return np.min(dists[iu])
    else:
        tree = cKDTree(positions)
        min_d = float('inf')
        for i in range(n):
            # query_pairs or just query each point to find nearest neighbor
            # query_pairs is efficient for global min
            pass
        # Simpler: query_pairs
        min_d = tree.pdist(positions).min()
        return min_d

def _generate_icosahedral_template(n, seed=42):
    """
    Generate a Mackay icosahedral cluster.
    """
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (normalized to distance 1 from center)
    phi = (1 + math.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)] # Center
    
    # Layer 1: Vertices
    for v in verts:
        points.append(v.copy())
        
    # Layer 2: Edges (midpoints)
    edge_points = []
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - 2.0) < 1e-6:
                mid = (verts[i] + verts[j]) / 2.0
                edge_points.append(mid)
                
    for p in edge_points:
        points.append(p)
        
    # Layer 3: Faces (centroids)
    face_points = []
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            for k in range(j+1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if abs(d1 - 2.0) < 1e-6 and abs(d2 - 2.0) < 1e-6 and abs(d3 - 2.0) < 1e-6:
                    centroid = (verts[i] + verts[j] + verts[k]) / 3.0
                    face_points.append(centroid)
                    
    for p in face_points:
        points.append(p)
        
    points = np.array(points)
    
    if len(points) < n:
        # If we need more atoms, we can't easily extend Mackay.
        # We will just scale the existing points to fill the volume?
        # Or just use the 63 points and let relaxation handle the rest?
        # The interface requires n atoms.
        # If n > 63, we will add random points near the surface.
        rng2 = np.random.default_rng(seed + 100)
        while len(points) < n:
            # Pick a random existing point and add a small offset
            idx = rng2.integers(0, len(points))
            new_p = points[idx] + rng2.normal(0, 0.2, 3)
            points = np.vstack([points, new_p.reshape(1,3)])
    else:
        # Sort by distance to center and take closest n
        distances = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(distances)
        points = points[sorted_indices[:n]]
        
    # Jitter
    jitter = rng.uniform(-0.02, 0.02, (n, 3))
    points = points + jitter
    return points

def _generate_fcc_template(n, seed=43):
    """
    Generate an FCC-like cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.8 * (n ** (1.0/3.0))
    points = []
    max_coord = int(math.ceil(radius)) + 2
    
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
                        
    if not points:
        return _generate_random_template(n, seed)
        
    points = np.array(points)
    distances = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(distances)
    cluster = points[sorted_indices[:n]]
    
    # Small jitter to break symmetry
    jitter = rng.uniform(-0.02, 0.02, (n, 3))
    cluster = cluster + jitter
    return cluster

def _generate_bcc_template(n, seed=44):
    """
    Generate a BCC-like cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.8 * (n ** (1.0/3.0))
    points = []
    max_coord = int(math.ceil(radius)) + 2
    
    # BCC basis: (0,0,0), (0.5,0.5,0.5)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
                        
    if not points:
        return _generate_random_template(n, seed)
        
    points = np.array(points)
    distances = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(distances)
    cluster = points[sorted_indices[:n]]
    
    jitter = rng.uniform(-0.02, 0.02, (n, 3))
    cluster = cluster + jitter
    return cluster

def _generate_random_template(n, seed=45):
    """
    Generate a random cluster with minimum distance constraint.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    attempts = 0
    max_attempts = 5000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            if points:
                pts_arr = np.array(points)
                dists = np.linalg.norm(pts_arr - p, axis=1)
                if np.min(dists) < 0.6:
                    valid = False
            if valid:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Generate diverse high-quality templates
    templates = []
    
    try:
        templates.append(_generate_icosahedral_template(n, seed=42))
    except Exception:
        pass
        
    try:
        templates.append(_generate_fcc_template(n, seed=43))
    except Exception:
        pass
        
    try:
        templates.append(_generate_bcc_template(n, seed=44))
    except Exception:
        pass
        
    try:
        templates.append(_generate_random_template(n, seed=45))
    except Exception:
        pass
        
    if not templates:
        templates.append(_generate_random_template(n, seed=0))
        
    # 2. Basin Hopping Strategy
    # For each template, run a short basin hopping sequence.
    
    rng = np.random.default_rng(12345)
    
    for t_idx, template in enumerate(templates):
        if _get_remaining_time() < 15.0:
            break
            
        # Check initial validity
        min_dist = _min_distance(template)
        if min_dist < 0.5:
            continue
            
        # Initialize current state
        try:
            current_energy, current_positions = relax(template, potential, rho)
        except Exception:
            continue
            
        # Check validity of relaxed structure
        min_dist = _min_distance(current_positions)
        if min_dist < 0.5:
            continue
            
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_positions.copy()
            
        # Basin Hopping loop
        step_size = 0.3
        temp = 1.0
        max_bh_steps = 100 # Safety cap, but time is the real limit
        
        for step in range(max_bh_steps):
            if _get_remaining_time() < 5.0:
                break
                
            # Perturb
            noise = rng.normal(0, step_size, current_positions.shape)
            perturbed = current_positions + noise
            
            # Relax
            try:
                new_energy, new_positions = relax(perturbed, potential, rho)
            except Exception:
                continue
                
            # Check validity
            min_dist = _min_distance(new_positions)
            if min_dist < 0.5:
                continue
                
            # Metropolis Acceptance
            if new_energy < current_energy:
                accept = True
            else:
                delta_e = new_energy - current_energy
                if temp > 1e-6:
                    accept_prob = math.exp(-delta_e / temp)
                    accept = rng.random() < accept_prob
                else:
                    accept = False
                    
            if accept:
                current_energy = new_energy
                current_positions = new_positions
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_positions.copy()
                    
                # Anneal
                temp *= 0.95
                step_size *= 0.98
            else:
                # Even if rejected, we might want to cool down
                temp *= 0.98
                step_size *= 0.99
                
    if best_positions is None:
        # Absolute fallback
        best_positions = templates[0]
        
    return best_positions
# EVOLVE-BLOCK-END
