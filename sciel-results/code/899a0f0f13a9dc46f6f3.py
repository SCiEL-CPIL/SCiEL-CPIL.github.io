"""Rewritten program for Dzugutov cluster optimization using Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

def _generate_fcc_cluster(n):
    """Generate a cluster of n atoms from an FCC lattice."""
    # FCC lattice points in a box
    points = []
    # Generate points in a cubic lattice
    # To get FCC, we take points where (i+j+k) is even? No, that's BCC.
    # FCC: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    # Let's generate a large grid of FCC points
    L = 10
    a = 1.0 # lattice constant
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # FCC basis
                for dx, dy, dz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    points.append([x, y, z])
    points = np.array(points)
    
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    # Take first n
    points = points[:n]
    
    # Center
    points -= np.mean(points, axis=0)
    
    return points

def _generate_random_cluster(n):
    """Generate a random cluster of n atoms in a sphere."""
    # Radius estimation
    # Volume ~ n * (atom volume)
    # Assume atom radius ~ 0.5
    # R ~ n^(1/3)
    R = 1.5 * n ** (1.0/3.0)
    
    points = []
    while len(points) < n:
        p = np.random.uniform(-R, R, 3)
        if np.linalg.norm(p) < R:
            # Check distance to existing points
            if all(np.linalg.norm(p - q) > 0.8 for q in points):
                points.append(p)
                
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points

def construct(instance):
    n = instance['n']
    start_time = time.time()
    # 300s total / 6 instances = 50s per instance.
    # Use 45s to be safe.
    deadline = start_time + 45.0
    
    # 1. Initialization
    candidates = []
    
    # FCC
    try:
        candidates.append(_generate_fcc_cluster(n))
    except Exception:
        pass
        
    # Random
    try:
        candidates.append(_generate_random_cluster(n))
    except Exception:
        pass
        
    # If no candidates, use random
    if not candidates:
        candidates.append(np.random.rand(n, 3) * 10)
        
    # 2. Relax candidates
    best_energy = float('inf')
    best_pos = None
    current_energy = float('inf')
    current_pos = None
    
    for pos in candidates:
        if time.time() > deadline - 10.0:
            break
        try:
            energy, relaxed_pos = relax(pos, 'dzugutov', None)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
            if energy < current_energy:
                current_energy = energy
                current_pos = relaxed_pos
        except Exception:
            pass
            
    if best_pos is None:
        # Fallback
        best_pos = candidates[0]
        current_pos = best_pos
        current_energy = float('inf')
        # Try to relax fallback
        try:
            current_energy, current_pos = relax(best_pos, 'dzugutov', None)
            best_energy = current_energy
            best_pos = current_pos
        except Exception:
            pass

    # 3. Basin Hopping
    # Adaptive sigma
    sigma = 0.2
    min_sigma = 0.05
    
    # Metropolis temperature
    T = 0.1
    
    while time.time() < deadline - 2.0:
        # Perturb
        noise = np.random.normal(0, sigma, size=current_pos.shape)
        perturbed_pos = current_pos + noise
        
        # Relax
        try:
            energy, relaxed_pos = relax(perturbed_pos, 'dzugutov', None)
        except Exception:
            continue
            
        # Acceptance
        delta = energy - current_energy
        accept = False
        if delta < 0:
            accept = True
        else:
            # Metropolis
            if T > 0 and np.random.rand() < np.exp(-delta / T):
                accept = True
                
        if accept:
            current_energy = energy
            current_pos = relaxed_pos
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
                
        # Anneal
        T *= 0.99
        if T < 0.01:
            T = 0.01
            
        # Adapt sigma?
        # If we accepted, maybe keep sigma. If rejected, decrease?
        # Simple: decrease slowly
        sigma *= 0.999
        if sigma < min_sigma:
            sigma = min_sigma
            
    return best_pos
# EVOLVE-BLOCK-END
