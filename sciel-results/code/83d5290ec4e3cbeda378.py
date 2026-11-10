import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_PARAMS = {
    "12-6": (12, 6, 144.41),
    "9-6":  (9, 6, 39.432),
    "10-8": (10, 8, 34.408),
}

_START_TIME = time.time()
_DEADLINE = 325.0 # Safety margin

def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_fcc_seed(n, rng):
    """Generate a compact FCC fragment."""
    # FCC lattice vectors
    # We generate a grid of FCC points
    # Point (i, j, k) is in FCC if i+j+k is even? No, that's BCC?
    # FCC points: (0,0,0), (0,1,1), (1,0,1), (1,1,0) + integer translations.
    # Let's just generate a dense grid and filter.
    # A simpler way: generate points in a sphere, then project to FCC lattice?
    # Or just generate a large FCC block and cut.
    
    # Lattice constant a = 1.0 (will be scaled by relax)
    # Generate indices
    max_idx = int(np.ceil(n ** (1/3) * 1.5)) + 5
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC condition: i, j, k all even or all odd? No.
                # FCC primitive vectors: a/2(1,1,0), a/2(1,0,1), a/2(0,1,1)
                # So points are (i+j, i+k, j+k) * 0.5?
                # Let's use the standard: (i, j, k) where i,j,k are integers and i+j+k is even.
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * 0.5
                    if np.linalg.norm(p) < max_idx: # Sphere cut
                        points.append(p)
    
    points = np.array(points)
    if len(points) < n:
        # Fallback to random if generation failed (unlikely)
        return rng.uniform(-2, 2, (n, 3))
    
    # Sort by distance from center
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    selected = points[indices[:n]]
    
    # Center the cluster
    selected -= np.mean(selected, axis=0)
    
    # Random rotation to break symmetry
    # Generate random rotation matrix
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    selected = selected @ rot.T
    
    return selected

def _generate_hcp_seed(n, rng):
    """Generate a compact HCP fragment."""
    # HCP: ABAB stacking
    # Layer A: hexagonal grid
    # Layer B: hexagonal grid shifted
    max_idx = int(np.ceil(n ** (1/3) * 1.5)) + 5
    points = []
    a = 1.0
    c = 1.633 * a # Ideal HCP c/a ratio
    
    for l in range(-max_idx, max_idx + 1):
        z = l * c
        # Hexagonal grid in xy plane
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                # Hex grid coordinates
                x = a * (i + 0.5 * (j % 2))
                y = a * (np.sqrt(3) / 2) * j
                if l % 2 == 0: # Layer A
                    x += 0
                else: # Layer B
                    x += a / 2
                    y += a * np.sqrt(3) / 6
                
                p = np.array([x, y, z])
                if np.linalg.norm(p) < max_idx * a:
                    points.append(p)
    
    points = np.array(points)
    if len(points) < n:
        return rng.uniform(-2, 2, (n, 3))
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    selected = points[indices[:n]]
    selected -= np.mean(selected, axis=0)
    
    # Random rotation
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    selected = selected @ rot.T
    
    return selected

def _generate_random_compact_seed(n, rng):
    """Generate a random compact seed."""
    radius = 1.0 * n ** (1/3)
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) < 0.8: # Too close
                    continue
            points.append(p)
    return np.array(points)

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    rng = np.random.default_rng(instance["n"])
    
    seeds = []
    
    # 1. FCC
    seeds.append(_generate_fcc_seed(n, rng))
    # 2. HCP
    seeds.append(_generate_hcp_seed(n, rng))
    # 3. Random Compact
    seeds.append(_generate_random_compact_seed(n, rng))
    
    best_energy = np.inf
    best_positions = seeds[0] # Fallback
    
    for i, seed in enumerate(seeds):
        if _remaining_time() < 5.0:
            break
        
        try:
            # Use relax which does L-BFGS-B
            energy, positions = relax(seed, 'sutton_chen', kind)
            
            # Validate
            if not np.all(np.isfinite(positions)):
                continue
            if np.min(np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)) < 0.35:
                # Check unique pairs
                for a in range(n):
                    for b in range(a+1, n):
                        if np.linalg.norm(positions[a] - positions[b]) < 0.35:
                            raise ValueError("Too close")
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                
        except Exception:
            continue
            
    return best_positions
