"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _hexagonal_lattice():
    """Generate hexagonal lattice points inside unit disk."""
    # Try different spacings to find the best hexagonal packing
    best_r = 0.0
    best_points = None
    
    for spacing in np.linspace(0.04, 0.12, 100):
        # Generate hexagonal lattice
        a = spacing
        h = a * np.sqrt(3) / 2
        
        points = []
        for j in range(-50, 51):
            for i in range(-50, 51):
                x = (i + 0.5 * (j % 2)) * a
                y = j * h
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            continue
        
        # Keep only points that fit with the spacing
        mask = (1.0 - np.linalg.norm(points, axis=1)) >= a/2
        points = points[mask]
        
        if len(points) < N:
            continue
        
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        selected = points[idx]
        
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_points = selected.copy()
    
    return best_points, best_r

def _concentric_rings():
    """Generate concentric ring packing."""
    best_r = 0.0
    best_points = None
    
    # Try different numbers of rings
    for n_rings in range(3, 20):
        # Distribute N points among rings
        # First ring has 1 point (center), rest distributed
        remaining = N - 1
        # Distribute remaining points among n_rings-1 rings
        # Use proportional distribution
        radii = np.linspace(0.1, 0.95, n_rings-1)
        counts = np.zeros(n_rings-1, dtype=int)
        
        # Distribute points proportional to circumference
        circumferences = 2 * np.pi * radii
        total_circ = np.sum(circumferences)
        counts = np.round(remaining * circumferences / total_circ).astype(int)
        
        # Adjust to ensure sum is correct
        while np.sum(counts) < remaining:
            counts[np.argmax(circumferences - counts)] += 1
        while np.sum(counts) > remaining:
            idx = np.argmax(counts - circumferences)
            if counts[idx] > 0:
                counts[idx] -= 1
            else:
                break
        
        points = [[0.0, 0.0]]
        for ring_idx, count in enumerate(counts):
            radius = radii[ring_idx]
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for angle in angles:
                points.append([radius * np.cos(angle), radius * np.sin(angle)])
        
        points = np.array(points)
        if len(points) != N:
            continue
        
        r = _compute_radius(points)
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    return best_points, best_r

def _optimize_hexagonal(points, max_iters=50):
    """Optimize hexagonal packing using gradient-free optimization."""
    if points is None:
        return None, 0.0
    
    current_r = _compute_radius(points)
    
    # Try to improve by moving points slightly
    best_points = points.copy()
    best_r = current_r
    
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            # Try small perturbations
            for dx in np.linspace(-0.005, 0.005, 5):
                for dy in np.linspace(-0.005, 0.005, 5):
                    new_points = best_points.copy()
                    new_points[i] = [best_points[i][0] + dx, best_points[i][1] + dy]
                    
                    # Check if still inside unit circle
                    if np.linalg.norm(new_points[i]) > 1.0:
                        continue
                    
                    r = _compute_radius(new_points)
                    if r > best_r + 1e-10:
                        best_r = r
                        best_points = new_points
                        improved = True
        
        if not improved:
            break
    
    return best_points, best_r

def construct():
    """Construct the optimal packing."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with hexagonal lattice
    hex_points, hex_r = _hexagonal_lattice()
    
    # Try concentric rings
    ring_points, ring_r = _concentric_rings()
    
    # Choose the better starting point
    if hex_r >= ring_r:
        points = hex_points
        best_r = hex_r
    else:
        points = ring_points
        best_r = ring_r
    
    if points is None:
        # Fallback to simple grid
        spacing = 0.06
        points = []
        for i in range(-17, 18):
            for j in range(-17, 18):
                x = i * spacing
                y = j * spacing
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            points = points[idx]
        best_r = _compute_radius(points)
    
    # Optimize if we have time
    if time.time() < deadline - 30:
        points, best_r = _optimize_hexagonal(points, max_iters=30)
    
    # Final optimization pass if time allows
    if time.time() < deadline - 10:
        # Try more aggressive optimization
        for _ in range(20):
            if time.time() >= deadline - 5:
                break
            
            improved = False
            for i in range(N):
                if time.time() >= deadline - 5:
                    break
                
                # Try larger perturbations
                for dx in np.linspace(-0.01, 0.01, 9):
                    for dy in np.linspace(-0.01, 0.01, 9):
                        new_points = points.copy()
                        new_points[i] = [points[i][0] + dx, points[i][1] + dy]
                        
                        if np.linalg.norm(new_points[i]) > 1.0:
                            continue
                        
                        r = _compute_radius(new_points)
                        if r > best_r + 1e-10:
                            best_r = r
                            points = new_points
                            improved = True
            
            if not improved:
                break
    
    return points
# EVOLVE-BLOCK-END
