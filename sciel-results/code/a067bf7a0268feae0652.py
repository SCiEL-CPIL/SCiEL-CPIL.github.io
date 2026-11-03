"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

def _score(pts):
    """Compute the packing radius: min of pairwise distances / 2 and boundary distances."""
    if len(pts) != N:
        return 0.0
    # Boundary constraint
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(boundary))

def _force_displacement(pts, r_target, iters=100, learning_rate=0.01):
    """Simple force-directed relaxation to increase minimum distance."""
    pts = pts.copy()
    for _ in range(iters):
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum distance pair
        min_dist = np.min(dists)
        if min_dist >= 2 * r_target:
            break
        
        # Compute forces: repulsion for close pairs, attraction to center for boundary violations
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion for overlapping pairs
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] < 2 * r_target:
                    d = pts[i] - pts[j]
                    norm = np.linalg.norm(d)
                    if norm > 1e-10:
                        force = (2 * r_target - dists[i, j]) * d / norm
                        forces[i] += force
                        forces[j] -= force
        
        # Boundary repulsion
        radii = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if radii[i] > 1.0 - r_target:
                d = pts[i] / radii[i]
                forces[i] -= (radii[i] - (1.0 - r_target)) * d
        
        # Apply forces
        pts += learning_rate * forces
        # Keep points within bounds
        pts[np.linalg.norm(pts, axis=1) > 1.0] = pts[np.linalg.norm(pts, axis=1) > 1.0] / np.linalg.norm(pts[np.linalg.norm(pts, axis=1) > 1.0], axis=1)[:, np.newaxis]
    
    return pts

def _delauany_swap_search(pts, max_swaps=500):
    """Use Delaunay triangulation to identify local neighborhoods and try swaps."""
    pts = pts.copy()
    current_score = _score(pts)
    
    for _ in range(max_swaps):
        try:
            tri = Delaunay(pts)
        except:
            break
        
        # Get simplexes (triangles) from Delaunay triangulation
        simplices = tri.simplices
        
        # For each simplex, try swapping one vertex with a nearby vertex outside the simplex
        improved = False
        for simplex in simplices:
            i, j, k = simplex
            # Find vertices near the triangle but not part of it
            centroid = (pts[i] + pts[j] + pts[k]) / 3.0
            distances = np.linalg.norm(pts - centroid, axis=1)
            distances[i] = np.inf
            distances[j] = np.inf
            distances[k] = np.inf
            
            # Get the closest 3 candidates for swap
            candidates = np.argsort(distances)[:3]
            
            for c in candidates:
                # Try swapping i with c
                pts_test = pts.copy()
                pts_test[i] = pts_test[c]
                new_score = _score(pts_test)
                if new_score > current_score:
                    pts = pts_test
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s buffer for safety
    
    # Generate multiple initial configurations
    best_pts = None
    best_score = 0.0
    
    # Configuration 1: Hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            hex_pts = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    hex_score = _score(hex_pts)
    if hex_score > best_score:
        best_pts = hex_pts
        best_score = hex_score
    
    # Configuration 2: Random points
    np.random.seed(42)
    for seed in range(5):
        if time.time() > deadline:
            break
        
        np.random.seed(seed + 100)
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0, 1, N)
        rand_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Relax
        r_target = _score(rand_pts) * 0.9
        relaxed = _force_displacement(rand_pts, r_target, iters=50, learning_rate=0.005)
        
        score = _score(relaxed)
        if score > best_score:
            best_pts = relaxed
            best_score = score
    
    # Configuration 3: Ring-based initialization
    if time.time() < deadline:
        # Place points in concentric rings
        ring_count = 20
        pts = []
        for ring in range(ring_count):
            r = (ring + 1) / ring_count
            count = int(2 * np.pi * r * 15)
            if count < 1:
                count = 1
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            pts.append(ring_pts)
        
        ring_pts = np.vstack(pts)[:N]
        if len(ring_pts) == N:
            score = _score(ring_pts)
            if score > best_score:
                best_pts = ring_pts
                best_score = score
    
    # Refine best configuration using force relaxation
    if time.time() < deadline and best_pts is not None:
        r_target = best_score * 1.01
        refined = _force_displacement(best_pts, r_target, iters=200, learning_rate=0.005)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_pts = refined
            best_score = refined_score
    
    # Try Delaunay-based swap search
    if time.time() < deadline and best_pts is not None:
        swapped, swap_score = _delauany_swap_search(best_pts, max_swaps=200)
        if swap_score > best_score:
            best_pts = swapped
            best_score = swap_score
    
    # Final polish with tighter force relaxation
    if time.time() < deadline and best_pts is not None:
        r_target = best_score * 1.005
        polished = _force_displacement(best_pts, r_target, iters=500, learning_rate=0.001)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Validate
    if best_pts is None:
        # Fallback to hexagonal
        return hex_pts
    
    # Ensure valid
    radii = np.linalg.norm(best_pts, axis=1)
    if np.any(radii > 1.0):
        best_pts = best_pts / radii[:, np.newaxis]
    
    return best_pts
# EVOLVE-BLOCK-END
