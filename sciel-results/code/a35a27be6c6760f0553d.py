# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Uses a geometric search strategy:
    1. Initialize points on the unit sphere using a deterministic symmetric seed (icosahedron + 2 points).
    2. Perform Simulated Annealing on the sphere to maximize the minimum distance (which maximizes the ratio when constrained to sphere).
    3. The ratio (d_min/d_max)^2 is maximized when points are as evenly distributed as possible on a sphere.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial configuration: Icosahedron (12 points) + 2 points
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 points. To maintain symmetry, we can try placing them at (0,0,1) and (0,0,-1)
    # but that might be too close to ico vertices. 
    # Better: Start with a random symmetric perturbation or just random on sphere.
    # Let's try a few different initial seeds.
    
    seeds = []
    
    # Seed 1: Icosahedron + 2 opposite points on Z-axis
    p1 = np.vstack([ico, np.array([[0, 0, 1], [0, 0, -1]])])
    p1 = p1 / np.linalg.norm(p1, axis=1, keepdims=True)
    seeds.append(p1)
    
    # Seed 2: Icosahedron + 2 points rotated
    rot = np.array([[1, 0, 0], [0, np.cos(np.pi/5), -np.sin(np.pi/5)], [0, np.sin(np.pi/5), np.cos(np.pi/5)]])
    extra = np.array([[0, 0, 1], [0, 0, -1]]) @ rot
    p2 = np.vstack([ico, extra])
    p2 = p2 / np.linalg.norm(p2, axis=1, keepdims=True)
    seeds.append(p2)
    
    # Seed 3: Random on sphere
    for _ in range(5):
        pts = np.random.randn(n, d)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
        
    best_points = None
    best_score = -np.inf
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Simulated Annealing on the sphere
        # Objective: Maximize d_min. Since d_max is roughly constant (diameter of sphere ~2),
        # maximizing d_min maximizes the ratio.
        # Actually, we should maximize (d_min/d_max)^2. 
        # On the unit sphere, d_max <= 2. d_min is the bottleneck.
        
        current_points = seed.copy()
        current_score = _score(current_points)
        
        # Annealing parameters
        T = 1.0
        T_min = 1e-6
        cooling = 0.9995
        steps_per_temp = 100
        
        for step in range(20000):
            if time.time() > deadline:
                break
                
            # Perturb a random point slightly on the sphere
            i = np.random.randint(n)
            # Generate a small random vector
            delta = np.random.randn(d) * 0.1
            new_point = current_points[i] + delta
            new_point = new_point / np.linalg.norm(new_point)
            
            new_points = current_points.copy()
            new_points[i] = new_point
            
            new_score = _score(new_points)
            
            # Acceptance probability
            delta_score = new_score - current_score
            if delta_score > 0:
                current_points = new_points
                current_score = new_score
            else:
                prob = np.exp(delta_score / T)
                if np.random.rand() < prob:
                    current_points = new_points
                    current_score = new_score
            
            # Update global best
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            
            # Cool down
            T *= cooling
            if T < T_min:
                break
        
    # If no best found (shouldn't happen), return seed
    if best_points is None:
        best_points = seeds[0]
        
    # Final polish: Local optimization using gradient-free method on the sphere
    # We can use a simple hill climbing step
    if time.time() < deadline - 5:
        current_points = best_points.copy()
        current_score = _score(current_points)
        
        for step in range(5000):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                # Try small moves in 3 directions
                for axis in range(3):
                    for sign in [1, -1]:
                        new_point = current_points[i].copy()
                        new_point[axis] += sign * 1e-3
                        new_point = new_point / np.linalg.norm(new_point)
                        
                        new_points = current_points.copy()
                        new_points[i] = new_point
                        new_score = _score(new_points)
                        
                        if new_score > current_score + 1e-10:
                            current_points = new_points
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        if current_score > best_score:
            best_points = current_points
            best_score = current_score

    return best_points

# EVOLVE-BLOCK-END
