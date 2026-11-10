# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 620  # Leave 10s margin from 630s

    # Precompute indices for all triples of points
    idx = np.array(list(itertools.combinations(range(n), 3)))

    def min_triangle_area_vectorized(points):
        """
        Calculate the minimum triangle area among all triples of points using vectorization.
        points: (n, 2)
        """
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        return np.min(areas)

    def hull_area(points):
        """Calculate the convex hull area of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def objective_exact(points):
        """
        Exact objective: minimize -A_min / A_hull.
        Uses actual ConvexHull area.
        """
        A_hull = hull_area(points)
        if A_hull <= 1e-10:
            return 0.0
        
        A_min = min_triangle_area_vectorized(points)
        
        if A_min < 1e-12:
            return 0.0
            
        return -A_min / A_hull

    def score(points):
        """Calculate the actual score: A_min / A_hull"""
        A_hull = hull_area(points)
        if A_hull <= 1e-10:
            return 0.0
        A_min = min_triangle_area_vectorized(points)
        if A_min < 1e-12:
            return 0.0
        return A_min / A_hull

    def physics_relaxation(points, n_iter=200, dt=0.01, repulse_strength=1.0, laplacian_strength=0.5):
        """
        Physics-based relaxation: Laplacian smoothing + repulsive forces to promote even distribution.
        This helps escape local minima in the nonsmooth min-area landscape.
        """
        points = points.copy()
        n = len(points)
        
        for it in range(n_iter):
            forces = np.zeros_like(points)
            
            # Repulsive forces between all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    diff = points[i] - points[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-10:
                        # Jitter to avoid singularity
                        diff = np.array([rng.normal(0, 0.01), rng.normal(0, 0.01)])
                        dist_sq = np.dot(diff, diff)
                    dist = np.sqrt(dist_sq)
                    force_mag = repulse_strength / (dist_sq + 0.01)
                    force_vec = force_mag * diff / dist
                    forces[i] += force_vec
                    forces[j] -= force_vec
            
            # Laplacian smoothing: pull each point toward the average of its neighbors
            # Use Delaunay triangulation to find neighbors
            try:
                tri = Delaunay(points)
                for i in range(n):
                    neighbors = []
                    for simplex in tri.simplices:
                        if i in simplex:
                            for j in simplex:
                                if j != i:
                                    neighbors.append(j)
                    if neighbors:
                        avg_neighbor = np.mean(points[neighbors], axis=0)
                        forces[i] += laplacian_strength * (avg_neighbor - points[i])
            except Exception:
                pass
            
            # Update positions
            points += dt * forces
            
            # Center the points
            center = np.mean(points, axis=0)
            points -= center
            
            # Normalize scale
            norm = np.linalg.norm(points)
            if norm > 1e-10:
                points = points / norm * 1.0
        
        return points

    def delaunay_optimize(points, n_iter=100, step_size=0.01):
        """
        Optimize using Delaunay triangulation: move internal vertices to maximize 
        the minimum circumradius of adjacent triangles.
        """
        points = points.copy()
        n = len(points)
        
        for it in range(n_iter):
            try:
                tri = Delaunay(points)
                hull = ConvexHull(points)
                hull_points = set(hull.vertices.tolist())
                
                for i in range(n):
                    if i in hull_points:
                        continue  # Only move internal points
                    
                    # Find triangles containing point i
                    adjacent_triangles = []
                    for simplex in tri.simplices:
                        if i in simplex:
                            adjacent_triangles.append(simplex)
                    
                    if not adjacent_triangles:
                        continue
                    
                    # Compute circumcenters of adjacent triangles
                    circumcenters = []
                    for simplex in adjacent_triangles:
                        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                        # Circumcenter calculation
                        ax, ay = p1
                        bx, by = p2
                        cx, cy = p3
                        
                        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                        if abs(d) < 1e-10:
                            continue
                        
                        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
                        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
                        
                        circumcenters.append(np.array([ux, uy]))
                    
                    if circumcenters:
                        # Move toward the average circumcenter (farthest-point Voronoi property)
                        avg_cc = np.mean(circumcenters, axis=0)
                        points[i] += step_size * (avg_cc - points[i])
                        
                        # Keep within bounds
                        points[i] = np.clip(points[i], -2, 2)
            except Exception:
                pass
            
            # Center and normalize
            center = np.mean(points, axis=0)
            points -= center
            norm = np.linalg.norm(points)
            if norm > 1e-10:
                points = points / norm * 1.0
        
        return points

    # Generate diverse initial configurations
    rng = np.random.default_rng(seed=42)
    
    initial_configs = []
    
    # 1. Regular polygon (baseline)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular polygon with center point and perturbed outer ring
    center_pt = np.array([0.0, 0.0])
    outer_angles = np.linspace(0, 2*np.pi, n-1, endpoint=False)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    initial_configs.append(np.vstack([outer_pts, [center_pt]]))
    
    # 3. Hexagonal lattice pattern
    hex_configs = []
    for offset in [0.0, 0.1, 0.2]:
        pts = []
        pts.append([0.0, 0.0])
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
            r = 1.0 + offset * rng.random()
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6:
            r = 1.8 + offset * rng.random()
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        hex_configs.append(np.array(pts[:13]))
    initial_configs.extend(hex_configs)
    
    # 4. Perturbed regular polygon
    for _ in range(3):
        perturbation = rng.normal(0, 0.15, (n, 2))
        initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]) + perturbation)
        
    # 5. Random points in unit circle
    for _ in range(5):
        r = np.sqrt(rng.random(n))
        theta = rng.random(n) * 2 * np.pi
        initial_configs.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 6. Grid-like with jitter
    grid_x = [-1.5, -0.5, 0.5, 1.5]
    grid_y = [-1.5, -0.5, 0.5, 1.5]
    for _ in range(3):
        pts = []
        used = set()
        while len(pts) < n:
            i = rng.integers(0, 4)
            j = rng.integers(0, 4)
            if (i, j) not in used:
                used.add((i, j))
                x = grid_x[i] + rng.normal(0, 0.1)
                y = grid_y[j] + rng.normal(0, 0.1)
                pts.append([x, y])
        initial_configs.append(np.array(pts))

    best_points = None
    best_score = -float('inf')
    
    # First pass: physics-based relaxation + Delaunay optimization + Nelder-Mead polish
    for i, init_points in enumerate(initial_configs):
        if time.time() - start_time > time_limit - 60:
            break
            
        # Center and normalize the initial points
        center = np.mean(init_points, axis=0)
        centered = init_points - center
        norm = np.linalg.norm(centered)
        if norm > 1e-10:
            centered = centered / norm * 1.0
        
        # Apply physics-based relaxation
        relaxed = physics_relaxation(centered, n_iter=100, dt=0.02, repulse_strength=0.5, laplacian_strength=0.3)
        
        # Apply Delaunay optimization
        delaunay_opt = delaunay_optimize(relaxed, n_iter=50, step_size=0.02)
        
        # Polish with Nelder-Mead
        try:
            result = minimize(
                objective_exact,
                delaunay_opt,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': 0.1
                }
            )
            
            if result.fun < 0:
                exact_score = -result.fun
                if exact_score > best_score:
                    best_score = exact_score
                    best_points = result.x.copy()
                    
        except Exception:
            pass

    # Second pass: refine best points with smaller simplex
    if best_points is not None and time.time() - start_time < time_limit - 30:
        for simplex_size in [0.05, 0.02, 0.01]:
            if time.time() - start_time > time_limit - 15:
                break
            try:
                result = minimize(
                    objective_exact,
                    best_points,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-11,
                        'fatol': 1e-13,
                        'adaptive': True,
                        'initial_simplex': simplex_size
                    }
                )
                
                if result.fun < 0:
                    exact_score = -result.fun
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
            except Exception:
                pass

    # Final fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
