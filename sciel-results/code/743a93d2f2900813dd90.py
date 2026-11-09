# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0
                areas[i, j, k] = area
    return areas

def _score(pts):
    """Compute A_min / A_hull."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
        areas = _triangle_areas(pts)
        # Only consider valid triples
        n = len(pts)
        valid_areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    valid_areas.append(areas[i, j, k])
        min_area = min(valid_areas)
        return min_area / hull_area
    except Exception:
        return 0.0

def _objective(pts_flat, beta):
    """Objective function to minimize: -soft_min of triangle areas + penalty for hull area."""
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0
                areas[i, j, k] = area
    valid_areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                valid_areas.append(areas[i, j, k])
    valid_areas = np.array(valid_areas)
    # Soft minimum
    soft_min = -np.log(np.sum(np.exp(-beta * valid_areas))) / beta
    # Hull area penalty (we want to maximize hull area relative to min triangle)
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        hull_area = 1.0
    # We want to maximize min_area / hull_area, so minimize -min_area/hull_area
    # Approximate: minimize soft_min + lambda * hull_area (since we want large hull)
    # Actually, the metric is A_min/A_hull, so we want to maximize that.
    # Let's use: minimize -(A_min/A_hull) approximately
    # Use soft_min as proxy for A_min, and normalize by hull_area
    score = -soft_min / max(hull_area, 1e-10)
    return score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses multi-phase optimization with diverse initial configurations.
    """
    start_time = time.time()
    deadline = start_time + 620  # 10s safety margin
    
    n = 13
    best_points = None
    best_score = 0.0
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(seed=42)
    
    # Configuration 1: Hexagonal lattice
    hex_pts = np.array([[0, 0]])
    # Inner ring (6 points)
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        hex_pts = np.vstack([hex_pts, [[0.5*np.cos(angle), 0.5*np.sin(angle)]]])
    # Outer ring (6 points)
    for angle in np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False):
        hex_pts = np.vstack([hex_pts, [[1.0*np.cos(angle), 1.0*np.sin(angle)]]])
    hex_pts = (hex_pts - hex_pts.mean(axis=0))
    hex_pts = hex_pts / np.max(np.abs(hex_pts)) * 0.5
    
    # Configuration 2: Regular polygon + center
    poly_pts = []
    poly_pts.append([0, 0])
    for i in range(6):
        angle = 2 * np.pi * i / 6
        poly_pts.append([0.5*np.cos(angle), 0.5*np.sin(angle)])
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi/6
        poly_pts.append([1.0*np.cos(angle), 1.0*np.sin(angle)])
    poly_pts = np.array(poly_pts)
    poly_pts = (poly_pts - poly_pts.mean(axis=0))
    poly_pts = poly_pts / np.max(np.abs(poly_pts)) * 0.5
    
    # Configuration 3: Random with structure
    rand_pts = rng.random((13, 2))
    rand_pts = (rand_pts - rand_pts.mean(axis=0))
    rand_pts = rand_pts / np.max(np.abs(rand_pts)) * 0.5
    
    # Configuration 4: Grid-like
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([i - 1.5, j - 1.5])
    grid_pts = np.array(grid_pts[:13])
    grid_pts = (grid_pts - grid_pts.mean(axis=0))
    grid_pts = grid_pts / np.max(np.abs(grid_pts)) * 0.5
    
    # Configuration 5: Another hexagonal variant
    hex2_pts = np.array([[0, 0]])
    for ring in [0.4, 0.8]:
        num = 6
        for angle in np.linspace(0, 2*np.pi, num, endpoint=False) + ring * 0.1:
            hex2_pts = np.vstack([hex2_pts, [[ring*np.cos(angle), ring*np.sin(angle)]]])
    hex2_pts = hex2_pts[:13]
    hex2_pts = (hex2_pts - hex2_pts.mean(axis=0))
    hex2_pts = hex2_pts / np.max(np.abs(hex2_pts)) * 0.5
    
    initial_configs = [hex_pts, poly_pts, rand_pts, grid_pts, hex2_pts]
    
    # Add more random configurations
    for _ in range(10):
        rpts = rng.random((13, 2))
        rpts = (rpts - rpts.mean(axis=0))
        rpts = rpts / np.max(np.abs(rpts)) * 0.5
        initial_configs.append(rpts)
    
    # Multi-phase optimization
    betas = [10, 50, 200, 1000, 5000]
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        current_pts = config.copy()
        current_flat = current_pts.flatten()
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            # Use L-BFGS-B
            try:
                result = minimize(
                    _objective,
                    current_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if result.success:
                    new_pts = result.x.reshape(-1, 2)
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts.copy()
                    current_flat = result.x
            except Exception:
                pass
        
        # Local refinement with small perturbations
        if best_points is not None:
            if time.time() > deadline:
                break
            for _ in range(20):
                if time.time() > deadline:
                    break
                perturbed = best_points + rng.normal(0, 0.01, (13, 2))
                perturbed_score = _score(perturbed)
                if perturbed_score > best_score:
                    best_score = perturbed_score
                    best_points = perturbed.copy()
    
    # Final polish
    if best_points is not None and time.time() < deadline:
        for beta in [10000, 50000]:
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    _objective,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15}
                )
                if result.success:
                    new_pts = result.x.reshape(-1, 2)
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: return hexagonal lattice
        best_points = hex_pts
    
    # Ensure valid output
    best_points = np.array(best_points, dtype=np.float64)
    if not np.all(np.isfinite(best_points)):
        best_points = hex_pts
    
    return best_points

# EVOLVE-BLOCK-END
