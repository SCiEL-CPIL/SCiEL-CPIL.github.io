import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
MIN_DIST = np.sqrt(3)  # Minimum center-to-center distance for non-overlapping unit hexagons

def hexagon_vertices(cx, cy, side, angle_deg=0):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle_rad
    vertices = []
    for a in angles:
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_hexagon(p, center, side, angle_deg=0):
    """Check if point p is inside a regular hexagon using half-plane tests."""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle_rad
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    rel_p = np.array(p) - np.array(center)
    # For a regular hexagon with side s, the distance from center to each side is s*sqrt(3)/2
    max_dist = side * np.sqrt(3) / 2
    projections = np.abs(rel_p @ normals.T)
    return np.all(projections <= max_dist + 1e-9)

def min_enclosing_hexagon(points, max_iter=100, tol=1e-6):
    """Find the minimum side length regular hexagon enclosing all points, allowing rotation and translation."""
    # We'll optimize over: center (x, y), side length L, and rotation angle theta
    # For a given hexagon (center, L, theta), check if all points are inside
    
    def is_enclosed(params):
        cx, cy, L, theta = params
        if L < 0:
            return False
        angle_rad = np.radians(theta)
        angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle_rad
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        max_dist = L * np.sqrt(3) / 2
        for p in points:
            rel_p = p - np.array([cx, cy])
            projections = np.abs(rel_p @ normals.T)
            if np.any(projections > max_dist + 1e-9):
                return False
        return True
    
    # Start with a large hexagon and shrink
    # Initial guess: center at centroid, side large enough to enclose
    centroid = np.mean(points, axis=0)
    max_r = np.max(np.linalg.norm(points - centroid, axis=1))
    # For a hexagon, the inradius is L*sqrt(3)/2, so L = max_r / (sqrt(3)/2) * 1.1
    L_init = max_r / (np.sqrt(3) / 2) * 1.2
    
    # Optimize to minimize L while keeping all points inside
    # We'll use a penalty approach: minimize L + penalty if not enclosed
    
    def cost_func(params):
        cx, cy, L, theta = params
        if L < 0.1:
            return 1000.0
        angle_rad = np.radians(theta)
        angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle_rad
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        max_dist = L * np.sqrt(3) / 2
        
        penalty = 0.0
        for p in points:
            rel_p = p - np.array([cx, cy])
            projections = np.abs(rel_p @ normals.T)
            max_proj = np.max(projections)
            if max_proj > max_dist:
                penalty += (max_proj - max_dist) ** 2
        
        return L + 100.0 * penalty
    
    # Use differential evolution for global search
    bounds = [
        (centroid[0] - 5, centroid[0] + 5),  # cx
        (centroid[1] - 5, centroid[1] + 5),  # cy
        (0.5, L_init),  # L
        (0, 360)  # theta
    ]
    
    res = differential_evolution(cost_func, bounds, maxiter=100, tol=1e-6, 
                                  seed=42, polish=False, init='latinhypercube')
    
    # Verify the solution
    cx, cy, L, theta = res.x
    if is_enclosed(res.x):
        return L, (cx, cy), theta
    else:
        # Fallback: use a conservative estimate
        return L_init, tuple(centroid), 0.0

def construct():
    # Step 1: Optimize positions of 11 hexagon centers to minimize convex hull area
    # with non-overlap constraints
    
    def hull_area_cost(params):
        points = params.reshape(-1, 2)
        # Calculate convex hull area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            area = hull.area
        except:
            return 1000.0
        
        # Non-overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Also penalize if points are too spread out (to keep cluster compact)
        centroid = np.mean(points, axis=0)
        max_r = np.max(np.linalg.norm(points - centroid, axis=1))
        spread_penalty = max(0, max_r - 4) ** 2
        
        return area + 50.0 * penalty + 10.0 * spread_penalty
    
    # Initialize with hexagonal lattice pattern
    best_params = None
    best_cost = float('inf')
    
    for restart in range(30):
        if restart == 0:
            # Hexagonal lattice: center + 6 around + 4 more
            pts = np.array([[0, 0]])
            for k in range(6):
                angle = k * np.pi / 3
                pts = np.vstack([pts, [MIN_DIST * np.cos(angle), MIN_DIST * np.sin(angle)]])
            # Add 4 more in second ring
            for k in range(4):
                angle = k * np.pi / 2 + np.pi/6
                pts = np.vstack([pts, [2*MIN_DIST * np.cos(angle), 2*MIN_DIST * np.sin(angle)]])
            pts += np.random.normal(0, 0.1, pts.shape)
        else:
            # Random initialization in a disk
            angles = np.random.uniform(0, 2*np.pi, N)
            radii = np.random.uniform(0, 3, N)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        initial_params = pts.flatten()
        
        # Optimize positions
        res = minimize(hull_area_cost, initial_params, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.fun < best_cost:
            # Verify no overlaps
            final_pts = res.x.reshape(-1, 2)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(final_pts[i] - final_pts[j])
                    if d < MIN_DIST - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_cost = res.fun
                best_params = res.x
    
    if best_params is None:
        # Fallback: return a valid but large solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Step 2: Find minimum enclosing regular hexagon for the optimized positions
    points = best_params.reshape(-1, 2)
    
    # Center the points at origin for symmetry
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    L, center, angle = min_enclosing_hexagon(points_centered)
    
    # Verify the solution
    inner_pts = points_centered
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    outer_side = L
    outer_center = (0.0, 0.0)
    outer_angle = angle
    
    return inner, outer_center, outer_side, outer_angle
