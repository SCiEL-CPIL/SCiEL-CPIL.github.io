import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11
MIN_DIST = np.sqrt(3)
APOTHEM_INNER = np.sqrt(3) / 2.0

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def get_hull_area(points):
        try:
            hull = ConvexHull(points)
            return hull.volume # In 2D, volume is area
        except:
            return 100.0 # Penalize degenerate hulls
            
    def cost_func(params):
        points = params.reshape(-1, 2)
        
        # Center the points
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Calculate hull area
        area = get_hull_area(points_centered)
        
        # Also calculate the max projection to guide towards hexagonal shape
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        
        # Combine: Minimize hull area and max projection
        # The max projection is directly related to the enclosing hexagon size
        # L_outer = 2 * (max_proj + APOTHEM_INNER) / sqrt(3)
        # So minimizing max_proj is the most direct objective.
        # But hull area helps keep the shape compact.
        
        return max_proj + 0.1 * area + 100.0 * penalty

    best_params = None
    best_max_proj = float('inf')
    
    # Initializations
    inits = []
    
    # 1. Hexagonal lattice
    pts = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = 1.5 * q
            y = np.sqrt(3)/2 * q + np.sqrt(3) * r
            pts.append([x, y])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)[:N]
    init_pts = pts[indices]
    inits.append(init_pts)
    
    # 2. Random
    for _ in range(5):
        inits.append(np.random.uniform(-3, 3, (N, 2)))
        
    # 3. Circle
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([2.0 * np.cos(theta), 2.0 * np.sin(theta)]))

    for restart, init_pts in enumerate(inits):
        # Pre-relaxation: simple gradient step to reduce overlaps
        pts = init_pts.copy()
        for step in range(50):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = pts[i] - pts[j]
                    d = np.linalg.norm(d_vec)
                    if d < MIN_DIST and d > 1e-6:
                        f = (MIN_DIST - d) / d
                        forces[i] += f * d_vec
                        forces[j] -= f * d_vec
            pts += 0.1 * forces
            # Center
            pts -= np.mean(pts, axis=0)
            
        initial_params = pts.flatten()
        
        # Optimize
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        final_pts = res.x.reshape(-1, 2)
        centroid = np.mean(final_pts, axis=0)
        final_pts_centered = final_pts - centroid
        
        # Check validity
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(final_pts_centered[i] - final_pts_centered[j])
                if d < MIN_DIST - 1e-5:
                    valid = False
                    break
                if not valid:
                    break
                    
        if valid:
            projections = np.abs(final_pts_centered @ normals.T)
            max_proj = np.max(projections)
            
            if max_proj < best_max_proj:
                best_max_proj = max_proj
                best_params = res.x

    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_pts = best_params.reshape(-1, 2)
    # Center the final points for the output
    centroid = np.mean(inner_pts, axis=0)
    inner_pts = inner_pts - centroid
    
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    # Calculate outer side length
    # L_outer = 2 * (max_proj + APOTHEM_INNER) / sqrt(3)
    # best_max_proj is the max projection of the centered points.
    outer_side = 2.0 * (best_max_proj + APOTHEM_INNER) / np.sqrt(3)
    
    return inner, (0.0, 0.0), outer_side, 0.0
