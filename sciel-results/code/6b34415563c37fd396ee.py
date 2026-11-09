import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    def compute_min_area(pts):
        """Compute minimum triangle area among all C(13,3)=286 triangles"""
        min_area = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    a = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                    if a < min_area:
                        min_area = a
        return min_area
    
    def convex_hull_area(pts):
        """Compute convex hull area"""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0
    
    def score(pts):
        """Compute the metric: A_min / A_hull"""
        hull_area = convex_hull_area(pts)
        if hull_area < 1e-10:
            return 0.0
        min_area = compute_min_area(pts)
        return min_area / hull_area
    
    def normalize_affine(pts, target_area=1.0):
        """Center and scale so convex hull area = target_area"""
        try:
            hull = ConvexHull(pts)
            current_area = hull.volume
            if current_area < 1e-10:
                return pts
            centroid = np.mean(pts, axis=0)
            pts_centered = pts - centroid
            scale = np.sqrt(target_area / current_area)
            return pts_centered * scale
        except Exception:
            return pts
    
    def is_valid(pts):
        """Check validity"""
        if not np.all(np.isfinite(pts)):
            return False
        if pts.shape != (n, 2):
            return False
        try:
            hull = ConvexHull(pts)
            return hull.volume > 1e-8
        except Exception:
            return False
    
    def objective(p):
        pts = p.reshape(n, 2)
        return -compute_min_area(pts)
    
    # Generate initial configurations
    def generate_configs():
        configs = []
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        # Regular polygons at various radii
        for r in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
            configs.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
        
        # Random in unit disk
        for seed in range(30):
            np.random.seed(seed * 7 + 3)
            r = np.sqrt(np.random.rand(n))
            theta = np.random.rand(n) * 2 * np.pi
            configs.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
        # Perturbed regular
        for seed in range(15):
            np.random.seed(seed * 13 + 1)
            base = np.column_stack([np.cos(angles), np.sin(angles)])
            noise = np.random.randn(n, 2) * 0.15
            configs.append(base + noise)
        
        return configs
    
    best_points = None
    best_score = -1.0
    
    configs = generate_configs()
    
    for config in configs:
        p0 = config.flatten()
        
        if not is_valid(config):
            continue
        
        # SLSQP with circle constraints
        cons = []
        for i in range(n):
            def circle_con(p, idx=i):
                return 1.0 - p[2*idx]**2 - p[2*idx+1]**2
            cons.append({'type': 'ineq', 'fun': circle_con})
        
        try:
            result = minimize(objective, p0, method='SLSQP',
                             constraints=cons,
                             options={'maxiter': 1500, 'ftol': 1e-12})
            
            pts = result.x.reshape(n, 2)
            
            if not is_valid(pts):
                continue
            
            # Project to unit circle if slightly outside
            r2 = np.sum(pts**2, axis=1)
            r = np.sqrt(r2)
            mask = r > 1.0
            if np.any(mask):
                pts[mask] = pts[mask] / r[mask, None]
            
            pts_norm = normalize_affine(pts, 1.0)
            
            if not is_valid(pts_norm):
                continue
            
            s = score(pts_norm)
            
            if s > best_score:
                best_score = s
                best_points = pts_norm.copy()
                
        except Exception:
            continue
    
    # Fallback
    if best_points is None or best_score < 1e-10:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = normalize_affine(best_points, 1.0)
    
    return best_points
