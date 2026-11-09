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
        """
        Objective function for optimization.
        We normalize the points to have centroid 0 and RMS radius 1 to remove scale invariance
        and keep the search space bounded.
        """
        pts = p.reshape(n, 2)
        
        # Check for degeneracy early
        if not np.all(np.isfinite(pts)):
            return 1e6
            
        # Normalize: Center
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        
        # Normalize: Scale
        rms = np.sqrt(np.mean(np.sum(pts_centered**2, axis=1)))
        if rms < 1e-8:
            return 1e6
        pts_norm = pts_centered / rms
        
        # Check validity
        hull_area = convex_hull_area(pts_norm)
        if hull_area < 1e-6:
            return 1e6
            
        min_area = compute_min_area(pts_norm)
        
        # We want to maximize min_area / hull_area.
        # So we minimize - (min_area / hull_area)
        ratio = min_area / hull_area
        return -ratio

    best_points = None
    best_score = -1.0
    
    # Use multiple random restarts
    n_restarts = 40
    np.random.seed(42)
    
    for i in range(n_restarts):
        # Generate random initial points
        # Mix of uniform in square and disk
        if i % 2 == 0:
            p0 = np.random.rand(n, 2) * 2 - 1
        else:
            r = np.sqrt(np.random.rand(n))
            theta = np.random.rand(n) * 2 * np.pi
            p0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
        p0_flat = p0.flatten()
        
        # Use Nelder-Mead or Powell
        # Powell is often more robust for this type of problem
        try:
            # Set initial simplex size
            # The objective is scale invariant (due to normalization), so standard simplex works
            result = minimize(objective, p0_flat, method='Powell', 
                              options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-12})
            
            if result.success or result.fun < 0: # If we found a negative value (positive score)
                pts = result.x.reshape(n, 2)
                
                # Re-normalize to standard form for final check
                if is_valid(pts):
                    pts_norm = normalize_affine(pts, 1.0)
                    s = score(pts_norm)
                    
                    if s > best_score:
                        best_score = s
                        best_points = pts_norm.copy()
                        
        except Exception:
            continue
            
        # Also try Nelder-Mead for some restarts
        if i % 4 == 0:
            try:
                result_nm = minimize(objective, p0_flat, method='Nelder-Mead', 
                                     options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                pts = result_nm.x.reshape(n, 2)
                if is_valid(pts):
                    pts_norm = normalize_affine(pts, 1.0)
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
