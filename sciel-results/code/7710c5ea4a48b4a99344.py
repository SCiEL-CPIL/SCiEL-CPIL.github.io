import numpy as np
from scipy.optimize import minimize

N = 13

def construct():
    # Angular positions for the N points
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # We parameterize the points using a Fourier series to ensure smoothness and convexity.
    # x_j = cos(theta_j) + sum_k (A_k cos(k*theta_j) + B_k sin(k*theta_j))
    # y_j = sin(theta_j) + sum_k (C_k cos(k*theta_j) + D_k sin(k*theta_j))
    # We include harmonics up to K=3.
    K = 3
    n_params = 4 * K
    # Params: [A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3]
    
    def get_points(params):
        A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3 = params
        
        # Base regular polygon
        x = np.cos(angles)
        y = np.sin(angles)
        
        # Add harmonics
        # k=1 is already in the base (A1=1, B1=0, C1=0, D1=1 effectively, but we treat base as fixed)
        # Wait, if we allow A1, we change the shape of the fundamental mode.
        # Better to fix the fundamental to be circular and only perturb with higher harmonics.
        # Or allow all? If we allow A1, we can skew the ellipse.
        # Let's allow perturbations on top of the unit circle.
        
        x += A1 * np.cos(angles) + B1 * np.sin(angles)
        y += C1 * np.cos(angles) + D1 * np.sin(angles)
        
        x += A2 * np.cos(2 * angles) + B2 * np.sin(2 * angles)
        y += C2 * np.cos(2 * angles) + D2 * np.sin(2 * angles)
        
        x += A3 * np.cos(3 * angles) + B3 * np.sin(3 * angles)
        y += C3 * np.cos(3 * angles) + D3 * np.sin(3 * angles)
        
        return np.column_stack((x, y))

    def score(points):
        # Calculate convex hull area
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        # Calculate minimum triangle area
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                    if area < min_area:
                        min_area = area
        
        if min_area == float('inf'):
            return 0.0
            
        return min_area / hull_area

    def objective(params):
        pts = get_points(params)
        # Penalize non-convexity? 
        # The convex hull area calculation handles the "outer" boundary.
        # If the points are non-convex (interior points), the hull area is smaller, 
        # but the triangles involving interior points might be small.
        # To ensure we stay in the "convex region" domain, we should probably 
        # ensure the points are in convex position.
        # A simple check: if the number of hull vertices < N, penalize.
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            if len(hull.vertices) < N:
                # Penalize strongly if not all points are on the hull
                return 1.0 # High penalty (we minimize -score, so this is bad)
        except Exception:
            return 1.0
            
        s = score(pts)
        return -s

    # Initial guess: regular polygon (all params 0)
    x0 = np.zeros(n_params)
    
    # Bounds: keep perturbations small to maintain convexity
    bounds = [(-0.5, 0.5)] * n_params
    
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = res.x
    best_points = get_points(best_params)
    
    # Ensure the result is valid (convex)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(best_points)
        if len(hull.vertices) < N:
             # Fallback to regular polygon if optimization broke convexity
             return np.column_stack((np.cos(angles), np.sin(angles)))
    except Exception:
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
