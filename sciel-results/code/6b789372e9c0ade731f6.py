import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    # Initial guess: Regular 13-gon in unit circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    p0 = x0.flatten()
    
    def area_min(p):
        pts = p.reshape(n, 2)
        # Calculate areas of all triangles
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    a = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                    areas.append(a)
        return min(areas)
    
    def objective(p):
        # We want to maximize min area, so minimize negative min area
        return -area_min(p)
        
    # Constraints: points must be inside unit circle
    # x^2 + y^2 <= 1
    # Inequality constraint: 1 - x^2 - y^2 >= 0
    cons = []
    for i in range(n):
        def make_constraint(i):
            def constraint(p):
                x = p[2*i]
                y = p[2*i+1]
                return 1 - x**2 - y**2
            return constraint
        cons.append({'type': 'ineq', 'fun': make_constraint(i)})
        
    # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness better
    # SLSQP might get stuck in local minima or have trouble with the non-differentiable min
    # Nelder-Mead is slower but more robust for this small problem
    result = minimize(objective, p0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    points = result.x.reshape(n, 2)
    
    # Verify validity
    if not np.all(np.isfinite(points)):
        return x0
        
    try:
        hull = ConvexHull(points)
        if hull.volume < 1e-6:
            return x0
    except Exception:
        return x0
        
    return points
