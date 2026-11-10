import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def compute_covering_radius(x):
    # x: (N, 3) array
    # Normalize
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    
    try:
        hull = ConvexHull(x)
    except:
        return 180.0 # Invalid
        
    max_rad = 0.0
    # Iterate over facets
    for simplex in hull.simplices:
        p1, p2, p3 = x[simplex[0]], x[simplex[1]], x[simplex[2]]
        
        # Normal to the plane
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # The circumcenter is either n or -n
        # Check which one is the "center" of the cap defined by the facet
        # The cap is the set of points on the sphere closer to the facet's plane?
        # Actually, the Voronoi vertex is the point on the sphere equidistant to p1, p2, p3.
        # There are two such points: n and -n.
        # The one that defines the covering radius is the one that is NOT contained in the convex hull of the other points?
        # No, it's the one that is the center of the empty cap.
        # The cap centered at n with radius d1 (angle to p1) is empty of other points if n is the correct pole.
        # Let's check the distance from n to p1.
        d1 = np.arccos(np.clip(np.dot(n, p1), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(-n, p1), -1.0, 1.0))
        
        # The circumradius of the facet is the smaller of the two?
        # Actually, the facet is a triangle on the convex hull. The "cap" is the region of the sphere.
        # The Voronoi vertex is the point on the sphere farthest from the sites p1,p2,p3 within the cell.
        # This is always the circumcenter.
        # Which pole? The one that is on the "outside" of the triangle relative to the sphere center?
        # The convex hull is around the origin. The facet normal n points outwards.
        # So the circumcenter is n.
        # Let's verify: The points p1,p2,p3 are on the sphere. The plane cuts the sphere.
        # The cap is the part of the sphere on the side of the plane where n points.
        # The center of this cap is n.
        # The radius is the angle between n and p1.
        rad = d1
        
        if rad > max_rad:
            max_rad = rad
            
    return np.degrees(max_rad)

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_x = x0
    best_val = compute_covering_radius(x0)
    
    # Optimization
    # Use Powell method
    # Flatten x to 1D for optimizer
    x_flat = x0.flatten()
    
    def objective_flat(xf):
        x = xf.reshape(N, 3)
        return compute_covering_radius(x)
        
    try:
        # Run minimization
        res = minimize(objective_flat, x_flat, method='Powell', 
                       options={'maxiter': 200, 'maxfev': 1000})
        
        if time.time() < deadline:
            x_opt = res.x.reshape(N, 3)
            val_opt = compute_covering_radius(x_opt)
            if val_opt < best_val:
                best_x = x_opt
                best_val = val_opt
                
        # If time permits, try another round with tighter tolerance or different start
        if time.time() < deadline - 20:
             # Perturb best_x and run again?
             # Or just run Powell again from best_x with more iterations
             x_flat2 = best_x.flatten()
             res2 = minimize(objective_flat, x_flat2, method='Powell', 
                            options={'maxiter': 300, 'maxfev': 1500})
             if time.time() < deadline:
                 x_opt2 = res2.x.reshape(N, 3)
                 val_opt2 = compute_covering_radius(x_opt2)
                 if val_opt2 < best_val:
                     best_x = x_opt2
                     best_val = val_opt2
                     
    except Exception as e:
        pass
        
    return best_x
