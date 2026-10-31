import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def perimeter_to_cart(t):
    """
    Map perimeter distance t in [0, 3) to Cartesian coordinates (x, y)
    on the boundary of the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    t = np.mod(t, 3.0)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    
    # Side 1: (0,0) to (1,0)
    mask1 = t < 1.0
    x[mask1] = t[mask1]
    y[mask1] = 0.0
    
    # Side 2: (1,0) to (0.5, sqrt(3)/2)
    mask2 = (t >= 1.0) & (t < 2.0)
    s = t[mask2] - 1.0
    x[mask2] = 1.0 - 0.5 * s
    y[mask2] = s * H
    
    # Side 3: (0.5, sqrt(3)/2) to (0,0)
    mask3 = t >= 2.0
    s = t[mask3] - 2.0
    x[mask3] = 0.5 * (1.0 - s)
    y[mask3] = H * (1.0 - s)
    
    return np.column_stack([x, y])

def calc_areas_cart(pts):
    """
    Calculate areas of all triangles formed by N points in Cartesian coords.
    pts: (N, 2) array
    """
    # Vectorized area calculation
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # Using determinant form: 0.5 * |det([x1 y1 1; x2 y2 1; x3 y3 1])|
    # Or simpler: 0.5 * |(x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)|
    
    p1 = pts[tri_idx[:, 0]]
    p2 = pts[tri_idx[:, 1]]
    p3 = pts[tri_idx[:, 2]]
    
    # Vectors from p1
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product in 2D: x1*y2 - y1*x2
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    areas = 0.5 * np.abs(cross)
    return areas

def objective_boundary(t_params):
    """
    Objective function for boundary optimization.
    Minimize negative of the minimum triangle area.
    """
    pts = perimeter_to_cart(t_params)
    areas = calc_areas_cart(pts)
    return -np.min(areas)

def objective_2d(coords):
    """
    Objective function for full 2D optimization.
    coords: (N*2,) flat array of (x,y) pairs.
    """
    pts = coords.reshape(N, 2)
    # Check if points are inside the triangle
    # Barycentric coordinates check
    # w0 = area(P, V1, V2) / Area(Triangle)
    # w1 = area(V0, P, V2) / Area(Triangle)
    # w2 = area(V0, V1, P) / Area(Triangle)
    # If any w < -eps, point is outside.
    
    # Area of triangle formed by P, A, B is 0.5 * |(Ax-px)(By-py) - (Ay-py)(Bx-px)|
    # Total Area = sqrt(3)/4
    
    # Let's just clamp or penalize? 
    # For now, assume optimizer stays close to valid region or we clamp.
    # Clamping to triangle is complex. 
    # Instead, we can use a barrier function or just rely on the optimizer staying in bounds if we set bounds.
    # Bounds for x: [0, 1], y: [0, H]. But this includes points outside the triangle.
    # We need a penalty if outside.
    
    # Calculate barycentric coordinates
    # w0 = 1 - w1 - w2
    # w1 = (y - y0) / H  (if x is projected... no, simple formula)
    
    # Using the cross product method for barycentric
    # w0 * V0 + w1 * V1 + w2 * V2 = P
    # w0 + w1 + w2 = 1
    
    # Area(T) = H/2
    # Area(P, V1, V2) = 0.5 * |(V1x-Px)(V2y-Py) - (V2x-Px)(V1y-Py)|
    
    A_tri = H / 2.0
    
    # w0
    v1x, v1y = V1 - pts
    v2x, v2y = V2 - pts
    area0 = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    w0 = area0 / A_tri
    
    # w1
    v0x, v0y = V0 - pts
    v2x, v2y = V2 - pts
    area1 = 0.5 * np.abs(v0x * v2y - v0y * v2x)
    w1 = area1 / A_tri
    
    # w2 = 1 - w0 - w1
    
    # Check validity
    # If any w < -1e-6, it's outside.
    # We can return a large penalty.
    
    min_w = np.min(np.column_stack([w0, w1, 1.0 - w0 - w1]), axis=1)
    
    if np.any(min_w < -1e-5):
        return 1e10
        
    areas = calc_areas_cart(pts)
    return -np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_pts = None
    
    def update_best(pts):
        nonlocal best_val, best_pts
        # Ensure points are valid (inside triangle)
        # Simple check: convert to barycentric
        # If invalid, return
        A_tri = H / 2.0
        v1x, v1y = V1 - pts
        v2x, v2y = V2 - pts
        area0 = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        w0 = area0 / A_tri
        
        v0x, v0y = V0 - pts
        v2x, v2y = V2 - pts
        area1 = 0.5 * np.abs(v0x * v2y - v0y * v2x)
        w1 = area1 / A_tri
        
        w2 = 1.0 - w0 - w1
        
        if np.any(w0 < -1e-6) or np.any(w1 < -1e-6) or np.any(w2 < -1e-6):
            return False
            
        areas = calc_areas_cart(pts)
        val = np.min(areas)
        
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
            return True
        return False

    # Phase 1: Boundary Optimization using Differential Evolution
    # Parameters: 11 perimeter distances
    bounds = [(0, 3.0)] * N
    
    # Initial population: evenly spaced
    # DE will generate its own, but we can provide a seed
    
    if time.time() < deadline - 40:
        try:
            result_de = differential_evolution(
                objective_boundary,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False,
                updating='deferred'
            )
            
            best_t = result_de.x
            pts_boundary = perimeter_to_cart(best_t)
            update_best(pts_boundary)
            
            # Local refinement on boundary
            if time.time() < deadline - 20:
                try:
                    result_refine = minimize(
                        objective_boundary,
                        best_t,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    pts_refined = perimeter_to_cart(result_refine.x)
                    update_best(pts_refined)
                except Exception:
                    pass
        except Exception:
            pass

    # Phase 2: Full 2D Optimization starting from best boundary solution
    # This allows points to move inside if it improves the min area
    if best_pts is not None and time.time() < deadline - 15:
        try:
            # Start from best boundary points
            x0 = best_pts.flatten()
            
            # Use Powell or Nelder-Mead for 2D
            # Powell is good for non-differentiable
            result_2d = minimize(
                objective_2d,
                x0,
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            
            pts_2d = result_2d.x.reshape(N, 2)
            update_best(pts_2d)
            
            # One more polish with Nelder-Mead
            if time.time() < deadline - 5:
                try:
                    result_nm = minimize(
                        objective_2d,
                        best_pts.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    pts_nm = result_nm.x.reshape(N, 2)
                    update_best(pts_nm)
                except Exception:
                    pass
        except Exception:
            pass

    # If no valid points found (should not happen), fallback to random
    if best_pts is None:
        rng = np.random.default_rng(42)
        # Generate random points inside triangle
        u = rng.random(N)
        v = rng.random(N)
        # Barycentric to Cart
        # P = w0*V0 + w1*V1 + w2*V2
        # w0 = 1-u-v, w1=u, w2=v
        w0 = 1 - u - v
        w1 = u
        w2 = v
        best_pts = w0[:, None] * V0 + w1[:, None] * V1 + w2[:, None] * V2
        update_best(best_pts)

    return best_pts

