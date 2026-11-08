import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def _get_all_triangle_areas(points):
    """
    Calculate areas of all C(N,3) triangles formed by the points.
    Returns a numpy array of areas.
    """
    n = len(points)
    indices = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    i_idx, j_idx, k_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_z)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _regular_polygon(n, r=1.0):
    """Generate a regular n-gon with given radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

def _params_to_points(params):
    """
    Convert perturbation parameters to point coordinates.
    params: array of shape (2*N,) containing (delta_r, delta_theta) for each point.
    Returns: array of shape (N, 2) with points in CCW order.
    """
    delta_r = params[:N]
    delta_theta = params[N:]
    
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_r = np.ones(N)
    
    r = base_r + delta_r
    theta = base_angles + delta_theta
    
    # Ensure points are in CCW order by sorting by angle
    order = np.argsort(theta)
    r = r[order]
    theta = theta[order]
    
    # Ensure angles are in [0, 2*pi)
    theta = theta % (2 * np.pi)
    
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Center the points
    points = points - np.mean(points, axis=0)
    
    return points

def _points_to_params(points):
    """
    Convert points to perturbation parameters relative to regular polygon.
    """
    points = points - np.mean(points, axis=0)
    
    # Sort by angle from origin
    angles = np.arctan2(points[:, 1], points[:, 0])
    order = np.argsort(angles)
    points_sorted = points[order]
    angles_sorted = angles[order]
    
    # Compute radius and angle for each point
    r = np.sqrt(points_sorted[:, 0]**2 + points_sorted[:, 1]**2)
    theta = angles_sorted
    
    # Compute deviations from regular polygon
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    delta_r = r - 1.0
    delta_theta = theta - base_angles
    
    # Unwrap delta_theta to be in [-pi, pi]
    delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
    
    params = np.concatenate([delta_r, delta_theta])
    return params

def _normalize_points(points):
    """Normalize points to unit hull area, centered at origin."""
    points = points - np.mean(points, axis=0)
    hull_area = _hull_area(points)
    if hull_area > 1e-10:
        points = points / np.sqrt(hull_area)
    return points

def _objective_smooth(params, tau=50.0):
    """
    Smooth surrogate for the negative min triangle area.
    """
    points = _params_to_points(params)
    
    # Check convexity: all cross products should have same sign
    # For CCW ordering, cross products of consecutive edges should be positive
    cross_products = []
    for i in range(N):
        p1 = points[i]
        p2 = points[(i+1) % N]
        p3 = points[(i+2) % N]
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    
    cross_products = np.array(cross_products)
    if np.any(cross_products <= 1e-8):
        return 1e6  # Penalty for non-convex
    
    # Normalize to unit hull area
    points = _normalize_points(points)
    
    # Calculate areas
    areas = _get_all_triangle_areas(points)
    
    # Smooth min approximation
    m = np.min(areas)
    shifted = areas - m
    sum_exp = np.sum(np.exp(-tau * shifted))
    
    val = (1.0 / tau) * np.log(sum_exp) - m
    
    return val

def _objective_exact(params):
    """
    Exact objective: -min_triangle_area, assuming points are normalized.
    """
    points = _params_to_points(params)
    
    # Check convexity
    cross_products = []
    for i in range(N):
        p1 = points[i]
        p2 = points[(i+1) % N]
        p3 = points[(i+2) % N]
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    
    cross_products = np.array(cross_products)
    if np.any(cross_products <= 1e-8):
        return 1e6  # Penalty for non-convex
    
    # Normalize to unit hull area
    points = _normalize_points(points)
    
    # Calculate min area
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    return -min_area

def _generate_initial_candidates():
    """
    Generate diverse initial configurations in parameter space.
    """
    candidates = []
    
    # 1. Regular 14-gon (zero perturbation)
    candidates.append(np.zeros(2 * N))
    
    # 2. Small random perturbations
    np.random.seed(42)
    for scale in [0.01, 0.05, 0.1, 0.2]:
        for _ in range(5):
            params = np.random.randn(2 * N) * scale
            candidates.append(params)
    
    # 3. Structured perturbations: alternating radius
    for amp in [0.05, 0.1, 0.15]:
        delta_r = amp * np.array([1 if i % 2 == 0 else -1 for i in range(N)])
        delta_theta = np.zeros(N)
        candidates.append(np.concatenate([delta_r, delta_theta]))
    
    # 4. Rotational symmetry breaking
    for shift in np.linspace(0, 2*np.pi/N, 7, endpoint=False):
        delta_theta = shift * np.ones(N)
        delta_r = np.zeros(N)
        candidates.append(np.concatenate([delta_r, delta_theta]))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    best_score = -1.0
    best_points = None
    best_params = None
    
    # Get initial candidates
    inits = _generate_initial_candidates()
    
    refined_candidates = []
    
    for x0 in inits:
        if time.time() > deadline:
            break
        
        try:
            # Phase 1: Smooth local optimization
            res1 = minimize(_objective_smooth, x0, method='Powell', 
                            options={'maxiter': 300, 'xtol': 1e-6, 'ftol': 1e-6, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p1 = res1.x
            refined_candidates.append(p1)
            
            # Phase 2: Exact local optimization (Polish)
            res2 = minimize(_objective_exact, p1, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p2 = res2.x
            
            # Convert to points and normalize
            pts = _params_to_points(p2)
            
            # Check convexity
            cross_products = []
            for i in range(N):
                p1i = pts[i]
                p2i = pts[(i+1) % N]
                p3i = pts[(i+2) % N]
                v1 = p2i - p1i
                v2 = p3i - p2i
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                cross_products.append(cross)
            
            cross_products = np.array(cross_products)
            if np.any(cross_products <= 1e-8):
                continue  # Skip non-convex
            
            pts = _normalize_points(pts)
            
            s = _score(pts)
            
            if s > best_score:
                best_score = s
                best_points = pts
                best_params = p2
                
        except Exception as e:
            pass
    
    # If we have time, try more random local starts around the best point
    if time.time() < deadline - 20.0 and best_params is not None:
        for _ in range(10):
            if time.time() > deadline:
                break
            noise = np.random.randn(2 * N) * 0.05
            p_perturbed = best_params + noise
            try:
                res_refine = minimize(_objective_exact, p_perturbed, method='Nelder-Mead', 
                                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                p_ref = res_refine.x
                pts = _params_to_points(p_ref)
                
                # Check convexity
                cross_products = []
                for i in range(N):
                    p1i = pts[i]
                    p2i = pts[(i+1) % N]
                    p3i = pts[(i+2) % N]
                    v1 = p2i - p1i
                    v2 = p3i - p2i
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    cross_products.append(cross)
                
                cross_products = np.array(cross_products)
                if np.any(cross_products <= 1e-8):
                    continue
                
                pts = _normalize_points(pts)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
                    best_params = p_ref
            except Exception:
                pass
    
    # If we still have time, try a global DE search in parameter space
    if time.time() < deadline - 30.0:
        try:
            bounds = [(-0.3, 0.3)] * N + [(-0.2, 0.2)] * N
            
            res_de = differential_evolution(
                _objective_smooth, 
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True,
                options={'maxiter': 100}
            )
            
            if time.time() < deadline:
                p_de = res_de.x
                pts = _params_to_points(p_de)
                
                # Check convexity
                cross_products = []
                for i in range(N):
                    p1i = pts[i]
                    p2i = pts[(i+1) % N]
                    p3i = pts[(i+2) % N]
                    v1 = p2i - p1i
                    v2 = p3i - p2i
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    cross_products.append(cross)
                
                cross_products = np.array(cross_products)
                if np.all(cross_products > 1e-8):
                    pts = _normalize_points(pts)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
                        best_params = p_de
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = best_points - np.mean(best_points, axis=0)
        hull_area = _hull_area(best_points)
        if hull_area > 1e-10:
            best_points = best_points / np.sqrt(hull_area)

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Points shape: {pts.shape}")
