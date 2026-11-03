import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = np.sqrt(3)
AREA_EQUI = 0.5 * 1.0 * (SQRT3 / 2)  # area of unit equilateral triangle = sqrt(3)/4

# Precompute all triples
_TRIPLES = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _TRIPLES.append((i, j, k))
_TRIPLES = np.array(_TRIPLES, dtype=np.int32)
NUM_TRIPLES = len(_TRIPLES)

def _all_areas(pts):
    """Compute all triangle areas for N points (N,2) Cartesian coords."""
    i1 = _TRIPLES[:, 0]
    i2 = _TRIPLES[:, 1]
    i3 = _TRIPLES[:, 2]
    p1 = pts[i1]
    p2 = pts[i2]
    p3 = pts[i3]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _min_area(pts):
    """Minimum triangle area."""
    return np.min(_all_areas(pts))

def _min_area_normalized(pts):
    """Minimum triangle area normalized by equilateral triangle area."""
    return _min_area(pts) / AREA_EQUI

def _project_to_triangle(pts):
    """Project points to be inside the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    pts = pts.copy()
    # The triangle is: x >= 0, y >= 0, y <= sqrt(3)*(1-x) for x in [0,1]
    # Or equivalently: barycentric coords >= 0
    # Vertex A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # Barycentric: P = a*A + b*B + c*C, a+b+c=1, a,b,c >= 0
    # x = b + 0.5*c, y = (sqrt(3)/2)*c
    # c = 2y/sqrt(3), b = x - 0.5*c, a = 1 - b - c
    y = pts[:, 1]
    x = pts[:, 0]
    c = 2.0 * y / SQRT3
    b = x - 0.5 * c
    a = 1.0 - b - c
    
    # Project to simplex a,b,c >= 0, a+b+c=1
    # Simple projection: clip and renormalize
    abc = np.column_stack([a, b, c])
    abc = np.maximum(abc, 0)
    abc_sum = abc.sum(axis=1, keepdims=True)
    abc_sum[abc_sum == 0] = 1.0
    abc = abc / abc_sum
    
    # Convert back to Cartesian
    pts_new = np.column_stack([abc[:, 1] + 0.5 * abc[:, 2], (SQRT3 / 2) * abc[:, 2]])
    return pts_new

def _objective_smooth(pts, beta=50.0):
    """Smooth objective: softmin of areas (negative for minimization)."""
    areas = _all_areas(pts)
    # Softmin: -(1/beta) * log(sum(exp(-beta * areas)))
    # This is differentiable and approximates -min(areas) as beta -> inf
    x = -beta * areas
    x_max = np.max(x)
    lse = x_max + np.log(np.sum(np.exp(x - x_max)))
    softmin = -lse / beta
    return -softmin  # We want to maximize min, so minimize -min

def _gradient_smooth(pts, beta=50.0):
    """Gradient of the smooth objective."""
    areas = _all_areas(pts)
    i1 = _TRIPLES[:, 0]
    i2 = _TRIPLES[:, 1]
    i3 = _TRIPLES[:, 2]
    
    d1 = pts[i2] - pts[i1]
    d2 = pts[i3] - pts[i1]
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    
    # Gradient of area_i with respect to points
    # area_i = 0.5 * |cross_i|
    # For smoothness, use sign-aware gradient
    sign = np.sign(cross)
    sign[sign == 0] = 1.0
    
    # d(area_i)/d(pts[i1]) = -0.5 * sign_i * cross(d1, d2) direction
    # Actually, let's compute numerically for safety, or use analytic
    # d(cross)/d(p1) = -(d1[1], d2[1]) + ... this gets complex
    # Let's use finite difference for gradient to avoid bugs
    
    eps = 1e-7
    grad = np.zeros_like(pts)
    f0 = _objective_smooth(pts, beta)
    
    for d in range(2 * N):
        pt = pts.copy()
        pt.flat[d] += eps
        f1 = _objective_smooth(pt, beta)
        grad.flat[d] = (f1 - f0) / eps
    
    return grad

def _generate_initial_points(seed=42):
    """Generate initial points using farthest-point sampling in the triangle."""
    rng = np.random.default_rng(seed)
    
    # Start with a point near center
    points = np.array([[0.5, SQRT3 / 6]])
    
    # Add vertices as starting points
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3 / 2]])
    
    # Use farthest-point sampling
    points = vertices.copy()
    
    while len(points) < N:
        # For each candidate, compute min distance to existing points
        candidates = rng.uniform(size=(500, 2))
        candidates[:, 0] = np.abs(candidates[:, 0])  # x in [0, 2] -> project
        candidates[:, 1] = np.abs(candidates[:, 1])
        candidates = _project_to_triangle(candidates)
        
        min_dists = np.min(
            np.linalg.norm(candidates[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2),
            axis=1
        )
        best_idx = np.argmax(min_dists)
        points = np.vstack([points, candidates[best_idx:best_idx+1]])
    
    return points

def _local_refine(pts, max_time=10.0):
    """Local refinement using L-BFGS-B with smooth objective."""
    start_time = time.time()
    best_pts = pts.copy()
    best_val = _min_area_normalized(pts)
    
    # Try multiple beta values
    for beta in [20.0, 50.0, 100.0]:
        if time.time() - start_time > max_time:
            break
        
        try:
            result = minimize(
                _objective_smooth,
                pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=_gradient_smooth,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            refined_pts = _project_to_triangle(result.x.reshape(N, 2))
            val = _min_area_normalized(refined_pts)
            
            if val > best_val:
                best_val = val
                best_pts = refined_pts.copy()
                
            # Use refined points for next beta
            pts = refined_pts
        except Exception:
            continue
    
    return best_pts, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_pts = None
    best_val = 0.0
    
    # Generate multiple initial solutions
    rng = np.random.default_rng(0)
    num_inits = 15
    
    for i in range(num_inits):
        if time.time() > deadline - 15.0:
            break
        
        # Different initialization strategies
        if i == 0:
            # Farthest point sampling
            init_pts = _generate_initial_points(seed=i * 100)
        elif i == 1:
            # Grid-like initialization
            init_pts = _project_to_triangle(np.array([
                [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                [0.2, 0.3], [0.4, 0.3], [0.6, 0.3], [0.8, 0.3],
                [0.3, 0.5], [0.7, 0.5]
            ]))
        elif i == 2:
            # Random with projection
            init_pts = _project_to_triangle(rng.uniform(size=(N, 2)) * 2 - 0.5)
        else:
            # Perturbed farthest point
            init_pts = _generate_initial_points(seed=i * 100 + 1)
            init_pts += rng.normal(size=(N, 2)) * 0.05
            init_pts = _project_to_triangle(init_pts)
        
        # Local refinement
        refined_pts, val = _local_refine(init_pts, max_time=8.0)
        
        if val > best_val:
            best_val = val
            best_pts = refined_pts.copy()
    
    # Final fine-tuning with very high beta
    if best_pts is not None and time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective_smooth,
                best_pts.flatten(),
                args=(200.0,),
                method='L-BFGS-B',
                jac=_gradient_smooth,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            final_pts = _project_to_triangle(result.x.reshape(N, 2))
            final_val = _min_area_normalized(final_pts)
            
            if final_val > best_val:
                best_pts = final_pts
                best_val = final_val
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        best_pts = _generate_initial_points(seed=0)
    
    return best_pts
