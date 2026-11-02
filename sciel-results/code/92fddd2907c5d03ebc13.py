"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _in_triangle(p, A, B, C):
    """Check if point p is inside triangle ABC (including boundary)."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, A, B)
    d2 = sign(p, B, C)
    d3 = sign(p, C, A)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def construct():
    """
    Construct 11 points in the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    maximizing the minimum triangle area.
    
    Uses a 3-4-4 grid topology with a rubber band model.
    Optimizes global affine transform (scale, rotation, translation) and local perturbations.
    """
    sqrt3 = np.sqrt(3.0)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, sqrt3 / 2.0])
    h = sqrt3 / 2.0
    
    # Base lattice: 3 points top, 4 middle, 4 bottom
    # Symmetric around x=0.5
    # Bottom row (y=0): 4 points
    x_bottom = np.array([0.15, 0.35, 0.65, 0.85])
    y_bottom = np.zeros(4)
    
    # Middle row (y=h/2): 4 points
    x_middle = np.array([0.25, 0.45, 0.55, 0.75])
    y_middle = np.full(4, h / 2.0)
    
    # Top row (y=h): 3 points
    x_top = np.array([0.35, 0.50, 0.65])
    y_top = np.full(3, h)
    
    base_x = np.concatenate([x_top, x_middle, x_bottom])
    base_y = np.concatenate([y_top, y_middle, y_bottom])
    base_pts = np.column_stack((base_x, base_y))
    
    # Parameters: 
    # 0: rotation (theta)
    # 1: scale (s)
    # 2: tx
    # 3: ty
    # 4-25: perturbations for 11 points (2 each)
    
    def decode_params(params):
        theta = params[0]
        s = params[1]
        tx = params[2]
        ty = params[3]
        
        # Rotation and scaling matrix
        c, sn = np.cos(theta), np.sin(theta)
        R = np.array([[c, -sn], [sn, c]]) * s
        
        # Transform base points
        pts = (base_pts @ R.T) + np.array([tx, ty])
        
        # Add perturbations
        perts = params[4:].reshape(11, 2)
        pts = pts + perts
        
        return pts

    def objective(params):
        pts = decode_params(params)
        
        # Penalty for points outside triangle
        pen = 0.0
        for p in pts:
            if not _in_triangle(p, A, B, C):
                # Calculate distance to outside for smooth penalty
                # Simple penalty: distance squared
                # To be robust, we project or just add a large constant
                # Let's use a distance-based penalty
                # Distance to nearest edge
                d1 = abs((B[0]-A[0])*(A[1]-p[1]) + (A[0]-p[0])*(B[1]-A[1])) / np.linalg.norm(B-A)
                d2 = abs((C[0]-B[0])*(B[1]-p[1]) + (B[0]-p[0])*(C[1]-B[1])) / np.linalg.norm(C-B)
                d3 = abs((A[0]-C[0])*(C[1]-p[1]) + (C[0]-p[0])*(A[1]-C[1])) / np.linalg.norm(A-C)
                # If any barycentric coord is negative, it's outside
                # Using sign check is non-differentiable. 
                # For optimization, a smooth penalty is better.
                # We can use the minimum of the three signed distances.
                # Signed distance to edge AB (normal pointing in):
                # AB vector: (1,0). Normal: (0,1). Distance: p[1].
                # BC vector: (-0.5, h/2). Normal: (h/2, 0.5). Normalized: (sqrt(3)/2, 1/2)? No.
                # Let's stick to a simple check and large penalty for L-BFGS-B which can handle discontinuities somewhat, 
                # or better, use a smooth approximation.
                # Given the complexity, I will use a penalty that grows quadratically with the "violation"
                # determined by barycentric coordinates.
                
                # Barycentric coordinates
                v0 = B - A
                v1 = C - A
                v2 = p - A
                d00 = np.dot(v0, v0)
                d01 = np.dot(v0, v1)
                d11 = np.dot(v1, v1)
                d20 = np.dot(v2, v0)
                d21 = np.dot(v2, v1)
                denom = d00 * d11 - d01 * d01
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1 - v - w
                
                # If u, v, w < 0, it's outside.
                # Penalty = max(0, -u)^2 + max(0, -v)^2 + max(0, -w)^2
                pen += max(0, -u)**2 + max(0, -v)**2 + max(0, -w)**2
        
        # Spring penalty to base lattice (to maintain structure)
        # The perturbations are already separate variables, so we don't need a spring to base_pts 
        # if we want free movement. But the directive says "rubber band model".
        # The "rubber band" usually means springs between points.
        # However, with the affine + perturbation parameterization, the "rubber band" is implicit 
        # in the fact that points start near each other and we minimize energy.
        # To enforce the lattice structure, we add a penalty for large perturbations.
        perts = params[4:].reshape(11, 2)
        spring_pen = 0.1 * np.sum(perts**2)
        
        min_area = _min_triangle_area(pts)
        
        return -min_area + 10.0 * pen + spring_pen

    # Initial parameters
    # theta=0, s=1, tx=0, ty=0, perts=0
    x0 = np.zeros(26)
    x0[1] = 1.0 # scale
    
    # Bounds for parameters
    # theta: [-pi/6, pi/6]
    # s: [0.8, 1.2]
    # tx, ty: [-0.5, 1.5]
    # perts: [-0.1, 0.1]
    bounds = [
        (-np.pi/6, np.pi/6),
        (0.8, 1.2),
        (-0.5, 1.5),
        (-0.5, 1.5)
    ]
    for i in range(22):
        bounds.append((-0.1, 0.1))
        
    # Run optimization from multiple starts
    best_points = None
    best_score = -float('inf')
    
    np.random.seed(42)
    
    # Start 1: Zero perturbations
    starts = [x0]
    
    # Start 2-5: Small random perturbations
    for _ in range(4):
        x0_pert = x0.copy()
        x0_pert[4:] = 0.02 * np.random.randn(22)
        starts.append(x0_pert)
        
    # Start 6: Slightly rotated/scaled
    x0_rot = x0.copy()
    x0_rot[0] = 0.1
    x0_rot[1] = 1.05
    starts.append(x0_rot)
    
    for start in starts:
        try:
            res = minimize(objective, start, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-10})
            pts = decode_params(res.x)
            
            # Verify validity
            valid = True
            for p in pts:
                if not _in_triangle(p, A, B, C):
                    valid = False
                    break
            
            if valid:
                score = _min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass

    if best_points is None:
        # Fallback to base lattice
        best_points = base_pts
        
    return np.array(best_points)
# EVOLVE-BLOCK-END
