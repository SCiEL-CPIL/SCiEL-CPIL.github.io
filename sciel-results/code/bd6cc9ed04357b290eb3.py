import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute hexagon vertices
    # Unit hexagon, pointy-top?
    # Vertices at angles 0, 60, ...?
    # If we use axial coords from before, they were flat-top?
    # axial_to_cartesian:
    # (1,0) -> (sqrt(3), 0). Vector is (sqrt(3), 0).
    # This is along x-axis.
    # A hexagon with a vertex on x-axis has vertices at 0, 60, 120...
    # So it's "pointy-top" in the sense that a vertex is at 0 deg?
    # Usually "pointy-top" means a vertex is at 90 deg.
    # Let's define vertices explicitly.
    angles = np.deg2rad(np.arange(0, 360, 60))
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Support function of a unit hexagon
    # h(theta) = max dot(v, n) for v in vertices
    # Due to symmetry, we can compute this analytically or numerically.
    # For optimization, we need gradients? Nelder-Mead doesn't need them.
    
    def get_hexagon_vertices(cx, cy, theta):
        """Returns 6 vertices of a hexagon centered at (cx, cy) rotated by theta."""
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return cx + vertices[0] * 0 + R @ vertices.T + np.array([cx, cy]) # Wait, vertices are relative to 0
        # Correct:
        verts = R @ vertices.T
        return verts + np.array([cx, cy])

    def min_distance(hex1_center, hex1_rot, hex2_center, hex2_rot):
        """
        Calculate the minimum distance required between centers to avoid overlap.
        d_min = max over all angles psi of (h1(psi) + h2(psi + delta_rot))
        where h is the support function.
        """
        # Relative rotation
        delta = hex2_rot - hex1_rot
        # We need to maximize h1(psi) + h2(psi + delta)
        # h1(psi) is periodic with pi/3.
        # We can sample psi in [0, pi/3]
        # To be fast, we can use the analytical form if possible, or coarse sampling.
        # h(psi) = APOTHEM / cos( (psi % (pi/3)) - pi/6 ) ?
        # Let's verify.
        # Normal at pi/6. h(pi/6) = APOTHEM.
        # Vertex at 0. h(0) = 1.
        # Formula: APOTHEM / cos(0 - pi/6) = (sqrt(3)/2) / (sqrt(3)/2) = 1. Correct.
        # Normal at pi/6 + pi/3 = pi/2.
        # h(pi/2) = APOTHEM / cos( (pi/2 % pi/3) - pi/6 ) = (sqrt(3)/2) / cos( pi/6 - pi/6 ) = APOTHEM. Correct.
        
        def h(psi):
            # Map psi to [0, pi/3)
            p = psi % (math.pi / 3.0)
            # Angle from the closest normal?
            # The normal is at pi/6 within the sector [0, pi/3].
            # So the angle difference is |p - pi/6|
            # But we need the angle relative to the normal.
            # The support function is APOTHEM / cos(angle_diff)
            # where angle_diff is the angle between the direction psi and the normal to the face.
            # The face normal is at pi/6.
            # So angle_diff = abs(p - pi/6) ?
            # If p=0, diff=pi/6. cos(pi/6)=sqrt(3)/2. h=1.
            # If p=pi/6, diff=0. cos(0)=1. h=APOTHEM.
            # If p=pi/3, diff=pi/6. h=1.
            # Yes.
            return APOTHEM / math.cos(abs(p - math.pi / 6.0))
            
        # Maximize h1(psi) + h2(psi + delta)
        # Sample psi
        max_d = 0
        for i in range(12): # 12 samples in pi/3 is enough?
            psi = i * (math.pi / 3.0) / 12.0
            d = h(psi) + h(psi + delta)
            if d > max_d:
                max_d = d
        return max_d

    def objective(params):
        # params: [x1, y1, t1, x2, y2, t2, ..., x11, y11, t11, cx, cy, phi]
        # 11 * 3 + 3 = 36 params
        pts = params[:33].reshape(11, 3)
        cx, cy, phi = params[33], params[34], params[35]
        
        # Check overlaps
        for i in range(11):
            for j in range(i + 1, 11):
                x1, y1, t1 = pts[i]
                x2, y2, t2 = pts[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                req_dist = min_distance((x1,y1), t1, (x2,y2), t2)
                if dist < req_dist - 1e-6:
                    # Penalty
                    return 1e6 + (req_dist - dist)
                    
        # Compute enclosing hexagon
        # Normals for outer hexagon
        normals = []
        for k in range(6):
            ang = phi + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
            
        max_proj = 0.0
        for i in range(11):
            x, y, t = pts[i]
            # Get vertices of this hexagon
            c, s = np.cos(t), np.sin(t)
            R = np.array([[c, -s], [s, c]])
            verts = R @ vertices.T + np.array([x, y])
            
            for nx, ny in normals:
                projs = verts[0] * nx + verts[1] * ny
                p_min = np.min(projs)
                p_max = np.max(projs)
                
                # Center projection
                c_proj = cx * nx + cy * ny
                
                # Max deviation from center
                dev1 = abs(p_max - c_proj)
                dev2 = abs(p_min - c_proj)
                dev = max(dev1, dev2)
                if dev > max_proj:
                    max_proj = dev
                    
        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S

    # Initial configurations
    # Use the best candidate from the old code as a start
    # Candidate 1: Hex cluster
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    pts_init = axial_to_cartesian(axial_1)
    
    # Initial params
    # Centers are pts_init, rotations are 0
    # Outer center is centroid, rotation is 0
    cx0 = float(np.mean(pts_init[:, 0]))
    cy0 = float(np.mean(pts_init[:, 1]))
    
    params0 = []
    for i in range(11):
        params0.append(pts_init[i, 0])
        params0.append(pts_init[i, 1])
        params0.append(0.0)
    params0.extend([cx0, cy0, 0.0])
    
    # Optimize
    best_S = float('inf')
    best_params = params0
    
    # Try a few random perturbations
    for trial in range(5):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        if trial == 0:
            p = params0
        else:
            p = params0 + np.random.normal(0, 0.1, 36)
            
        try:
            res = minimize(objective, p, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract results
    pts_opt = best_params[:33].reshape(11, 3)
    cx, cy, phi = best_params[33], best_params[34], best_params[35]
    
    inner = []
    for i in range(11):
        x, y, t = pts_opt[i]
        inner.append([float(x), float(y), float(math.degrees(t))])
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(math.degrees(phi))
    
    return inner, center, side, rotation
