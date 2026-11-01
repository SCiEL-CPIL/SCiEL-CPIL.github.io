import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def get_lattice_points(shape):
    points = []
    if shape == 'rect_4x3':
        for q in range(4):
            for r in range(3):
                points.append((q, r))
    elif shape == 'rect_3x4':
        for q in range(3):
            for r in range(4):
                points.append((q, r))
    elif shape == 'hex_7_5':
        # 7 center
        points.append((0,0))
        # 6 neighbors
        points.extend([(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)])
        # 5 from next ring
        # Let's pick (2,0), (2,-1), (1,-2), (0,-2), (-1,-2)
        points.extend([(2,0), (2,-1), (1,-2), (0,-2), (-1,-2)])
    elif shape == 'hex_7_5b':
        points.append((0,0))
        points.extend([(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)])
        # Try different 5
        points.extend([(2,0), (1,1), (0,2), (-1,2), (-2,1)])
    return points

def axial_to_cartesian(q, r):
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def construct():
    best_S = float('inf')
    best_config = None
    
    candidates = ['rect_4x3', 'rect_3x4', 'hex_7_5', 'hex_7_5b']
    
    # Pre-calculate unit hexagon vertices at origin, angle 0
    # Flat top? Pointy top?
    # Standard hexagon side 1.
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # x = cos(a), y = sin(a)
    unit_verts = []
    for k in range(6):
        a = np.deg2rad(60 * k)
        unit_verts.append((np.cos(a), np.sin(a)))
    unit_verts = np.array(unit_verts)
    
    for shape in candidates:
        pts = get_lattice_points(shape)
        if len(pts) != N:
            continue
            
        # Convert to cartesian
        cart_pts = []
        for q, r in pts:
            x, y = axial_to_cartesian(q, r)
            cart_pts.append((x, y))
        cart_pts = np.array(cart_pts)
        
        # Center the cluster at origin
        cart_pts -= np.mean(cart_pts, axis=0)
        
        def objective(theta):
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            
            max_S = 0.0
            for i in range(N):
                center = R @ cart_pts[i]
                # Rotate vertices by theta
                verts = unit_verts @ R.T + center
                # For each vertex, calculate required S
                for x, y in verts:
                    s1 = abs(x)
                    s2 = abs(np.sqrt(3)*x + y) / np.sqrt(3)
                    s3 = abs(np.sqrt(3)*x - y) / np.sqrt(3)
                    s_req = max(s1, s2, s3)
                    if s_req > max_S:
                        max_S = s_req
            return max_S
        
        res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded')
        if res.success:
            S = res.fun
            if S < best_S:
                best_S = S
                best_theta = res.x
                best_pts = cart_pts
                best_shape = shape
                
    if best_config is None and best_S != float('inf'):
        # Reconstruct the inner list
        R = np.array([
            [np.cos(best_theta), -np.sin(best_theta)],
            [np.sin(best_theta), np.cos(best_theta)]
        ])
        inner = []
        for i in range(N):
            x, y = R @ best_pts[i]
            inner.append([x, y, np.rad2deg(best_theta)])
        return inner, (0.0, 0.0), best_S, 0.0
    else:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
