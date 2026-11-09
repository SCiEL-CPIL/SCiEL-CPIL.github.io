import numpy as np
from scipy.optimize import linprog
import time

def construct():
    start_time = time.time()
    
    # Hexagon geometry
    # Side length 1
    # Vertices of a unit hexagon centered at (0,0) with angle 0
    # Angle 0 means one vertex at (1,0)? Or one side horizontal?
    # Standard: vertex at (1,0) is angle 0.
    # Let's use vertices at angles 0, 60, 120, 180, 240, 300.
    angles = np.deg2rad(np.arange(0, 360, 60))
    base_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Lattice vectors
    sqrt3 = np.sqrt(3)
    u = np.array([sqrt3, 0])
    v = np.array([sqrt3/2, 1.5])
    
    # Base 7 hexagons centers
    centers = [
        np.array([0.0, 0.0]),
        u, v, v-u, -u, -v, u-v
    ]
    
    # Second ring candidates (corners)
    # Distance 2*sqrt(3)
    corners = [
        2*u,
        2*v,
        2*(v-u),
        -2*u,
        -2*v,
        -2*(v-u)
    ]
    
    # We need 4 more. Choose from corners.
    # Generate all combinations of 4 from 6 corners
    import itertools
    candidate_sets = []
    for combo in itertools.combinations(range(6), 4):
        extra_centers = [corners[i] for i in combo]
        all_centers = centers + extra_centers
        candidate_sets.append(all_centers)
        
    best_side = float('inf')
    best_config = None
    best_theta = 0.0
    best_center = (0.0, 0.0)
    
    # To save time, we can limit the number of candidate sets or theta steps
    # 15 sets * 120 steps (0.5 deg) = 1800 LPs.
    # Each LP is small. Should be fast.
    
    for all_centers in candidate_sets:
        # Get all vertices of these hexagons
        # Since they are aligned, vertices are center + base_vertices
        points = []
        for c in all_centers:
            verts = c + base_vertices
            points.extend(verts)
        points = np.array(points)
        
        # Sweep theta
        for theta_deg in np.arange(0, 60, 0.5):
            if time.time() - start_time > 110:
                break
            
            theta = np.deg2rad(theta_deg)
            # Normals for the outer hexagon
            # The sides are perpendicular to the normals.
            # If the hexagon is rotated by theta, the normals are at theta + 30 + k*60?
            # Let's define the hexagon such that at theta=0, one side is vertical?
            # It doesn't matter, as long as we are consistent.
            # Let's say at theta=0, normals are at 0, 60, 120, 180, 240, 300.
            normal_angles = theta + np.deg2rad(np.arange(0, 360, 60))
            normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
            
            # LP: Minimize h
            # Variables: cx, cy, h
            # Constraints:
            # (p - c) . n <= h  =>  -p.n + c.n + h >= 0  =>  -c.n - h <= -p.n
            # -(p - c) . n <= h =>  p.n - c.n + h >= 0   =>   c.n - h <=  p.n
            
            A = []
            b = []
            
            for p in points:
                for n in normals:
                    # Constraint 1: (p - c) . n <= h
                    # p.n - c.n <= h
                    # -c.n + h >= p.n  (Wait, p.n - c.n - h <= 0)
                    # c.n + h >= p.n
                    # -c.n - h <= -p.n
                    A.append([-n[0], -n[1], -1])
                    b.append(-np.dot(p, n))
                    
                    # Constraint 2: -(p - c) . n <= h
                    # -p.n + c.n <= h
                    # c.n - h <= p.n
                    A.append([n[0], n[1], -1])
                    b.append(np.dot(p, n))
            
            A = np.array(A)
            b = np.array(b)
            
            c_obj = np.array([0, 0, 1])
            
            try:
                res = linprog(c_obj, A_ub=A, b_ub=b, method='highs')
                if res.success:
                    h = res.x[2]
                    side = 2 * h / sqrt3
                    if side < best_side:
                        best_side = side
                        best_config = all_centers
                        best_theta = theta_deg
                        best_center = (res.x[0], res.x[1])
            except:
                pass
        
        if time.time() - start_time > 110:
            break

    if best_config is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-5, 6)], (0.0, 0.0), 40.0, 0.0

    # Construct inner list
    inner = []
    for c in best_config:
        inner.append([c[0], c[1], 0.0])
        
    return inner, best_center, best_side, best_theta

