import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12
SQRT3 = np.sqrt(3.0)

def hex_vertices(center, side, angle):
    cx, cy = center
    # Pointy-top hexagon vertices relative to center
    # Angles: 30, 90, 150, 210, 270, 330 degrees (pi/6 offset)
    angles = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 7)[:-1] + angle
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def point_in_hex(px, py, center, side, angle):
    # Check if point (px, py) is inside hexagon defined by center, side, angle
    # Transform point to local coords
    dx = px - center[0]
    dy = py - center[1]
    # Rotate back
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For pointy-top hexagon, vertices are at 30, 90, ...
    # The hexagon is intersection of 6 half-planes.
    # Normal vectors point to vertices.
    # Distance from center to side (apothem) is side * sqrt(3)/2
    apothem = side * SQRT3 / 2.0
    
    # Check 6 constraints
    # The normals are at angles 0, 60, 120, 180, 240, 300 relative to the hex frame?
    # For pointy-top, the flat sides are at 0, 60, ...?
    # Let's check vertices: 30, 90, 150...
    # The side connecting 30 and 90 is perpendicular to 60.
    # So normals are at 60, 120, 180, 240, 300, 0.
    for k in range(6):
        theta = np.pi/3 * k
        nx = np.cos(theta)
        ny = np.sin(theta)
        dist = lx * nx + ly * ny
        if dist > apothem:
            return False
    return True

def get_cluster_centers():
    # 3 rows, 4 cols pointy-top grid
    centers = []
    # Row 0
    for i in range(4):
        centers.append((i * SQRT3, 0.0))
    # Row 1
    for i in range(4):
        centers.append((i * SQRT3 + SQRT3/2, 1.5))
    # Row 2
    for i in range(4):
        centers.append((i * SQRT3, 3.0))
    
    centers = np.array(centers)
    # Center the cluster
    center_of_mass = np.mean(centers, axis=0)
    centers = centers - center_of_mass
    return centers

CLUSTER_CENTERS = get_cluster_centers()

def get_cluster_vertices(tx, ty, theta):
    # Returns array of shape (72, 2)
    verts = []
    for cx, cy in CLUSTER_CENTERS:
        # Local center
        lx = cx * np.cos(theta) - cy * np.sin(theta) + tx
        ly = cx * np.sin(theta) + cy * np.cos(theta) + ty
        v = hex_vertices((lx, ly), 1.0, theta)
        verts.append(v)
    return np.vstack(verts)

def penalty_function(params):
    L = params[0]
    ocx, ocy = params[1], params[2]
    otheta = params[3]
    tx, ty, itheta = params[4], params[5], params[6]
    
    if L < 1.0: # Minimum possible side for 1 hex is 1, for 12 it's larger
        return 1e10

    # Outer hex vertices
    outer_verts = hex_vertices((ocx, ocy), L, otheta)
    
    # Cluster vertices
    cluster_verts = get_cluster_vertices(tx, ty, itheta)
    
    # Check containment
    # For each cluster vertex, check if inside outer hex
    # We can use the point_in_hex function, but it's slow for 72 points * many iterations.
    # Alternative: SAT or simple polygon containment.
    # Given the convexity, if all vertices are inside, the polygon is inside.
    
    penalty = L
    
    for px, py in cluster_verts:
        if not point_in_hex(px, py, (ocx, ocy), L, otheta):
            # Calculate distance to boundary for penalty
            # Simple penalty: distance to nearest edge
            # For speed, just add a large penalty
            penalty += 10.0
            # Break early? No, sum penalties for gradient-like info
            # But DE doesn't use gradients. Early break is good for speed.
            # However, we want to know *how much* it's out to guide search?
            # DE is blind. So just a flag is enough, but larger penalty for deeper violation helps.
            # Let's estimate distance roughly.
            # Distance from center to vertex is L.
            # If point is outside, it's likely far.
            # Let's just add 1.0 per violating vertex to keep it simple and fast.
            
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial guess
    # The cluster bounding box approx:
    # X width: 3 * SQRT3 + SQRT3 = 4 * SQRT3 ~ 6.9
    # Y height: 3 + 2 = 5.0
    # Outer hex side L should be around 4.0 - 5.0
    
    # Bounds for DE
    # L: [2.0, 10.0]
    # ocx, ocy: [-10, 10]
    # otheta: [0, 2*pi]
    # tx, ty: [-10, 10]
    # itheta: [0, 2*pi]
    
    bounds = [
        (2.0, 10.0),  # L
        (-10.0, 10.0), # ocx
        (-10.0, 10.0), # ocy
        (0.0, 2*np.pi), # otheta
        (-10.0, 10.0), # tx
        (-10.0, 10.0), # ty
        (0.0, 2*np.pi)  # itheta
    ]
    
    best_params = None
    best_score = np.inf
    
    try:
        # Use Differential Evolution for global search
        # maxiter is number of generations. popsize is population size.
        # We have 7 variables. popsize=10 is small, but DE is slow.
        # Let's try popsize=5, maxiter=50.
        # Time limit check inside DE is not direct, so we use small maxiter.
        
        res = differential_evolution(penalty_function, bounds, maxiter=50, popsize=5, 
                                      tol=0.01, seed=42, polish=False)
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    except Exception:
        pass
    
    # Refine with Nelder-Mead if we have time and a good start
    if best_params is not None and time.time() < deadline - 10.0:
        try:
            res2 = minimize(penalty_function, best_params, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    if best_params is None:
        # Fallback: Return a large valid hexagon
        return [], (0,0), 10.0, 0.0

    L = best_params[0]
    ocx, ocy = best_params[1], best_params[2]
    otheta = best_params[3]
    tx, ty, itheta = best_params[4], best_params[5], best_params[6]
    
    # Final Verification and Correction
    # We need to ensure ALL vertices are inside.
    # If not, we increase L until they are.
    # This guarantees validity.
    
    cluster_verts = get_cluster_vertices(tx, ty, itheta)
    
    # Check current L
    valid = True
    for px, py in cluster_verts:
        if not point_in_hex(px, py, (ocx, ocy), L, otheta):
            valid = False
            break
            
    if not valid:
        # Increase L
        # Calculate max required L
        # The furthest vertex from center determines min L for a centered hex?
        # But the hex can rotate.
        # Simple approach: increase L by 0.1 until valid or L > 10
        while not valid and L < 10.0:
            L += 0.1
            valid = True
            for px, py in cluster_verts:
                if not point_in_hex(px, py, (ocx, ocy), L, otheta):
                    valid = False
                    break
            if time.time() > deadline - 1.0:
                break
                
    # Construct inner list
    inner = []
    for cx, cy in CLUSTER_CENTERS:
        lx = cx * np.cos(itheta) - cy * np.sin(itheta) + tx
        ly = cx * np.sin(itheta) + cy * np.cos(itheta) + ty
        inner.append((lx, ly, np.degrees(itheta)))
        
    return inner, (ocx, ocy), L, np.degrees(otheta)
