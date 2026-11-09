import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])
VERTS = np.array([A, B, C])

def project_to_triangle(p):
    """
    Project point p onto the triangle defined by A, B, C.
    """
    # Barycentric coordinates
    # p = u*A + v*B + w*C, u+v+w=1
    # Using standard formula
    x, y = p
    # d = (Bx - Cx)(Ay - Cy) - (Ax - Cx)(By - Cy)
    # u = ((Bx - Cx)(y - Cy) - (By - Cy)(x - Cx)) / d
    # v = ((Cx - Ax)(y - Ay) - (Cy - Ay)(x - Ax)) / d
    # w = 1 - u - v
    
    # For our specific triangle:
    # A=(0,0), B=(1,0), C=(0.5, h)
    # h = sqrt(3)/2
    h = np.sqrt(3) / 2
    # u corresponds to weight of A? No, let's derive carefully.
    # p = u*A + v*B + w*C
    # x = 0*u + 1*v + 0.5*w => x = v + 0.5w
    # y = 0*u + 0*v + h*w => y = h*w => w = y/h
    # v = x - 0.5w = x - 0.5*y/h
    # u = 1 - v - w = 1 - (x - 0.5y/h) - y/h = 1 - x - 0.5y/h
    
    w = y / h
    v = x - 0.5 * w
    u = 1.0 - v - w
    
    # If inside, return p
    if u >= 0 and v >= 0 and w >= 0:
        return p
    
    # If outside, project to the boundary.
    # The projection of a point onto a convex polygon is either the point itself (if inside),
    # or the closest point on one of the edges, or one of the vertices.
    # We can simply clip the barycentric coordinates to [0,1] and renormalize?
    # No, clipping barycentric coordinates and renormalizing gives the closest point in the simplex?
    # Actually, for a simplex, the Euclidean projection is obtained by setting negative barycentric coords to 0 and renormalizing the rest?
    # Let's check. If w < 0, the point is "below" the triangle. The closest point is on the edge AB (w=0).
    # If we set w=0 and renormalize u, v, we get a point on AB. Is it the closest?
    # Yes, for a simplex, the projection is obtained by zeroing out negative barycentric coordinates and renormalizing.
    
    bary = np.array([u, v, w])
    bary[bary < 0] = 0
    s = np.sum(bary)
    if s == 0:
        # Should not happen unless p is far away and all negative, which is impossible for a triangle covering origin?
        # Actually if p is far, one coord will be positive.
        return A
    bary /= s
    return bary[0]*A + bary[1]*B + bary[2]*C

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    """
    n = len(points)
    min_area = np.inf
    # Iterate over all combinations of 3 points
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            for k in range(j + 1, n):
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def objective(flat_points):
    """
    Objective function for minimization.
    Minimizes negative min area.
    """
    points = flat_points.reshape((N, 2))
    # Project to triangle to ensure validity
    projected = np.array([project_to_triangle(p) for p in points])
    area = min_triangle_area(projected)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_area = 0.0
    
    # Initial configurations
    inits = []
    
    # 1. Grid-like initialization
    # We can use a simple grid in barycentric coords
    # For N=11, maybe 3x3 grid + 2 extra?
    # Let's just use random restarts with good spread.
    
    # Strategy: 
    # 1. Generate a set of diverse initial points.
    # 2. Run Nelder-Mead on each.
    # 3. Keep the best.
    
    # To ensure diversity, we can use a few different random seeds and also a structured start.
    
    # Structured Start: Place points at vertices, midpoints, and center, then jitter.
    # Vertices: A, B, C
    # Midpoints: (A+B)/2, (B+C)/2, (C+A)/2
    # Center: (A+B+C)/3
    # That's 7 points. We need 4 more.
    # Add points at 1/3 and 2/3 along edges?
    
    structured = []
    structured.append(A)
    structured.append(B)
    structured.append(C)
    structured.append((A+B)/2)
    structured.append((B+C)/2)
    structured.append((C+A)/2)
    structured.append((A+B+C)/3)
    # 4 more points. Let's put them in the middle of the sub-triangles.
    # Sub-triangles formed by center and edge midpoints.
    # There are 6 such triangles. We need 4 points.
    # Let's just use random for the rest in this init.
    rng = np.random.RandomState(42)
    for _ in range(4):
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            p = u*A + v*B + (1-u-v)*C
            structured.append(p)
        else:
            # Retry
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                p = u*A + v*B + (1-u-v)*C
                structured.append(p)
            else:
                structured.append(rng.random(2) * 0.5 + 0.25) # Fallback inside
    
    inits.append(np.array(structured).flatten())
    
    # Random restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        rng = np.random.RandomState(100 + i)
        # Generate N random points inside triangle
        pts = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    p = u*A + v*B + (1-u-v)*C
                    pts.append(p)
                    break
        inits.append(np.array(pts).flatten())
        
        # Run optimization
        try:
            res = minimize(objective, inits[-1], method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-10, 'adaptive': True})
            
            if res.success:
                opt_points = res.x.reshape((N, 2))
                opt_points = np.array([project_to_triangle(p) for p in opt_points])
                area = min_triangle_area(opt_points)
                
                if area > best_area:
                    best_area = area
                    best_points = opt_points
        except Exception:
            pass

    # If no optimization succeeded or we haven't run any, use the best init
    if best_points is None:
        # Evaluate all inits
        for init in inits:
            pts = np.array(init).reshape((N, 2))
            pts = np.array([project_to_triangle(p) for p in pts])
            area = min_triangle_area(pts)
            if area > best_area:
                best_area = area
                best_points = pts
    else:
        # Check if any init was better (unlikely if opt worked, but possible if opt failed to improve)
        pass

    return best_points
