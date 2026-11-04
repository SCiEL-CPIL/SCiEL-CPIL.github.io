import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11

def get_hex_verts(cx, cy, s, ang):
    # ang in radians
    angles = ang + np.arange(6) * np.pi / 3
    return np.column_stack((cx + s * np.cos(angles), cy + s * np.sin(angles)))

def point_in_hex(px, py, cx, cy, s, ang):
    # Check if point (px, py) is inside hexagon centered at (cx, cy) with side s and angle ang
    # Transform point to local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -ang
    ca = np.cos(-ang)
    sa = np.sin(-ang)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    # Hexagon with vertices at angles 0, 60, ... has inradius r = s * sqrt(3)/2
    # The boundaries are lines x = +/- r/2? No.
    # Vertices at (s, 0), (s/2, s*sqrt(3)/2), etc.
    # Inradius is distance from center to midpoint of side.
    # Midpoint of side between (s,0) and (s/2, s*sqrt(3)/2) is (3s/4, s*sqrt(3)/4).
    # Distance is sqrt(9s^2/16 + 3s^2/16) = sqrt(12s^2/16) = s*sqrt(3)/2.
    # The normal to that side is at angle 30 deg.
    # So the hexagon is defined by |x cos(30) + y sin(30)| <= r? No.
    # The 3 pairs of parallel sides are at angles 30, 90, 150?
    # Let's check vertex (s, 0).
    # Distance to line x = s*3/4? No.
    
    # Easier: Check max of dot products with normals.
    # Normals to sides are at angles pi/6, pi/6 + pi/3, pi/6 + 2*pi/3.
    # n0 = (cos(pi/6), sin(pi/6))
    # n1 = (cos(pi/2), sin(pi/2))
    # n2 = (cos(5*pi/6), sin(5*pi/6))
    # Inradius r = s * sqrt(3) / 2
    # Condition: abs(dot(l, n_i)) <= r for all i.
    
    n_angles = np.pi/6 + np.arange(3) * np.pi/3
    normals = np.column_stack((np.cos(n_angles), np.sin(n_angles)))
    dots = np.abs(normals @ np.array([lx, ly]))
    r = s * np.sqrt(3) / 2
    return np.all(dots <= r)

def min_dist_hex(h1, h2):
    # h1, h2 are (cx, cy, s, ang)
    # If they overlap, return negative distance? Or 0?
    # For penalty, we want distance to be >= 0.
    # If overlap, distance is negative (penalty).
    # We can check if any vertex of h1 is in h2.
    v1 = get_hex_verts(h1[0], h1[1], h1[2], h1[3])
    v2 = get_hex_verts(h2[0], h2[1], h2[2], h2[3])
    
    # Check vertices of v1 in h2
    for p in v1:
        if point_in_hex(p[0], p[1], h2[0], h2[1], h2[2], h2[3]):
            return -1.0 # Deep overlap
    
    # Check vertices of v2 in h1
    for p in v2:
        if point_in_hex(p[0], p[1], h1[0], h1[1], h1[2], h1[3]):
            return -1.0

    # If no overlap, min distance is min dist between vertices and edges
    min_d = float('inf')
    
    # Dist from v1 points to h2 edges
    for p in v1:
        # Check dist to each edge of h2
        for i in range(6):
            e1 = v2[i]
            e2 = v2[(i+1)%6]
            # Point to line segment distance
            dx = e2[0] - e1[0]
            dy = e2[1] - e1[1]
            len_sq = dx*dx + dy*dy
            if len_sq == 0: continue
            t = np.clip(((p[0]-e1[0])*dx + (p[1]-e1[1])*dy) / len_sq, 0, 1)
            proj_x = e1[0] + t * dx
            proj_y = e1[1] + t * dy
            d = np.sqrt((p[0]-proj_x)**2 + (p[1]-proj_y)**2)
            if d < min_d: min_d = d
            
    # Dist from v2 points to h1 edges
    for p in v2:
        for i in range(6):
            e1 = v1[i]
            e2 = v1[(i+1)%6]
            dx = e2[0] - e1[0]
            dy = e2[1] - e1[1]
            len_sq = dx*dx + dy*dy
            if len_sq == 0: continue
            t = np.clip(((p[0]-e1[0])*dx + (p[1]-e1[1])*dy) / len_sq, 0, 1)
            proj_x = e1[0] + t * dx
            proj_y = e1[1] + t * dy
            d = np.sqrt((p[0]-proj_x)**2 + (p[1]-proj_y)**2)
            if d < min_d: min_d = d
            
    return min_d

def objective(params):
    # params: x0, y0, a0, ..., x10, y10, a10, S, A
    # 33 + 2 = 35 vars? No, 11*3=33. S, A. Total 35.
    # Wait, I said 34 before. 11*3=33. S, A. 35.
    
    start_time = time.time()
    
    inner_hexes = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2] * np.pi / 180.0
        inner_hexes.append((x, y, 1.0, a))
        
    S = params[33]
    A = params[34] * np.pi / 180.0
    
    # Penalty for containment
    pen = 0.0
    outer_h = (0.0, 0.0, S, A)
    
    for h in inner_hexes:
        v = get_hex_verts(h[0], h[1], h[2], h[3])
        for p in v:
            if not point_in_hex(p[0], p[1], 0.0, 0.0, S, A):
                # Calculate distance to boundary?
                # Approximate: if not inside, add penalty based on how far out
                # Simple penalty: 1.0 is enough to push it in if we use DE?
                # Better: calculate min dist to outer hex boundary.
                # But point_in_hex is boolean.
                # Let's just use a large penalty if any vertex is out.
                pen += 10.0
                
    # Penalty for overlap
    for i in range(N):
        for j in range(i+1, N):
            d = min_dist_hex(inner_hexes[i], inner_hexes[j])
            if d < 0:
                pen += 10.0
            elif d < 0.01:
                # Smooth penalty near 0
                pen += (0.01 - d) * 100
                
    # Objective: Minimize S + Penalty
    return S + pen

def construct():
    start = time.time()
    deadline = 100.0 # Safety margin
    
    bounds = [(-5, 5), (-5, 5), (0, 60)] * N
    bounds.append((2, 8)) # Side
    bounds.append((0, 60)) # Angle
    
    # Initial guess: Line
    x0 = []
    for k in range(-(N//2), N - N//2):
        x0.append(1.732 * k) # spacing ~ sqrt(3)
        x0.append(0.0)
        x0.append(0.0)
    x0.append(4.0)
    x0.append(0.0)
    
    # Run DE
    best_result = None
    best_score = float('inf')
    
    # Use a smaller population and iterations to fit in time
    # 35 vars. Pop 20. Iter 50.
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            x0=np.array(x0),
            seed=42,
            maxiter=100, 
            popsize=20, 
            tol=1e-4,
            polish=True
        )
        if time.time() - start < deadline:
             best_score = res.fun
             best_result = res.x
    except Exception:
        pass
        
    if best_result is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract results
    inner = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        a = best_result[i*3+2]
        inner.append([x, y, a])
        
    S = best_result[33]
    A = best_result[34]
    
    return inner, (0.0, 0.0), S, A
