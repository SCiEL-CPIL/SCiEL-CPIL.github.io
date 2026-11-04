import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, angle_deg):
    """
    Returns the 6 vertices of a unit regular hexagon centered at (cx, cy)
    with rotation angle_deg.
    Vertices are in order.
    """
    # Unit hexagon vertices (side 1)
    # Angle 0: Vertex at (1, 0)? No, standard is usually flat top or pointy top.
    # Let's define vertices relative to center for angle 0.
    # If angle 0, let's say vertex 0 is at (1, 0).
    # Then vertices are at angles 0, 60, 120, 180, 240, 300.
    # Side length 1 means distance from center to vertex is 1.
    # Distance between adjacent vertices is 1.
    # This is a "pointy-top" if rotated 90? No, (1,0) is pointy right.
    # Let's stick to this definition.
    angles = np.deg2rad(angle_deg) + np.deg2rad(np.arange(0, 360, 60))
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _separations(H1, H2):
    """
    Calculate the 12 separation values for two hexagons.
    H1, H2 are (6, 2) arrays of vertices.
    Returns array of 12 floats.
    Positive means disjoint in that direction.
    """
    # Normals of H1
    n1 = np.zeros((6, 2))
    for i in range(6):
        v1 = H1[i]
        v2 = H1[(i + 1) % 6]
        edge = v2 - v1
        # Normal pointing outwards
        # Cross product z-component: edge.x * 0 - edge.y * 0? No.
        # For CCW vertices, normal is (edge.y, -edge.x) or (-edge.y, edge.x)?
        # Let's check. (1,0) to (0.5, 0.866). Edge = (-0.5, 0.866).
        # Outward normal should be (0.5, 0.866) ? No, that's towards center?
        # Center is (0,0). Midpoint is (0.75, 0.43).
        # Normal (0.866, 0.5) is outward.
        # Edge (-0.5, 0.866). Normal (0.866, 0.5).
        # Formula: n = (edge[1], -edge[0])?
        # (-0.5, 0.866) -> (0.866, 0.5). Yes.
        n1[i] = np.array([edge[1], -edge[0]])
        n1[i] /= np.linalg.norm(n1[i])
        
    n2 = np.zeros((6, 2))
    for i in range(6):
        v1 = H2[i]
        v2 = H2[(i + 1) % 6]
        edge = v2 - v1
        n2[i] = np.array([edge[1], -edge[0]])
        n2[i] /= np.linalg.norm(n2[i])

    seps = []
    for n in n1:
        # Project H1 and H2 onto n
        p1 = np.dot(H1, n)
        p2 = np.dot(H2, n)
        min1, max1 = np.min(p1), np.max(p1)
        min2, max2 = np.min(p2), np.max(p2)
        # Separation is min of gaps
        # If max1 < min2, gap is min2 - max1
        # If max2 < min1, gap is min1 - max2
        # Else 0 (overlap in this direction)
        # We want a signed distance.
        # Let's define sep = min( min1 - max2, min2 - max1 )
        # If both negative, they overlap.
        # Actually, the "separation" along axis n is the distance between the closest points.
        # If disjoint, it's positive.
        # If overlapping, it's negative (penetration depth).
        # sep = min( min1 - max2, min2 - max1 )
        sep = min(min1 - max2, min2 - max1)
        seps.append(sep)
        
    for n in n2:
        p1 = np.dot(H1, n)
        p2 = np.dot(H2, n)
        min1, max1 = np.min(p1), np.max(p1)
        min2, max2 = np.min(p2), np.max(p2)
        sep = min(min1 - max2, min2 - max1)
        seps.append(sep)
        
    return np.array(seps)

def _soft_min(arr, tau=10.0):
    # -1/tau * log(sum(exp(-tau * arr)))
    # To avoid overflow, subtract max
    m = np.min(arr)
    return m - (1.0/tau) * np.log(np.sum(np.exp(-tau * (arr - m))))

def _objective(params):
    # Unpack
    # 11 hexagons: 11 * (x, y, angle)
    # Outer: S, theta
    n_hex = 11
    hex_params = params[:n_hex*3].reshape(n_hex, 3)
    S = params[n_hex*3]
    theta = params[n_hex*3+1]
    
    # Ensure S is positive
    if S < 0.5: S = 0.5 # Hard constraint for optimizer stability
    
    # Calculate vertices for all hexagons
    verts = []
    for i in range(n_hex):
        x, y, ang = hex_params[i]
        verts.append(_hex_vertices(x, y, ang))
        
    penalty = 0.0
    
    # Containment penalties
    # Outer hexagon vertices (for normals)
    outer_verts = _hex_vertices(0.0, 0.0, theta)
    outer_normals = np.zeros((6, 2))
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        edge = v2 - v1
        n = np.array([edge[1], -edge[0]])
        n /= np.linalg.norm(n)
        outer_normals[i] = n
        
    apothem = S * np.sqrt(3) / 2.0
    
    for i in range(n_hex):
        H = verts[i]
        for k in range(6):
            n = outer_normals[k]
            # Project H onto n
            proj = np.dot(H, n)
            max_proj = np.max(proj)
            # The boundary of the outer hexagon in direction n is at distance apothem from center (0,0)
            # Since center is 0,0, the max allowed projection is apothem.
            margin = apothem - max_proj
            if margin < 0:
                penalty += margin * margin
                
    # Disjointness penalties
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            seps = _separations(verts[i], verts[j])
            # We want min(seps) >= 0
            # Use soft min to get a smooth approximation of min(seps)
            # If min(seps) is negative, it's a violation.
            # soft_min returns a value <= min(seps).
            # If min(seps) = -0.1, soft_min ~ -0.1.
            # We penalize max(0, -soft_min)^2
            s_min = _soft_min(seps, tau=5.0)
            if s_min < 0:
                penalty += s_min * s_min
                
    # Objective: Minimize S + Penalty
    # We want to minimize S, but S is in params.
    # If we minimize S directly, the optimizer will try to make S small,
    # which increases the containment penalty if hexagons are too big.
    # This creates a trade-off.
    # However, if we just return S + penalty, the optimizer might find a local min
    # where S is small but penalty is huge, or S is large but penalty is 0.
    # We want the smallest S such that penalty is 0.
    # A better objective might be:
    # If penalty > threshold, return large penalty.
    # Else return S.
    # But this is non-differentiable.
    # Let's try: S + 100 * penalty.
    # As S decreases, penalty increases. The minimum will be where the cost of decreasing S is balanced by the increase in penalty.
    # Ideally, penalty should be 0 at the optimal S.
    
    return S + 100.0 * penalty

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guess: Seed configuration
    # 1 center, 6 ring, 4 ring
    # Side length ~ 10
    # Centers from seed:
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
        
    # Initial angles: 30 degrees for all?
    # Seed used 30.
    angles = [30.0] * 11
    
    # Initial S
    S_init = 10.0
    theta_init = 30.0
    
    x0 = []
    for i in range(11):
        x0.append(centers[i][0])
        x0.append(centers[i][1])
        x0.append(angles[i])
    x0.append(S_init)
    x0.append(theta_init)
    
    # Bounds
    # S must be > 0
    # Positions can be anywhere, but let's bound them to [-20, 20]
    # Angles [0, 360]
    bounds = []
    for i in range(11):
        bounds.append((-20.0, 20.0))
        bounds.append((-20.0, 20.0))
        bounds.append((0.0, 360.0))
    bounds.append((1.0, 20.0)) # S
    bounds.append((0.0, 360.0)) # theta
    
    best_result = None
    best_score = _objective(x0)
    
    # Use L-BFGS-B
    try:
        # Run optimization
        res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_result = res.x
            
        # Check time
        if time.time() > deadline:
            pass # Exit loop if multiple restarts were planned, but we only do one for now
            
    except Exception as e:
        pass
        
    # If best_result is None, use x0
    if best_result is None:
        best_result = x0
        
    # Extract final parameters
    hex_params = best_result[:33].reshape(11, 3)
    S_final = best_result[33]
    theta_final = best_result[34]
    
    # Verify validity
    # Check if any overlap or containment violation remains
    # If so, we might need to adjust S slightly or return the seed if invalid
    # For now, trust the optimizer if it converged.
    # To be safe, if the penalty is still high, we can increase S.
    
    # Calculate penalty to check
    verts = []
    for i in range(11):
        x, y, ang = hex_params[i]
        verts.append(_hex_vertices(x, y, ang))
        
    outer_verts = _hex_vertices(0.0, 0.0, theta_final)
    outer_normals = np.zeros((6, 2))
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        edge = v2 - v1
        n = np.array([edge[1], -edge[0]])
        n /= np.linalg.norm(n)
        outer_normals[i] = n
        
    apothem = S_final * np.sqrt(3) / 2.0
    
    valid = True
    for i in range(11):
        H = verts[i]
        for k in range(6):
            n = outer_normals[k]
            proj = np.dot(H, n)
            max_proj = np.max(proj)
            if max_proj > apothem + 1e-9:
                valid = False
                break
        if not valid: break
        
    if valid:
        for i in range(11):
            for j in range(i+1, 11):
                seps = _separations(verts[i], verts[j])
                if np.min(seps) < -1e-9:
                    valid = False
                    break
            if not valid: break
            
    if not valid:
        # Fallback to seed if invalid
        # This shouldn't happen if optimizer is good, but just in case
        # We can also try to increase S until valid
        # Let's just increase S by a small amount and re-check?
        # Or return the seed configuration which is known to be valid.
        # Seed S was ~10.
        # Let's calculate seed S precisely.
        max_x = (2 + 1e-9) * np.sqrt(3)
        max_y = 4 + 2 * 1e-9
        s_x = 2 * max_x / np.sqrt(3)
        s_y = max_y
        S_seed = max(s_x, s_y) + 1e-9
        
        inner = []
        for x, y in centers:
            inner.append([x, y, 30.0])
        return inner, (0.0, 0.0), S_seed, 30.0

    # Construct return values
    inner = []
    for i in range(11):
        x, y, ang = hex_params[i]
        inner.append([x, y, ang])
        
    return inner, (0.0, 0.0), S_final, theta_final

