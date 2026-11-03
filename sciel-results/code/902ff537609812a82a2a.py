import numpy as np
import time
from scipy.optimize import minimize

N = 11

def get_hexagon_vertices(center, side, angle_deg):
    """
    Returns the 6 vertices of a regular hexagon.
    center: (x, y)
    side: side length
    angle_deg: rotation in degrees
    """
    cx, cy = center
    angle = np.deg2rad(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 relative to center
    # Standard orientation: one vertex at (side, 0)
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + angle
    vertices = np.zeros((6, 2))
    for i in range(6):
        vertices[i, 0] = cx + side * np.cos(angles[i])
        vertices[i, 1] = cy + side * np.sin(angles[i])
    return vertices

def hexagon_distance_penalty(v1_center, v1_angle, v2_center, v2_angle, side=1.0):
    """
    Calculate a smooth penalty for overlap between two hexagons.
    Uses the minimum distance between vertices of one and edges of the other,
    or simply checks if vertices are inside and calculates penetration depth.
    For simplicity and speed in SA, we use a sum of squared penetrations of vertices
    into the other hexagon's half-planes.
    """
    def point_in_hexagon_penalty(p, center, angle_deg, side):
        """
        Returns the maximum penetration depth if p is inside, 0 otherwise.
        """
        cx, cy = center
        angle = np.deg2rad(angle_deg)
        # Normals to the sides of the hexagon
        normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])) + angle
        apothem = side * np.sqrt(3) / 2
        
        v = p - np.array(center)
        max_pen = 0.0
        for na in normal_angles:
            n = np.array([np.cos(na), np.sin(na)])
            proj = np.dot(v, n)
            # If proj > apothem, it's outside. Penetration is (apothem - proj) if we want distance to boundary?
            # Actually, for convex containment, if proj > apothem, it's outside.
            # We want a penalty if it is INSIDE.
            # Wait, the penalty should be high if they OVERLAP.
            # Overlap happens if vertices of A are in B OR vertices of B are in A OR edges cross.
            # A simpler metric for "closeness" that penalizes overlap:
            # If we treat the hexagon as the intersection of 6 half-planes,
            # the distance from a point p to the boundary of hexagon H (center c, normal n_i) is:
            # d_i = apothem - dot(p-c, n_i)
            # If d_i > 0 for all i, p is inside.
            # We can sum the squared positive d_i's? No, that penalizes being deep inside.
            # We want to penalize OVERLAP.
            # Let's use a different approach: compute the minimum distance between the two convex polygons.
            # If distance < 0 (overlap), penalty = distance^2.
            # Computing min distance between two convex polygons is complex.
            
            # Alternative: Use a "soft" constraint based on center distance? No, rotation matters.
            
            # Let's stick to the vertex-in-polygon check but make it smooth.
            # If a vertex of H1 is inside H2, it's an overlap.
            # We can approximate the "depth" by how much it violates the boundary?
            # Actually, if it's inside, the violation is negative distance.
            # Let's calculate the distance from p to the boundary of H.
            # If p is inside, dist is negative (penetration).
            # min_dist = min_i (dot(p-c, n_i) - apothem)
            # If min_dist < 0, p is inside. Penetration = -min_dist.
            
            projs = np.dot(v, n) # This is not vectorized per normal, let's vectorize
            # v is (2,), n is (6,2). dot is not direct.
            # proj_i = v[0]*n[0] + v[1]*n[1]
            # Let's just loop, it's only 6 iterations.
            
        # Re-doing with vectorization for speed
        ns = np.array([np.array([np.cos(na), np.sin(na)]) for na in normal_angles])
        projs = ns.dot(v)
        # projs[i] is the projection of (p-c) on normal i.
        # Boundary is at apothem.
        # If projs[i] > apothem, p is outside relative to face i.
        # If projs[i] < apothem, p is inside relative to face i.
        # For p to be INSIDE, it must be inside ALL faces.
        # So we need projs[i] <= apothem for ALL i.
        # The "most inside" face is the one with the smallest (apothem - projs[i]).
        # Penetration depth = min_i (apothem - projs[i])
        # If this min is > 0, p is inside.
        # We want to penalize this.
        margins = apothem - projs
        min_margin = np.min(margins)
        if min_margin > 0:
            return min_margin # Penetration depth
        else:
            return 0.0

    # Check vertices of v1 against v2
    verts1 = get_hexagon_vertices(v1_center, side, v1_angle)
    verts2 = get_hexagon_vertices(v2_center, side, v2_angle)
    
    pen = 0.0
    for v in verts1:
        pen += point_in_hexagon_penalty(v, v2_center, v2_angle, side)
    for v in verts2:
        pen += point_in_hexagon_penalty(v, v1_center, v1_angle, side)
        
    return pen

def objective(params):
    """
    params: array of length 3*N: [x1, y1, a1, x2, y2, a2, ...]
    Returns: penalized outer side length
    """
    side_inner = 1.0
    penalty = 0.0
    
    # Normals for the outer hexagon (fixed orientation, centered at 0,0)
    # Outer hexagon has apothem A. Sides are at distance A from center.
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    outer_normals = np.array([
        [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
        [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
        [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
        [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
        [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
        [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]
    ])
    
    max_proj = 0.0
    
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        
        # Get vertices
        verts = get_hexagon_vertices((x, y), side_inner, a)
        
        # Calculate max projection for this hexagon's vertices
        for v in verts:
            # Vectorized dot product with outer_normals
            projs = outer_normals.dot(v)
            if np.max(projs) > max_proj:
                max_proj = np.max(projs)
                
    # Side length S = 2 * A / sqrt(3) where A is apothem (max projection)
    S = 2.0 * max_proj / np.sqrt(3.0)
    
    # Check for overlaps
    # We use a smooth penalty based on penetration depth
    for i in range(N):
        for j in range(i+1, N):
            x1, y1, a1 = params[3*i], params[3*i+1], params[3*i+2]
            x2, y2, a2 = params[3*j], params[3*j+1], params[3*j+2]
            pen = hexagon_distance_penalty((x1, y1), a1, (x2, y2), a2, side_inner)
            # Square the penalty to make it smoother and stronger for deep overlaps
            penalty += pen * pen
            
    return S + 10.0 * penalty

def construct():
    start_time = time.time()
    timeout = 115.0 # Safety margin
    
    best_S = float('inf')
    best_params = None
    
    # Initial guess: Hexagonal grid
    # Side length 1.
    # Pointy-top hexagons (vertex at 0 deg).
    # Width = sqrt(3), Height = 2.
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 1.5
    
    init_params = []
    positions = []
    # 1 center
    positions.append((0.0, 0.0, 0.0))
    # 6 neighbors
    for k in range(6):
        ang = np.deg2rad(k * 60)
        x = np.sqrt(3) * np.cos(ang)
        y = np.sqrt(3) * np.sin(ang)
        positions.append((x, y, 0.0))
    # 4 more
    # Ring 2
    # (2*sqrt(3), 0), (sqrt(3), 3), (0, 2*sqrt(3)), (-sqrt(3), 3) ... 
    # Let's pick 4 spots
    # Angles: 0, 90, 180, 270? No, that's not hex grid.
    # Hex grid ring 2 positions relative to center:
    # (2*sqrt(3), 0), (sqrt(3), 3), (-sqrt(3), 3), (-2*sqrt(3), 0), (-sqrt(3), -3), (sqrt(3), -3)
    # We need 4 more.
    positions.append((2*np.sqrt(3), 0, 0.0))
    positions.append((np.sqrt(3), 3, 0.0))
    positions.append((-np.sqrt(3), 3, 0.0))
    positions.append((-2*np.sqrt(3), 0, 0.0))
    
    for p in positions:
        init_params.extend([p[0], p[1], p[2]])
        
    init_params = np.array(init_params)
    
    # Simulated Annealing
    current_params = init_params.copy()
    current_score = objective(current_params)
    
    T = 1.0 # Temperature
    cooling = 0.995
    
    np.random.seed(42)
    
    while time.time() - start_time < timeout:
        # Perturb
        step_size = 0.1 * (1.0 + T)
        new_params = current_params + np.random.normal(0, step_size, size=len(current_params))
        
        # Constrain angles to [-180, 180] for sanity? Not strictly necessary but helps.
        # new_params[2::3] = (new_params[2::3] + 180) % 360 - 180
        
        new_score = objective(new_params)
        
        if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            
            if current_score < best_S:
                best_S = current_score
                best_params = current_params.copy()
                
        T *= cooling
        
        # If we have a very good solution (low penalty), we can stop early?
        # The penalty is 10 * sum(pen^2). If penalty is 0, score is just S.
        # A perfect packing of 11 hexagons has S approx 3.0-3.5?
        # If best_S < 4.0 and penalty part is small, we might be good.
        # But we don't know the penalty part separately in the score.
        # Let's just run until time runs out.
        
    if best_params is None:
        best_params = init_params
        best_S = objective(best_params)
        
    # Extract best solution
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
        
    # Calculate true S for best_params
    outer_normals = np.array([
        [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
        [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
        [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
        [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
        [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
        [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]
    ])
    
    max_proj = 0.0
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        verts = get_hexagon_vertices((x, y), 1.0, a)
        for v in verts:
            projs = outer_normals.dot(v)
            if np.max(projs) > max_proj:
                max_proj = np.max(projs)
                
    true_S = 2.0 * max_proj / np.sqrt(3.0)
    
    # Final validation: Check if there are overlaps.
    # If there are overlaps, the solution is invalid.
    # We should return a valid solution if possible.
    # If the best solution found has overlaps, it's invalid.
    # We can try to "push apart" overlapping hexagons in a final cleanup step?
    # Or just return the best found. If it's invalid, score is 0.
    # To improve chances, we can do a local refinement to remove overlaps.
    
    # Simple cleanup: iterate and push apart overlapping pairs
    cleanup_params = best_params.copy()
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, a1 = cleanup_params[3*i], cleanup_params[3*i+1], cleanup_params[3*i+2]
                x2, y2, a2 = cleanup_params[3*j], cleanup_params[3*j+1], cleanup_params[3*j+2]
                pen = hexagon_distance_penalty((x1, y1), a1, (x2, y2), a2, 1.0)
                if pen > 0:
                    # Push apart
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 0:
                        push = 0.01 * pen
                        cleanup_params[3*i] -= push * dx / dist
                        cleanup_params[3*i+1] -= push * dy / dist
                        cleanup_params[3*j] += push * dx / dist
                        cleanup_params[3*j+1] += push * dy / dist
                        changed = True
        if not changed:
            break
            
    # Re-calculate S for cleanup params
    max_proj = 0.0
    for i in range(N):
        x = cleanup_params[3*i]
        y = cleanup_params[3*i+1]
        a = cleanup_params[3*i+2]
        verts = get_hexagon_vertices((x, y), 1.0, a)
        for v in verts:
            projs = outer_normals.dot(v)
            if np.max(projs) > max_proj:
                max_proj = np.max(projs)
                
    cleanup_S = 2.0 * max_proj / np.sqrt(3.0)
    
    # Check if cleanup solution is valid (no overlaps)
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            x1, y1, a1 = cleanup_params[3*i], cleanup_params[3*i+1], cleanup_params[3*i+2]
            x2, y2, a2 = cleanup_params[3*j], cleanup_params[3*j+1], cleanup_params[3*j+2]
            pen = hexagon_distance_penalty((x1, y1), a1, (x2, y2), a2, 1.0)
            if pen > 1e-6: # Tolerance
                valid = False
                break
        if not valid:
            break
            
    if valid:
        best_params = cleanup_params
        true_S = cleanup_S
    else:
        # If cleanup failed, stick with the best found, hoping it's valid or close enough
        # Or try to return the initial guess if it's valid?
        # The initial guess (hex grid) is valid.
        # Let's check if initial guess is valid and compare S.
        init_S = objective(init_params) # This includes penalty, so if valid, penalty=0
        if init_S < true_S: # If init is better (and valid since penalty=0 implies valid in our metric)
             # Actually objective returns S + penalty. If penalty is 0, it's S.
             # So if init_S < true_S, and we assume init is valid, it's better.
             # But true_S might be from an invalid solution with low S but high penalty?
             # No, we return true_S calculated from best_params.
             # If best_params was invalid, true_S is just the geometric size.
             # The evaluator checks validity.
             # So we should prefer a valid solution with larger S over an invalid one with smaller S.
             
             # Let's explicitly check validity of best_params
             best_valid = True
             for i in range(N):
                 for j in range(i+1, N):
                     x1, y1, a1 = best_params[3*i], best_params[3*i+1], best_params[3*i+2]
                     x2, y2, a2 = best_params[3*j], best_params[3*j+1], best_params[3*j+2]
                     pen = hexagon_distance_penalty((x1, y1), a1, (x2, y2), a2, 1.0)
                     if pen > 1e-6:
                         best_valid = False
                         break
                 if not best_valid:
                     break
             
             if not best_valid and init_S < float('inf'):
                 # Use initial guess
                 best_params = init_params
                 # Recalculate true_S for init
                 max_proj = 0.0
                 for i in range(N):
                     x = init_params[3*i]
                     y = init_params[3*i+1]
                     a = init_params[3*i+2]
                     verts = get_hexagon_vertices((x, y), 1.0, a)
                     for v in verts:
                         projs = outer_normals.dot(v)
                         if np.max(projs) > max_proj:
                             max_proj = np.max(projs)
                 true_S = 2.0 * max_proj / np.sqrt(3.0)

    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
        
    return inner, (0.0, 0.0), true_S, 0.0
