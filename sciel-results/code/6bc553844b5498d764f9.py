import numpy as np
import time
import random

N = 12
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin for 120s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    elif a < 1e-12:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        return np.linalg.norm(r + t * d2)
    elif e < 1e-12:
        s = -np.dot(d1, r) / a
        s = max(0.0, min(1.0, s))
        return np.linalg.norm(s * d1 - r)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    
    if denom < 1e-12:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        s = max(0.0, min(1.0, (b * t - c) / a))
    else:
        s = (b * f - c * e) / denom
        if s < 0.0 or s > 1.0:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
        else:
            t = (b * s + f) / e
            t = max(0.0, min(1.0, t))
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
    
    return np.linalg.norm(s * d1 - r - t * d2)

def _polygons_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap (including boundary)."""
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            if _seg_seg_dist(p1, p2, q1, q2) < eps:
                return True
    return False

def _point_in_hex(p, center, side, angle_deg, eps=1e-7):
    """Check if point p is inside the hexagon."""
    x, y = p
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    # Transform point to hexagon local frame
    dx = x - cx
    dy = y - cy
    # Rotate by -theta
    rx = dx * np.cos(-theta) - dy * np.sin(-theta)
    ry = dx * np.sin(-theta) + dy * np.cos(-theta)
    
    # Hexagon with side S, flat top (angle 0 means vertices at 0, 60... or flat sides at 30, 90...?)
    # _hex_vertices uses angles theta + i*pi/3. 
    # If theta=0, vertices are at 0, 60, 120, 180, 240, 300.
    # This is a "pointy-top" hexagon relative to the x-axis? No, vertex at 0 is (S, 0).
    # The boundaries are lines passing through vertices.
    # Easier: check distance to each edge.
    # Or use the support function.
    # For a regular hexagon centered at origin with vertices at angles alpha + k*pi/3:
    # The max projection of any point inside onto any direction n is S * cos(pi/6) = S * sqrt(3)/2?
    # No, the apothem is S * sqrt(3)/2.
    # The condition for being inside is that the projection onto the normal of each side is <= apothem.
    # The normals to the sides are at angles alpha + pi/6 + k*pi/3.
    
    # Let's just check if p is to the left of all edges (CCW).
    verts = _hex_vertices(center, side, angle_deg)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i+1)%6]
        # Cross product of (p2-p1) and (p-p1)
        cross = (p2[0]-p1[0])*(y-p1[1]) - (p2[1]-p1[1])*(x-p1[0])
        # Vertices are generated CCW. So inside is to the left (positive cross).
        if cross < -eps:
            return False
    return True

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    # Check disjointness
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _polygons_overlap(inner_verts[i], inner_verts[j], eps=1e-9):
                return False, "overlap"
            
            # Check containment
            # Check if all vertices of inner i are inside outer
            for p in inner_verts[i]:
                if not _point_in_hex(p, outer_center, outer_side, outer_angle, eps=eps):
                    return False, "outside"
                
    return True, "valid"

def _objective(config):
    """
    config: list of (x, y, angle) for 12 inner hexagons.
    Returns: (score, valid) where score is outer side length.
    We optimize the outer hexagon center and angle to minimize side length for a fixed inner config?
    Or do we fix outer at origin/0 and just move inner?
    Fixing outer at origin and angle 0 is a valid restriction, but might be suboptimal.
    However, minimizing side length for fixed inner config involves optimizing outer center and angle.
    Let's optimize outer center and angle as part of the state?
    State: 12 * 3 + 3 = 39 variables.
    This is high dimensional.
    Alternative: 
    1. Fix outer hexagon at origin, angle 0.
    2. Optimize inner positions to fit inside.
    This is simpler. The optimal outer hexagon for a given set of points is not necessarily at origin/0, 
    but if we can move the inner points, we can effectively shift them to center the cluster.
    So fixing outer at origin is WLOG for the *relative* configuration, provided we allow translation of inner points.
    But we must also consider rotation. If we fix outer angle to 0, we restrict the relative orientation.
    However, regular hexagons have 6-fold symmetry. If the optimal packing has a relative angle, 
    we can rotate the whole system. So fixing outer angle to 0 is WLOG if we allow inner rotations.
    Wait, if we fix outer angle to 0, and inner hexagons can rotate, we cover all relative orientations.
    So: Fix outer center (0,0) and angle 0. Optimize 12 inner (x,y,angle).
    """
    inner_hexes = config
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Find minimal side length S such that all inner are inside outer (centered at 0, angle 0).
    # For a fixed inner config, the minimal S is determined by the "furthest" extent.
    # S = max over all inner vertices V of ( projection of V onto normals of outer ) / (sqrt(3)/2)
    # Normals of outer (angle 0) are at 30, 90, 150, 210, 270, 330 degrees.
    
    max_proj = 0.0
    for h in inner_hexes:
        verts = _hex_vertices(h[0:2], 1.0, h[2])
        for v in verts:
            # Project onto 6 normals
            for k in range(6):
                angle = np.deg2rad(30 + k * 60)
                n = np.array([np.cos(angle), np.sin(angle)])
                proj = abs(np.dot(v, n))
                if proj > max_proj:
                    max_proj = proj
    
    side = max_proj * 2.0 / np.sqrt(3.0)
    
    # Check validity (disjointness)
    # If disjoint, return side. If not, return penalty.
    # We need a differentiability-like or at least continuous penalty for local search.
    # Penalty = sum of overlaps + large constant if any overlap.
    
    penalty = 0.0
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    for i in range(N):
        for j in range(i+1, N):
            # Distance between polygons
            # If they overlap, dist is 0 (or negative in some definitions).
            # Our _seg_seg_dist returns min distance between segments.
            # If polygons overlap, segments intersect, dist = 0.
            # We want to penalize overlap.
            # We can use the minimum distance between vertices of one to edges of another?
            # Simpler: if _polygons_overlap is True, add penalty.
            # But boolean is not smooth.
            # Let's use a continuous measure: 
            # For each pair, compute the minimum distance between any vertex of i and any edge of j, 
            # and vice versa. If negative (inside), it's overlap.
            # Actually, for convex polygons, if they are disjoint, min dist > 0.
            # If they overlap, min dist = 0.
            # We can define penalty as sum of (max(0, min_dist_required - actual_dist))^2 ?
            # But we don't have a signed distance function easily.
            
            # Let's just check for overlap and add a large penalty.
            if _polygons_overlap(inner_verts[i], inner_verts[j], eps=0):
                penalty += 100.0
    
    if penalty > 0:
        return side + penalty, False
    else:
        return side, True

def construct():
    start_time = time.time()
    
    # Initialize with a random valid-ish configuration
    # Greedy placement on a lattice first to get a valid starting point
    np.random.seed(42)
    random.seed(42)
    
    # Generate a valid starting config using the greedy lattice method from the seed
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.0 + j * 0.5
            y = j * (np.sqrt(3) / 2.0)
            if x*x + y*y <= 4.0**2:
                candidates.append((x, y))
    
    lattice_points = np.array(candidates)
    sorted_indices = np.argsort(np.sum(lattice_points**2, axis=1))
    
    picked = []
    for idx in sorted_indices:
        if len(picked) >= N:
            break
        p = lattice_points[idx]
        valid = True
        for q in picked:
            if np.linalg.norm(p - q) < 1.8:
                valid = False
                break
        if valid:
            picked.append(p)
            
    if len(picked) < N:
        # Fallback random
        picked = []
        for _ in range(N):
            p = np.random.uniform(-5, 5, 2)
            picked.append(p)
            
    # Initial config: positions from picked, angles 0
    current_config = [ [p[0], p[1], 0.0] for p in picked ]
    
    # Evaluate initial
    curr_score, curr_valid = _objective(current_config)
    best_config = [ [x, y, a] for x, y, a in current_config ]
    best_score = curr_score
    best_valid = curr_valid
    
    if not curr_valid:
        # Try to fix by randomizing angles? Or just start annealing with high T
        pass
        
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.9995
    steps = 0
    max_steps = 50000 # Adjust based on time
    
    while time.time() - start_time < TIME_LIMIT:
        if T < 1e-4:
            break
            
        # Pick a random hexagon to move
        idx = random.randint(0, N-1)
        dx = random.gauss(0, 0.5)
        dy = random.gauss(0, 0.5)
        dangle = random.gauss(0, 10) # degrees
        
        new_config = [ [x, y, a] for x, y, a in current_config ]
        new_config[idx] = [
            current_config[idx][0] + dx,
            current_config[idx][1] + dy,
            current_config[idx][2] + dangle
        ]
        
        new_score, new_valid = _objective(new_config)
        
        delta = new_score - curr_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_config = new_config
            curr_score = new_score
            curr_valid = new_valid
            
            if curr_valid and curr_score < best_score:
                best_score = curr_score
                best_config = [ [x, y, a] for x, y, a in current_config ]
                best_valid = True
        
        T *= cooling_rate
        steps += 1
        
    # Final check and return
    if best_valid:
        # Re-verify best config
        is_valid, _ = _check_validity(best_config, best_score, (0.0, 0.0), 0.0)
        if is_valid:
            return (best_config, (0.0, 0.0), best_score, 0.0)
            
    # If SA failed to find a valid one, return the initial greedy one if valid
    is_valid, _ = _check_validity(current_config, curr_score, (0.0, 0.0), 0.0)
    if is_valid:
        return (current_config, (0.0, 0.0), curr_score, 0.0)
        
    # Fallback: Seed
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

# Keep the helper functions from the original if needed, but we redefined them.
# Ensure _hex_vertices, _seg_seg_dist, _polygons_overlap, _point_in_hex, _check_validity, _objective are defined.

