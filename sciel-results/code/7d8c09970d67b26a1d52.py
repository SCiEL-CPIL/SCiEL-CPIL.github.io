import math
import time
import numpy as np
import random

N = 11
_DEADLINE = 145.0
_C30 = math.cos(math.radians(30))

def _hex_inradius(s):
    return s * _C30

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _signed_dist_hex(hex1, hex2):
    """
    Returns signed distance between two convex hexagons.
    Positive if disjoint, negative if overlapping (penetration depth).
    hex1, hex2: (6, 2) numpy arrays.
    """
    n1 = len(hex1)
    n2 = len(hex2)
    
    # Get normals for all edges
    normals = []
    for i in range(n1):
        p1 = hex1[i]
        p2 = hex1[(i+1)%n1]
        e = p2 - p1
        n = np.array([-e[1], e[0]])
        norms = np.linalg.norm(n)
        if norms > 1e-9:
            normals.append(n / norms)
    for i in range(n2):
        p1 = hex2[i]
        p2 = hex2[(i+1)%n2]
        e = p2 - p1
        n = np.array([-e[1], e[0]])
        norms = np.linalg.norm(n)
        if norms > 1e-9:
            normals.append(n / norms)
    
    min_sep = float('inf')
    max_pen = 0.0
    overlapping = True
    
    for n in normals:
        # Project hex1
        proj1 = hex1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        # Project hex2
        proj2 = hex2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check overlap in this direction
        if max1 < min2 or max2 < min1:
            # Separated
            dist = min(abs(max1 - min2), abs(max2 - min1))
            if dist < min_sep:
                min_sep = dist
            overlapping = False
        else:
            # Overlapping in this direction
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                 # Should not happen if not separated, but due to float errors
                 overlap = 0
            if overlap > max_pen:
                max_pen = overlap
            # If they overlap in this direction, they might be overlapping globally
            # But we need to check if ANY direction separates them.
            
    if not overlapping:
        return min_sep
    else:
        # They overlap in all checked directions?
        # The penetration depth is the minimum overlap over all axes?
        # Actually, the penetration depth is the minimum translation to separate.
        # This is equivalent to the minimum overlap length over all separating axes?
        # Wait, if they overlap, the minimum overlap over the normals is the penetration depth?
        # Yes, for convex bodies, the minimum distance to separate is the minimum overlap length over the support function differences?
        # Let's verify. If I push A into B, the depth is how far I need to push A out.
        # This is the minimum over all directions u of the overlap length in u.
        # So we need min_overlap, not max_pen.
        
        # Re-calculate min overlap
        min_overlap = float('inf')
        for n in normals:
            proj1 = hex1 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            proj2 = hex2 @ n
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 or max2 < min1:
                # Separated, so this is not an overlap axis
                continue
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
        
        return -min_overlap

def _enclosing_side(hexagons):
    """
    Calculate the side length of the minimum enclosing regular hexagon
    centered at (0,0) with fixed orientation (sides at 30, 90, ...).
    hexagons: list of (6,2) arrays.
    """
    # Normals for the enclosing hexagon sides
    # Sides at 30, 90, 150, 210, 270, 330 degrees
    # Normals point inwards? No, we need distance from origin to support line.
    # The condition for point p to be inside hexagon with inradius r is:
    # |n_k . p| <= r for all 6 normals.
    # So r_req = max_k max_p |n_k . p|
    
    normals = []
    for k in range(6):
        angle = math.radians(30 + 60*k)
        normals.append(np.array([math.cos(angle), math.sin(angle)]))
    
    max_r = 0.0
    for h in hexagons:
        for p in h:
            for n in normals:
                dist = abs(np.dot(p, n))
                if dist > max_r:
                    max_r = dist
                    
    # max_r is the inradius
    side = max_r / _C30
    return side

def construct():
    t_start = time.time()
    
    def _remaining():
        return _DEADLINE - (time.time() - t_start)
    
    # Initial configuration: Honeycomb
    # Center (0,0), 6 around, 4 more
    init_pos = []
    init_pos.append((0.0, 0.0, 0.0))
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        init_pos.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        init_pos.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
        
    # State: x1, y1, theta1, ..., x11, y11, theta11
    # We will optimize 33 variables.
    
    def _eval(state):
        # state: (3*N,) array
        hexes = []
        for i in range(N):
            x = state[3*i]
            y = state[3*i+1]
            theta = state[3*i+2]
            hexes.append(_hex_verts(x, y, 1.0, theta))
            
        # Enclosing side
        S = _enclosing_side(hexes)
        
        # Overlap penalty
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Quick check
                dc = math.sqrt((state[3*i]-state[3*j])**2 + (state[3*i+1]-state[3*j+1])**2)
                if dc > 2.5:
                    continue
                d = _signed_dist_hex(hexes[i], hexes[j])
                if d < 0:
                    pen += d * d * 100.0
        
        return S + pen

    # Simulated Annealing
    best_state = np.array([init_pos[i][k] for i in range(N) for k in range(3)])
    best_cost = _eval(best_state)
    
    current_state = best_state.copy()
    current_cost = best_cost
    
    T = 1.0
    T_min = 0.001
    alpha = 0.999
    steps_per_temp = 100
    
    rng = np.random.RandomState(42)
    
    while T > T_min and _remaining() > 10.0:
        for _ in range(steps_per_temp):
            if _remaining() < 10.0:
                break
            
            # Perturb
            new_state = current_state.copy()
            i = rng.randint(N)
            # Move position
            new_state[3*i] += rng.randn() * 0.1
            new_state[3*i+1] += rng.randn() * 0.1
            # Rotate
            new_state[3*i+2] += rng.randn() * 0.1
            
            new_cost = _eval(new_state)
            
            delta = new_cost - current_cost
            if delta < 0 or rng.rand() < math.exp(-delta / T):
                current_state = new_state
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = current_state.copy()
        
        T *= alpha
    
    # Final polish with Nelder-Mead
    if _remaining() > 20.0:
        from scipy.optimize import minimize
        try:
            res = minimize(_eval, best_state, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_cost:
                best_state = res.x
                best_cost = res.fun
        except Exception:
            pass
    
    # Extract result
    inner = []
    for i in range(N):
        x = best_state[3*i]
        y = best_state[3*i+1]
        theta = best_state[3*i+2]
        inner.append([float(x), float(y), float(math.degrees(theta))])
    
    # Calculate final S
    hexes = [_hex_verts(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2])) for i in range(N)]
    S = _enclosing_side(hexes)
    
    # Check validity
    # If overlaps exist, it's invalid. We need to ensure no overlaps.
    # The SA should have minimized them. If not, we might need to increase S to accommodate overlaps?
    # No, the task requires disjoint hexagons.
    # If there are overlaps, the configuration is invalid.
    # We should check and if invalid, return a valid fallback (e.g. larger S with no overlaps).
    
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            d = _signed_dist_hex(hexes[i], hexes[j])
            if d < -1e-5:
                valid = False
    
    if not valid:
        # Fallback: Use the initial honeycomb with a larger S
        # The initial honeycomb has no overlaps.
        # We just need to calculate S for it.
        init_hexes = [_hex_verts(init_pos[i][0], init_pos[i][1], 1.0, init_pos[i][2]) for i in range(N)]
        S = _enclosing_side(init_hexes)
        inner = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
    
    center = (0.0, 0.0)
    angle_out = 0.0
    
    return inner, center, float(S), angle_out
