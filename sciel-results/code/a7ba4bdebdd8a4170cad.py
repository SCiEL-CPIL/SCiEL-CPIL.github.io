"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
import random

N = 11
_T0 = time.time()
_DEADLINE = 145.0

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _point_in_hex(px, py, cx, cy, s, rot):
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = s * math.cos(math.radians(30))
    c30 = math.cos(math.radians(30))
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _calc_penalty(positions, angles, S):
    """
    Calculate penalty for a configuration.
    positions: list of (x, y)
    angles: list of radians
    S: outer hexagon side length
    Returns: (penalty, max_violation)
    """
    penalty = 0.0
    max_violation = 0.0
    
    # Check containment in outer hexagon
    # Outer hexagon centered at (0,0) with side S, rot 0
    # Vertices of outer hexagon are at distance S from center? No, circumradius is S.
    # Inradius is S * cos(30).
    
    for i in range(N):
        xi, yi = positions[i]
        ai = angles[i]
        vi = _hex_verts(xi, yi, 1.0, ai)
        
        # Check if any vertex is outside outer hexagon
        for k in range(6):
            vx, vy = vi[k]
            if not _point_in_hex(vx, vy, 0.0, 0.0, S, 0.0):
                # Calculate distance to boundary roughly
                # Project onto normal of the violated side
                # Simpler: use distance from center as proxy or just count
                dist = math.sqrt(vx*vx + vy*vy)
                # Max dist in outer hex is S (vertices), min is S*cos(30) (midpoints)
                # If point is outside, it's beyond the boundary.
                # Approximate violation as (dist - S) if dist > S, but that's not always correct for flat sides.
                # For penalty, just adding a fixed amount or distance to center works.
                # Let's use distance from origin - S as a measure of "how far out"
                viol = max(0, dist - S)
                penalty += viol * viol
                max_violation = max(max_violation, viol)

    # Check overlaps between inner hexagons
    for i in range(N):
        xi, yi = positions[i]
        ai = angles[i]
        vi = _hex_verts(xi, yi, 1.0, ai)
        
        for j in range(i+1, N):
            xj, yj = positions[j]
            aj = angles[j]
            
            dc = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            # If centers are far apart, no overlap
            if dc > 2.0 * math.cos(math.radians(30)) * 2: # 2 * inradius * 2? No.
                # Max distance for overlap is 2 * circumradius = 2.0
                # Min distance for no overlap is 2 * inradius = sqrt(3)
                # If dc > 2.0, definitely no overlap.
                continue
            
            vj = _hex_verts(xj, yj, 1.0, aj)
            
            # Check if any vertex of i is in j
            overlap_pen = 0.0
            for k in range(6):
                if _point_in_hex(vi[k,0], vi[k,1], xj, yj, 1.0, aj):
                    overlap_pen += 1.0
            for k in range(6):
                if _point_in_hex(vj[k,0], vj[k,1], xi, yi, 1.0, ai):
                    overlap_pen += 1.0
            
            if overlap_pen > 0:
                penalty += overlap_pen * 10.0
                max_violation = max(max_violation, 1.0)

    return penalty, max_violation

def construct():
    t_start = time.time()
    n = N
    
    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    # Discrete angles in radians
    angle_options = [math.radians(a) for a in [0, 30, 60, 90, 120, 150]]
    
    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_penalty = float('inf')
    
    rng = np.random.RandomState(42)
    
    # Initial guess: Hexagonal tiling
    # Center at (0,0)
    # 6 around at distance sqrt(3) (touching)
    # 4 more further out?
    # For 11 hexagons, a common arrangement is 1 center, 6 in first ring, 4 in second ring (partial)
    # Or 1 center, 6 in first ring, 4 in "corners" of second ring.
    
    init_positions = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60.0 * k + 30.0) # Pointy top hexagons? 
        # If we use flat-top (rot 0), vertices are at 0, 60...
        # Neighbors in flat-top tiling are at distance sqrt(3) at angles 30, 90, 150, 210, 270, 330.
        init_positions.append((math.sqrt(3)*math.cos(a), math.sqrt(3)*math.sin(a)))
        
    # Add 4 more
    # Place them at the "gaps" or further out.
    # Distance for second ring is 2*sqrt(3) for direct neighbors, but we can pack tighter with rotation.
    # Let's just place them at distance 2*sqrt(3) * 0.9 to encourage tightening.
    for k in range(4):
        a = math.radians(60.0 * k)
        init_positions.append((2.0*sqrt(3)*0.9*math.cos(a), 2.0*sqrt(3)*0.9*math.sin(a)))
        
    sqrt3 = math.sqrt(3)
    init_positions = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        init_positions.append((sqrt3*math.cos(a), sqrt3*math.sin(a)))
    for k in range(4):
        a = math.radians(60.0 * k)
        init_positions.append((2.0*sqrt3*0.85*math.cos(a), 2.0*sqrt3*0.85*math.sin(a)))

    # Initial S
    # Max distance from origin in init config
    max_d = 0
    for x, y in init_positions:
        max_d = max(max_d, math.sqrt(x*x+y*y))
    # Outer hex side S needs to cover circumradius of inner hex + max_d
    # Circumradius of inner hex is 1.
    # So S >= max_d + 1.
    # But we can rotate outer hex? No, outer hex rot is 0 in our check.
    # We can rotate inner hexes.
    
    S = max_d + 1.5 # Some buffer
    
    current_positions = [p for p in init_positions]
    current_angles = [0.0] * N
    # Randomize angles initially
    for i in range(N):
        current_angles[i] = angle_options[rng.randint(0, len(angle_options))]
        
    # Calculate initial penalty
    pen, _ = _calc_penalty(current_positions, current_angles, S)
    if pen == 0:
        best_S = S
        best_positions = [p for p in current_positions]
        best_angles = [a for a in current_angles]
        
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    
    steps = 0
    while _remaining() > 5.0 and T > T_min:
        steps += 1
        
        # Pick a random hexagon to move
        i = rng.randint(0, N-1)
        
        # Move position
        dx = rng.randn() * 0.2
        dy = rng.randn() * 0.2
        new_positions = [p for p in current_positions]
        new_positions[i] = (current_positions[i][0] + dx, current_positions[i][1] + dy)
        
        # Maybe change angle
        new_angles = [a for a in current_angles]
        if rng.random() < 0.1:
            new_angles[i] = angle_options[rng.randint(0, len(angle_options))]
            
        # Check new penalty
        new_pen, _ = _calc_penalty(new_positions, new_angles, S)
        
        # Acceptance criterion
        # We want to minimize S. But S is fixed in the penalty check for now?
        # Actually, we should try to shrink S if penalty is 0.
        # Strategy: 
        # 1. If new_pen == 0, try to reduce S slightly.
        # 2. If new_pen > 0, accept with probability exp(-(new_pen - old_pen)/T) if it's not too bad?
        #    Or simply: if new_pen < old_pen, accept.
        
        old_pen = pen
        # We need to track old_pen
        if 'old_pen' not in locals():
            old_pen = pen

        if new_pen < old_pen:
            current_positions = new_positions
            current_angles = new_angles
            pen = new_pen
            # If no penalty, try to shrink S
            if pen == 0:
                # Try to reduce S
                # How much? 
                # We can binary search or just reduce by small amount and check.
                # Let's reduce by 0.01 and check.
                new_S = S - 0.01
                if new_S > 0:
                    test_pen, _ = _calc_penalty(current_positions, current_angles, new_S)
                    if test_pen == 0:
                        S = new_S
                        if S < best_S:
                            best_S = S
                            best_positions = [p for p in current_positions]
                            best_angles = [a for a in current_angles]
        else:
            # Metropolis accept
            delta = new_pen - old_pen
            if delta < 0:
                pass
            else:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_positions = new_positions
                    current_angles = new_angles
                    pen = new_pen
        
        T *= cooling
        
        # Periodically try to shrink S if we have a valid config
        if pen == 0 and steps % 100 == 0:
            # Aggressive shrink
            for _ in range(10):
                new_S = S * 0.99
                test_pen, _ = _calc_penalty(current_positions, current_angles, new_S)
                if test_pen == 0:
                    S = new_S
                else:
                    break
            if S < best_S:
                best_S = S
                best_positions = [p for p in current_positions]
                best_angles = [a for a in current_angles]

    if best_positions is None:
        # Fallback
        inner = [[float(k*2.0), 0.0, 0.0] for k in range(n)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(n):
        inner.append([float(best_positions[i][0]), float(best_positions[i][1]), float(math.degrees(best_angles[i]))])
    
    return inner, (0.0, 0.0), float(best_S), 0.0
