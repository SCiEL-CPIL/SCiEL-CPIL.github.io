"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def hex_in_hexagon(v_hex, outer_center, outer_side, outer_angle):
    """Check if all vertices of hex are inside the outer hexagon."""
    ox, oy = outer_center
    oa = math.radians(outer_angle)
    # Outer hexagon vertices
    outer_verts = []
    for i in range(6):
        theta = oa + i * math.pi / 3
        outer_verts.append((ox + outer_side * math.cos(theta), oy + outer_side * math.sin(theta)))
    outer_verts = np.array(outer_verts)
    
    # Check if all hex vertices are inside outer hexagon using point-in-convex-polygon
    for vx, vy in v_hex:
        inside = True
        for i in range(6):
            j = (i + 1) % 6
            ax, ay = outer_verts[i]
            bx, by = outer_verts[j]
            # Cross product to check same side
            cross = (bx - ax) * (vy - ay) - (by - ay) * (vx - ax)
            if cross < -1e-9:
                inside = False
                break
        if not inside:
            return False
    return True

def hexagons_disjoint(v1, v2, tol=1e-9):
    """Check if two convex hexagons are disjoint (no overlap)."""
    # SAT test: check all axes
    for poly in [v1, v2]:
        for i in range(6):
            j = (i + 1) % 6
            # Edge normal
            ex = poly[j][0] - poly[i][0]
            ey = poly[j][1] - poly[i][1]
            # Normal perpendicular to edge
            nx, ny = -ey, ex
            # Project both polygons onto this normal
            min1 = max1 = np.dot(v1, [nx, ny])[0]
            for k in range(1, 6):
                proj = v1[k][0]*nx + v1[k][1]*ny
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
            min2 = max2 = np.dot(v2, [nx, ny])[0]
            for k in range(1, 6):
                proj = v2[k][0]*nx + v2[k][1]*ny
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
            # Check separation
            if max1 < min2 - tol or max2 < min1 - tol:
                return True  # Disjoint (separating axis found)
    return False  # No separating axis found, they overlap

def compute_min_outer_side(inner, outer_center, outer_angle):
    """Compute minimum outer side length such that all inner hexagons fit inside."""
    ox, oy = outer_center
    oa = math.radians(outer_angle)
    
    # For each inner hexagon, find the minimum side length needed
    # The outer hexagon has vertices at distance 'side' from center at angles oa + k*pi/3
    # A point (x,y) is inside outer hexagon of side s if:
    # For each of 6 directions, the projection is <= s * cos(pi/6)
    # Actually, for a regular hexagon with side length s, the apothem is s*cos(pi/6) = s*sqrt(3)/2
    # The condition for a point to be inside is that for each of the 6 normals,
    # the dot product with the normal <= apothem
    
    min_side = 0.0
    for hex_info in inner:
        hx, hy, ha = hex_info
        v_hex = hex_vertices(hx, hy, ha, 1.0)
        for vx, vy in v_hex:
            # Relative to outer center
            dx = vx - ox
            dy = vy - oy
            # The 6 normals of the outer hexagon (pointing outward)
            for k in range(6):
                theta = oa + k * math.pi / 3
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = dx * nx + dy * ny
                # For the point to be inside, proj <= apothem = side * sqrt(3)/2
                # So side >= proj * 2/sqrt(3) if proj > 0
                if proj > 0:
                    required_side = proj * 2.0 / math.sqrt(3.0)
                    if required_side > min_side:
                        min_side = required_side
    return min_side

def validate(inner, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: all hexagons inside and disjoint."""
    # Check disjointness
    for i in range(N):
        for j in range(i+1, N):
            v1 = hex_vertices(inner[i][0], inner[i][1], inner[i][2], 1.0)
            v2 = hex_vertices(inner[j][0], inner[j][1], inner[j][2], 1.0)
            if not hexagons_disjoint(v1, v2):
                return False
    # Check all inside
    for i in range(N):
        v = hex_vertices(inner[i][0], inner[i][1], inner[i][2], 1.0)
        if not hex_in_hexagon(v, outer_center, outer_side, outer_angle):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s out of 150s budget
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Initial guess: hexagonal packing
    # Place hexagons in a hexagonal arrangement
    # For 12 hexagons, we can do a center + ring pattern or linear arrangement
    
    # Try several initial configurations
    random.seed(42)
    
    # Configuration 1: Linear arrangement along x-axis
    # 12 hexagons in a line
    inner1 = [[4.0 * k, 0.0, 0.0] for k in range(-5, 7)]
    # That's 12 positions: k = -5,-4,-3,-2,-1,0,1,2,3,4,5,6
    # Actually that's 12 values
    # But they overlap! Hexagons of side 1 have width 2, so need spacing > 2
    # With spacing 4, they're disjoint
    
    # Configuration 2: Hexagonal cluster
    # Center at origin, 6 around it, 6 more
    inner2 = []
    # Center
    inner2.append([0.0, 0.0, 0.0])
    # First ring: 6 hexagons at distance ~2.1 from center
    for k in range(6):
        angle = k * 60.0
        r = 2.15
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        inner2.append([x, y, 0.0])
    # Second ring: 5 more hexagons (need 12 total, have 7, need 5 more)
    for k in range(5):
        angle = 30.0 + k * 60.0
        r = 4.0
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        inner2.append([x, y, 0.0])
    
    # Configuration 3: 2 rows of 6
    inner3 = []
    for row in range(2):
        for col in range(6):
            x = col * 2.15
            y = row * 1.9
            inner3.append([x, y, 0.0])
    
    configs = [inner1, inner2, inner3]
    
    # For each config, compute the optimal outer hexagon
    for config in configs:
        # Try different outer angles
        for outer_angle in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
            # Compute optimal center and side
            # Simple approach: center at centroid of inner hexagons
            cx = sum(h[0] for h in config) / N
            cy = sum(h[1] for h in config) / N
            center = (cx, cy)
            
            side = compute_min_outer_side(config, center, outer_angle)
            
            # Verify
            if validate(config, center, side, outer_angle):
                if side < best_score:
                    best_score = side
                    best_inner = [list(h) for h in config]
                    best_center = center
                    best_angle = outer_angle
    
    # Now do local optimization using simulated annealing / gradient-free
    if time.time() > deadline:
        if best_inner is not None:
            return best_inner, best_center, best_score, best_angle
        else:
            # Fallback
            return inner1, (0.0, 0.0), compute_min_outer_side(inner1, (0.0, 0.0), 0.0), 0.0
    
    # Start from best found so far
    if best_inner is None:
        best_inner = [list(h) for h in inner2]
        best_center = (sum(h[0] for h in best_inner)/N, sum(h[1] for h in best_inner)/N)
        best_angle = 0.0
        best_score = compute_min_outer_side(best_inner, best_center, best_angle)
    
    # Simulated annealing
    current_inner = [list(h) for h in best_inner]
    current_center = list(best_center)
    current_angle = best_angle
    current_score = best_score
    
    T = 1.0  # Temperature
    cooling = 0.999
    
    while time.time() < deadline - 5.0:  # Keep 5s margin
        if T < 0.001:
            break
        
        # Random perturbation
        new_inner = [list(h) for h in current_inner]
        new_center = list(current_center)
        new_angle = current_angle
        
        # Perturb one hexagon
        idx = random.randint(0, N-1)
        new_inner[idx][0] += random.gauss(0, 0.5)
        new_inner[idx][1] += random.gauss(0, 0.5)
        new_inner[idx][2] = (new_inner[idx][2] + random.gauss(0, 15.0)) % 360.0
        
        # Sometimes perturb center
        if random.random() < 0.1:
            new_center[0] += random.gauss(0, 0.2)
            new_center[1] += random.gauss(0, 0.2)
        
        # Sometimes perturb outer angle
        if random.random() < 0.1:
            new_angle = (new_angle + random.gauss(0, 5.0)) % 180.0
        
        # Compute new score
        new_score = compute_min_outer_side(new_inner, tuple(new_center), new_angle)
        
        # Check validity (disjointness)
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                v1 = hex_vertices(new_inner[i][0], new_inner[i][1], new_inner[i][2], 1.0)
                v2 = hex_vertices(new_inner[j][0], new_inner[j][1], new_inner[j][2], 1.0)
                if not hexagons_disjoint(v1, v2):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Reject
            T *= cooling
            continue
        
        # Accept/reject
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_inner = new_inner
            current_center = new_center
            current_angle = new_angle
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_inner = [list(h) for h in current_inner]
                best_center = tuple(current_center)
                best_angle = current_angle
        
        T *= cooling
    
    return best_inner, best_center, best_score, best_angle

# EVOLVE-BLOCK-END
