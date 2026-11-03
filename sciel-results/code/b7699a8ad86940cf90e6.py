"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + i * math.pi / 3.0
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return np.array(verts)
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        a = math.radians(angle_deg)
        # Transform point to hexagon's local coordinate system
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        cos_a = math.cos(-a)
        sin_a = math.sin(-a)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # In local coords, hexagon vertices are at angles 0, 60, 120, 180, 240, 300
        # The hexagon is the intersection of 6 half-planes
        # For a regular hexagon with side s, the distance from center to each edge is s * sqrt(3)/2
        # The edges are at angles 30, 90, 150, 210, 270, 330 degrees from center
        d = side * math.sqrt(3) / 2.0
        for i in range(6):
            edge_angle = math.radians(30 + i * 60)
            ex = math.cos(edge_angle)
            ey = math.sin(edge_angle)
            dist = lx * ex + ly * ey
            if dist > d + 1e-9:
                return False
        return True
    
    def hexagon_distance(h1_center, h1_side, h1_angle, h2_center, h2_side, h2_angle):
        """Compute minimum distance between two hexagons (0 if overlapping)."""
        # Use vertex-edge and edge-vertex distance checks
        v1 = hexagon_vertices(*h1_center, h1_side, h1_angle)
        v2 = hexagon_vertices(*h2_center, h2_side, h2_angle)
        
        min_dist = float('inf')
        
        # Check all vertex-edge pairs
        for i in range(6):
            p1 = v1[i]
            e1_start = v1[(i+1)%6]
            e2_start = v2[(i+1)%6]
            
            # Distance from p1 to each edge of hex2
            for j in range(6):
                e2_end = v2[(j+1)%6]
                # Point-to-segment distance
                ax, ay = e2_start
                bx, by = e2_end
                px, py = p1
                vx, vy = bx - ax, by - ay
                wx, wy = px - ax, py - ay
                c1 = vx * wx + vy * wy
                if c1 <= 0:
                    d = math.hypot(px - ax, py - ay)
                else:
                    c2 = vx * (bx - px) + vy * (by - py)
                    if c2 <= 0:
                        d = math.hypot(px - bx, py - by)
                    else:
                        t = c1 / (vx*vx + vy*vy)
                        proj = (ax + t*vx, ay + t*vy)
                        d = math.hypot(px - proj[0], py - proj[1])
                min_dist = min(min_dist, d)
            
            # Distance from p2 to each edge of hex1
            p2 = v2[i]
            for j in range(6):
                e1_end = v1[(j+1)%6]
                ax, ay = v1[j]
                bx, by = e1_end
                px, py = p2
                vx, vy = bx - ax, by - ay
                wx, wy = px - ax, py - ay
                c1 = vx * wx + vy * wy
                if c1 <= 0:
                    d = math.hypot(px - ax, py - ay)
                else:
                    c2 = vx * (bx - px) + vy * (by - py)
                    if c2 <= 0:
                        d = math.hypot(px - bx, py - by)
                    else:
                        t = c1 / (vx*vx + vy*vy)
                        proj = (ax + t*vx, ay + t*vy)
                        d = math.hypot(px - proj[0], py - proj[1])
                min_dist = min(min_dist, d)
        
        return min_dist
    
    def is_valid_placement(positions, angles, outer_side, outer_center, outer_angle):
        """Check if all inner hexagons fit inside outer and don't overlap."""
        ocx, ocy = outer_center
        # Check each inner hexagon is inside outer
        for i in range(len(positions)):
            cx, cy = positions[i]
            # Check all 6 vertices of inner hex are inside outer hex
            inner_verts = hexagon_vertices(cx, cy, 1.0, angles[i])
            for vx, vy in inner_verts:
                if not point_in_hexagon(vx, vy, ocx, ocy, outer_side, outer_angle):
                    return False
        
        # Check no overlaps between inner hexagons
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = hexagon_distance(positions[i], 1.0, angles[i], 
                                       positions[j], 1.0, angles[j])
                if dist < -1e-6:  # Overlapping
                    return False
        return True
    
    def compute_outer_side(positions, angles, outer_center, outer_angle):
        """Compute minimum outer side length needed to contain all inner hexagons."""
        ocx, ocy = outer_center
        a = math.radians(outer_angle)
        # For a regular hexagon centered at (ocx, ocy) with side s and rotation a,
        # a point (px, py) is inside if for all 6 edges, the distance to edge <= s*sqrt(3)/2
        # The maximum required s is determined by the point that is farthest in any edge direction
        
        max_ratio = 0.0
        for i in range(len(positions)):
            cx, cy = positions[i]
            inner_verts = hexagon_vertices(cx, cy, 1.0, angles[i])
            for vx, vy in inner_verts:
                dx = vx - ocx
                dy = vy - ocy
                cos_a = math.cos(-a)
                sin_a = math.sin(-a)
                lx = dx * cos_a - dy * sin_a
                ly = dx * sin_a + dy * cos_a
                for k in range(6):
                    edge_angle = math.radians(30 + k * 60)
                    ex = math.cos(edge_angle)
                    ey = math.sin(edge_angle)
                    dist = lx * ex + ly * ey
                    if dist > 0:
                        ratio = dist / (math.sqrt(3) / 2.0)
                        max_ratio = max(max_ratio, ratio)
        return max_ratio
    
    def objective(params):
        """Optimization objective: minimize outer side length."""
        if time.time() > deadline - 5:
            return float('inf')
        
        # Parse parameters: 2*N positions + N angles + 2 outer center + 1 outer angle
        n = N
        pos_x = params[0:n]
        pos_y = params[n:2*n]
        angles = params[2*n:3*n]
        ocx = params[3*n]
        ocy = params[3*n+1]
        outer_angle = params[3*n+2]
        
        positions = [(pos_x[i], pos_y[i]) for i in range(n)]
        
        # Check validity
        outer_side = compute_outer_side(positions, angles, (ocx, ocy), outer_angle)
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dist = hexagon_distance(positions[i], 1.0, angles[i], 
                                       positions[j], 1.0, angles[j])
                if dist < -0.01:  # Allow small tolerance
                    return 1000.0 + outer_side
        
        return outer_side
    
    def try_initial_layouts():
        """Try several initial layouts and return the best one."""
        best_result = None
        best_score = float('inf')
        
        # Layout 1: Linear chain
        for start_offset in [0, 1, 2]:
            positions = []
            angles = []
            for i in range(N):
                x = (i - N//2) * 2.0
                y = 0.0
                positions.append((x, y))
                angles.append(0.0)
            
            # Compute bounding
            min_x = min(p[0] for p in positions) - 1.5
            max_x = max(p[0] for p in positions) + 1.5
            min_y = min(p[1] for p in positions) - 1.5
            max_y = max(p[1] for p in positions) + 1.5
            
            for ocx in [0.0, (min_x+max_x)/2]:
                for ocy in [0.0, (min_y+max_y)/2]:
                    for oa in [0.0, 30.0]:
                        outer_side = compute_outer_side(positions, angles, (ocx, ocy), oa)
                        if outer_side < best_score:
                            best_score = outer_side
                            best_result = (positions, angles, (ocx, ocy), oa)
        
        # Layout 2: Honeycomb cluster
        # Arrange in a hexagonal pattern
        hex_positions = []
        # Center
        hex_positions.append((0, 0))
        # Ring 1: 6 hexagons
        for i in range(6):
            angle = i * 60
            x = 2.0 * math.cos(math.radians(angle))
            y = 2.0 * math.sin(math.radians(angle))
            hex_positions.append((x, y))
        # Ring 2: 4 more to make 11
        for i in [0, 2, 4, 6]:
            angle = i * 60
            x = 4.0 * math.cos(math.radians(angle))
            y = 4.0 * math.sin(math.radians(angle))
            hex_positions.append((x, y))
        
        positions = [(x, y) for x, y in hex_positions[:N]]
        angles = [0.0] * N
        
        for oa in [0.0, 30.0]:
            outer_side = compute_outer_side(positions, angles, (0.0, 0.0), oa)
            if outer_side < best_score:
                best_score = outer_side
                best_result = (positions, angles, (0.0, 0.0), oa)
        
        # Layout 3: Compact cluster with some rotations
        positions = []
        angles = []
        # Try a 3-4-4 arrangement
        for row, y in enumerate([-2, 0, 2]):
            count = 3 if row == 0 else 4
            for i in range(count):
                if len(positions) >= N:
                    break
                x = (i - (count-1)/2) * 2.0
                positions.append((x, y))
                angles.append(0.0)
        
        for oa in [0.0, 30.0]:
            outer_side = compute_outer_side(positions, angles, (0.0, 0.0), oa)
            if outer_side < best_score:
                best_score = outer_side
                best_result = (positions, angles, (0.0, 0.0), oa)
        
        return best_result, best_score
    
    # Get initial best layout
    best_result, best_score = try_initial_layouts()
    if best_result is None:
        # Fallback
        positions = [(i * 2.0, 0.0) for i in range(N)]
        angles = [0.0] * N
        best_result = (positions, angles, (0.0, 0.0), 0.0)
        best_score = compute_outer_side(positions, angles, (0.0, 0.0), 0.0)
    
    # Try to improve with optimization
    positions, angles, outer_center, outer_angle = best_result
    
    # Prepare initial parameters for optimization
    init_params = []
    for i in range(N):
        init_params.append(positions[i][0])
    for i in range(N):
        init_params.append(positions[i][1])
    for i in range(N):
        init_params.append(angles[i])
    init_params.append(outer_center[0])
    init_params.append(outer_center[1])
    init_params.append(outer_angle)
    
    init_params = np.array(init_params)
    
    # Try a few local optimizations with different starting points
    n_evals = 0
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Perturb initial parameters
        if restart == 0:
            x0 = init_params
        else:
            x0 = init_params + np.random.normal(0, 0.5, len(init_params))
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_score:
                best_score = result.fun
                params = result.x
                positions = [(params[i], params[N+i]) for i in range(N)]
                angles = params[2*N:3*N].tolist()
                outer_center = (params[3*N], params[3*N+1])
                outer_angle = params[3*N+2]
        except Exception:
            pass
    
    # Final verification and return
    # Ensure the result is valid
    positions, angles, outer_center, outer_angle = best_result if best_score > 100 else \
        ([(params[i], params[N+i]) for i in range(N)], params[2*N:3*N].tolist(), 
         (params[3*N], params[3*N+1]), params[3*N+2])
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], angles[i]])
    
    # Verify validity
    if not is_valid_placement(positions, angles, best_score, outer_center, outer_angle):
        # Fallback to a simple valid layout
        positions = [(i * 2.0, 0.0) for i in range(N)]
        angles = [0.0] * N
        outer_side = compute_outer_side(positions, angles, (0.0, 0.0), 0.0)
        inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
        return inner, (0.0, 0.0), outer_side, 0.0
    
    return inner, outer_center, best_score, outer_angle
# EVOLVE-BLOCK-END
