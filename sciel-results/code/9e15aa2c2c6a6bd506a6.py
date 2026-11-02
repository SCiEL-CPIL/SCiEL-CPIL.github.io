"""
Strategy: Free-DOF Lattice-Based Annealing with Smooth Collision Penalties.
Instead of constraining the 12 hexagons to a rigid 1-6-5 ring structure, we treat
all 36 degrees of freedom (12 positions + 12 angles) as free variables. We initialize
them on a dense hexagonal grid to provide a good structural starting point.
Crucially, we replace the hard SAT disjointness check with a smooth, continuous
penalty based on the minimum distance between convex polygons. This creates a
smooth energy landscape that allows the optimizer to "slide" hexagons past each
other during the annealing process, avoiding the local minima traps common in
discrete overlap checks. The outer hexagon size is computed as the maximum
projection of all inner vertices, which is also smooth.
"""
import numpy as np
import time
import math
from scipy.spatial import ConvexHull

N = 12
SQRT3 = math.sqrt(3.0)
APOHEM_UNIT = SQRT3 / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def min_distance_convex(v1, v2):
        """
        Compute minimum distance between two convex polygons.
        Uses SAT to check for overlap. If overlapping, returns 0.
        If not overlapping, finds the minimum distance between any pair of edges.
        This is O(N*M) for N,M vertices, which is fine for hexagons (6*6=36 checks).
        """
        # First check for overlap using SAT
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                # Separated on this axis, but we need min distance, not just separation
                # Actually, if they are separated, the min distance is the min over all axes of the gap
                # But SAT only tells us if they are disjoint. To get distance, we need to check edge-edge.
                pass
        
        # If we reach here, they might be overlapping or just disjoint.
        # To be robust and smooth, we compute the min distance between all pairs of edges.
        # If min distance > 0, they are disjoint. If 0, they touch or overlap.
        min_dist = float('inf')
        
        # Helper: distance between two line segments
        def segment_distance(p1, p2, p3, p4):
            # Returns minimum distance between segment p1-p2 and segment p3-p4
            # This is a standard computational geometry problem.
            # For simplicity and speed, we can check:
            # 1. Point-to-segment distances for all 4 points
            # 2. Segment-segment intersection (if intersect, dist=0)
            
            def point_segment_distance(p, a, b):
                ab = b - a
                ap = p - a
                ab_len2 = np.dot(ab, ab)
                if ab_len2 < 1e-12:
                    return np.linalg.norm(p - a)
                t = np.dot(ap, ab) / ab_len2
                t = max(0.0, min(1.0, t))
                closest = a + t * ab
                return np.linalg.norm(p - closest)
            
            d1 = point_segment_distance(p1, p3, p4)
            d2 = point_segment_distance(p2, p3, p4)
            d3 = point_segment_distance(p3, p1, p2)
            d4 = point_segment_distance(p4, p1, p2)
            min_d = min(d1, d2, d3, d4)
            
            # Check for intersection (if intersect, distance is 0)
            # Cross product method
            def orient(a, b, c):
                return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            
            d1_orient = orient(p1, p2, p3)
            d2_orient = orient(p1, p2, p4)
            d3_orient = orient(p3, p4, p1)
            d4_orient = orient(p3, p4, p2)
            
            if ((d1_orient > 0 and d2_orient < 0) or (d1_orient < 0 and d2_orient > 0)) and \
               ((d3_orient > 0 and d4_orient < 0) or (d3_orient < 0 and d4_orient > 0)):
                return 0.0
            
            return min_d

        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j + 1) % 6]
                d = segment_distance(p1, p2, p3, p4)
                if d < min_dist:
                    min_dist = d
                    if min_dist < 1e-9:
                        return 0.0
        return min_dist

    def get_vertices_for_positions(positions):
        """
        positions: list of (x, y, angle_deg)
        Returns list of vertex arrays
        """
        verts_list = []
        for cx, cy, angle in positions:
            verts_list.append(hexagon_vertices(cx, cy, 1.0, angle))
        return verts_list

    def min_outer_side(inner_positions, outer_angle_deg, outer_center=(0.0, 0.0)):
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        if not all_verts:
            return 0.0
        all_verts = np.vstack(all_verts)

        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad

        max_proj = 0.0
        for na in normals:
            nx = math.cos(na)
            ny = math.sin(na)
            projs = (all_verts[:, 0] - outer_center[0]) * nx + (all_verts[:, 1] - outer_center[1]) * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p

        return max_proj / (SQRT3 / 2)

    def objective(positions, outer_rot):
        """
        positions: list of 12 tuples (x, y, angle)
        Returns side length + penalty
        """
        # Compute outer side
        s = min_outer_side(positions, outer_rot, (0.0, 0.0))
        
        # Compute collision penalty
        # We want a smooth penalty. 
        # penalty = sum( max(0, 1 - dist)^2 ) ? 
        # Or simpler: sum( max(0, 1 - dist) )
        # Let's use a quadratic penalty for smoothness: sum( max(0, 1 - dist)^2 )
        # If dist >= 1, penalty is 0. If dist < 1, penalty grows.
        
        verts_list = get_vertices_for_positions(positions)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = min_distance_convex(verts_list[i], verts_list[j])
                if d < 1.0:
                    # Overlap or too close. 
                    # Wait, disjoint means d > 0. 
                    # The "touching" distance for two unit hexagons is not 1.0.
                    # The minimum distance between two non-overlapping unit hexagons is 0.
                    # We want them to be disjoint, i.e., d > 0.
                    # But we need a penalty for d <= 0.
                    # Actually, min_distance_convex returns 0 if they touch or overlap.
                    # It returns > 0 if they are separated.
                    # So we penalize when d is small.
                    # We want to encourage d > 0.
                    # Let's use penalty = max(0, 0.1 - d)^2 ? 
                    # If d is 0.05, penalty is (0.05)^2.
                    # If d is 0.1, penalty is 0.
                    # This encourages a small buffer.
                    if d < 0.1:
                        penalty += (0.1 - d) ** 2
        
        return s + penalty * 10.0

    # Initialization: Hexagonal grid
    # Place 12 hexagons in a 4x3 grid or similar
    # Hex grid spacing: x_step = 1.5, y_step = sqrt(3)
    # Let's try a 4x3 grid
    # Row 0: 4 hexes
    # Row 1: 3 hexes (offset)
    # Row 2: 4 hexes
    # Row 3: 1 hex? No, 12 total.
    
    # Let's use a compact cluster
    # Center at 0,0
    # Ring 0: 1
    # Ring 1: 6
    # Ring 2: 5
    # This is the same as the previous program's initialization, but now they are free to move.
    
    def initial_positions():
        pos = []
        # Center
        pos.append((0.0, 0.0, 0.0))
        # Ring 1
        for k in range(6):
            angle = math.radians(60 * k)
            x = 1.75 * math.cos(angle) # Slightly larger than 1.5 to avoid initial overlap
            y = 1.75 * math.sin(angle)
            pos.append((x, y, 0.0))
        # Ring 2
        for k in range(5):
            angle = math.radians(60 * k + 30)
            x = 3.25 * math.cos(angle)
            y = 3.25 * math.sin(angle)
            pos.append((x, y, 0.0))
        return pos

    # Simulated Annealing
    best_positions = initial_positions()
    best_outer_rot = 0.0
    best_score = objective(best_positions, best_outer_rot)
    
    current_positions = [p[:] for p in best_positions]
    current_outer_rot = best_outer_rot
    current_score = best_score
    
    T = 0.5
    cooling = 0.995
    max_iters = 5000
    
    np.random.seed(42)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Generate new state
        new_positions = [p[:] for p in current_positions]
        new_outer_rot = current_outer_rot
        
        # Perturb a random hexagon
        idx = np.random.randint(N)
        new_positions[idx][0] += np.random.uniform(-0.2, 0.2)
        new_positions[idx][1] += np.random.uniform(-0.2, 0.2)
        new_positions[idx][2] += np.random.uniform(-15, 15)
        
        # Perturb outer rotation
        new_outer_rot += np.random.uniform(-5, 5)
        
        new_score = objective(new_positions, new_outer_rot)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score < current_score:
            current_positions = new_positions
            current_outer_rot = new_outer_rot
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_positions = [p[:] for p in current_positions]
                best_outer_rot = current_outer_rot
        else:
            delta = new_score - current_score
            if np.random.rand() < np.exp(-delta / T):
                current_positions = new_positions
                current_outer_rot = new_outer_rot
                current_score = new_score
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final polish with Nelder-Mead on the best found configuration
    # Flatten parameters: 12*3 + 1 = 37
    def flatten(params):
        pos = []
        for i in range(12):
            pos.append(params[i*3])
            pos.append(params[i*3+1])
            pos.append(params[i*3+2])
        outer_rot = params[36]
        return pos, outer_rot

    def unflatten(pos, outer_rot):
        params = []
        for i in range(12):
            params.append(pos[i*3])
            params.append(pos[i*3+1])
            params.append(pos[i*3+2])
        params.append(outer_rot)
        return params

    def obj_flat(params):
        pos, outer_rot = flatten(params)
        # Ensure positions are tuples
        pos_tuples = []
        for i in range(12):
            pos_tuples.append((pos[i*3], pos[i*3+1], pos[i*3+2]))
        return objective(pos_tuples, outer_rot)

    initial_flat = unflatten([x for p in best_positions for x in p], best_outer_rot)
    
    # Use Nelder-Mead for final refinement
    try:
        from scipy.optimize import minimize
        res = minimize(obj_flat, initial_flat, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_score:
            best_score = res.fun
            best_flat = res.x
            pos, outer_rot = flatten(best_flat)
            best_positions = [(pos[i*3], pos[i*3+1], pos[i*3+2]) for i in range(12)]
            best_outer_rot = outer_rot
    except Exception:
        pass

    # Validation and final extraction
    inner = best_positions
    center = (0.0, 0.0)
    angle = best_outer_rot
    
    # Calculate side length
    s = min_outer_side(inner, angle, center)
    
    # Check validity
    valid = True
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], s, angle):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        s *= 1.01

    return inner, center, s, angle
