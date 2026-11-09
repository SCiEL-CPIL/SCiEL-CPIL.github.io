import math
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    sqrt3 = math.sqrt(3.0)
    
    # Initial configuration: Compact cluster
    # Center (0,0), 6 neighbors, 5 from next ring
    # Axial to Cartesian for flat-top hexagons (angle 0)
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y
        
    positions = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
        (2, -1),
        (1, 1),
        (-1, 2),
        (-2, 1),
        (2, 0) # Added one more to make 12? Wait, the previous list had 12.
    ]
    
    # Let's verify the count and compactness.
    # Previous list:
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) -> 6 neighbors
    # (2,0), (2,-1), (1,1), (-1,2), (-2,1) -> 5 more
    # Total 12.
    # Let's stick to the previous list which was a valid subset of the 19-cluster.
    positions = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
        (2, 0),
        (2, -1),
        (1, 1),
        (-1, 2),
        (-2, 1)
    ]

    inner = []
    for q, r in positions:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
        
    # Vertices of a unit flat-top hexagon relative to center
    hex_verts = [
        (1.0, 0.0),
        (0.5, sqrt3 / 2.0),
        (-0.5, sqrt3 / 2.0),
        (-1.0, 0.0),
        (-0.5, -sqrt3 / 2.0),
        (0.5, -sqrt3 / 2.0)
    ]

    def get_max_side(inner_pts, outer_center, outer_angle_deg):
        """
        Calculate the minimum side length of the outer hexagon required to contain
        all inner hexagons, given outer center and rotation.
        Returns inf if invalid (though we assume inner is valid).
        """
        cx, cy = outer_center
        angle_rad = math.radians(outer_angle_deg)
        
        # Normals of the outer hexagon
        # Edge normals are at angles: outer_angle + 30 + k*60
        normals = []
        for k in range(6):
            normal_angle = angle_rad + math.pi / 6 + k * math.pi / 3
            normals.append((math.cos(normal_angle), math.sin(normal_angle)))
        
        max_s = 0.0
        
        for ix, iy, i_angle in inner_pts:
            # Since inner angles are 0, vertices are fixed relative to ix, iy
            # Rotate vertices by i_angle (0)
            for vx, vy in hex_verts:
                px = ix + vx
                py = iy + vy
                
                # Project onto all normals
                for nx, ny in normals:
                    # Distance from center (cx, cy) in direction (nx, ny)
                    # The hexagon boundary is at distance s * sqrt(3) / 2 from center along normal?
                    # For a regular hexagon of side s, the apothem (distance from center to edge) is s * sqrt(3) / 2.
                    # So if the projection of a point (relative to center) is d, we need s * sqrt(3) / 2 >= d.
                    # s >= 2 * d / sqrt(3)
                    rel_x = px - cx
                    rel_y = py - cy
                    d = rel_x * nx + rel_y * ny
                    if d > 0:
                        s_req = 2.0 * d / sqrt3
                        if s_req > max_s:
                            max_s = s_req
                            
        return max_s

    def is_valid(inner_pts):
        """
        Check if inner hexagons are disjoint.
        Assuming all inner hexagons have angle 0.
        Condition: distance between centers >= sqrt(3).
        """
        n = len(inner_pts)
        for i in range(n):
            x1, y1, _ = inner_pts[i]
            for j in range(i + 1, n):
                x2, y2, _ = inner_pts[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx * dx + dy * dy
                if dist_sq < (sqrt3 * sqrt3) * 0.9999: # Slight tolerance for floating point
                    return False
        return True

    # Optimization Loop
    # We will optimize:
    # 1. Outer center (cx, cy)
    # 2. Outer angle (angle_deg)
    # 3. Inner positions (x, y)
    
    # Start with best guess
    best_inner = [p[:] for p in inner]
    best_cx = 0.0
    best_cy = 0.0
    best_angle = 0.0
    best_side = get_max_side(best_inner, (best_cx, best_cy), best_angle)
    
    # Simple Random Search / Hill Climbing
    # Parameters to perturb:
    # - Inner positions (24 params)
    # - Outer center (2 params)
    # - Outer angle (1 param)
    
    step_size = 0.1
    num_steps = 2000
    
    for i in range(num_steps):
        if time.time() > deadline:
            break
            
        # Create a candidate
        cand_inner = [p[:] for p in best_inner]
        cand_cx = best_cx
        cand_cy = best_cy
        cand_angle = best_angle
        
        # Randomly perturb a few inner points
        for _ in range(3):
            idx = random.randint(0, N - 1)
            cand_inner[idx][0] += random.uniform(-step_size, step_size)
            cand_inner[idx][1] += random.uniform(-step_size, step_size)
            
        # Randomly perturb outer center
        cand_cx += random.uniform(-step_size, step_size)
        cand_cy += random.uniform(-step_size, step_size)
        
        # Randomly perturb outer angle
        cand_angle += random.uniform(-2.0, 2.0)
        
        # Check validity
        if not is_valid(cand_inner):
            continue
            
        # Calculate score
        cand_side = get_max_side(cand_inner, (cand_cx, cand_cy), cand_angle)
        
        if cand_side < best_side:
            best_inner = cand_inner
            best_cx = cand_cx
            best_cy = cand_cy
            best_angle = cand_angle
            best_side = cand_side
            # Reduce step size slightly on improvement
            step_size *= 0.95
            if step_size < 0.01:
                step_size = 0.01
        else:
            # Occasionally increase step size if stuck?
            pass

    # Final refinement: try fine-tuning the outer angle and center with the best inner positions
    # Since inner positions are fixed now, we can solve for optimal outer center/angle more precisely?
    # Actually, just do a few more iterations with smaller steps
    
    step_size = 0.01
    for i in range(500):
        if time.time() > deadline:
            break
            
        cand_inner = [p[:] for p in best_inner]
        cand_cx = best_cx
        cand_cy = best_cy
        cand_angle = best_angle
        
        # Only perturb outer params for now to tighten the fit
        cand_cx += random.uniform(-step_size, step_size)
        cand_cy += random.uniform(-step_size, step_size)
        cand_angle += random.uniform(-0.5, 0.5)
        
        cand_side = get_max_side(best_inner, (cand_cx, cand_cy), cand_angle)
        
        if cand_side < best_side:
            best_cx = cand_cx
            best_cy = cand_cy
            best_angle = cand_angle
            best_side = cand_side

    # Add safety margin
    side = best_side * 1.001
    center = (best_cx, best_cy)
    angle = best_angle
    
    # Construct final inner list
    final_inner = []
    for x, y, a in best_inner:
        final_inner.append([x, y, a])
        
    return final_inner, center, side, angle
# EVOLVE-BLOCK-END
