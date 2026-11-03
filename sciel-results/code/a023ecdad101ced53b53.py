import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Objective function for optimization
    # Variables: 
    # x[0:22] -> 11 inner hexagons (x, y)
    # x[22:33] -> 11 inner hexagons (angle)
    # x[33:36] -> outer hexagon (cx, cy, side)
    # x[36] -> outer hexagon angle
    def objective(params):
        if time.time() > deadline:
            return 1e6  # Return large penalty to stop optimization

        # Unpack
        inner_pos = params[0:22].reshape(11, 2)
        inner_angles = params[22:33]
        outer_cx, outer_cy, outer_side, outer_angle = params[33], params[34], params[35], params[36]

        # Penalty 1: Outer hexagon containment
        # We want outer_side to be as small as possible, but it must contain all inner hexagons.
        # Instead of checking containment and returning inf, we add a penalty for violation.
        # However, differential_evolution works better if we minimize the side length directly
        # subject to constraints? No, DE minimizes a scalar.
        # Strategy: Minimize (outer_side + lambda * violation_penalty)
        # But if we minimize outer_side, it will go to 0 unless penalized for violations.
        # Let's compute the required outer side for the current configuration and add penalty if outer_side < required_side.
        
        # Calculate required outer side
        required_side = 0.0
        a_out = math.radians(outer_angle)
        cos_a = math.cos(a_out)
        sin_a = math.sin(a_out)
        
        # Precompute inner hex vertices for speed? 
        # Inner hex side is 1.0
        # Vertices relative to center:
        # v_k = (cos(theta_k), sin(theta_k)) where theta_k = inner_angle + k*pi/3
        
        for i in range(N):
            cx_i = inner_pos[i, 0]
            cy_i = inner_pos[i, 1]
            a_in = math.radians(inner_angles[i])
            
            # Vertices of inner hex i
            # To check if vertex is inside outer hex, we transform vertex to outer hex local frame
            # Outer hex local frame: rotated by -outer_angle, centered at (0,0)
            # Point in global: (vx, vy)
            # Point in outer local: 
            # dx = vx - outer_cx
            # dy = vy - outer_cy
            # rx = dx * cos(outer_angle) + dy * sin(outer_angle)  (rotation by -a? No, rotation of frame by a means point rotation by -a)
            # Let's use standard rotation.
            # Frame R: axes rotated by a.
            # Point P in global. P_local = R^T * (P - C).
            # R = [[cos a, -sin a], [sin a, cos a]]
            # R^T = [[cos a, sin a], [-sin a, cos a]]
            
            for k in range(6):
                theta = a_in + k * math.pi / 3
                vx = cx_i + math.cos(theta)
                vy = cy_i + math.sin(theta)
                
                dx = vx - outer_cx
                dy = vy - outer_cy
                
                # Transform to outer local frame
                rx = dx * cos_a + dy * sin_a
                ry = -dx * sin_a + dy * cos_a
                
                # Check if (rx, ry) is inside regular hexagon of side S centered at origin.
                # Hexagon with flat top/bottom? No, standard vertex at angle 0?
                # Let's assume standard orientation: vertex at (S, 0).
                # Wait, my previous code used normals at 30, 90, 150.
                # This corresponds to a hexagon with flat sides at x= +/- S*sqrt(3)/2?
                # Let's stick to the support function definition used in the prompt's inspirations.
                # A regular hexagon with side S can be defined by the intersection of 3 strips.
                # If vertices are at angles 0, 60, ..., 300.
                # Normals are at 30, 90, 150, 210, 270, 330.
                # Apothem a = S * sqrt(3) / 2.
                # Condition: |proj| <= a for normals at 30, 90, 150.
                
                # Normal 1: angle 30 deg. n1 = (cos 30, sin 30) = (sqrt(3)/2, 1/2)
                # Normal 2: angle 90 deg. n2 = (0, 1)
                # Normal 3: angle 150 deg. n3 = (-sqrt(3)/2, 1/2)
                
                p1 = rx * (SQRT3/2) + ry * 0.5
                p2 = ry
                p3 = rx * (-SQRT3/2) + ry * 0.5
                
                max_proj = max(abs(p1), abs(p2), abs(p3))
                
                # Required apothem for this vertex is max_proj
                # Required side is max_proj / (sqrt(3)/2) = max_proj * 2 / sqrt(3)
                req_side_for_vertex = max_proj * 2.0 / SQRT3
                if req_side_for_vertex > required_side:
                    required_side = req_side_for_vertex

        # If current outer_side is less than required_side, add penalty
        # We want to encourage outer_side to be at least required_side.
        # If we just minimize outer_side, it will shrink until it hits required_side.
        # So objective = outer_side + penalty * max(0, required_side - outer_side)^2
        # But actually, if we allow outer_side to be a variable, we can just set outer_side = required_side in the final result?
        # No, the variable is part of the optimization.
        # Better approach: 
        # Objective = outer_side + 100 * (max(0, required_side - outer_side))**2
        # This forces outer_side to be >= required_side, and minimizes outer_side.
        
        violation = max(0, required_side - outer_side)
        penalty_containment = 100.0 * violation * violation
        
        # Penalty 2: Inner hexagon overlap
        # Check pairwise distance between inner hexagons.
        # For two hexagons with centers c1, c2 and angles a1, a2.
        # We can check if the distance between centers is less than the minimum distance for non-overlap.
        # Minimum distance depends on relative angle.
        # For aligned hexagons, min dist is sqrt(3).
        # For general angles, we can use a conservative bound or exact check.
        # Exact check: Minkowski sum of two rotated hexagons.
        # This is expensive.
        # Approximation: 
        # The minimum distance between centers of two non-overlapping unit hexagons is at least sqrt(3) (when aligned)
        # and at most 2 (when rotated 30 deg relative? No, circumradius is 1. Sum of circumradii is 2.
        # Actually, the minimum distance is the support function of the Minkowski sum.
        # To keep it fast, let's use a simpler penalty:
        # If dist < sqrt(3), they definitely overlap (for aligned).
        # If dist < 2, they might overlap.
        # Let's use a penalty based on the distance being too small.
        # A robust penalty: sum of max(0, d_min - dist)^2
        # What is d_min? 
        # For any two unit hexagons, the minimum center distance to avoid overlap is at least sqrt(3).
        # Wait, is it?
        # If one is rotated 30 deg, the "flat" side of one touches the "point" of another?
        # Let's check the distance between centers when they touch.
        # The minimum distance is the apothem of the Minkowski sum.
        # The Minkowski sum of two unit hexagons (any rotation) contains a disk of radius sqrt(3)/2 + sqrt(3)/2 = sqrt(3)?
        # No, the inradius of the Minkowski sum is the sum of inradii if they are aligned?
        # Actually, the inradius of the Minkowski sum of two convex bodies A and B is at least r_A + r_B.
        # Here r = sqrt(3)/2. So inradius of sum is at least sqrt(3).
        # So if distance < sqrt(3), they MUST overlap.
        # If distance > circumradius of sum (which is 2), they MUST NOT overlap.
        # So we can use a penalty that activates when dist < sqrt(3) + epsilon.
        # But this might be too loose?
        # Let's use a slightly tighter bound or just penalize dist < sqrt(3).
        # Actually, if they are rotated, they can be closer?
        # No, the inradius is the minimum distance from center to boundary.
        # If d < r1 + r2, they overlap.
        # So d_min = sqrt(3).
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = inner_pos[i, 0] - inner_pos[j, 0]
                dy = inner_pos[i, 1] - inner_pos[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < SQRT3:
                    # They overlap. Penalty proportional to overlap amount.
                    overlap = SQRT3 - dist
                    overlap_penalty += overlap * overlap
        
        total_penalty = penalty_containment + 10.0 * overlap_penalty
        return outer_side + total_penalty

    # Bounds for differential evolution
    # Inner positions: likely within radius 5
    # Inner angles: 0 to 360
    # Outer center: within radius 5
    # Outer side: 3 to 10 (guess)
    # Outer angle: 0 to 360
    
    bounds = []
    for i in range(11):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
    for i in range(11):
        bounds.append((0.0, 2 * math.pi)) # angle
    bounds.append((-5.0, 5.0))  # outer cx
    bounds.append((-5.0, 5.0))  # outer cy
    bounds.append((3.0, 10.0))  # outer side
    bounds.append((0.0, 2 * math.pi)) # outer angle

    # Run differential evolution
    # maxiter and popsize need to be balanced with time.
    # 150s limit. DE can be slow.
    # Let's try a few iterations.
    
    best_side = float('inf')
    best_result = None

    # To improve speed and reliability, we can run multiple shorter DE runs or one longer one.
    # Given the dimensionality (37), DE might struggle.
    # Let's try to reduce dimensions by fixing outer angle to 0? 
    # No, the prompt says outer can rotate. But symmetry might allow fixing it?
    # If we fix outer angle to 0, we lose generality.
    # Let's try DE with moderate settings.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=0.01, 
            seed=42,
            polish=True
        )
        
        if res.success or res.fun < best_side:
            best_params = res.x
            # Re-calculate the true outer side for the final configuration
            # The DE result might have outer_side slightly larger than required due to penalty smoothing.
            # We should snap outer_side to the exact required_side.
            
            inner_pos = best_params[0:22].reshape(11, 2)
            inner_angles = best_params[22:33]
            outer_cx, outer_cy = best_params[33], best_params[34]
            outer_angle = best_params[36]
            
            # Calculate exact required side
            required_side = 0.0
            a_out = math.radians(outer_angle)
            cos_a = math.cos(a_out)
            sin_a = math.sin(a_out)
            
            for i in range(N):
                cx_i = inner_pos[i, 0]
                cy_i = inner_pos[i, 1]
                a_in = math.radians(inner_angles[i])
                
                for k in range(6):
                    theta = a_in + k * math.pi / 3
                    vx = cx_i + math.cos(theta)
                    vy = cy_i + math.sin(theta)
                    
                    dx = vx - outer_cx
                    dy = vy - outer_cy
                    
                    rx = dx * cos_a + dy * sin_a
                    ry = -dx * sin_a + dy * cos_a
                    
                    p1 = rx * (SQRT3/2) + ry * 0.5
                    p2 = ry
                    p3 = rx * (-SQRT3/2) + ry * 0.5
                    
                    max_proj = max(abs(p1), abs(p2), abs(p3))
                    req_side_for_vertex = max_proj * 2.0 / SQRT3
                    if req_side_for_vertex > required_side:
                        required_side = req_side_for_vertex
                        
            # Verify no overlaps (with a small tolerance)
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    dx = inner_pos[i, 0] - inner_pos[j, 0]
                    dy = inner_pos[i, 1] - inner_pos[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < SQRT3 - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_side = required_side
                best_result = (inner_pos, inner_angles, outer_cx, outer_cy, required_side, outer_angle)
                
    except Exception as e:
        print(f"Optimization failed: {e}")

    # If DE didn't find a valid solution, try a fallback or simple heuristic
    if best_result is None:
        # Fallback: simple grid or linear arrangement
        # Linear arrangement along x-axis
        inner_pos = np.zeros((N, 2))
        for i in range(N):
            inner_pos[i, 0] = i * SQRT3
            inner_pos[i, 1] = 0.0
        inner_angles = np.zeros(N)
        
        # Calculate bounding box
        min_x = np.min(inner_pos[:, 0]) - 1
        max_x = np.max(inner_pos[:, 0]) + 1
        min_y = np.min(inner_pos[:, 1]) - 1
        max_y = np.max(inner_pos[:, 1]) + 1
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Approximate side
        width = max_x - min_x
        height = max_y - min_y
        # For a hexagon, side approx max(width/2 / cos(30), height/2 / 1) ... rough
        # Let's just use a large side to be safe
        side = max(width, height) + 2
        
        best_side = side
        best_result = (inner_pos, inner_angles, center_x, center_y, side, 0.0)

    inner_pos, inner_angles, outer_cx, outer_cy, outer_side, outer_angle = best_result
    
    # Format output
    inner_rows = []
    for i in range(N):
        inner_rows.append([float(inner_pos[i, 0]), float(inner_pos[i, 1]), float(math.degrees(inner_angles[i]))])
    
    outer_center = (float(outer_cx), float(outer_cy))
    outer_side_val = float(outer_side)
    outer_angle_deg = float(math.degrees(outer_angle))
    
    return inner_rows, outer_center, outer_side_val, outer_angle_deg

if __name__ == "__main__":
    r = construct()
    print(r)
