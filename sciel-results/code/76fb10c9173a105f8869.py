import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Vertices of a unit regular hexagon centered at origin, angle 0
    # Flat-top orientation: vertices at 0, 60, 120, 180, 240, 300 degrees
    base_hex_vertices = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    def get_hex_vertices(center, angle_deg, side_len=1.0):
        angle_rad = np.deg2rad(angle_deg)
        R = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)]
        ])
        return center + side_len * (R @ base_hex_vertices.T).T

    def min_dist_between_hexes(v1, v2):
        """
        v1, v2: arrays of shape (6, 2)
        Returns minimum distance between any edge of v1 and any edge of v2.
        Positive if separated, negative if overlapping (approximation using segment distances).
        For regular convex polygons, we can check distance between all pairs of segments.
        """
        min_d = float('inf')
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                # Distance between segment p1-p2 and p3-p4
                # Using vector projection
                d1 = p2 - p1
                d2 = p4 - p3
                denom = np.dot(d1, d1) * np.dot(d2, d2) - np.dot(d1, d2)**2
                if abs(denom) < 1e-10:
                    # Parallel
                    d = np.linalg.norm(p1 - p3)
                else:
                    s = (np.dot(d1, d2) * np.dot(d2, p1 - p3) - np.dot(d1, d1) * np.dot(d2, p1 - p4)) / denom
                    t = (np.dot(d1, d2) * np.dot(d1, p1 - p4) - np.dot(d2, d2) * np.dot(d1, p1 - p3)) / denom
                    s = max(0.0, min(1.0, s))
                    t = max(0.0, min(1.0, t))
                    p_closest1 = p1 + s * d1
                    p_closest2 = p3 + t * d2
                    d = np.linalg.norm(p_closest1 - p_closest2)
                if d < min_d:
                    min_d = d
        return min_d

    def check_containment(inner_verts, outer_center, outer_side, outer_angle_deg):
        """
        Checks if inner_verts (6x2) are inside the outer hexagon.
        Returns the maximum penetration depth (positive if outside).
        """
        outer_verts = get_hex_vertices(outer_center, outer_angle_deg, outer_side)
        # Normal vectors for the 6 edges of the outer hexagon
        # Edge i is from outer_verts[i] to outer_verts[(i+1)%6]
        max_penetration = 0.0
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1)%6]
            edge_vec = p2 - p1
            # Inward normal
            # For a convex polygon defined CCW, inward normal is (edge_y, -edge_x) rotated?
            # Let's just use the fact that center is inside.
            # Normal pointing towards center
            mid_edge = (p1 + p2) / 2.0
            normal = outer_center - mid_edge
            normal = normal / np.linalg.norm(normal)
            
            # Project inner vertices onto normal
            # Distance from edge line
            # Line eq: normal . (x - p1) = 0
            # Inside if normal . (x - p1) <= 0 (since normal points to center, and center is on inside)
            # Wait, if normal points to center, then for any point x inside, normal . (x - p1) should be <= 0?
            # Let's check: center C. normal = C - mid. normal . (C - p1) > 0.
            # So inside condition is normal . (x - p1) <= 0 is WRONG.
            # Let's define normal pointing OUTWARDS.
            # Outward normal is -normal.
            # Inside condition: outward_normal . (x - p1) <= 0.
            # Penetration = max(0, outward_normal . (x - p1))
            
            outward_normal = -normal
            outward_normal = outward_normal / np.linalg.norm(outward_normal)
            
            dists = np.dot(inner_verts - p1, outward_normal)
            pen = np.max(dists)
            if pen > max_penetration:
                max_penetration = pen
        return max_penetration

    def calculate_energy(state, inner_centers, inner_angles, outer_center, outer_side, outer_angle):
        overlap_pen = 0.0
        contain_pen = 0.0
        
        # Calculate vertices for all inner hexagons
        inner_verts_list = []
        for i in range(N):
            v = get_hex_vertices(inner_centers[i], inner_angles[i])
            inner_verts_list.append(v)
            
            # Containment
            pen = check_containment(v, outer_center, outer_side, outer_angle)
            contain_pen += pen * pen # Squared penalty for smoothness
            
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_between_hexes(inner_verts_list[i], inner_verts_list[j])
                if d < 0:
                    overlap_pen += (-d) ** 2

        # Total energy: We want to minimize outer_side, but we must satisfy constraints.
        # If constraints are violated, energy is huge.
        # If valid, energy is outer_side.
        # To allow optimization, we use a weighted sum.
        # However, for SA, it's better to have a single scalar.
        # Let's use: E = outer_side + 100 * (overlap_pen + contain_pen)
        # But if we just minimize outer_side, we might get stuck with overlaps.
        # Better: E = outer_side if valid, else outer_side + 1000 * violation.
        
        violation = overlap_pen + contain_pen
        if violation > 1e-6:
            return outer_side + 100.0 * violation
        else:
            return outer_side

    # Initialize
    # Hexagonal grid of 12 hexagons
    # Center 1, ring 6, ring 5? No, 1+6=7. 1+6+5=12?
    # Let's place them in a honeycomb pattern.
    # Row 0: 2 hexes
    # Row 1: 3 hexes
    # Row 2: 3 hexes
    # Row 3: 2 hexes
    # Row 4: 2 hexes?
    # Total 12.
    
    # Simpler: Start with a loose grid and let SA tighten it.
    inner_centers = np.zeros((N, 2))
    inner_angles = np.zeros(N)
    
    # Generate hex grid positions
    # Offset coordinates
    # q, r
    # x = size * 3/2 * q
    # y = size * sqrt(3) * (r + q/2)
    # For unit hexes, distance between centers is sqrt(3) if edge-to-edge?
    # Side length 1. Distance between centers of adjacent hexes is sqrt(3).
    dx = 1.5
    dy = math.sqrt(3)/2
    # Actually, for flat-top hexes:
    # Horizontal distance: 1.5 * side * 2? No.
    # Width = 2 * side. Height = sqrt(3) * side.
    # Horizontal step: 1.5 * side.
    # Vertical step: sqrt(3) * side.
    # For adjacent columns, vertical shift is sqrt(3)/2 * side.
    
    # Let's just use a simple circular packing initialization to avoid grid bias
    for i in range(N):
        angle = 2 * math.pi * i / N
        r = 2.0
        inner_centers[i] = [r * math.cos(angle), r * math.sin(angle)]
        inner_angles[i] = 0.0
        
    outer_center = np.array([0.0, 0.0])
    # Estimate initial side length
    # Max distance from center to vertex of inner hexes
    max_r = 0
    for i in range(N):
        v = get_hex_vertices(inner_centers[i], inner_angles[i])
        dists = np.linalg.norm(v, axis=1)
        max_r = max(max_r, np.max(dists))
    # Circumradius of outer hex is R = side.
    # Inradius r = side * sqrt(3)/2.
    # We need inradius >= max_r? No, we need the hex to contain them.
    # If aligned, side >= max_r / cos(angle_offset).
    # Just set side to max_r + 1.0
    outer_side = max_r + 1.0
    outer_angle = 0.0
    
    best_state = {
        'inner_centers': inner_centers.copy(),
        'inner_angles': inner_angles.copy(),
        'outer_center': outer_center.copy(),
        'outer_side': outer_side,
        'outer_angle': outer_angle
    }
    best_energy = calculate_energy(best_state, inner_centers, inner_angles, outer_center, outer_side, outer_angle)
    
    current_state = best_state.copy()
    current_energy = best_energy
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    step_size_pos = 0.5
    step_size_angle = 10.0
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate neighbor
        new_state = current_state.copy()
        new_inner_centers = current_state['inner_centers'].copy()
        new_inner_angles = current_state['inner_angles'].copy()
        new_outer_center = current_state['outer_center'].copy()
        new_outer_side = current_state['outer_side']
        new_outer_angle = current_state['outer_angle']
        
        move_type = np.random.randint(0, 4)
        if move_type == 0:
            # Move an inner hex
            i = np.random.randint(0, N)
            new_inner_centers[i] += np.random.normal(0, step_size_pos, 2)
        elif move_type == 1:
            # Rotate an inner hex
            i = np.random.randint(0, N)
            new_inner_angles[i] += np.random.normal(0, step_size_angle)
        elif move_type == 2:
            # Move outer center
            new_outer_center += np.random.normal(0, step_size_pos * 0.5, 2)
        else:
            # Adjust outer side or angle
            if np.random.rand() < 0.5:
                new_outer_side *= (1.0 + np.random.normal(0, 0.05))
                if new_outer_side < 1.0: new_outer_side = 1.0
            else:
                new_outer_angle += np.random.normal(0, step_size_angle)
                
        new_energy = calculate_energy(new_state, new_inner_centers, new_inner_angles, new_outer_center, new_outer_side, new_outer_angle)
        
        delta = new_energy - current_energy
        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            current_state = new_state
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = current_state.copy()
                
        T *= cooling_rate
        
    # Final validation and cleanup
    # Ensure the best state is valid
    bc = best_state['inner_centers']
    ba = best_state['inner_angles']
    oc = best_state['outer_center']
    os_ = best_state['outer_side']
    oa = best_state['outer_angle']
    
    # Check validity
    valid = True
    inner_verts_list = []
    for i in range(N):
        v = get_hex_vertices(bc[i], ba[i])
        inner_verts_list.append(v)
        if check_containment(v, oc, os_, oa) > 1e-6:
            valid = False
            break
            
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_between_hexes(inner_verts_list[i], inner_verts_list[j])
                if d < -1e-6:
                    valid = False
                    break
            if not valid: break
            
    if not valid:
        # Fallback to a known valid configuration (e.g., seed-like but tighter)
        # Return a valid configuration even if not optimal
        inner = []
        for k in range(-(N // 2), N - N // 2):
            inner.append([4.0 * k, 0.0, 0.0])
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Format output
    inner = []
    for i in range(N):
        x, y = bc[i]
        angle_deg = ba[i] % 360.0
        inner.append([x, y, angle_deg])
        
    return inner, (float(oc[0]), float(oc[1])), float(os_), float(oa)

# EVOLVE-BLOCK-END
