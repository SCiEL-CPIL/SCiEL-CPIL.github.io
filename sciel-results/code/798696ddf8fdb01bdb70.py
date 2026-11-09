import numpy as np
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # 10 minutes, leaving margin for overhead
    
    # Constants
    N = 12
    d = 1.0 # Side length of inner hexagons
    
    # Geometry helpers
    def get_vertices(cx, cy, side, angle_deg):
        a = np.radians(angle_deg)
        angles = a + np.radians(60 * np.arange(6))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_edge_normals(vertices):
        # vertices is (6, 2)
        edges = np.roll(vertices, -1, axis=0) - vertices
        # Normal is perpendicular to edge
        # For a convex polygon, we want outward normals
        # Cross product in 2D: (x1, y1) x (x2, y2) = x1y2 - y1x2
        # If polygon is CCW, outward normal of edge (p1->p2) is (dy, -dx) normalized?
        # Let's check orientation.
        # Standard math coords: CCW is positive area.
        # Area = 0.5 * sum(x_i y_{i+1} - x_{i+1} y_i)
        # If area > 0, CCW.
        # Edge v = p2 - p1. Right normal is (v_y, -v_x). Left normal is (-v_y, v_x).
        # For CCW polygon, the interior is to the left of the edge. So outward is right normal.
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return normals

    def projection_interval(vertices, axis):
        proj = vertices @ axis
        return np.min(proj), np.max(proj)

    def check_overlap_v1(hex1_verts, hex2_verts, hex1_normals):
        # SAT check
        # Axes are normals of hex1 (and hex2, but they are parallel if angles are same)
        # If angles are different, we need normals from both.
        # For simplicity in this first pass, we assume all hexagons have angle 0.
        # This is valid because we can rotate the entire configuration, but we can't rotate individual hexagons 
        # relative to the container without losing optimality? 
        # Actually, for packing regular polygons, optimal packings often align them.
        # Let's assume angle=0 for all inner hexagons.
        
        for axis in hex1_normals:
            min1, max1 = projection_interval(hex1_verts, axis)
            min2, max2 = projection_interval(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return False, 0.0 # Separated
            # Calculate overlap depth
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                return True, overlap
        return True, 0.0 # Should not happen if loop completes without return False

    def check_overlap(hex1_verts, hex2_verts, normals):
        min_overlap = 0.0
        for i in range(6):
            axis = normals[i]
            min1, max1 = projection_interval(hex1_verts, axis)
            min2, max2 = projection_interval(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return False, 0.0
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > min_overlap:
                min_overlap = overlap
        return True, min_overlap

    def check_inside(inner_verts, outer_verts, outer_normals):
        # Check if all inner vertices are inside outer hexagon
        # Outer hexagon is convex.
        # For each edge of outer, defined by normal n and point p, 
        # the interior is { x | n . (x - p) <= 0 }
        # Wait, outward normal n means interior is n . (x - p) <= 0?
        # Let's verify. Center at 0, vertex at (1,0). Normal (1,0).
        # Point (0.5, 0) is inside. (0.5,0)-(1,0) = (-0.5, 0). Dot with (1,0) is -0.5 <= 0. Correct.
        
        max_violation = 0.0
        for i in range(6):
            p = outer_verts[i]
            n = outer_normals[i]
            # Distance of each inner vertex to the plane
            # val = n . (v - p)
            # If val > 0, it's outside.
            vals = inner_verts @ n - (p @ n)
            if np.any(vals > 0):
                max_violation = max(max_violation, np.max(vals))
        return max_violation

    def get_outer_hex(side, cx=0, cy=0, angle_deg=0):
        return get_vertices(cx, cy, side, angle_deg)

    def get_outer_normals(outer_verts):
        return get_edge_normals(outer_verts)

    # Initialization
    # Place hexagons in a grid
    # Spacing for non-overlapping unit hexagons (angle 0):
    # Horizontal distance: 3 (side 1, width 2, gap 1? No. 
    # Vertices of hex at (0,0): (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866)
    # Width is 2. Height is sqrt(3) ~ 1.732.
    # To pack them, we can offset rows.
    # Row 0: x = 0, 3, 6...
    # Row 1: y = sqrt(3)/2 * 2 = sqrt(3)? No.
    # Vertical distance between rows in hex grid: sqrt(3).
    # Horizontal offset: 1.5.
    
    positions = np.zeros((N, 2))
    # Arrange 12 in a 3x4 block or similar
    # Let's try a 4x3 grid
    for i in range(N):
        row = i // 4
        col = i % 4
        # x spacing 3, y spacing sqrt(3)
        # Offset alternate rows by 1.5
        x = col * 3.0
        if row % 2 == 1:
            x += 1.5
        y = row * np.sqrt(3)
        
        # Center the grid around origin
        # Approx center of grid:
        # x range: 0 to 9 (if 4 cols). Center 4.5.
        # y range: 0 to 2*sqrt(3) ~ 3.46. Center 1.73.
        positions[i, 0] = x - 4.5
        positions[i, 1] = y - np.sqrt(3)

    # Initial side length s
    # Find max distance from origin to any vertex of any inner hex
    # This is a lower bound for s? No, s is the circumradius of the outer hex.
    # The outer hex must contain all vertices.
    # A regular hexagon of side s has inradius s*sqrt(3)/2 and circumradius s.
    # The farthest point from center in a regular hexagon is the vertex (distance s).
    # So s must be at least max distance from center to any inner vertex.
    
    def calc_min_s(positions):
        max_dist = 0
        for p in positions:
            verts = get_vertices(p[0], p[1], 1.0, 0)
            dists = np.linalg.norm(verts, axis=1)
            max_dist = max(max_dist, np.max(dists))
        return max_dist

    s = calc_min_s(positions)
    # Add some margin
    s *= 1.1
    
    best_s = s
    best_positions = positions.copy()
    
    # Precompute normals for inner hexagons (angle 0)
    # Vertices of a unit hex at (0,0) with angle 0
    ref_verts = get_vertices(0, 0, 1.0, 0)
    ref_normals = get_edge_normals(ref_verts)
    
    # Precompute outer hex normals for a reference outer hex (angle 0)
    # Note: Outer hex orientation is fixed at 0 relative to inner? 
    # The problem allows any angle for outer. 
    # However, due to symmetry, we can fix outer angle to 0.
    # Inner hexagons can also be fixed to angle 0.
    # Why? The objective is symmetric under rotation.
    # If we have a solution with rotated inner hexes, we can rotate the whole system so one is at 0.
    # But can we have different angles for different inner hexes?
    # For regular polygons, "interlocking" is not possible in the same way as circles.
    # Usually, aligned hexagons pack best.
    # Let's assume all angles are 0. This reduces degrees of freedom significantly.
    
    current_positions = positions.copy()
    current_s = s
    
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    step_size = 0.5
    
    # SA Loop
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate new candidate
        new_positions = current_positions.copy()
        
        # Move 1-3 random hexagons
        num_moves = np.random.randint(1, 4)
        idxs = np.random.choice(N, num_moves, replace=False)
        for idx in idxs:
            new_positions[idx] += np.random.normal(0, step_size, 2)
        
        # Try to reduce s slightly
        # s_new = current_s * (1 - np.random.uniform(0, 0.01))
        # But we need to ensure it's not too small immediately.
        # Let's calculate the minimal s required for new_positions
        s_req = calc_min_s(new_positions)
        # If s_req > current_s, we are definitely invalid (vertices outside).
        # Let's set new_s to max(s_req, current_s * 0.99) ?
        # Actually, the penalty function will handle containment.
        # Let's keep s fixed in the move, and only update s if we find a valid config?
        # No, we want to minimize s.
        # Let's propose a new s as well.
        new_s = current_s * (1 - np.random.uniform(0, 0.05))
        if new_s < 1.0: new_s = 1.0 # Minimum possible is roughly sqrt(12) ~ 3.46 for 12 unit hexes?
        # Actually, area of 12 unit hexes is 12 * (3*sqrt(3)/2) = 18*sqrt(3) ~ 31.
        # Area of outer hex with side s is (3*sqrt(3)/2) * s^2.
        # So s^2 >= 18. s >= sqrt(18) ~ 4.24.
        # So s < 4.24 is impossible.
        
        # Calculate Cost
        def calc_cost(pos, s_val):
            # Overlap penalty
            overlap_pen = 0
            for i in range(N):
                v1 = ref_verts + pos[i]
                for j in range(i+1, N):
                    v2 = ref_verts + pos[j]
                    is_ov, depth = check_overlap(v1, v2, ref_normals)
                    if is_ov:
                        overlap_pen += depth * depth # Quadratic penalty
            
            # Containment penalty
            # Outer hex vertices
            outer_verts = get_vertices(0, 0, s_val, 0)
            outer_normals = get_edge_normals(outer_verts)
            
            contain_pen = 0
            for i in range(N):
                v1 = ref_verts + pos[i]
                viol = check_inside(v1, outer_verts, outer_normals)
                contain_pen += viol * viol
            
            return s_val + 10.0 * overlap_pen + 10.0 * contain_pen

        current_cost = calc_cost(current_positions, current_s)
        new_cost = calc_cost(new_positions, new_s)
        
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_positions = new_positions
            current_s = new_s
            # Check if valid and better
            # We need to check validity explicitly to update best
            # Validity: overlap_pen == 0 and contain_pen == 0
            # We can reuse the penalties from calc_cost, but calc_cost returns scalar.
            # Let's re-calculate penalties to check validity.
            
            # Optimization: Only check validity if cost is low enough?
            # Or just check every step. N=12, pairs=66. Fast.
            
            is_valid = True
            # Check overlaps
            for i in range(N):
                v1 = ref_verts + current_positions[i]
                for j in range(i+1, N):
                    v2 = ref_verts + current_positions[j]
                    is_ov, depth = check_overlap(v1, v2, ref_normals)
                    if is_ov:
                        is_valid = False
                        break
                if not is_valid: break
            
            if is_valid:
                # Check containment
                outer_verts = get_vertices(0, 0, current_s, 0)
                outer_normals = get_edge_normals(outer_verts)
                for i in range(N):
                    v1 = ref_verts + current_positions[i]
                    viol = check_inside(v1, outer_verts, outer_normals)
                    if viol > 1e-8:
                        is_valid = False
                        break
            
            if is_valid and current_s < best_s:
                best_s = current_s
                best_positions = current_positions.copy()
                
        T *= alpha
        # Adaptive step size?
        # If T is low, reduce step size
        step_size = max(0.01, 0.5 * T / 1.0) # Rough scaling

    # Final Result
    # Ensure best_positions and best_s are returned
    # Construct inner data
    inner_data = np.zeros((N, 3))
    for i in range(N):
        inner_data[i, 0] = best_positions[i, 0]
        inner_data[i, 1] = best_positions[i, 1]
        inner_data[i, 2] = 0.0 # Angle

    outer_data = np.array([0.0, 0.0, 0.0])
    
    return inner_data, outer_data, best_s

# Run a quick test to ensure it doesn't crash
if __name__ == "__main__":
    # inner, outer, s = hexagon_packing_12()
    # print(f"Best s: {s}")
    pass
