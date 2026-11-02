"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def hexagons_disjoint_fast(h1, h2):
        """Fast check for disjointness using SAT (Separating Axis Theorem)."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2.0, definitely disjoint
        if dist_sq > 4.0:
            return True
        
        # Precise SAT check
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # Check all 12 edge normals (6 from each hexagon)
        for i in range(6):
            # Edge normal from h1
            e1 = v1[(i+1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            nn1 = np.linalg.norm(n1)
            if nn1 > 1e-12:
                n1 = n1 / nn1
                proj1 = np.dot(v1, n1)
                proj2 = np.dot(v2, n1)
                if np.max(proj1) < np.min(proj2) - 1e-10 or np.max(proj2) < np.min(proj1) - 1e-10:
                    return True
            
            # Edge normal from h2
            e2 = v2[(i+1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            nn2 = np.linalg.norm(n2)
            if nn2 > 1e-12:
                n2 = n2 / nn2
                proj1 = np.dot(v1, n2)
                proj2 = np.dot(v2, n2)
                if np.max(proj1) < np.min(proj2) - 1e-10 or np.max(proj2) < np.min(proj1) - 1e-10:
                    return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def compute_min_side_detailed(inner, center, angle):
        """Compute min side and return which constraint is active."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def is_valid(inner):
        """Check if all hexagons are pairwise disjoint."""
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return False
        return True

    def generate_honeycomb(scale=1.0, offset_x=0.0, offset_y=0.0, rot_deg=0.0):
        """Generate a compact honeycomb arrangement of 12 unit hexagons.
        Uses a 3x4 grid with hexagonal spacing for tight packing.
        Returns list of (x, y, angle_deg) for each hexagon.
        """
        positions = []
        # For flat-top hexagons of side 1:
        # - Horizontal spacing between columns: 3/2 * 1 * 2 = 3? No.
        # Let's use the standard honeycomb spacing:
        # For flat-top hexagons, adjacent columns are offset by sqrt(3)/2 vertically
        # and 3/2 horizontally (for touching hexagons)
        # Actually, let me use a simpler approach: place in a 3x4 grid with 
        # spacing that ensures no overlap.
        
        # Standard honeycomb for flat-top hexagons:
        # Column spacing: 3/2 * side * 2 = 3? No, that's for pointy-top.
        # For flat-top: width = 2, height = sqrt(3)
        # Horizontal neighbors in same row: distance = 2 (touching at vertices)
        # But in honeycomb, adjacent columns are offset by sqrt(3)/2 and 3/2 apart horizontally
        
        # Let me use a proven compact arrangement:
        # 4 columns, 3 rows, with hexagonal offset
        col_spacing = 1.5  # 3/2 * side for flat-top honeycomb
        row_spacing = np.sqrt(3)  # height of hexagon
        
        for col in range(4):
            for row in range(3):
                x = col * col_spacing
                y = row * row_spacing
                if col % 2 == 1:
                    y += row_spacing / 2  # Offset odd columns
                positions.append((x, y, 0.0))
        
        # Center the arrangement
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        # Apply rotation and translation
        rot_rad = np.radians(rot_deg)
        c, s = np.cos(rot_rad), np.sin(rot_rad)
        
        rotated_positions = []
        for x, y, a in positions:
            x_c = x - cx
            y_c = y - cy
            x_r = x_c * c - y_c * s + offset_x
            y_r = x_c * s + y_c * c + offset_y
            a_r = a + rot_deg
            rotated_positions.append((x_r, y_r, a_r))
        
        return rotated_positions

    def objective_cluster(offset_x, offset_y, rot_deg, outer_angle):
        """Objective for cluster-based approach: minimize outer hexagon side length.
        The cluster topology is fixed (honeycomb), we optimize its placement and rotation.
        """
        positions = generate_honeycomb(1.0, offset_x, offset_y, rot_deg)
        inner = [(p[0], p[1], p[2]) for p in positions]
        center = (0.0, 0.0)
        side = compute_min_side_fast(inner, center, outer_angle)
        return side

    def find_critical_orientations(inner, center):
        """Find all critical outer hexagon orientations where a vertex lies on a support line.
        Returns a sorted list of unique critical angles in [0, 60).
        """
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        
        critical_angles = set()
        
        # For each vertex, the critical outer angle is when the vertex direction
        # aligns with an outer hexagon edge normal
        for vx, vy in offsets:
            # The angle from center to this vertex
            vert_angle = np.degrees(np.arctan2(vy, vx)) % 60.0
            
            # The outer hexagon edge normals are at angle + 30 + k*60
            # Critical when vert_angle = angle + 30 + k*60 (mod 360)
            # So angle = vert_angle - 30 - k*60 (mod 60)
            for k in range(6):
                crit_angle = (vert_angle - 30.0 - k * 60.0) % 60.0
                critical_angles.add(round(crit_angle, 6))
        
        # Also add 0 and 30 as baseline orientations
        critical_angles.add(0.0)
        critical_angles.add(30.0)
        
        return sorted(critical_angles)

    def full_objective(params):
        """Full objective: params = [offset_x, offset_y, rot_deg, outer_angle]
        Uses fixed honeycomb topology.
        """
        offset_x, offset_y, rot_deg, outer_angle = params
        return objective_cluster(offset_x, offset_y, rot_deg, outer_angle)

    # Strategy: Use critical orientation analysis.
    # For a fixed cluster (position + rotation), the optimal outer hexagon
    # orientation must be at a "critical" angle where a vertex of the cluster
    # lies exactly on a support line of the outer hexagon.
    
    # Phase 1: Coarse search over cluster rotation (rot_deg) and outer angle
    best_params = np.array([0.0, 0.0, 0.0, 0.0])
    best_score = float('inf')
    
    # Try multiple cluster rotations
    for rot_deg in np.linspace(0, 60, 13):
        if time.time() > deadline - 40.0:
            break
        
        # For this cluster rotation, find critical outer angles
        positions = generate_honeycomb(1.0, 0.0, 0.0, rot_deg)
        inner = [(p[0], p[1], p[2]) for p in positions]
        center = (0.0, 0.0)
        
        critical_angles = find_critical_orientations(inner, center)
        
        # Evaluate each critical angle
        for outer_angle in critical_angles:
            if time.time() > deadline - 35.0:
                break
            side = compute_min_side_fast(inner, center, outer_angle)
            if side < best_score:
                best_score = side
                best_params = np.array([0.0, 0.0, rot_deg, outer_angle])
    
    # Phase 2: Refine the best cluster rotation using local search
    if time.time() < deadline - 30.0:
        best_rot = best_params[2]
        best_outer = best_params[3]
        
        # Local search around best rotation
        for step in [0.5, 0.2, 0.1, 0.05]:
            improved = True
            while improved and time.time() < deadline - 25.0:
                improved = False
                for da in np.linspace(-step, step, 21):
                    test_rot = best_rot + da
                    positions = generate_honeycomb(1.0, 0.0, 0.0, test_rot)
                    inner = [(p[0], p[1], p[2]) for p in positions]
                    center = (0.0, 0.0)
                    
                    critical_angles = find_critical_orientations(inner, center)
                    for outer_angle in critical_angles:
                        side = compute_min_side_fast(inner, center, outer_angle)
                        if side < best_score:
                            best_score = side
                            best_rot = test_rot
                            best_outer = outer_angle
                            best_params = np.array([0.0, 0.0, best_rot, best_outer])
                            improved = True
                            break
                    if improved:
                        break
    
    # Phase 3: Fine-grained refinement of outer angle around the best critical angle
    if time.time() < deadline - 10.0:
        best_rot = best_params[2]
        best_outer = best_params[3]
        
        positions = generate_honeycomb(1.0, 0.0, 0.0, best_rot)
        inner = [(p[0], p[1], p[2]) for p in positions]
        center = (0.0, 0.0)
        
        # Fine scan around best outer angle
        for step in [0.1, 0.05, 0.02]:
            for da in np.linspace(-step*2, step*2, 41):
                test_outer = best_outer + da
                side = compute_min_side_fast(inner, center, test_outer)
                if side < best_score:
                    best_score = side
                    best_outer = test_outer
        
        best_params[3] = best_outer
    
    # Extract final result
    offset_x, offset_y, rot_deg, outer_angle = best_params
    positions = generate_honeycomb(1.0, offset_x, offset_y, rot_deg)
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in positions]
    center = (0.0, 0.0)
    side = compute_min_side_fast(inner, center, outer_angle)
    
    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use simple grid
        inner = []
        spacing = 2.1
        for row in range(3):
            for col in range(4):
                x = col * spacing - 3 * spacing / 2
                y = row * spacing - spacing
                inner.append([float(x), float(y), 0.0])
        center = (0.0, 0.0)
        outer_angle = 0.0
        side = compute_min_side_fast(inner, center, outer_angle)

    return inner, center, side, float(outer_angle)
# EVOLVE-BLOCK-END
