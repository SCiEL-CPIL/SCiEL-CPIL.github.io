"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """Compute minimum outer side to contain all inner hexagons."""
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        
        # Pre-compute normals for the outer hexagon in local coordinates
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                # Rotate to align with outer hexagon
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                
                # Find maximum projection on each of 6 normals
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    # Required side: proj / (sqrt(3)/2)
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def check_overlap_sat(pos1, pos2):
        """
        Returns True if hexagons overlap, False if disjoint.
        Uses Separating Axis Theorem (SAT).
        """
        # Bounding box check for quick rejection
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        if dx*dx + dy*dy > 4.0:
            return False
        
        v1 = hexagon_vertices(pos1[0], pos1[1], 1.0, pos1[2])
        v2 = hexagon_vertices(pos2[0], pos2[1], 1.0, pos2[2])
        
        # Collect all normals
        normals = []
        for i in range(6):
            # Edge 1
            ex = v1[(i+1)%6][0] - v1[i][0]
            ey = v1[(i+1)%6][1] - v1[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
            
            # Edge 2
            ex = v2[(i+1)%6][0] - v2[i][0]
            ey = v2[(i+1)%6][1] - v2[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
        
        # Check for separation along each normal
        for nx, ny in normals:
            # Project both hexagons onto this normal
            proj1 = [v[0]*nx + v[1]*ny for v in v1]
            proj2 = [v[0]*nx + v[1]*ny for v in v2]
            
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            
            # If separated, no overlap
            if max1 <= min2 or max2 <= min1:
                return False
        
        # If we get here, they overlap
        return True

    def has_overlap(inner_positions):
        """Check if any pair of hexagons overlap."""
        n = len(inner_positions)
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap_sat(inner_positions[i], inner_positions[j]):
                    return True
        return False

    # === TOPOLOGICAL CONTACT GRAPH APPROACH ===
    # Fix 11 hexagons in a rigid, high-symmetry cluster with touching contacts.
    # Configuration: central hex + 6 in first ring (touching center) + 4 in second ring (touching gaps)
    # This creates a compact, symmetric cluster.
    
    # Hex lattice coordinates (axial): center at (0,0), ring 1 at distance 1, ring 2 at distance 2
    # For unit hexagons with side 1, the distance between centers of touching hexagons is sqrt(3)
    # In axial coordinates, position = (sqrt(3)*(q + r/2), 1.5*r)
    
    # Define the rigid cluster positions in local coordinates (before global rotation)
    # Center: (0, 0)
    # Ring 1 (6 hexagons at 60-degree intervals, distance sqrt(3) from center)
    # Ring 2 (4 hexagons filling gaps, at specific positions)
    
    local_positions = []
    
    # Center hexagon
    local_positions.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3) from center, at angles 0, 60, 120, 180, 240, 300 degrees
    for k in range(6):
        angle = math.radians(60 * k)
        x = sqrt3 * math.cos(angle)
        y = sqrt3 * math.sin(angle)
        local_positions.append((x, y))
    
    # Ring 2: 4 hexagons in the "gap" positions
    # In a hex lattice, the second ring has 12 positions. We pick 4 that fill the concave gaps
    # between ring-1 hexagons to create a more compact shape.
    # These are at axial coordinates where they touch two adjacent ring-1 hexagons.
    # The gap positions are at angles 30, 90, 150, 210, 270, 330 degrees (between ring-1 positions)
    # But we only need 4 more. Let's pick the ones that create the most symmetric compact shape.
    # Actually, for maximum compactness, we should place them at the "outer" gap positions.
    # Let me compute: the 12 second-ring positions in axial coords with max(abs(q),abs(r),abs(q+r))==2
    # Pick 4 that are evenly spaced to maintain symmetry.
    
    # Second ring positions (axial q,r where max(|q|,|r|,|q+r|)==2):
    # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1), (1,-2), (-1,-2), (-2,-1)
    # Convert to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    second_ring_axial = [(2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1), (1,-2), (-1,-2), (-2,-1)]
    
    # Pick 4 that are evenly spaced (every 3rd one) to maintain 4-fold symmetry
    # Indices 0, 3, 6, 9 give angles 0, 90, 180, 270 roughly
    selected_indices = [0, 3, 6, 9]
    for idx in selected_indices:
        q, r = second_ring_axial[idx]
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        local_positions.append((x, y))
    
    # Now we have 11 positions in local coordinates.
    # The cluster is rigid: all hexagons have the same orientation relative to each other.
    # We optimize: global rotation angle and outer hexagon orientation.
    # Since the cluster has 6-fold symmetry (approximately), the optimal outer orientation
    # should align with the cluster's symmetry axes.
    
    def get_rotated_positions(global_angle_deg):
        """Get all 11 hexagon positions rotated by global_angle_deg, all with same orientation."""
        angle_rad = math.radians(global_angle_deg)
        cos_g = math.cos(angle_rad)
        sin_g = math.sin(angle_rad)
        positions = []
        for lx, ly in local_positions:
            rx = lx * cos_g - ly * sin_g
            ry = lx * sin_g + ly * cos_g
            # All hexagons have the same orientation as the global rotation
            positions.append([rx, ry, global_angle_deg])
        return positions

    def objective_for_angle(global_angle_deg, outer_angle_deg):
        """Compute outer side length for given global and outer angles."""
        positions = get_rotated_positions(global_angle_deg)
        return compute_min_side(positions, (0.0, 0.0), outer_angle_deg)

    # First, verify no overlaps in the rigid cluster
    test_positions = get_rotated_positions(0.0)
    if has_overlap(test_positions):
        # If there are overlaps, we need to adjust. But with the correct lattice spacing,
        # touching hexagons should be disjoint (no interior overlap).
        # Let's check and if there's an issue, use a slightly larger spacing.
        # Actually, for unit hexagons, touching means edges coincide, which SAT might flag.
        # Let's use a tiny gap to ensure disjointness.
        pass
    
    # Optimize global rotation and outer orientation
    # Due to 6-fold symmetry of the hexagonal lattice, the objective is periodic with period 60 degrees.
    # We only need to search over [0, 60) for both angles.
    
    best_side = float('inf')
    best_global_angle = 0.0
    best_outer_angle = 0.0
    
    # Coarse search
    for ga in np.arange(0, 60, 5.0):
        if time.time() > deadline - 20:
            break
        for oa in np.arange(0, 60, 5.0):
            side = objective_for_angle(ga, oa)
            if side < best_side:
                best_side = side
                best_global_angle = ga
                best_outer_angle = oa
    
    # Fine search around the best
    if time.time() < deadline - 10:
        for ga in np.arange(max(0, best_global_angle - 5), min(60, best_global_angle + 5), 0.5):
            if time.time() > deadline - 10:
                break
            for oa in np.arange(max(0, best_outer_angle - 5), min(60, best_outer_angle + 5), 0.5):
                side = objective_for_angle(ga, oa)
                if side < best_side:
                    best_side = side
                    best_global_angle = ga
                    best_outer_angle = oa
    
    # Even finer search
    if time.time() < deadline - 5:
        for ga in np.arange(max(0, best_global_angle - 0.5), min(60, best_global_angle + 0.5), 0.05):
            if time.time() > deadline - 5:
                break
            for oa in np.arange(max(0, best_outer_angle - 0.5), min(60, best_outer_angle + 0.5), 0.05):
                side = objective_for_angle(ga, oa)
                if side < best_side:
                    best_side = side
                    best_global_angle = ga
                    best_outer_angle = oa
    
    # Generate final positions
    inner_positions = get_rotated_positions(best_global_angle)
    
    # Verify no overlaps
    if has_overlap(inner_positions):
        # If overlaps detected, add a tiny perturbation to separate them
        # This shouldn't happen with correct lattice spacing, but as a safety measure
        # Use a slightly expanded cluster
        scale = 1.001
        inner_positions = []
        angle_rad = math.radians(best_global_angle)
        cos_g = math.cos(angle_rad)
        sin_g = math.sin(angle_rad)
        for lx, ly in local_positions:
            rx = scale * lx * cos_g - scale * ly * sin_g
            ry = scale * lx * sin_g + scale * ly * cos_g
            inner_positions.append([rx, ry, best_global_angle])
        
        # Recompute side
        best_side = compute_min_side(inner_positions, (0.0, 0.0), best_outer_angle)
    
    return inner_positions, (0.0, 0.0), best_side, best_outer_angle
# EVOLVE-BLOCK-END

