import numpy as np
import time
import math
from itertools import product

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_inner_hexagons_config(config_type="flower"):
        """
        Generates a static, dense configuration of 12 unit hexagons.
        The 'flower' config is a known tight packing: 1 center, 6 around it, 5 more around those.
        Actually, a better packing for 12 is often a 2x2x2 block or a distorted hexagon.
        Let's use a "super-hexagon" of side 2 (which has 1+6+12=19 hexes) and pick the 12 most central.
        Or simply a 12-hexagon cluster that fits in a hexagon of side ~2.5.
        
        A very tight packing of 12 hexagons is a 2x2x2 rhombus? No, hexagons don't tile in 2x2.
        Let's use the "compact" shape:
        Row 0: 2 hexes
        Row 1: 3 hexes
        Row 2: 3 hexes
        Row 3: 2 hexes
        Row 4: 2 hexes
        Total: 12.
        This forms a hexagon-like shape.
        """
        sqrt3 = math.sqrt(3)
        positions = []
        
        if config_type == "flower":
            # Center at 0,0
            positions.append((0, 0))
            # 6 around center
            for i in range(6):
                angle = i * 60
                rad = math.radians(angle)
                x = sqrt3 * math.cos(rad)
                y = 1.5 * math.sin(rad) # Wait, standard hex grid coords:
                # Distance between centers is 2 * (sqrt(3)/2 * side) = sqrt(3) for side 1?
                # No, for side 1, distance between centers of touching hexes is 2 * apothem = sqrt(3).
                # Vertices are at distance 1.
                # So yes, distance is sqrt(3).
                positions.append((x, y))
            
            # We have 7. Need 5 more.
            # Place them on the "corners" of the 6 surrounding ones to extend the shape.
            # The 6 surrounding form a hexagon of side 2 (in terms of hex units).
            # The outer vertices of this ring are at distance 2*sqrt(3) from center?
            # Let's just add 5 more in a ring further out.
            # Actually, a better strategy: use a pre-computed optimal-ish layout.
            # Layout: 
            #   . X X .
            #   X X X X
            #   X X X X
            #   . X X .
            # This is 12 hexes.
            # Axial coordinates (q, r):
            # (1, 0), (0, 0)
            # (2, 0), (1, 0), (0, 0), (-1, 0) -> wait, overlap.
            
            # Let's use a simple dense grid:
            # q, r coords
            coords = [
                (0, 0), (1, 0), (2, 0), (3, 0), # Row 0
                (-1, 1), (0, 1), (1, 1), (2, 1), # Row 1
                (-2, 2), (-1, 2), (0, 2), (1, 2)  # Row 2
            ]
            # This is a parallelogram 4x3. Not optimal for hexagonal symmetry.
            
            # Let's stick to a symmetric "snowflake" minus some.
            # 1 center.
            # 6 neighbors.
            # 5 of the 12 next-neighbors.
            # Next neighbors are at distance 2*sqrt(3) from center? No.
            # Distance from center to neighbor is sqrt(3).
            # Distance from center to next neighbor (sharing a face with a neighbor) is sqrt(3) * 2 * cos(30)?
            # Let's calculate positions explicitly.
            
            pos = []
            pos.append((0, 0))
            
            # Ring 1: 6 hexes
            for i in range(6):
                angle = math.radians(60 * i)
                pos.append((math.sqrt(3) * math.cos(angle), 1.5 * math.sin(angle) * 2 / math.sqrt(3) * math.sqrt(3)/1.5)) 
                # Simpler:
                # x = sqrt(3) * (q + r/2)
                # y = 1.5 * r
                # Ring 1 axial: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
            
            # Let's use axial to cartesian conversion for precision
            def axial_to_cartesian(q, r, size=1.0):
                x = size * math.sqrt(3) * (q + r / 2.0)
                y = size * 1.5 * r
                return (x, y)
            
            # 12 hexes in a compact hexagonal shape
            # A hexagon of side 2 has 19 hexes.
            # Remove the 7 outermost corners?
            # Side 2 hexagon coords:
            # r from -1 to 1
            # r=-1: q from 0 to 1 (2 hexes)
            # r=0: q from -1 to 1 (3 hexes)
            # r=1: q from -2 to 0 (3 hexes)
            # Total 8? No.
            # Hexagon of side L has 3L(L-1)+1 hexes.
            # L=1: 1
            # L=2: 7
            # L=3: 19
            # We want 12.
            # Take the L=2 hexagon (7 hexes) and add 5 more.
            # The 5 most "central" of the next ring.
            
            base_coords = [
                (0,0),
                (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
            ]
            
            # Next ring candidates (distance 2 from center in hex grid steps)
            # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,0)?? No.
            # Let's just pick 5 that maintain symmetry or fill gaps.
            # To minimize bounding box, we should fill the "corners" of the L=2 hexagon?
            # The L=2 hexagon has corners at (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
            # Adding hexes at (2,0), (0,2), (-2,1), (-2,0), (0,-2) extends the shape.
            
            add_coords = [
                (2, 0),
                (0, 2),
                (-2, 1),
                (-2, 0),
                (0, -2)
            ]
            
            all_coords = base_coords + add_coords
            
            for q, r in all_coords:
                x, y = axial_to_cartesian(q, r)
                pos.append((x, y))
                
            # Remove duplicates if any (should be none)
            unique_pos = list(set(pos))
            # Ensure we have exactly 12
            if len(unique_pos) != 12:
                 # Fallback to a simple grid
                 unique_pos = [axial_to_cartesian(q, r) for q, r in [(0,0), (1,0), (2,0), (-1,1), (0,1), (1,1), (-2,2), (-1,2), (0,2), (1,2), (2,2), (-1,0)]]
            
            return np.array(unique_pos[:12])

    def required_outer_side_for_cluster(positions, outer_angle_deg=0.0):
        """
        Given a set of unit hexagons (at angle 0) at positions 'positions',
        find the minimum side length of a regular hexagon (at outer_angle_deg)
        that contains them all.
        
        Since the inner hexagons are convex and the outer is convex,
        the minimum side length is determined by the maximum support function
        of the union of inner hexagons in the 6 directions normal to the outer hexagon sides.
        """
        outer_angle_rad = np.radians(outer_angle_deg)
        
        # The 6 normals for a regular hexagon of angle 0 are at 0, 60, 120, 180, 240, 300 degrees.
        # If the outer hexagon is rotated by outer_angle_rad, the normals are rotated too.
        normals = []
        for i in range(6):
            angle = outer_angle_rad + i * np.pi / 3.0
            normals.append(np.array([math.cos(angle), math.sin(angle)]))
        
        # Calculate support function for each normal
        # Support h(n) = max_i (n . p_i + h_inner(n))
        # h_inner(n) for a unit hexagon at angle 0:
        # The support function of a regular hexagon of side 1 is:
        # h(theta) = (1/sqrt(3)) * (1 + 2 * |sin(theta)|) ? No.
        # Apothem is sqrt(3)/2.
        # The support function of a hexagon with apothem a is a * (1 + 2 * |sin(3*theta)|) / (1 + 2*|sin(3*theta)|)??
        # Actually, for a regular hexagon, the support function h(theta) is periodic with period 60 deg.
        # In the sector [0, 30], h(theta) = a / cos(theta - 30)? No.
        # Let's just compute it numerically for the 6 vertices of a unit hexagon.
        # Vertices of unit hexagon at (0,0), angle 0:
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        
        inner_verts = hexagon_vertices(0, 0, 1.0, 0.0)
        
        max_support = 0.0
        for n in normals:
            # Max projection of any inner hexagon in direction n
            # = max over all inner hexes i of (n . pos_i + max over vertices v of (n . v))
            
            # Pre-compute max projection of a single unit hexagon in direction n
            # proj = inner_verts.dot(n)
            # max_proj_inner = np.max(proj)
            
            # To be efficient, we can compute this for each normal once.
            # But N is small (12), so just loop.
            
            min_required_offset = -np.inf
            for i in range(len(positions)):
                p = positions[i]
                # Projection of center
                center_proj = p.dot(n)
                # Max projection of shape relative to center
                shape_max = np.max(inner_verts.dot(n))
                
                total_proj = center_proj + shape_max
                if total_proj > min_required_offset:
                    min_required_offset = total_proj
            
            # The side length S of the outer hexagon must satisfy:
            # The distance from center to side is S * sqrt(3) / 2.
            # So S * sqrt(3) / 2 >= min_required_offset
            # S >= min_required_offset * 2 / sqrt(3)
            
            required_side = min_required_offset * 2.0 / math.sqrt(3.0)
            if required_side > max_support:
                max_support = required_side
        
        return max_support

    def check_validity(positions, tol=1e-6):
        """
        Check if all hexagons are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                disjoint = False
                # Check separating axes from hexagon 1
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        disjoint = True
                        break
                
                if not disjoint:
                    # Check separating axes from hexagon 2
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    # 1. Generate a static, dense cluster
    # We will try a few static clusters and pick the best one.
    
    clusters = []
    
    # Cluster 1: The "12-hexagon" shape derived above
    pos1 = get_inner_hexagons_config("flower")
    if check_validity(pos1):
        clusters.append(pos1)
    
    # Cluster 2: A 2x2x2 block (if it fits)
    # Axial: 
    # (0,0), (1,0), (0,1), (1,1) -> 4
    # (2,0), (3,0), (2,1), (3,1) -> 4
    # (4,0), (5,0), (4,1), (5,1) -> 4
    # This is a 6x2 block.
    coords2 = []
    for r in [0, 1]:
        for q in range(6):
            coords2.append((q, r))
    pos2 = np.array([axial_to_cartesian(q, r) for q, r in coords2])
    if check_validity(pos2):
        clusters.append(pos2)
        
    # Cluster 3: A hexagon of side 2 (7 hexes) + 5 scattered to minimize bounding box?
    # Let's stick to the first one which is more compact.
    
    if not clusters:
        # Fallback to a simple valid grid
        coords = [
            (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        pos_fallback = np.array([axial_to_cartesian(q, r) for q, r in coords])
        if check_validity(pos_fallback):
            clusters.append(pos_fallback)
        else:
            # Should not happen
            return [[0,0,0]]*12, (0,0), 10.0, 0.0

    # 2. For each cluster, find the optimal orientation and translation
    best_score = float('inf')
    best_inner = None
    best_center = (0,0)
    best_side = float('inf')
    best_angle = 0.0
    
    for cluster in clusters:
        # The cluster is centered at some point. We can translate it to (0,0) to simplify.
        # But the "required_outer_side" function assumes the outer hexagon is centered at (0,0).
        # So we should center the cluster at (0,0).
        centroid = np.mean(cluster, axis=0)
        centered_cluster = cluster - centroid
        
        # Now, we need to find the optimal rotation of the outer hexagon.
        # Since the inner hexagons are fixed (angle 0), and we can rotate the outer hexagon,
        # we can try all 60-degree increments? No, the optimal angle might be anywhere.
        # However, due to symmetry of the hexagon, the support function is periodic.
        # The objective function (min outer side) is a function of the outer hexagon's angle.
        # It is piecewise linear/convex?
        # Actually, for a fixed set of convex bodies, the minimum enclosing regular hexagon
        # can be found by checking the angles where a vertex of an inner hexagon aligns
        # with a side normal of the outer hexagon.
        
        # Let's sample angles. The function is periodic with 60 deg.
        # We can use a golden-section search or just sample finely.
        
        def optimize_angle(cluster_pos):
            # Sample angles from 0 to 60 degrees
            angles = np.linspace(0, 60, 361)
            best_a = 0
            best_s = float('inf')
            for a in angles:
                s = required_outer_side_for_cluster(cluster_pos, a)
                if s < best_s:
                    best_s = s
                    best_a = a
            return best_a, best_s
            
        # Refine the best angle found by sampling
        a0, s0 = optimize_angle(centered_cluster)
        
        # Fine search around a0
        angles_fine = np.linspace(a0 - 1, a0 + 1, 101)
        best_a = a0
        best_s = s0
        for a in angles_fine:
            s = required_outer_side_for_cluster(centered_cluster, a)
            if s < best_s:
                best_s = s
                best_a = a
                
        if best_s < best_score:
            best_score = best_s
            best_inner = centered_cluster
            best_center = (0.0, 0.0) # Since we centered it
            best_side = best_s
            best_angle = best_a

    # 3. Construct the return value
    # The inner hexagons are at best_inner, with angle 0.
    # The outer hexagon is at best_center, with side best_side and angle best_angle.
    
    inner = []
    for i in range(N):
        inner.append([float(best_inner[i, 0]), float(best_inner[i, 1]), 0.0])
        
    return inner, best_center, float(best_side), float(best_angle)

# EVOLVE-BLOCK-END
