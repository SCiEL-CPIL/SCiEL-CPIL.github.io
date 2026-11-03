"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Helper: get vertices of a regular hexagon with side length 1
    # We assume the hexagon is defined by vertices at angles 0, 60, 120, ... + rotation
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    # Optimized for speed: precomputed normals if possible, but here we compute on the fly
    def hexagons_overlap(v1, v2):
        """Check if two convex hexagons overlap using SAT."""
        # 12 axes: 6 from v1, 6 from v2
        # We can unroll this for speed if needed, but N=12 is small
        axes = []
        for i in range(6):
            edge1 = v1[(i+1) % 6] - v1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            
            edge2 = v2[(i+1) % 6] - v2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False
        return True

    def check_all_overlaps(positions, angles):
        """Check if any pair of hexagons overlaps."""
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    return True
        return False

    # Helper: compute minimum outer hexagon side length needed to contain all inner hexagons
    def compute_min_side(positions, angles, outer_angle_deg):
        """
        Returns the minimum side length of outer hexagon to contain all inner hexagons.
        """
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, a)
            all_verts.append(verts)
            
        if not all_verts:
            return 0.0
            
        all_verts = np.vstack(all_verts)
        
        theta = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            a = theta + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        max_proj = 0
        for normal in normals:
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        
        # For a regular hexagon, if the max projection on the normal axis is R,
        # the side length s is related to R by R = s * sqrt(3) / 2 * 2? 
        # Let's derive:
        # A regular hexagon of side s has "radius" (center to vertex) R_v = s.
        # The distance from center to side (apothem) is R_a = s * sqrt(3) / 2.
        # The normals in compute_min_side point to the vertices of the dual hexagon?
        # No, the normals are at angles theta + k*pi/3.
        # If the outer hexagon has vertices at angles theta + k*pi/3, then the normals to its sides are at theta + k*pi/3 + pi/6.
        # Wait, the standard definition:
        # If outer hexagon has side s, its vertices are at distance s from center.
        # The projection of a point (x,y) onto the normal of a side.
        # The sides are perpendicular to the directions theta + pi/6 + k*pi/3.
        # Let's check the geometry.
        # A hexagon with vertices at angles alpha + k*pi/3.
        # The side between vertex k and k+1 is perpendicular to the bisector angle alpha + (k+0.5)*pi/3.
        # So the normals to the sides are at angles alpha + pi/6 + k*pi/3.
        # The code uses normals at theta + k*pi/3.
        # This implies the "normals" in the code correspond to the vertex directions of the dual hexagon?
        # Actually, if we project onto 6 axes separated by 60 degrees, we are effectively checking the support function.
        # For a regular hexagon containing the origin, centered at origin, side s:
        # The maximum projection onto a normal at angle phi is s * cos(phi - vertex_angle)?
        # Let's stick to the formula derived in previous successful runs or verify.
        # If max_proj is the maximum distance from center to any vertex of the inner set along the normal direction.
        # If the outer hexagon has side s, its "radius" (center to vertex) is s.
        # The distance from center to side (apothem) is s * sqrt(3)/2.
        # The normals in the loop are at angles theta + k*pi/3.
        # If these are the normals to the sides, then the max allowed projection is the apothem.
        # So apothem >= max_proj.
        # s * sqrt(3)/2 >= max_proj => s >= 2 * max_proj / sqrt(3).
        # This matches the formula in the seed.
        
        side = 2 * max_proj / math.sqrt(3)
        return side

    def get_config_score(positions, angles, outer_angle):
        """Return side length if valid, else infinity."""
        if check_all_overlaps(positions, angles):
            return float('inf')
        return compute_min_side(positions, angles, outer_angle)

    # === NEW: Structural Honeycomb Generation ===
    
    def axial_to_cartesian(q, r):
        """Convert axial coordinates to Cartesian for a hex grid with side length 1."""
        # Standard pointy-top or flat-top?
        # Seed uses: x = c * sqrt(3) + (r % 2) * sqrt(3) / 2, y = r * 1.5
        # This is a "pointy-top" hexagon arrangement where rows are horizontal?
        # Let's check vertices of a single hexagon with angle 0.
        # Vertices: (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5,-0.866), (0.5,-0.866).
        # This is a "pointy-top" hexagon (vertices at 0, 60, 120...).
        # Distance between centers in a honeycomb of pointy-top hexagons:
        # Horizontal distance: sqrt(3) * side.
        # Vertical distance: 1.5 * side.
        # Offset for alternating rows: sqrt(3)/2 * side.
        # This matches the seed's make_honeycomb logic.
        
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def generate_honeycomb_positions():
        """
        Generate the 12 positions for the optimal honeycomb packing.
        Topology: 1 center, 6 neighbors, 5 from the next ring.
        Axial coords (q, r):
        Center: (0,0)
        Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        Ring 2: We need 5 more.
        The full ring 2 has 12 hexagons.
        To maintain symmetry and compactness, we pick 5 that form a contiguous block or symmetric set.
        A common compact shape for 12 is a "hexagon of radius 1" (7 hexes) plus 5 more.
        Actually, a regular hexagon of side 2 has 19 hexagons.
        A regular hexagon of side 1 has 7 hexagons.
        We have 12.
        
        Let's try a specific layout that is known to be tight.
        Layout:
        Row 0: 2 hexes
        Row 1: 3 hexes
        Row 2: 3 hexes
        Row 3: 3 hexes
        Row 4: 1 hex? No, that's 12.
        
        Let's use the axial coordinate generation from the inspiration which produced a good score,
        but verify it's the "flower" shape.
        
        Ring 0: (0,0)
        Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        Total 7.
        
        We need 5 more.
        Candidates in Ring 2:
        (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
        
        To make it symmetric, we can pick 5 that are roughly evenly spaced, or just pick a contiguous block to minimize the bounding box.
        However, the "best" packing is often the one that forms a larger regular hexagon shape as much as possible.
        
        Let's try a "staggered rectangle" approach which is often optimal for N not being a hex number.
        12 hexagons can be arranged in a 3x4 grid (staggered).
        
        Let's generate a few candidate topologies and pick the one with the smallest initial side length.
        """
        
        topologies = []
        
        # Topology 1: 4 rows of 3 (Staggered)
        # Row 0: 3 hexes
        # Row 1: 3 hexes (shifted)
        # Row 2: 3 hexes
        # Row 3: 3 hexes
        # Axial coords for row r, col c:
        # x = sqrt(3) * (c + (r%2)*0.5)
        # y = 1.5 * r
        # This is a rectangular block.
        
        def make_grid(rows, cols):
            coords = []
            for r in range(rows):
                for c in range(cols):
                    # Axial to cartesian mapping used in axial_to_cartesian:
                    # x = sqrt(3) * (q + r/2)
                    # y = 1.5 * r
                    # Here r is the row index.
                    # If we use axial coords directly:
                    # Let's just use the cartesian formula directly to avoid confusion.
                    x = math.sqrt(3) * (c + (r % 2) * 0.5)
                    y = 1.5 * r
                    coords.append((x, y))
            return coords

        # Topology 2: 3 rows of 4
        topologies.append(make_grid(3, 4))
        
        # Topology 3: 4 rows of 3
        topologies.append(make_grid(4, 3))
        
        # Topology 4: 2 rows of 6
        topologies.append(make_grid(2, 6))
        
        # Topology 5: 6 rows of 2
        topologies.append(make_grid(6, 2))
        
        # Topology 6: "Flower" (1+6+5)
        # Center (0,0)
        # Neighbors: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # Next 5: Let's pick (2,0), (2,-1), (1,-2), (0,-2), (-1,-1) ?? 
        # It's hard to guess the best 5. 
        # However, the grid topologies (3x4 and 4x3) are usually very competitive and easy to optimize.
        # The "Flower" might have a smaller bounding box if the 5 outer ones are placed well.
        
        # Let's also try a "Hexagonal" crop of a larger hexagon.
        # A hexagon of side 2 has 19 hexes.
        # Remove 7 from the corners/edges to get 12?
        
        # For now, the grid topologies are robust.
        # We will center them.
        
        positions_list = []
        for coords in topologies:
            # Center the layout
            cx = sum(p[0] for p in coords) / len(coords)
            cy = sum(p[1] for p in coords) / len(coords)
            centered = [(x - cx, y - cy) for x, y in coords]
            positions_list.append(centered)
            
        return positions_list

    # Main Logic
    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    # Generate candidate topologies
    candidate_positions = generate_honeycomb_positions()
    
    # For each topology, we assume the inner hexagons are aligned (angle 0).
    # We sweep the outer hexagon's rotation angle to find the minimal containing side.
    # The problem is symmetric, so we only need to sweep 0 to 30 degrees (or 60, depending on symmetry).
    # Since the inner hexagons are fixed at angle 0, and the outer one rotates,
    # the function f(outer_angle) is periodic with period 60 degrees.
    # Due to symmetry of the hexagon, we can sweep 0 to 30 degrees.
    
    # We will evaluate a fine grid of outer angles.
    angle_steps = np.linspace(0, 30, 100)
    
    for pos in candidate_positions:
        angles = [0.0] * N
        
        # Quick check: is this topology valid (no overlaps)?
        # For a honeycomb grid, they are exactly touching, so they should be valid.
        # SAT with floating point might detect tiny overlaps if not careful.
        # We use a strict check. If invalid, we might need to nudge, but honeycombs are exact.
        
        if check_all_overlaps(pos, angles):
            continue
            
        for oa in angle_steps:
            if time.time() > deadline - 5:
                break
                
            score = compute_min_side(pos, angles, oa)
            
            if score < best_score:
                best_score = score
                best_positions = pos
                best_angles = angles
                best_outer_angle = oa
        
        # If we found a good score, we can try to refine the outer angle locally
        # But the grid is fine enough (100 steps over 30 deg is 0.3 deg resolution).
        # The error in side length is proportional to angle error * derivative.
        # Derivative is small near minimum.
        
    # If no valid configuration was found (should not happen for honeycombs), fallback
    if best_positions is None or best_score == float('inf'):
        # Fallback to a simple grid, even if it has overlaps (should score inf, but better than crash)
        # Or just a large square grid
        positions = []
        for i in range(N):
            r = i // 4
            c = i % 4
            positions.append((c * 4.0, r * 4.0))
        best_positions = positions
        best_angles = [0.0] * N
        best_outer_angle = 0.0
        best_score = compute_min_side(positions, best_angles, 0.0)

    # Final cleanup
    # Ensure positions are centered at (0,0) for the return value
    if best_positions:
        cx = sum(p[0] for p in best_positions) / N
        cy = sum(p[1] for p in best_positions) / N
        final_positions = [(x - cx, y - cy) for x, y in best_positions]
    else:
        final_positions = [(0,0)] * N
        best_angles = [0] * N
        best_score = 10.0

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]
    
    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END
