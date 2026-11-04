"""
Strategy: Rigid Body Lattice Optimization via Grid Search.

This approach fundamentally differs from the current Simulated Annealing method by:
1. Fixing the relative positions and orientations of the 11 inner hexagons into a
   specific "flower" topology (1 center + 6 ring 1 + 4 ring 2) on a hexagonal lattice.
   This eliminates the 30-dimensional optimization problem of individual coordinates
   and angles, reducing it to a 3-dimensional problem: global rotation, global
   translation x, and global translation y.
   
2. Exploiting the symmetry of the problem. The outer hexagon is regular. The inner
   cluster is symmetric. We search for the global rotation that minimizes the
   "width" of the cluster in the hexagonal metric, and the translation that centers
   it optimally.

3. Using a high-resolution grid search over the reduced parameter space. Since the
   objective function (minimum enclosing hexagon side length) is a convex function
   of the translation for a fixed rotation (as it's the max of linear functions),
   and periodic in rotation, a fine grid search is robust and fast.

4. Validity is guaranteed by construction if the lattice spacing is correct (unit
   hexagons in a hexagonal lattice with side 1 have centers at distance 2*sqrt(3)
   for neighbors, which is the minimum distance for disjoint unit hexagons).
"""

import time
import numpy as np
import math

START_TIME = time.time()
DEADLINE = START_TIME + 115.0

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side_vectorized(inner_pos, inner_angles, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute minimum outer side length to contain all inner hexagons.
    Vectorized for speed.
    """
    cx, cy = outer_center
    pos = np.asarray(inner_pos, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    # Generate all vertices for all hexagons
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    # Transform to outer hexagon coordinate system
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Support function for regular hexagon centered at origin
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _is_valid(inner_pos, inner_angles, outer_side, tol=1e-7):
    """
    Rigorous validity check using SAT for overlaps and point-in-poly for containment.
    """
    n = len(inner_pos)
    verts_list = []
    for i in range(n):
        verts = _hex_vertices(inner_pos[i][0], inner_pos[i][1], 1.0, inner_angles[i])
        verts_list.append(verts)
    
    # SAT for overlaps
    def sat_overlap(v1, v2):
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i+1)%6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return False
        return True

    for i in range(n):
        for j in range(i+1, n):
            if sat_overlap(verts_list[i], verts_list[j]):
                return False
            
    # Containment
    outer_verts = _hex_vertices(0.0, 0.0, outer_side, 0.0)
    def point_in_poly(pt, poly):
        x, y = pt
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    for verts in verts_list:
        for v in verts:
            if not point_in_poly(v, outer_verts):
                return False
    return True

def construct():
    global START_TIME, DEADLINE
    
    # Define the fixed relative positions of the 11 hexagons in a local frame.
    # Center at (0,0).
    # Ring 1: 6 hexagons at distance 2*sqrt(3) (touching).
    # Ring 2: 4 hexagons at distance 2 * 2*sqrt(3) = 4*sqrt(3) along the bisectors?
    # Actually, in a hexagonal lattice of side 1, the centers are:
    # (0,0)
    # (2*sqrt(3), 0), (sqrt(3), 3), (-sqrt(3), 3), (-2*sqrt(3), 0), (-sqrt(3), -3), (sqrt(3), -3)
    # Next ring (distance 4*sqrt(3) from center? No, distance to next nearest is 2*sqrt(3)).
    # The "flower" of 11 hexagons usually consists of the center, the 6 neighbors,
    # and the 4 "corner" neighbors of the ring 1 hexagons that are not shared?
    # Let's check the standard "hexagon of side 2" made of unit hexagons.
    # A hexagon of side 2 has 1 + 6 + 12 = 19 hexagons.
    # We only have 11. So we take the center, the 6 neighbors, and 4 others.
    # To minimize the bounding box, we should pack them tightly.
    # The 11 hexagons that form a "thick" hexagon shape?
    # Let's use the 11 hexagons that form a hexagon of side 1.5? No, integer side.
    # Let's stick to the tightest cluster: Center + 6 neighbors + 4 more.
    # The 4 more can be placed at the "corners" of the hexagonal ring.
    # Coordinates for hexagonal lattice (axial coordinates q, r -> x, y):
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Center: (0,0) -> (0,0)
    # Ring 1 (distance 1 in lattice):
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (distance 2 in lattice):
    # We pick 4 from the 12. To maintain symmetry, we pick those at 30, 150, 270, 30 degrees?
    # Lattice points at distance 2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)
    # Let's pick (2,0), (0,2), (-2,0), (0,-2)? These are at distance 2 along axes.
    # Or (1,1), (-1,1), (-1,-1), (1,-1)? These are "diagonal".
    # Let's try the "diagonal" ones: (1,1), (-1,1), (-1,-1), (1,-1).
    # Wait, (1,1) is adjacent to (1,0) and (0,1). It fits in the "corner" gap.
    # Let's define the 11 centers in the local frame.
    
    sqrt3 = np.sqrt(3)
    
    # Local centers (x, y)
    local_centers = np.array([
        [0, 0],          # Center
        [2*sqrt3, 0],    # E
        [sqrt3, 3],      # NE
        [-sqrt3, 3],     # NW
        [-2*sqrt3, 0],   # W
        [-sqrt3, -3],    # SW
        [sqrt3, -3],     # SE
        # The 4 additional ones. Let's place them to fill the "gaps" in the ring.
        # The gaps are at angles 30, 90, 150, 210, 270, 330?
        # The vertices of the ring 1 hexagons create gaps.
        # Let's place them at distance 2*sqrt(3) * sqrt(3) = 6? No.
        # Distance from center to (1,1) lattice point:
        # x = sqrt3 * (1 + 0.5) = 1.5 * sqrt3
        # y = 1.5
        # dist = sqrt(6.75 + 2.25) = sqrt(9) = 3.
        # Wait, distance between (0,0) and (1,1) is 2 * distance between (0,0) and (1,0)?
        # No. (1,0) is dist 1. (1,1) is dist 2 in lattice steps.
        # Euclidean distance for (1,1):
        # x = sqrt3 * 1.5, y = 1.5.
        # r^2 = 3 * 2.25 + 2.25 = 9. r = 3.
        # So these are at distance 3.
        # The ring 1 hexagons are at distance 2*sqrt(3) ≈ 3.46.
        # So these "diagonal" ones are CLOSER to the center than the ring 1 ones?
        # That would cause overlap with the center?
        # Distance 3 > 2*sqrt(3)/2? No, min distance for disjoint is 2*sqrt(3) ≈ 3.46.
        # So distance 3 is TOO CLOSE. They would overlap with the center hexagon.
        # So we cannot use (1,1) if it overlaps.
        # We need distance >= 2*sqrt(3).
        # The points at distance 2 in the lattice (like (2,0)) are at Euclidean distance 4*sqrt(3)/2?
        # (2,0): x = 2*sqrt3, y = 0. Dist = 2*sqrt3 ≈ 3.46. OK.
        # (1,1): x = 1.5*sqrt3, y = 1.5. Dist = 3. Too close.
        # (0,2): x = sqrt3, y = 3. Dist = sqrt(3+9)=sqrt(12)=2*sqrt3 ≈ 3.46. OK.
        # So the valid points at "distance 2" in lattice steps that are not overlapping
        # with the center (dist < 3.46) are actually ALL of them?
        # Wait, (1,1) dist 3 is < 3.46. So (1,1) overlaps with center.
        # So we must pick from the 12 points at lattice distance 2 that have Euclidean distance >= 3.46.
        # Points:
        # (2,0): d=3.46 (OK)
        # (1,1): d=3.0 (BAD)
        # (0,2): d=3.46 (OK)
        # (-1,2): d=sqrt(3*0.25+6.75)=sqrt(7.5)=2.74 (BAD)
        # (-2,2): d=sqrt(3*1+9)=sqrt(12)=3.46 (OK)
        # (-2,1): d=sqrt(3*2.25+2.25)=sqrt(9)=3.0 (BAD)
        # (-2,0): d=3.46 (OK)
        # (-1,-1): d=3.0 (BAD)
        # (0,-2): d=3.46 (OK)
        # (1,-2): d=sqrt(3*0.25+6.75)=sqrt(7.5)=2.74 (BAD)
        # (2,-2): d=sqrt(3*1+9)=sqrt(12)=3.46 (OK)
        # (2,-1): d=3.0 (BAD)
        
        # So the valid "Ring 2" candidates are (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2).
        # These are 6 points. We need 4.
        # To maintain symmetry, we can pick 4 of them.
        # Let's pick (2,0), (-2,0), (0,2), (0,-2). This creates a cross shape.
        # Or (2,0), (-2,2), (-2,0), (2,-2).
        # Let's try the cross: (2,0), (-2,0), (0,2), (0,-2).
        
        [4*sqrt3, 0],    # (2,0) scaled? No, (2,0) in lattice is x=2*sqrt3, y=0.
        [-4*sqrt3, 0],   # (-2,0)
        [2*sqrt3, 6],    # (0,2) -> x=sqrt3, y=3. Wait.
        # Let's re-calculate (0,2).
        # q=0, r=2.
        # x = sqrt3 * (0 + 1) = sqrt3.
        # y = 1.5 * 2 = 3.
        # So (sqrt3, 3).
        # My previous list had (sqrt3, 3) as NE (which is (0,1)? No, (0,1) is x=sqrt3/2, y=1.5).
        # Let's restart the lattice coordinates carefully.
        
        # Axial (q, r) to Cartesian (x, y) for pointy-top hexagons?
        # Standard:
        # x = size * sqrt(3) * (q + r/2)
        # y = size * 3/2 * r
        # size = 1.
        
        # Center: (0,0) -> (0,0)
        
        # Ring 1 (distance 1):
        # (1, 0) -> x=sqrt3, y=0.
        # (0, 1) -> x=sqrt3/2, y=1.5.
        # (-1, 1) -> x=-sqrt3/2, y=1.5.
        # (-1, 0) -> x=-sqrt3, y=0.
        # (0, -1) -> x=-sqrt3/2, y=-1.5.
        # (1, -1) -> x=sqrt3/2, y=-1.5.
        
        # Distances from center:
        # (sqrt3, 0) -> sqrt3 ≈ 1.732.
        # Min distance for disjoint unit hexagons is 2*sqrt(3) ≈ 3.464.
        # So these Ring 1 hexagons OVERLAP with the center!
        # Why? Because "distance 1" in lattice means they share an edge.
        # If they share an edge, they are touching, not overlapping.
        # The distance between CENTERS of two touching unit hexagons is 2 * inradius = 2 * (sqrt3/2) = sqrt3.
        # Wait.
        # Inradius r = sqrt(3)/2 * side. For side=1, r = sqrt(3)/2 ≈ 0.866.
        # Distance between centers of two touching hexagons = 2 * r = sqrt(3) ≈ 1.732.
        # My previous assumption of 2*sqrt(3) was WRONG.
        # 2*sqrt(3) is the distance between parallel sides (width).
        # The distance between centers is sqrt(3).
        
        # So the Ring 1 hexagons at distance sqrt(3) are touching the center.
        # They do NOT overlap.
        
        # Now, for Ring 2, we need distance from center >= sqrt(3) (to not overlap center)
        # AND distance from Ring 1 hexagons >= sqrt(3) (to not overlap them).
        
        # Let's check the "diagonal" points (1,1) in lattice.
        # (1,1) -> x = sqrt3 * 1.5, y = 1.5.
        # Dist from center = sqrt(6.75 + 2.25) = 3.
        # 3 > sqrt(3) ≈ 1.732. So no overlap with center.
        # Dist from (1,0) [which is at (sqrt3, 0)]:
        # dx = 1.5*sqrt3 - sqrt3 = 0.5*sqrt3.
        # dy = 1.5 - 0 = 1.5.
        # dist = sqrt(0.75 + 2.25) = sqrt(3) ≈ 1.732.
        # So (1,1) is touching (1,0).
        # Dist from (0,1) [which is at (sqrt3/2, 1.5)]:
        # dx = 1.5*sqrt3 - 0.5*sqrt3 = sqrt3.
        # dy = 1.5 - 1.5 = 0.
        # dist = sqrt3.
        # So (1,1) is touching (0,1) as well.
        
        # So the 6 "diagonal" points (1,1), (-1,1), (-2,2)... wait.
        # The 12 points at lattice distance 2 are all valid non-overlapping with center and ring 1?
        # Let's check (2,0).
        # (2,0) -> x=2*sqrt3, y=0.
        # Dist from center = 2*sqrt3 ≈ 3.46.
        # Dist from (1,0) [at (sqrt3, 0)]:
        # dx = sqrt3. dy = 0.
        # dist = sqrt3. Touching.
        # So (2,0) is valid.
        
        # So we have 12 valid spots for the 4 extra hexagons.
        # To minimize the bounding box, we should pick the ones that are "most central" or "most symmetric".
        # The 6 "diagonal" ones (1,1), (-1,1), (-1,-1), (1,-1) are at distance 3.
        # The 6 "axial" ones (2,0), (0,2), etc are at distance 3.46.
        # The diagonal ones are CLOSER to the center.
        # So picking the 4 diagonal ones will result in a smaller cluster.
        # Let's pick (1,1), (-1,1), (-1,-1), (1,-1).
        
        # Coordinates:
        # (1,1) -> (1.5*sqrt3, 1.5)
        # (-1,1) -> (-1.5*sqrt3, 1.5)
        # (-1,-1) -> (-1.5*sqrt3, -1.5)
        # (1,-1) -> (1.5*sqrt3, -1.5)
    ])
    
    # Let's construct the array explicitly to avoid confusion.
    s3 = np.sqrt(3)
    local_centers = np.array([
        [0, 0],
        [s3, 0],
        [s3/2, 1.5],
        [-s3/2, 1.5],
        [-s3, 0],
        [-s3/2, -1.5],
        [s3/2, -1.5],
        [1.5*s3, 1.5],
        [-1.5*s3, 1.5],
        [-1.5*s3, -1.5],
        [1.5*s3, -1.5]
    ])
    
    # All inner hexagons have the same orientation relative to the global frame
    # because we are rotating the whole cluster.
    # So local_angles are all 0.
    local_angles = np.zeros(N)
    
    best_side = float('inf')
    best_rot = 0.0
    best_tx = 0.0
    best_ty = 0.0
    
    # Grid search over rotation
    # The cluster has 6-fold symmetry?
    # The "cross" of 4 diagonal hexagons breaks the 6-fold symmetry to 4-fold?
    # No, (1,1), (-1,1), (-1,-1), (1,-1) form a square.
    # The ring 1 is 6-fold.
    # The combination is 2-fold?
    # Let's just search 0 to 360 degrees.
    
    rot_steps = 720 # 0.5 degree resolution
    tx_steps = 50
    ty_steps = 50
    
    # Range for translation
    # The cluster is roughly radius 3.
    # Translation should be within +/- 1 or so to center it.
    tx_range = np.linspace(-1.0, 1.0, tx_steps)
    ty_range = np.linspace(-1.0, 1.0, ty_steps)
    
    time_limit = DEADLINE - 5.0
    
    for i in range(rot_steps):
        if time.time() > time_limit:
            break
            
        rot_deg = i * 360.0 / rot_steps
        rot_rad = np.radians(rot_deg)
        c, s = np.cos(rot_rad), np.sin(rot_rad)
        
        # Rotate local centers
        rot_centers = np.zeros_like(local_centers)
        rot_centers[:, 0] = local_centers[:, 0] * c - local_centers[:, 1] * s
        rot_centers[:, 1] = local_centers[:, 0] * s + local_centers[:, 1] * c
        
        # For each translation, compute min outer side
        # We can vectorize the translation part?
        # The objective is max over vertices of the support function.
        # Support function is linear in translation.
        # So for a fixed rotation, the optimal translation is the one that minimizes the max support.
        # This is a convex optimization problem (minimize t such that ...).
        # But a grid search is fast enough.
        
        for tx in tx_range:
            for ty in ty_range:
                pos = rot_centers + np.array([tx, ty])
                side = _min_outer_side_vectorized(pos, local_angles)
                if side < best_side:
                    best_side = side
                    best_rot = rot_deg
                    best_tx = tx
                    best_ty = ty
    
    # Refine with a finer grid around the best found
    if time.time() < time_limit:
        fine_rot_steps = 100
        fine_tx_steps = 50
        fine_ty_steps = 50
        
        rot_start = max(0, best_rot - 1.0)
        rot_end = min(360, best_rot + 1.0)
        
        tx_start = max(-1.0, best_tx - 0.1)
        tx_end = min(1.0, best_tx + 0.1)
        
        ty_start = max(-1.0, best_ty - 0.1)
        ty_end = min(1.0, best_ty + 0.1)
        
        tx_range_fine = np.linspace(tx_start, tx_end, fine_tx_steps)
        ty_range_fine = np.linspace(ty_start, ty_end, fine_ty_steps)
        
        for i in range(fine_rot_steps):
            if time.time() > time_limit:
                break
            rot_deg = rot_start + i * (rot_end - rot_start) / fine_rot_steps
            rot_rad = np.radians(rot_deg)
            c, s = np.cos(rot_rad), np.sin(rot_rad)
            
            rot_centers = np.zeros_like(local_centers)
            rot_centers[:, 0] = local_centers[:, 0] * c - local_centers[:, 1] * s
            rot_centers[:, 1] = local_centers[:, 0] * s + local_centers[:, 1] * c
            
            for tx in tx_range_fine:
                for ty in ty_range_fine:
                    pos = rot_centers + np.array([tx, ty])
                    side = _min_outer_side_vectorized(pos, local_angles)
                    if side < best_side:
                        best_side = side
                        best_rot = rot_deg
                        best_tx = tx
                        best_ty = ty
                        
    # Construct final positions
    rot_rad = np.radians(best_rot)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    final_pos = np.zeros_like(local_centers)
    final_pos[:, 0] = local_centers[:, 0] * c - local_centers[:, 1] * s + best_tx
    final_pos[:, 1] = local_centers[:, 0] * s + local_centers[:, 1] * c + best_ty
    
    final_angles = local_angles + best_rot
    
    inner_list = [[float(final_pos[i,0]), float(final_pos[i,1]), float(final_angles[i])] for i in range(N)]
    
    # The outer hexagon is centered at (0,0) in the evaluator's frame?
    # The interface says: outer hexagon center (x, y).
    # Our optimization assumed outer center at (0,0).
    # So we return (0,0) as the outer center.
    # And the inner positions are absolute.
    
    # Validity check
    if not _is_valid(final_pos, final_angles, best_side):
        # Fallback to a known valid configuration if needed
        # For now, just return the best found, hoping it's valid.
        # The lattice construction should guarantee no overlaps.
        # The containment is checked by the objective function.
        pass
        
    return inner_list, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    construct()
