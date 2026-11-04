"""
Strategy: Combinatorial Topology Search with Rigid Body Optimization.

Key Differences from Current Program:
1. Representation: Instead of optimizing 22 independent coordinates (11 x, 11 y, 11 angles),
   we define the packing by its topology (adjacency graph). The positions are determined
   exactly by the geometry of touching regular hexagons.
2. Search Space: We search over a set of plausible topologies (graphs) for packing 11 hexagons.
   For each valid topology, the cluster is rigid. We only need to optimize the global
   rotation and translation of this rigid cluster relative to the container to minimize
   the container size.
3. Optimization: For a fixed topology, the problem reduces to finding the rotation angle
   theta that minimizes the support function of the cluster's vertices against the container's
   flat sides. This is a 1D optimization problem, which is much more robust and faster than
   22D gradient descent.
4. Validity: Topologies are generated to ensure non-overlap by construction (using exact
   offsets for touching hexagons).
"""

import time
import numpy as np
import itertools
from scipy.optimize import minimize_scalar

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

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

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-5):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
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

    for i in range(N):
        for j in range(i+1, N):
            if sat_overlap(verts_list[i], verts_list[j]):
                return False
            
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    
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

def _build_cluster_from_adjacency(adjacency, start_angle=0.0):
    """
    Given an adjacency list for N hexagons, build the cluster positions.
    Hexagon 0 is at origin.
    We assume the graph is a tree or has a consistent embedding.
    We use BFS to propagate positions.
    
    For two touching unit hexagons, the center distance is 2*sqrt(3).
    The angle between centers depends on the relative orientation.
    If hexagon A has angle alpha and hexagon B has angle beta, and they touch,
    the vector from A to B is determined by the contact edge.
    
    Simplification: Assume all hexagons in the cluster have the same orientation relative to the cluster frame.
    Then, if hexagon i and j are adjacent, the vector from i to j is one of 6 fixed vectors:
    v_k = 2*sqrt(3) * (cos(k*pi/3), sin(k*pi/3)) for k in 0..5.
    
    We need to determine which k for each edge. This is a constraint satisfaction problem.
    To keep it simple, we will define specific "topologies" as fixed sets of relative vectors.
    """
    # This function is a placeholder for a more complex solver.
    # For this implementation, we will hardcode a few known good topologies
    # and generate variations by permuting the "gap" fillers.
    pass

def _get_topology_candidates():
    """
    Generate candidate topologies for 11 hexagons.
    Topology 1: 1 center, 6 ring, 4 outer (current best structure).
    Topology 2: 2 centers, ...
    
    We define topologies as a list of (index, dx, dy) relative to hexagon 0.
    """
    sqrt3 = np.sqrt(3)
    d = 2 * sqrt3
    
    # Standard "Flower" + 4 outer
    # 0: (0,0)
    # 1-6: Ring 1
    ring1 = []
    for i in range(6):
        a = i * np.pi / 3
        ring1.append((i+1, d * np.cos(a), d * np.sin(a)))
    
    # 7-10: Outer ring, in the gaps
    # The gaps are at angles pi/6, 5pi/6, etc.
    # Distance from center to outer hexagon center:
    # If they touch two ring hexagons, the geometry fixes the position.
    # Let's calculate:
    # Ring hex 1 is at (d, 0). Ring hex 2 is at (d/2, d*sqrt(3)/2).
    # An outer hex touching both would be at distance d from both.
    # Intersection of circles radius d centered at (d,0) and (d/2, d*sqrt(3)/2).
    # Symmetry suggests it's along the bisector angle pi/6.
    # Distance R from origin:
    # R^2 = (d cos(pi/6))^2 + (d sin(pi/6))^2? No.
    # Let's use the property that 3 hexagons (center, ring1, ring2) form a rhombus or triangle.
    # Actually, in a hexagonal lattice, the second shell is at distance 4*sqrt(3)? No.
    # Let's just use the lattice coordinates.
    # Axial coordinates (q, r).
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 (12 positions): (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # We pick 4 from Ring 2 that are symmetric.
    # (2,0), (0,2), (-2,0), (0,-2) -> These are "across" from each other.
    # Or (2,-1), (0,2), (-2,1), (0,-2)?
    
    # Let's convert axial to cartesian for a "pointy-top" or "flat-top" hex.
    # Let's stick to the cartesian vectors derived from the lattice.
    # For flat-top hexagons (side 1), the distance between centers is 2*sqrt(3).
    # The lattice vectors are:
    # v0 = (2*sqrt(3), 0)
    # v1 = (sqrt(3), 3)
    # v2 = (-sqrt(3), 3)
    # etc.
    
    # Let's define the 11 positions explicitly for the "standard" packing.
    # 0: (0,0)
    # 1: (2s, 0)
    # 2: (s, 3)
    # 3: (-s, 3)
    # 4: (-2s, 0)
    # 5: (-s, -3)
    # 6: (s, -3)
    # s = sqrt(3)
    
    s = sqrt3
    pos_11 = [
        (0, 0),
        (2*s, 0),
        (s, 3),
        (-s, 3),
        (-2*s, 0),
        (-s, -3),
        (s, -3),
        # Outer 4:
        # Placed in the "corners" of the hexagonal cluster.
        # The cluster of 7 has a width of 4s (from -2s to 2s).
        # The outer hexagons should extend this.
        # Let's try placing them at distance 4s from center, at angles 0, 60, 120, 180?
        # That would collide with ring 1? No, ring 1 is at 2s.
        # If we place at 4s, they are 2s away from ring 1 hexagons?
        # Distance from (2s,0) to (4s,0) is 2s. They touch!
        # So (4s, 0), (2s, 3*sqrt(3)), (-2s, 3*sqrt(3)), (-4s, 0), (-2s, -3*sqrt(3)), (2s, -3*sqrt(3))
        # We only need 4. Let's pick 4 symmetric ones.
        (4*s, 0),
        (-4*s, 0),
        (2*s, 3*sqrt3),
        (-2*s, -3*sqrt3)
    ]
    
    # This specific topology might not be optimal.
    # Let's generate a few variations by rotating the outer 4.
    # But first, let's verify this is a valid non-overlapping set.
    # We will rely on the optimizer to find the best rotation/translation.
    
    candidates = []
    
    # Topology A: The one above
    candidates.append(np.array(pos_11))
    
    # Topology B: Different selection of outer 4
    pos_11_b = [
        (0, 0),
        (2*s, 0),
        (s, 3),
        (-s, 3),
        (-2*s, 0),
        (-s, -3),
        (s, -3),
        (2*s, 3*sqrt3),
        (-2*s, 3*sqrt3),
        (2*s, -3*sqrt3),
        (-2*s, -3*sqrt3)
    ]
    candidates.append(np.array(pos_11_b))
    
    # Topology C: Compact "diamond" shape?
    # Maybe 3-4-3-1?
    # Row 1: 3 hexes
    # Row 2: 4 hexes
    # Row 3: 3 hexes
    # Row 4: 1 hex
    # This is a triangular arrangement.
    # Let's construct it.
    # Row 1 (y=0): x = -s, 0, s ? No, distance must be 2s.
    # Let's use axial coordinates again.
    # (0,0), (1,0), (2,0) -> 3 in a row.
    # (1,-1), (2,-1), (3,-1), (0,-1) -> 4 in next row.
    # (1,-2), (2,-2), (3,-2) -> 3 in next row.
    # (2,-3) -> 1 in next row.
    
    def axial_to_cartesian(q, r, side=1.0):
        # Flat-top
        x = side * 3/2 * q
        y = side * sqrt3 * (r + q/2)
        return x, y
        
    # Note: standard axial to cartesian for flat-top hex with side 1:
    # x = 3/2 * q
    # y = sqrt(3) * (r + q/2)
    # Distance between (q,r) and (q+1,r) is sqrt( (1.5)^2 + (sqrt(3)/2)^2 ) = sqrt(2.25 + 0.75) = sqrt(3).
    # Wait, side length 1 hexagon has center-to-center distance 2*sqrt(3) for touching?
    # No, for regular hexagons of side 1, the distance between centers of touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) * 1 = sqrt(3)?
    # Let's re-verify.
    # Apothem a = s * sqrt(3)/2.
    # Distance between centers = 2 * a = s * sqrt(3).
    # So distance is sqrt(3) * s. If s=1, distance is sqrt(3).
    # My previous code used 2*sqrt(3). That is WRONG.
    # Let's check the current program's assumption.
    # Current program: `min_center_dist = 2 * np.sqrt(3)`.
    # Is that correct?
    # If side length is 1, the width (flat-to-flat) is sqrt(3).
    # So two hexagons side-by-side have center distance sqrt(3).
    # Why did the current program use 2*sqrt(3)?
    # Maybe the side length is not 1?
    # Task: "Unit regular hexagons". Side length 1.
    # So distance is sqrt(3).
    # The current program's `min_center_dist` is 2*sqrt(3) which is double the actual touching distance.
    # This implies the current program's "validity" check might be too loose or the objective function penalizes distances < 2*sqrt(3) as overlaps, which is incorrect (they can be closer than 2*sqrt(3) as long as they don't overlap).
    # Actually, if they touch, distance is sqrt(3). If distance is 2*sqrt(3), there is a gap of sqrt(3).
    # So the current program is forcing a large gap between hexagons!
    # This is a HUGE insight. The current program is suboptimal because it artificially inflates the required space by using the wrong minimum distance in the penalty term.
    
    # Let's correct this in the new strategy.
    # Correct touching distance: d_touch = sqrt(3) * side = sqrt(3).
    
    # Let's rebuild the topologies with d_touch = sqrt(3).
    
    s = sqrt3
    d = s # distance between centers
    
    # Topology A: 1-6-4
    # 0: (0,0)
    # 1-6: Ring 1 at distance d
    ring1 = []
    for i in range(6):
        a = i * np.pi / 3
        ring1.append((i+1, d * np.cos(a), d * np.sin(a)))
        
    # 7-10: Outer 4
    # If ring 1 is at d, outer ring touching ring 1 is at distance 2d?
    # Let's check geometry.
    # Hex 0 at (0,0). Hex 1 at (d, 0).
    # Hex 7 touches Hex 1.
    # If Hex 7 is at (2d, 0), it touches Hex 1 (dist d) and is far from Hex 0 (dist 2d).
    # This forms a line.
    # Let's try a more compact arrangement.
    # Hex 7 at angle pi/6 from center, distance R.
    # It should touch Hex 1 (at 0 deg) and Hex 2 (at 60 deg).
    # By symmetry, it is at 30 deg.
    # Distance from Hex 1 to Hex 7 is d.
    # Distance from Hex 2 to Hex 7 is d.
    # Hex 1 is at (d, 0). Hex 2 is at (d/2, d*sqrt(3)/2).
    # Midpoint of Hex1-Hex2 is (3d/4, d*sqrt(3)/4).
    # Vector Hex1->Hex2 is (-d/2, d*sqrt(3)/2). Perpendicular is (d*sqrt(3)/2, d/2).
    # Unit perp: (sqrt(3)/2, 1/2).
    # Height of equilateral triangle with side d is d*sqrt(3)/2.
    # So Hex 7 is at Midpoint + (d*sqrt(3)/2) * (sqrt(3)/2, 1/2)
    # = (3d/4 + 3d/4, d*sqrt(3)/4 + d/4)
    # = (3d/2, d*(sqrt(3)+1)/4)
    # Let's check distance from origin:
    # R^2 = (3d/2)^2 + ...
    # This is getting complicated. Let's just use the lattice vectors.
    
    # Lattice vectors for flat-top hexagons with side 1:
    # v0 = (sqrt(3), 0)
    # v1 = (sqrt(3)/2, 1.5)  -> Wait, y-component?
    # If flat-top, vertices are at 0, 60, 120...
    # Center to vertex is 1.
    # Vertex 0 is at (1, 0). Vertex 1 is at (1/2, sqrt(3)/2).
    # Edge 0-1 is from (1,0) to (1/2, sqrt(3)/2).
    # Normal to this edge is (sqrt(3)/2, 1/2).
    # Distance between centers = 2 * apothem = 2 * sqrt(3)/2 = sqrt(3).
    # So the vector to the neighbor across edge 0-1 is sqrt(3) * (sqrt(3)/2, 1/2) = (3/2, sqrt(3)/2).
    # So v1 = (1.5, sqrt(3)/2).
    # v2 = (-1.5, sqrt(3)/2).
    # v3 = (-sqrt(3), 0).
    # v4 = (-1.5, -sqrt(3)/2).
    # v5 = (1.5, -sqrt(3)/2).
    
    v0 = np.array([np.sqrt(3), 0])
    v1 = np.array([1.5, np.sqrt(3)/2])
    v2 = np.array([-1.5, np.sqrt(3)/2])
    v3 = np.array([-np.sqrt(3), 0])
    v4 = np.array([-1.5, -np.sqrt(3)/2])
    v5 = np.array([1.5, -np.sqrt(3)/2])
    
    # Topology A: 1 center, 6 ring, 4 outer
    # 0: 0
    # 1-6: v0..v5
    # 7-10: Outer.
    # Let's pick 4 outer positions that are "corners".
    # Corner 1: v0 + v1 = (sqrt(3)+1.5, sqrt(3)/2)
    # Corner 2: v2 + v3 = (-sqrt(3)-1.5, sqrt(3)/2)
    # Corner 3: v4 + v5 = (-sqrt(3)+1.5, -sqrt(3)/2)
    # Corner 4: v5 + v0 = (sqrt(3)+1.5, -sqrt(3)/2)
    
    pos_A = [np.zeros(2)]
    for i in range(6):
        pos_A.append(np.array([v0, v1, v2, v3, v4, v5][i]))
    pos_A.append(v0 + v1)
    pos_A.append(v2 + v3)
    pos_A.append(v4 + v5)
    pos_A.append(v5 + v0)
    
    candidates.append(np.array(pos_A))
    
    # Topology B: Different outer 4
    # Maybe v0+v5, v1+v2, v3+v4, v4+v3?
    pos_B = [np.zeros(2)]
    for i in range(6):
        pos_B.append(np.array([v0, v1, v2, v3, v4, v5][i]))
    pos_B.append(v0 + v5)
    pos_B.append(v1 + v2)
    pos_B.append(v3 + v4)
    pos_B.append(v4 + v3) # Duplicate? v3+v4 is same as v4+v3.
    # Let's use v2+v1, v3+v4, v4+v5, v5+v0
    pos_B = [np.zeros(2)]
    for i in range(6):
        pos_B.append(np.array([v0, v1, v2, v3, v4, v5][i]))
    pos_B.append(v1 + v2)
    pos_B.append(v2 + v3)
    pos_B.append(v3 + v4)
    pos_B.append(v4 + v5)
    
    candidates.append(np.array(pos_B))
    
    # Topology C: 2 centers
    # 0: (0,0), 1: (sqrt(3), 0)
    # 2-5: Ring around 0 (excluding 1)
    # 6-9: Ring around 1 (excluding 0)
    # 10: Outer
    
    pos_C = [np.zeros(2), v0]
    # Ring around 0: v1, v2, v3, v4, v5 (v0 is taken by 1)
    pos_C.append(v1)
    pos_C.append(v2)
    pos_C.append(v3)
    pos_C.append(v4)
    pos_C.append(v5)
    # Ring around 1 (center v0):
    # Neighbors of v0 relative to v0 are v0..v5.
    # So absolute positions: v0+v0, v0+v1, v0+v2, v0+v3, v0+v4, v0+v5.
    # v0+v3 is (0,0) which is 0. So skip v0+v3.
    pos_C.append(v0 + v0)
    pos_C.append(v0 + v1)
    pos_C.append(v0 + v2)
    pos_C.append(v0 + v4)
    pos_C.append(v0 + v5)
    # We have 1+1+5+5 = 12. We need 11.
    # Remove one. Let's remove v0+v0 (the one at 2*sqrt(3)).
    pos_C.pop()
    
    candidates.append(np.array(pos_C))
    
    return candidates

def _optimize_rigid_cluster(cluster_pos):
    """
    Optimize the rotation and translation of a rigid cluster to minimize
    the bounding hexagon side length.
    """
    # Center the cluster at origin to simplify translation optimization
    centroid = np.mean(cluster_pos, axis=0)
    centered_pos = cluster_pos - centroid
    
    def objective(theta):
        # Rotate cluster
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        rotated_pos = centered_pos @ rot_mat.T
        
        # Calculate min outer side
        # We can optimize translation too, but for a centrally symmetric container,
        # centering the cluster's centroid is often a good starting point.
        # However, the optimal translation might shift the centroid.
        # Let's fix translation to (0,0) for now and just optimize rotation.
        # Then do a local search on translation.
        
        # For now, just return the side length with center at (0,0)
        # and inner angles all 0 (since they are rigid, relative angles are fixed,
        # but global angle is theta).
        # Wait, if we rotate the cluster, the inner hexagons also rotate.
        # So inner_angles are all theta.
        
        ang_deg = np.degrees(theta)
        side = _min_outer_side(rotated_pos, np.full(N, ang_deg), (0.0, 0.0), 0.0)
        return side
    
    # Optimize theta in [0, pi/3] due to symmetry
    res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-6})
    
    best_theta = res.x
    best_side = res.fun
    
    # Now optimize translation (tx, ty) for this theta
    c, s = np.cos(best_theta), np.sin(best_theta)
    rot_mat = np.array([[c, -s], [s, c]])
    rotated_pos = centered_pos @ rot_mat.T
    
    def obj_trans(params):
        tx, ty = params
        pos = rotated_pos + np.array([tx, ty])
        ang_deg = np.degrees(best_theta)
        side = _min_outer_side(pos, np.full(N, ang_deg), (0.0, 0.0), 0.0)
        return side
    
    res_trans = minimize_scalar(lambda p: obj_trans([p, 0]), bounds=(-2, 2), method='bounded')
    # This is a 1D search. Let's do a simple grid or Nelder-Mead on 2D.
    from scipy.optimize import minimize
    res_trans2 = minimize(obj_trans, [0, 0], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6})
    
    if res_trans2.fun < best_side:
        best_side = res_trans2.fun
        best_tx, best_ty = res_trans2.x
    else:
        best_tx, best_ty = 0, 0
        
    # Final positions
    final_pos = rotated_pos + np.array([best_tx, best_ty])
    final_ang = np.full(N, np.degrees(best_theta))
    
    return final_pos, final_ang, best_side

def construct():
    global START_TIME, DEADLINE
    
    best_score = np.inf
    best_inner = None
    
    candidates = _get_topology_candidates()
    
    for i, cluster_pos in enumerate(candidates):
        if time.time() > DEADLINE - 10.0:
            break
            
        # Check validity of the topology (non-overlap)
        # Since we built them from touching vectors, they should be valid.
        # But let's verify with SAT just in case of floating point issues or bad topology.
        
        # Optimize
        try:
            pos, ang, side = _optimize_rigid_cluster(cluster_pos)
            
            if side < best_score:
                # Verify validity
                inner_list = [[float(pos[j,0]), float(pos[j,1]), float(ang[j])] for j in range(N)]
                if _check_validity(inner_list, (0.0, 0.0), side, 0.0):
                    best_score = side
                    best_inner = inner_list
        except Exception as e:
            pass
    
    if best_inner is None:
        # Fallback to a simple configuration
        # 1 center, 6 ring, 4 outer
        s = np.sqrt(3)
        d = s
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        
        pos[0] = [0, 0]
        for i in range(6):
            a = i * np.pi / 3
            pos[i+1] = [d * np.cos(a), d * np.sin(a)]
            
        # Outer 4
        pos[7] = [2*d, 0]
        pos[8] = [-2*d, 0]
        pos[9] = [d, 1.5*d]
        pos[10] = [-d, -1.5*d]
        
        inner_list = [[float(pos[i,0]), float(pos[i,1]), float(ang[i])] for i in range(N)]
        best_score = _min_outer_side(pos, ang, (0.0, 0.0), 0.0)
        
    return best_inner, (0.0, 0.0), float(best_score), 0.0

if __name__ == "__main__":
    construct()
