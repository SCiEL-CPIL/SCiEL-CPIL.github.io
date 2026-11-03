import numpy as np
from scipy.optimize import linprog
import time

N = 12

# Directions for a regular hexagon (flat-top): edge normals at 0, 60, 120, 180, 240, 300 degrees
# For a regular hexagon with side length 1, inradius = sqrt(3)/2
INRADIUS_1 = np.sqrt(3) / 2.0

# The 6 outward edge-normal directions for a flat-top regular hexagon
# Vertex at angle 0 is to the right. Edge normals are perpendicular to edges.
# For flat-top hexagon, edges are at angles 30, 90, 150, 210, 270, 330 from center
# Edge normals (outward) are at angles 0, 60, 120, 180, 240, 300
def hex_edge_normals():
    angles = np.radians([0, 60, 120, 180, 240, 300])
    return np.column_stack([np.cos(angles), np.sin(angles)])

# Vertices of a unit regular hexagon (flat-top, side=1), relative to center
def hex_vertices_unit():
    # Flat-top: vertices at angles 0, 60, 120, 180, 240, 300
    angles = np.radians([0, 60, 120, 180, 240, 300])
    # For side length 1, the distance from center to vertex (circumradius) = 1
    return np.column_stack([np.cos(angles), np.sin(angles)])

NORMALS = hex_edge_normals()  # 6 x 2
VERTS_UNIT = hex_vertices_unit()  # 6 x 2

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    # Variables: for each inner hex i: x_i, y_i (2N vars)
    # Plus: outer center X, Y (2 vars), outer side s (1 var)
    # Total: 2N + 3 variables
    n_vars = 2 * N + 3
    
    # Objective: minimize s -> c = [0, 0, ..., 0, 0, 0, 1]
    c_obj = np.zeros(n_vars)
    c_obj[-1] = 1.0  # minimize s
    
    A_ub = []
    b_ub = []
    
    # Non-overlap constraints between inner hexagons:
    # For each pair (i, j), i < j, and each of 6 directions d_k:
    # d_k . (c_i - c_j) >= 2 * INRADIUS_1  OR  d_k . (c_i - c_j) <= -2 * INRADIUS_1
    # This is disjunctive. We need to handle this.
    # Alternative: use the fact that two convex polygons are disjoint iff
    # there exists a separating direction. For regular hexagons with same orientation,
    # it's sufficient to check the 6 edge-normal directions.
    # But the disjunction makes it non-linear for LP.
    
    # Better approach: Use a different formulation.
    # Two regular hexagons (same orientation, side 1) with centers c_i, c_j are disjoint
    # iff min_k |d_k . (c_i - c_j)| >= 2 * INRADIUS_1 is NOT the right condition.
    
    # Actually, two convex polygons are disjoint iff there exists a direction u such that
    # the projections are disjoint. For regular hexagons with the same orientation,
    # the separating axis is one of the 6 edge normals (by SAT theorem).
    # So: disjoint iff exists k in {0..5} such that |d_k . (c_i - c_j)| >= 2 * INRADIUS_1
    # This is still disjunctive.
    
    # Let me use a different strategy: local search / simulated annealing with
    # geometric feasibility checks, then optimize the outer hexagon.
    
    # Actually, let me try a simpler approach: use scipy's minimize with
    # penalty for overlaps, and binary search on outer side length.
    
    # First, let me set up the problem properly.
    # For a given outer side length S and outer center C, outer rotation 0,
    # check if we can place N inner hexagons (each with center and rotation)
    # such that they don't overlap each other and are inside the outer hexagon.
    
    # This is a non-convex problem. Let me use a heuristic approach:
    # 1. Start with a reasonable configuration
    # 2. Use local optimization to reduce the required outer side length
    
    # Let me think about the geometry more carefully.
    # A regular hexagon with side length S has inradius S*sqrt(3)/2.
    # The outer hexagon must contain all inner hexagons.
    
    # For now, let me use a grid-based search with local refinement.
    
    # Initial configuration: place hexagons in a honeycomb pattern
    # Honeycomb spacing: adjacent hexagons in honeycomb have center distance 2*INRADIUS_1 = sqrt(3)
    
    # Let me use a simple local search approach
    def hex_overlap(c1, a1, c2, a2):
        """Check if two unit hexagons overlap. c1, c2 are centers (2,), a1, a2 are angles in radians."""
        # Get vertices of both hexagons
        def get_verts(center, angle):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            return center + VERTS_UNIT.T @ R.T  # 6 x 2 vertices
        
        v1 = get_verts(c1, a1)
        v2 = get_verts(c2, a2)
        
        # SAT check: for each edge normal of both polygons, check separation
        # Edge normals of hexagon 1
        def edge_normals(verts):
            edges = np.roll(verts, -1, axis=0) - verts
            # Normal to each edge (outward)
            normals = np.column_stack([edges[:, 1], -edges[:, 0]])
            # Ensure outward: check against center
            center = np.mean(verts, axis=0)
            for i in range(len(normals)):
                if np.dot(normals[i], verts[i] - center) < 0:
                    normals[i] *= -1
            return normals
        
        n1 = edge_normals(v1)
        n2 = edge_normals(v2)
        
        for n in list(n1) + list(n2):
            proj1 = v1 @ n
            proj2 = v2 @ n
            if proj1.max() <= proj2.min() or proj2.max() <= proj1.min():
                return False  # Separated, no overlap
        return True  # Overlap
    
    def inside_hexagon(vert, center, side_len, angle=0.0):
        """Check if a point is inside a regular hexagon."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        v_local = R @ (vert - center)
        # For a flat-top hexagon with side s, the constraints are:
        # |x| <= s, |y| <= s*sqrt(3)/2, |x/2 + y*sqrt(3)/2| <= s
        # Actually, let me use the edge-normal formulation
        # For side length s, inradius = s*sqrt(3)/2
        # Edge normals at 0, 60, 120, 180, 240, 300 degrees
        # Constraint: d_k . v_local <= s * sqrt(3)/2 for all k
        inradius = side_len * np.sqrt(3) / 2
        for d in NORMALS:
            if np.dot(d, v_local) > inradius + 1e-9:
                return False
        return True
    
    def check_feasible(inner_centers, inner_angles, outer_center, outer_side, outer_angle=0.0):
        """Check if configuration is feasible."""
        # Check pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                if hex_overlap(inner_centers[i], inner_angles[i], inner_centers[j], inner_angles[j]):
                    return False
        # Check containment
        for i in range(N):
            def get_verts(center, angle):
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                return center + VERTS_UNIT.T @ R.T
            verts = get_verts(inner_centers[i], inner_angles[i])
            for v in verts:
                if not inside_hexagon(v, outer_center, outer_side, outer_angle):
                    return False
        return True
    
    def total_overlap_penalty(inner_centers, inner_angles):
        """Compute total overlap penalty (number of overlapping pairs)."""
        penalty = 0
        for i in range(N):
            for j in range(i + 1, N):
                if hex_overlap(inner_centers[i], inner_angles[i], inner_centers[j], inner_angles[j]):
                    penalty += 1
        return penalty
    
    # Initial configuration: honeycomb-like arrangement
    # For 12 hexagons, a 3x4 or similar arrangement
    # Honeycomb: row offset
    # Let me place them in a pattern
    spacing = 2 * INRADIUS_1 * 1.01  # slight gap to avoid numerical issues
    
    # 3 rows, 4 columns, with hexagonal offset
    inner_centers = []
    inner_angles = [0.0] * N
    row_count = [4, 4, 4]  # 3 rows of 4
    row = 0
    idx = 0
    for r in range(3):
        offset = (r % 2) * spacing / 2
        for c in range(4):
            x = c * spacing + offset
            y = r * spacing * np.sqrt(3) / 2
            inner_centers.append([x, y])
            idx += 1
    
    inner_centers = np.array(inner_centers)
    
    # Center the configuration
    centroid = np.mean(inner_centers, axis=0)
    inner_centers -= centroid
    outer_center = np.array([0.0, 0.0])
    
    # Estimate initial outer side length
    # Find max distance from center to any vertex
    max_dist = 0
    for i in range(N):
        cos_a = np.cos(inner_angles[i])
        sin_a = np.sin(inner_angles[i])
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = inner_centers[i] + VERTS_UNIT.T @ R.T
        for v in verts:
            d = np.linalg.norm(v - outer_center)
            max_dist = max(max_dist, d)
    # For a hexagon, the max distance from center to a point inside relates to side length
    # inradius = side * sqrt(3)/2, so side = 2 * inradius / sqrt(3)
    # But we need to account for the hexagon shape
    # Conservative estimate: side = max_dist * 2 / sqrt(3) * 1.1
    outer_side = max_dist * 2 / np.sqrt(3) * 1.2
    
    # Local search to reduce outer_side
    best_config = (inner_centers.copy(), inner_angles.copy(), outer_center.copy(), outer_side)
    
    # Simulated annealing
    import random
    random.seed(42)
    np.random.seed(42)
    
    T = 1.0
    cooling = 0.999
    steps = 5000
    
    current_inner = inner_centers.copy()
    current_angles = inner_angles.copy()
    current_outer_center = outer_center.copy()
    current_outer_side = outer_side
    
    # Check initial feasibility
    if check_feasible(current_inner, current_angles, current_outer_center, current_outer_side):
        best_config = (current_inner.copy(), current_angles.copy(), current_outer_center.copy(), current_outer_side)
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        # Propose a small move
        move_type = random.random()
        if move_type < 0.6:
            # Move a random inner hexagon
            i = random.randint(0, N - 1)
            dx = random.gauss(0, 0.1)
            dy = random.gauss(0, 0.1)
            new_inner = current_inner.copy()
            new_inner[i] += [dx, dy]
            new_angles = current_angles.copy()
            new_outer_center = current_outer_center.copy()
            new_outer_side = current_outer_side
        elif move_type < 0.8:
            # Rotate a random inner hexagon
            i = random.randint(0, N - 1)
            new_inner = current_inner.copy()
            new_angles = current_angles.copy()
            new_angles[i] += random.gauss(0, 0.1)
            new_outer_center = current_outer_center.copy()
            new_outer_side = current_outer_side
        elif move_type < 0.9:
            # Move outer center
            new_inner = current_inner.copy()
            new_angles = current_angles.copy()
            new_outer_center = current_outer_center + np.array([random.gauss(0, 0.05), random.gauss(0, 0.05)])
            new_outer_side = current_outer_side
        else:
            # Adjust outer side
            new_inner = current_inner.copy()
            new_angles = current_angles.copy()
            new_outer_center = current_outer_center.copy()
            new_outer_side = current_outer_side * (1 + random.gauss(0, 0.01))
        
        # Check feasibility of new config
        if check_feasible(new_inner, new_angles, new_outer_center, new_outer_side):
            # Accept if it improves or with some probability
            if new_outer_side < current_outer_side:
                current_inner = new_inner
                current_angles = new_angles
                current_outer_center = new_outer_center
                current_outer_side = new_outer_side
                if current_outer_side < best_config[3]:
                    best_config = (current_inner.copy(), current_angles.copy(), current_outer_center.copy(), current_outer_side)
            else:
                # Accept with probability exp(-(new_side - current_side) / T)
                delta = new_outer_side - current_outer_side
                if random.random() < np.exp(-delta / T):
                    current_inner = new_inner
                    current_angles = new_angles
                    current_outer_center = new_outer_center
                    current_outer_side = new_outer_side
        else:
            # Infeasible: accept with small probability to escape local minima
            if random.random() < 0.01:
                current_inner = new_inner
                current_angles = new_angles
                current_outer_center = new_outer_center
                current_outer_side = new_outer_side
        
        T *= cooling
        if T < 0.001:
            T = 0.001
    
    # Return best configuration
    bi, ba, bc, bs = best_config
    inner = [[float(bi[i][0]), float(bi[i][1]), float(np.degrees(ba[i]))] for i in range(N)]
    return inner, (float(bc[0]), float(bc[1])), float(bs), 0.0

