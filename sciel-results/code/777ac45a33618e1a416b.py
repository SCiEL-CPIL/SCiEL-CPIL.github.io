import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_result = None
    best_side = float('inf')
    
    # Hexagonal lattice basis for pointy-top unit hexagons
    # Center-to-center distance for adjacent hexagons
    # For pointy-top: width = sqrt(3), height = 2
    # Adjacent centers: horizontal dist = sqrt(3), diagonal = sqrt(3) at 60 deg
    basis1 = np.array([np.sqrt(3), 0.0])
    basis2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate hexagonal cluster arrangements
    # For a hexagonal grid, use axial coordinates (q, r)
    # Convert axial to cartesian: x = sqrt(3)*(q + r/2), y = 3*r/2
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    # Generate all hexagonal clusters of size N
    # A hexagonal cluster of "radius" k has hexagons at distance <= k from center
    # Number of hexagons at distance d from center: 6*d for d>=1, 1 for d=0
    # Total for radius k: 1 + 6*(1+2+...+k) = 1 + 3k(k+1)
    # k=0: 1, k=1: 7, k=2: 19
    
    # For N=11, we can use a radius-2 cluster (19 positions) and pick 11 compactly
    # Or use other arrangements
    
    # Strategy: try multiple arrangement patterns
    arrangements = []
    
    # Pattern 1: Hexagonal cluster, take first N in spiral order
    # Generate spiral order from center
    def hex_spiral_positions(n):
        """Generate n hexagon positions in spiral order from center."""
        positions = [np.array([0.0, 0.0])]  # center
        if n == 1:
            return positions
        
        # Ring 1: 6 positions
        ring1 = []
        for i in range(6):
            angle = np.pi / 3 * i  # 0, 60, 120, 180, 240, 300 degrees
            # For pointy-top, the 6 neighbors are at these angles from center
            # Distance = sqrt(3)
            x = np.sqrt(3) * np.cos(angle)
            y = np.sqrt(3) * np.sin(angle)
            ring1.append(np.array([x, y]))
        
        # Actually, let me use the proper hex grid neighbor directions
        # For pointy-top hexagons, the 6 neighbor directions (center-to-center) are:
        # angles: 0, 60, 120, 180, 240, 300 degrees, distance = sqrt(3)
        # But wait, that's not right for a hex grid. Let me reconsider.
        
        # In a hex grid with pointy-top hexagons:
        # The 6 neighbors of a hexagon are at:
        # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), 
        # (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
        # These are at angles 0, 60, 120, 180, 240, 300 from horizontal, distance sqrt(3)
        
        # So ring 1 positions:
        neighbors = []
        for angle_deg in [0, 60, 120, 180, 240, 300]:
            angle = np.radians(angle_deg)
            pos = np.sqrt(3) * np.array([np.cos(angle), np.sin(angle)])
            neighbors.append(pos)
        
        positions.extend(neighbors)
        
        if len(positions) >= n:
            return positions[:n]
        
        # Ring 2: 12 positions
        # These are at distance 2*sqrt(3) in the 6 main directions, plus intermediate
        # Actually, ring 2 has 12 hexagons. Let me generate them properly.
        # Using axial coordinates: ring k has hexagons with max(|q|,|r|,|q+r|) = k
        # Convert to cartesian: x = sqrt(3)*(q + r/2), y = 3*r/2
        
        # Let me use a different approach: generate all positions in a large hex cluster
        # and sort by distance from center
        
        all_positions = []
        max_ring = 3  # enough for N=11
        for r in range(-max_ring, max_ring + 1):
            for q in range(-max_ring, max_ring + 1):
                # Check if in hex cluster of radius max_ring
                if abs(q + r) <= max_ring and abs(q) <= max_ring and abs(r) <= max_ring:
                    # Actually, hex distance is max(|q|, |r|, |q+r|) for some conventions
                    # Let me use: distance = (|q| + |r| + |q+r|) / 2
                    dist = (abs(q) + abs(r) + abs(q + r)) / 2
                    if dist <= max_ring:
                        pos = axial_to_cartesian(q, r)
                        all_positions.append((dist, q, r, pos))
        
        # Sort by distance, then by angle for deterministic ordering
        all_positions.sort(key=lambda x: (x[0], x[1], x[2]))
        
        spiral = [p[3] for p in all_positions]
        return spiral[:n]
    
    # Try spiral arrangement
    spiral_positions = hex_spiral_positions(N)
    arrangements.append(spiral_positions)
    
    # Pattern 2: Row-based arrangement (3-4-4 or similar)
    # For pointy-top hexagons in rows:
    # Row spacing: 1.5 (vertical distance between row centers)
    # Horizontal offset within row: sqrt(3)
    # Adjacent rows offset by sqrt(3)/2
    
    def row_arrangement(counts):
        """Generate positions for rows with given counts."""
        positions = []
        for row_idx, count in enumerate(counts):
            y = 1.5 * row_idx
            x_offset = (np.sqrt(3) / 2) * (row_idx % 2)  # stagger
            for i in range(count):
                x = x_offset + np.sqrt(3) * (i - (count - 1) / 2.0)
                positions.append(np.array([x, y]))
        return positions
    
    # Try different row patterns for N=11
    row_patterns = [
        [3, 4, 4],      # 3+4+4=11
        [4, 4, 3],      # 4+4+3=11
        [2, 4, 4, 1],   # 2+4+4+1=11
        [1, 4, 4, 2],   # 1+4+4+2=11
        [3, 3, 3, 2],   # 3+3+3+2=11
        [2, 3, 3, 3],   # 2+3+3+3=11
        [1, 3, 4, 3],   # 1+3+4+3=11
        [3, 4, 3, 1],   # 3+4+3+1=11
    ]
    
    for pattern in row_patterns:
        if sum(pattern) == N:
            arr = row_arrangement(pattern)
            arrangements.append(arr)
    
    # For each arrangement, compute the minimal enclosing hexagon
    def compute_min_enclosing_hexagon(positions):
        """
        Compute the minimal side length of a regular hexagon (same orientation as unit hexagons)
        that contains all given hexagon centers, accounting for the hexagon size.
        
        A unit hexagon centered at p extends to distance 1 in all vertex directions.
        For the outer hexagon with center c and side length s, a point p is inside if
        the projection of (p - c) onto each of the 6 side normals is <= s * sqrt(3)/2.
        
        The 6 side normals for a pointy-top hexagon are at angles 30, 90, 150, 210, 270, 330 degrees.
        Actually, for a pointy-top hexagon, the flat sides are at angles 0, 60, 120, 180, 240, 300
        from horizontal, so the normals are at 30, 90, 150, 210, 270, 330.
        
        Wait, let me reconsider. For a pointy-top hexagon:
        - Vertices at angles 90, 150, 210, 270, 330, 30 (pointing up)
        - Edges are between consecutive vertices
        - The edge normals point outward at angles 60, 120, 180, 240, 300, 0
        
        Actually, the inradius (apothem) is sqrt(3)/2 for a unit hexagon.
        The 6 support directions (normals to edges) are at angles 0, 60, 120, 180, 240, 300.
        
        For a hexagon with center c and side length s, the constraint for a point p to be
        inside (with margin for the hexagon's own extent) is:
        For each normal direction n_i: dot(p - c, n_i) + 1 <= s * sqrt(3)/2
        
        Because the unit hexagon extends to distance 1 in the vertex direction, but the
        relevant extent in the normal direction is the inradius = sqrt(3)/2.
        
        Hmm, I need to be more careful. The unit hexagon centered at p has vertices at
        distance 1 from p. The maximum extent in direction n (a unit vector) is the
        support function h(n) = max dot(v, n) over vertices v of the hexagon.
        
        For a pointy-top unit hexagon with vertices at angles 90, 150, 210, 270, 330, 30:
        h(n) for n at angle 0 (horizontal) = max(cos(90), cos(150), cos(210), cos(270), cos(330), cos(30))
        = max(0, -sqrt(3)/2, -sqrt(3)/2, 0, sqrt(3)/2, sqrt(3)/2) = sqrt(3)/2
        
        For n at angle 90 (vertical) = max(sin(90), sin(150), sin(210), sin(270), sin(330), sin(30))
        = max(1, 0.5, -0.5, -1, -0.5, 0.5) = 1
        
        So the support function depends on direction.
        
        For the outer hexagon with side length s, center c, its interior is defined by:
        dot(x - c, n_i) <= s * sqrt(3)/2 for i=0..5, where n_i are the 6 edge normals.
        
        For a unit hexagon centered at p to be inside, we need:
        dot(p - c, n_i) + h_inner(n_i) <= s * sqrt(3)/2
        
        where h_inner(n_i) is the support function of the unit hexagon in direction n_i.
        
        The 6 edge normals of a pointy-top hexagon are at angles 0, 60, 120, 180, 240, 300.
        For these directions, h_inner:
        - n at 0 deg: h = sqrt(3)/2
        - n at 60 deg: h = 1 (vertex direction)
        - n at 120 deg: h = sqrt(3)/2
        - n at 180 deg: h = sqrt(3)/2
        - n at 240 deg: h = 1
        - n at 300 deg: h = sqrt(3)/2
        
        Wait, that's not right. Let me recalculate.
        
        Vertices of pointy-top unit hexagon (center at origin):
        V0 = (0, 1)           angle 90
        V1 = (-sqrt(3)/2, 1/2) angle 150
        V2 = (-sqrt(3)/2, -1/2) angle 210
        V3 = (0, -1)          angle 270
        V4 = (sqrt(3)/2, -1/2) angle 330
        V5 = (sqrt(3)/2, 1/2) angle 30
        
        Edge normals (outward) for a pointy-top hexagon:
        The edges are: V0-V1, V1-V2, V2-V3, V3-V4, V4-V5, V5-V0
        
        Edge V0-V1: from (0,1) to (-sqrt(3)/2, 1/2). Direction: (-sqrt(3)/2, -1/2).
        Normal (outward): rotate 90 deg clockwise: (-1/2, sqrt(3)/2). Angle: 120 deg.
        
        Hmm, this is getting complicated. Let me use a simpler approach.
        
        For a regular hexagon with side length s and center c, the condition that a point p
        is inside is:
        max over 6 directions of |dot(p - c, d_i)| <= s * sqrt(3)/2
        
        where d_i are unit vectors at angles 0, 60, 120, 180, 240, 300.
        
        Wait, that's for a flat-top hexagon. For pointy-top, the normals are at 30, 90, 150, 210, 270, 330.
        
        Let me just use: for a regular hexagon with "radius" R (center to vertex), the
        inradius is R * sqrt(3)/2. The 6 support directions are at the edge normals.
        
        For simplicity, let me define the outer hexagon by its inradius r_in = s * sqrt(3)/2.
        A point p is inside if for all 6 normal directions n_i: dot(p - c, n_i) <= r_in.
        
        For the unit hexagon centered at p to fit, we need:
        dot(p - c, n_i) + h_unit(n_i) <= r_in
        
        where h_unit(n_i) is the support function of the unit hexagon in direction n_i.
        
        For a pointy-top unit hexagon, the 6 vertex directions are at 90, 150, 210, 270, 330, 30.
        The 6 edge normal directions are at 60, 120, 180, 240, 300, 0.
        
        h_unit at angle 0: max dot(V, (1,0)) = max(0, -sqrt(3)/2, -sqrt(3)/2, 0, sqrt(3)/2, sqrt(3)/2) = sqrt(3)/2
        h_unit at angle 60: max dot(V, (1/2, sqrt(3)/2)) = 
          V0: 0*1/2 + 1*sqrt(3)/2 = sqrt(3)/2
          V1: -sqrt(3)/2*1/2 + 1/2*sqrt(3)/2 = -sqrt(3)/4 + sqrt(3)/4 = 0
          V2: -sqrt(3)/2*1/2 + (-1/2)*sqrt(3)/2 = -sqrt(3)/4 - sqrt(3)/4 = -sqrt(3)/2
          V3: 0*1/2 + (-1)*sqrt(3)/2 = -sqrt(3)/2
          V4: sqrt(3)/2*1/2 + (-1/2)*sqrt(3)/2 = sqrt(3)/4 - sqrt(3)/4 = 0
          V5: sqrt(3)/2*1/2 + 1/2*sqrt(3)/2 = sqrt(3)/4 + sqrt(3)/4 = sqrt(3)/2
          max = sqrt(3)/2
        
        Hmm, so h_unit is sqrt(3)/2 for all 6 normal directions? That would make sense if the
        hexagon is symmetric. Let me verify at angle 90:
        h_unit at angle 90: max dot(V, (0,1)) = max(1, 1/2, -1/2, -1, -1/2, 1/2) = 1
        
        But 90 is not one of the 6 normal directions for a pointy-top hexagon. The normals are
        at 0, 60, 120, 180, 240, 300. So h_unit at these 6 directions is sqrt(3)/2.
        
        Wait, that can't be right because the hexagon extends to distance 1 in the vertex directions.
        
        Oh I see my confusion. The support function h(n) = max dot(v, n) over all points v in the
        hexagon. For a convex polygon, this is achieved at a vertex.
        
        For n at angle 0: h = sqrt(3)/2 (achieved at V4 and V5)
        For n at angle 60: h = sqrt(3)/2 (achieved at V0 and V5)
        
        So yes, for the 6 edge normal directions, h_unit = sqrt(3)/2.
        
        Therefore, the constraint is:
        dot(p - c, n_i) + sqrt(3)/2 <= s * sqrt(3)/2
        
        for i = 0, ..., 5, where n_i are the 6 unit normals at angles 0, 60, 120, 180, 240, 300.
        
        This simplifies to:
        dot(p - c, n_i) <= s * sqrt(3)/2 - sqrt(3)/2 = (s - 1) * sqrt(3)/2
        
        So the effective "inradius" for the centers is (s - 1) * sqrt(3)/2.
        
        To find the minimal s, we need to find the center c that minimizes the maximum over
        all points p and all 6 directions n_i of dot(p - c, n_i).
        
        Let M(c) = max over p in positions, i in 0..5 of dot(p - c, n_i)
        
        We want to minimize M(c), and then s = 1 + 2 * M(c) / sqrt(3).
        
        The optimal c is the center of the minimum enclosing hexagon, which can be found by
        linear programming or by noting that it's related to the Chebyshev center.
        
        For a hexagon, the minimum enclosing hexagon center can be found by solving:
        minimize t
        subject to: dot(p_j - c, n_i) <= t for all j, i
                    dot(c - p_j, n_i) <= t for all j, i
        
        This is a linear program. But for simplicity, we can use the fact that the optimal center
        is the average of the extreme points in each direction, or use scipy.
        
        Actually, for a regular hexagon, the minimum enclosing hexagon (with fixed orientation)
        has its center at the centroid of the "dual" problem. Let me just use a simple approach:
        the center that minimizes the maximum projection is found by bisection or LP.
        
        For simplicity, let me use the fact that for a set of points, the minimum enclosing
        regular hexagon (with fixed orientation) can be found by:
        1. For each of the 6 directions, find the min and max projection
        2. The width in direction n_i is max_proj - min_proj
        3. The required inradius is max over i of (width_i) / 2 + h_unit
        4. The center is at the midpoint of the projections
        
        Wait, that's for an axis-aligned bounding box. For a hexagon, it's different because
        the constraints are coupled.
        
        Let me just use scipy.optimize to find the center c that minimizes the maximum
        violation.
        """
        from scipy.optimize import minimize_scalar, minimize
        
        positions = np.array(positions)
        
        # 6 normal directions for pointy-top hexagon
        angles = [0, 60, 120, 180, 240, 300]
        normals = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in angles])
        
        # h_unit for these directions
        h_unit = np.sqrt(3) / 2
        
        def objective(c):
            c = np.array(c)
            # Compute dot(p - c, n_i) for all p, i
            diffs = positions - c  # shape (N, 2)
            dots = diffs @ normals.T  # shape (N, 6)
            max_dot = np.max(dots)
            return max_dot
        
        # Initial guess: centroid
        c0 = np.mean(positions, axis=0)
        
        result = minimize(objective, c0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        c_opt = result.x
        max_dot = objective(c_opt)
        
        # s * sqrt(3)/2 = max_dot + h_unit
        # s = (max_dot + h_unit) * 2 / sqrt(3)
        s = (max_dot + h_unit) * 2 / np.sqrt(3)
        
        return s, c_opt
    
    # Evaluate each arrangement
    for arr in arrangements:
        if time.time() > deadline:
            break
        try:
            side, center = compute_min_enclosing_hexagon(arr)
            if side < best_side:
                best_side = side
                # Convert to the required format
                inner = []
                for pos in arr:
                    inner.append([float(pos[0]), float(pos[1]), 0.0])
                best_result = (inner, (float(center[0]), float(center[1])), float(side), 0.0)
        except Exception:
            continue
    
    # If no valid result, return a fallback
    if best_result is None:
        # Fallback: simple row arrangement
        positions = row_arrangement([3, 4, 4])
        inner = [[float(p[0]), float(p[1]), 0.0] for p in positions]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END

