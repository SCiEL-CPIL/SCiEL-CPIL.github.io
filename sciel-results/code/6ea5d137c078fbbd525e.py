"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Generate candidate configurations of inner hexagon centers
    # Using hexagonal lattice arrangements
    candidates = []
    
    # Helper: hexagonal lattice coordinates to Cartesian
    # Using axial coordinates (q, r) with pointy-top hexagons
    # Center-to-center distance for touching unit hexagons is 2
    def axial_to_cartesian(q, r, size=2.0):
        x = size * (q + r/2.0)
        y = size * (3.0/2.0) * r
        return x, y
    
    # Generate several candidate arrangements for 11 hexagons
    
    # Arrangement 1: 7-hex cluster + 4 more extending in 4 directions
    # 7-hex cluster: center + 6 neighbors at distance 2
    base_7 = [(0, 0)]
    for k in range(6):
        angle = np.pi/3 * k
        base_7.append((2*np.cos(angle), 2*np.sin(angle)))
    
    # Add 4 more hexagons at distance 4 in 4 directions (extending the cluster)
    # The 6 directions from center are at angles 0, 60, 120, 180, 240, 300
    # Extending means placing at distance 4 in those directions
    # Try all combinations of 4 directions out of 6
    from itertools import combinations
    dirs = [np.pi/3 * k for k in range(6)]
    for combo in combinations(range(6), 4):
        positions = list(base_7)
        for d in combo:
            positions.append((4*np.cos(dirs[d]), 4*np.sin(dirs[d])))
        candidates.append(positions)
    
    # Arrangement 2: 2 rows of hexagons
    # Row 1: 6 hexagons, Row 2: 5 hexagons offset
    for offset in [0.0, 1.0, -1.0, 0.5, -0.5]:
        positions = []
        # Row 1: 6 hexagons at y=0
        for i in range(6):
            positions.append((2.0 * (i - 2.5), 0.0))
        # Row 2: 5 hexagons at y=sqrt(3) (for pointy-top, vertical spacing is 3/2 * size = 3)
        # Wait, for unit hexagons with center distance 2, vertical spacing for adjacent rows is sqrt(3) * 2 = 2*sqrt(3)? 
        # Let me reconsider. For pointy-top hexagons with side 1, the distance between centers of adjacent hexagons in the same row is 2.
        # The vertical distance between rows is 2 * (sqrt(3)/2) * ... actually for regular hexagons tiling:
        # Horizontal spacing: 2 (for pointy-top)
        # Vertical spacing: sqrt(3) * 2 / 2 = sqrt(3)? No.
        # For a regular hexagon with side s, the width (flat-to-flat) is s*sqrt(3), and the height (point-to-point) is 2s.
        # For pointy-top orientation: width = s*sqrt(3), height = 2s
        # Center-to-center horizontal: s*sqrt(3) for adjacent in row? No, for tiling, adjacent centers in same row are 2*s apart for flat-top, or...
        # Let me just use the axial coordinate system properly.
        # For pointy-top hexagons with side length 1:
        # Axial to cartesian: x = size * (q + r/2), y = size * (3/2) * r, where size = 2 (center-to-center distance)
        # So adjacent in q direction: (2, 0)
        # Adjacent in r direction: (1, 3/2 * 2) = (1, 3)
        # So vertical spacing between rows is 3 (for size=2)
        # Let me regenerate using axial coordinates properly
        
    # Let me redo with proper axial coordinates
    candidates = []
    
    # Generate hexagonal lattice points
    # For a hexagonal cluster, use cube coordinates or axial
    # Axial coordinates (q, r): neighbors are (q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)
    
    def get_hexagon_positions(max_radius, n_target):
        """Generate hexagonal lattice positions up to max_radius, return all subsets of size n_target"""
        positions = []
        for r in range(-max_radius, max_radius+1):
            for q in range(-max_radius, max_radius+1):
                # Check if within hexagonal distance
                if max(abs(q), abs(r), abs(-q-r)) <= max_radius:
                    x = 2.0 * (q + r/2.0)
                    y = 2.0 * (3.0/2.0) * r
                    positions.append((x, y))
        return positions
    
    # For 11 hexagons, try various arrangements
    # Arrangement A: hexagonal cluster of radius 1 (7 hexagons) + 4 more
    pos_7 = get_hexagon_positions(1, 7)
    # This gives 7 positions: center and 6 neighbors
    
    # Add 4 more from radius 2 positions
    pos_2_only = []
    for r in range(-2, 3):
        for q in range(-2, 3):
            if max(abs(q), abs(r), abs(-q-r)) == 2:
                x = 2.0 * (q + r/2.0)
                y = 2.0 * (3.0/2.0) * r
                pos_2_only.append((x, y))
    
    # Try all combinations of 4 from the 12 radius-2 positions
    for combo in combinations(range(len(pos_2_only)), 4):
        positions = pos_7 + [pos_2_only[i] for i in combo]
        candidates.append(positions)
    
    # Also try arrangements that are not purely hexagonal lattice
    # E.g., 2 rows: 6 and 5
    for y_offset in [0.0, 1.5, -1.5, 3.0, -3.0]:
        positions = []
        # 6 in a row
        for i in range(6):
            positions.append((2.0 * (i - 2.5), 0.0))
        # 5 in another row
        for i in range(5):
            positions.append((2.0 * (i - 2.0), y_offset))
        candidates.append(positions)
    
    # 3 rows: 4, 3, 4
    for y1 in [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
        for y2 in [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
            if abs(y1 - y2) < 0.1:
                continue
            positions = []
            # Row 1: 4 hexagons
            for i in range(4):
                positions.append((2.0 * (i - 1.5), 0.0))
            # Row 2: 3 hexagons
            for i in range(3):
                positions.append((2.0 * (i - 1.0), y1))
            # Row 3: 4 hexagons
            for i in range(4):
                positions.append((2.0 * (i - 1.5), y2))
            candidates.append(positions)
    
    # Deduplicate candidates (approximately)
    unique_candidates = []
    for cand in candidates:
        # Sort by position for comparison
        sorted_cand = sorted([(round(x, 6), round(y, 6)) for x, y in cand])
        is_dup = False
        for uc in unique_candidates:
            if sorted_cand == uc:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(sorted_cand)
            candidates = candidates  # Keep original for now
    
    # Evaluate each candidate
    best_side = float('inf')
    best_result = None
    
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # Get the vertex positions of all inner hexagons
        # Each inner hexagon is unit, axis-aligned (rotation 0)
        vertices = []
        for (xi, yi) in cand:
            for k in range(6):
                angle = np.pi/3 * k
                vx = xi + np.cos(angle)
                vy = yi + np.sin(angle)
                vertices.append((vx, vy))
        
        vertices = np.array(vertices)
        
        # Optimize outer hexagon: center (cx, cy), rotation theta, side s
        # For a given theta, find optimal (cx, cy, s)
        
        # The constraint is: for each edge direction k (0..5), and each vertex v:
        # (vx - cx) * cos(theta + pi/6 + k*pi/3) + (vy - cy) * sin(theta + pi/6 + k*pi/3) <= s * sqrt(3)/2
        
        # This is equivalent to: for each k, max over v of [vx*cos_k + vy*sin_k] - cx*cos_k - cy*sin_k <= s*sqrt(3)/2
        # So: s >= (max_v [vx*cos_k + vy*sin_k] - cx*cos_k - cy*sin_k) / (sqrt(3)/2) for all k
        # And we want to minimize s, so for given (cx, cy, theta):
        # s = max_k { [max_v (vx*cos_k + vy*sin_k) - cx*cos_k - cy*sin_k] / (sqrt(3)/2) }
        
        # To minimize s, we need to choose (cx, cy) optimally for each theta.
        # This is a convex optimization in (cx, cy, s).
        
        # Simplification: for a given theta, the problem is to find (cx, cy, s) minimizing s
        # subject to: for all k, cx*cos_k + cy*sin_k + s*sqrt(3)/2 >= max_v(vx*cos_k + vy*sin_k)
        
        # Let a_k = cos(theta + pi/6 + k*pi/3), b_k = sin(theta + pi/6 + k*pi/3)
        # Let M_k = max_v(vx*a_k + vy*b_k)
        # Constraints: cx*a_k + cy*b_k + s*sqrt(3)/2 >= M_k for all k
        # Minimize s
        
        # This is a linear program in (cx, cy, s). We can solve it.
        
        # For simplicity, let's try a grid search over theta and then optimize (cx, cy) analytically or with a simple method.
        
        # For a given theta, the optimal (cx, cy) can be found by noting that the constraints are linear.
        # Let's use a simple approach: try several theta values and for each, find the minimal bounding hexagon.
        
        theta_values = np.linspace(0, np.pi/3, 13)  # 0 to 60 degrees, 13 points
        
        for theta in theta_values:
            if time.time() > deadline:
                break
            
            # Compute edge normal directions
            a = np.array([np.cos(theta + np.pi/6 + k*np.pi/3) for k in range(6)])
            b = np.array([np.sin(theta + np.pi/6 + k*np.pi/3) for k in range(6)])
            
            # Compute M_k = max over vertices of (vx*a_k + vy*b_k)
            # vertices shape: (n_vertices, 2)
            # projections shape: (6, n_vertices)
            projections = vertices[:, 0] * a[:, None] + vertices[:, 1] * b[:, None]
            M = np.max(projections, axis=1)  # shape (6,)
            
            # Now find (cx, cy, s) minimizing s subject to:
            # cx*a_k + cy*b_k + s*sqrt(3)/2 >= M_k for all k
            
            # This is a small LP. Let's use a simple approach: 
            # The optimal s is the minimum such that the system is feasible.
            # By duality or direct reasoning, we can find it.
            
            # For a given s, the constraints are:
            # cx*a_k + cy*b_k >= M_k - s*sqrt(3)/2 for all k
            # This is feasible iff the point (cx, cy) is in the intersection of 6 half-planes.
            # The minimal s is when the hexagon defined by these half-planes has zero area (degenerate).
            
            # Alternatively, we can use the fact that the optimal (cx, cy) is the center of the minimal enclosing hexagon.
            
            # Let's use a different approach: the minimal enclosing regular hexagon (with fixed orientation) 
            # for a set of points can be found by binary search on s and checking feasibility.
            
            # For a given s, the constraints on (cx, cy) are:
            # cx*a_k + cy*b_k >= M_k - s*sqrt(3)/2 for all k=0..5
            
            # This defines a convex polygon (possibly empty). We need the smallest s such that it's non-empty.
            
            # The polygon is the intersection of 6 half-planes. It's non-empty iff the 6 lines form a bounded region.
            
            # Let's find the minimal s by binary search.
            
            # Lower bound: s >= 0
            # Upper bound: s = max(M) + 10 (generous)
            
            s_lo = 0.0
            s_hi = np.max(M) + 10.0
            
            # Binary search
            for _ in range(40):
                s_mid = (s_lo + s_hi) / 2.0
                rhs = M - s_mid * np.sqrt(3) / 2.0
                
                # Check if there exists (cx, cy) such that cx*a_k + cy*b_k >= rhs_k for all k
                # This is a 2D feasibility problem with 6 constraints.
                # We can solve it by finding the intersection of half-planes.
                
                # Use a simple method: the feasible region is a convex polygon.
                # Find its vertices by intersecting pairs of boundary lines.
                
                # Line k: a_k * cx + b_k * cy = rhs_k
                # Normal vector: (a_k, b_k)
                
                # Find all pairwise intersections of the 6 lines
                feasible = False
                for i in range(6):
                    for j in range(i+1, 6):
                        # Solve:
                        # a_i * cx + b_i * cy = rhs_i
                        # a_j * cx + b_j * cy = rhs_j
                        
                        det = a[i] * b[j] - a[j] * b[i]
                        if abs(det) < 1e-12:
                            continue
                        
                        cx_val = (rhs[i] * b[j] - rhs[j] * b[i]) / det
                        cy_val = (a[i] * rhs[j] - a[j] * rhs[i]) / det
                        
                        # Check if this point satisfies all 6 constraints
                        ok = True
                        for k in range(6):
                            if cx_val * a[k] + cy_val * b[k] < rhs[k] - 1e-9:
                                ok = False
                                break
                        if ok:
                            feasible = True
                            break
                    if feasible:
                        break
                
                # Also check if the origin or some obvious point works
                if not feasible:
                    # Try a few test points
                    test_points = [(0, 0), (np.mean(vertices[:,0]), np.mean(vertices[:,1]))]
                    for (tx, ty) in test_points:
                        ok = True
                        for k in range(6):
                            if tx * a[k] + ty * b[k] < rhs[k] - 1e-9:
                                ok = False
                                break
                        if ok:
                            feasible = True
                            break
                
                if feasible:
                    s_hi = s_mid
                else:
                    s_lo = s_mid
            
            s_val = s_hi
            
            if s_val < best_side:
                best_side = s_val
                # Find the center (cx, cy) for this s
                rhs = M - s_val * np.sqrt(3) / 2.0
                
                # Find a feasible (cx, cy)
                best_center = (0.0, 0.0)
                found = False
                for i in range(6):
                    for j in range(i+1, 6):
                        det = a[i] * b[j] - a[j] * b[i]
                        if abs(det) < 1e-12:
                            continue
                        cx_val = (rhs[i] * b[j] - rhs[j] * b[i]) / det
                        cy_val = (a[i] * rhs[j] - a[j] * rhs[i]) / det
                        ok = True
                        for k in range(6):
                            if cx_val * a[k] + cy_val * b[k] < rhs[k] - 1e-9:
                                ok = False
                                break
                        if ok:
                            best_center = (cx_val, cy_val)
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    best_center = (np.mean(vertices[:,0]), np.mean(vertices[:,1]))
                
                # Construct the result
                inner = [[x, y, 0.0] for x, y in cand]
                best_result = (inner, best_center, s_val, theta)
    
    if best_result is None:
        # Fallback: return something valid
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END

