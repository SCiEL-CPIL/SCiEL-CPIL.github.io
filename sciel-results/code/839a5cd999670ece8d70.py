import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _hex_corners(cx, cy, side, angle_deg):
    """Return 6 corners of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        # rotate
        rx = ca * x - sa * y
        ry = sa * x + ca * y
        pts.append((cx + rx, cy + ry))
    return pts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # transform point to hex local coords
    dx, dy = px - cx, py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # regular hexagon with flat top: vertices at angles 0,60,...300
    # In local coords, hexagon is defined by |x| <= side, |y| <= side*sqrt(3)/2, |x + y/sqrt(3)| <= side etc.
    # Actually for a regular hexagon with side s, the distance from center to edge (apothem) is s*sqrt(3)/2
    # The hexagon can be described as intersection of 6 half-planes
    # For vertex at angle 0: the edge opposite is at x = -s, normal (1,0), so x >= -s -> x + s >= 0
    # Let's use the standard: for a regular hexagon with "pointy" vertices at angles 0,60,...
    # The supporting lines are at distance apothem = s*sqrt(3)/2 from center, with normals at angles 30, 90, 150, 210, 270, 330
    # Actually let me just check using the corner-based approach or half-plane
    
    # For a regular hexagon with side s, vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The edges are between consecutive vertices. The inward normals point toward center.
    # Edge between vertex at angle 0 and 60: midpoint at angle 30, distance from center to edge = s * sqrt(3)/2
    # Normal direction: 30 degrees (outward)
    # So constraint: dot((lx,ly), (cos(30), sin(30))) <= s*sqrt(3)/2
    # Similarly for other edges at normals 90, 150, 210, 270, 330
    
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx, ny = math.cos(theta), math.sin(theta)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def _hexagon_overlap(c1x, c1y, s1, a1, c2x, c2y, s2, a2):
    """Check if two hexagons overlap using separating axis theorem."""
    # Get all corners
    corners1 = _hex_corners(c1x, c1y, s1, a1)
    corners2 = _hex_corners(c2x, c2y, s2, a2)
    
    # All edge normals from both hexagons
    normals = []
    for i in range(6):
        # Edge from corner i to corner i+1
        x1, y1 = corners1[i]
        x2, y2 = corners1[(i + 1) % 6]
        dx, dy = x2 - x1, y2 - y1
        # Normal
        nx, ny = -dy, dx
        norm = math.sqrt(nx * nx + ny * ny)
        if norm > 1e-10:
            normals.append((nx / norm, ny / norm))
    
    for i in range(6):
        x1, y1 = corners2[i]
        x2, y2 = corners2[(i + 1) % 6]
        dx, dy = x2 - x1, y2 - y1
        nx, ny = -dy, dx
        norm = math.sqrt(nx * nx + ny * ny)
        if norm > 1e-10:
            normals.append((nx / norm, ny / norm))
    
    for nx, ny in normals:
        # Project all corners onto this normal
        min1, max1 = float('inf'), float('-inf')
        for cx, cy in corners1:
            proj = cx * nx + cy * ny
            min1 = min(min1, proj)
            max1 = max(max1, proj)
        
        min2, max2 = float('inf'), float('-inf')
        for cx, cy in corners2:
            proj = cx * nx + cy * ny
            min2 = min(min2, proj)
            max2 = max(max2, proj)
        
        if max1 < min2 or max2 < min1:
            return False
    
    return True

def _all_disjoint(inner, outer_center, outer_side, outer_angle):
    """Check all inner hexagons are disjoint from each other and inside outer hexagon."""
    n = len(inner)
    ocx, ocy = outer_center
    # Check inner-inner disjointness
    for i in range(n):
        x1, y1, a1 = inner[i]
        for j in range(i + 1, n):
            x2, y2, a2 = inner[j]
            if _hexagon_overlap(x1, y1, 1.0, a1, x2, y2, 1.0, a2):
                return False
    # Check inner inside outer
    for i in range(n):
        x, y, a = inner[i]
        corners = _hex_corners(x, y, 1.0, a)
        for cx, cy in corners:
            if not _point_in_hex(cx, cy, ocx, ocy, outer_side, outer_angle):
                return False
    return True

def _min_outer_side(inner, outer_angle=0.0):
    """Find minimum outer side length such that all inner hexagons fit inside a regular hexagon with given rotation."""
    # For a given set of inner hexagons and outer rotation, we need to find the minimum
    # outer hexagon (center + side) that contains all inner hexagons.
    # This is equivalent to finding the minimum enclosing regular hexagon.
    # For a fixed rotation, we can binary search on side length and check if there exists
    # a center such that all inner hexagons fit.
    
    # Alternative: for each inner hexagon corner, the outer hexagon must contain it.
    # The outer hexagon with center (cx,cy), side s, rotation a contains point (px,py) iff
    # for all 6 edge normals n_k: dot((px-cx, py-cy), n_k) <= s * sqrt(3)/2
    # So: cx * n_kx + cy * n_ky >= px * n_kx + py * n_ky - s * sqrt(3)/2
    # And: -cx * n_kx - cy * n_ky >= -px * n_kx - py * n_ky - s * sqrt(3)/2
    # i.e., |dot((px-cx, py-cy), n_k)| <= s * sqrt(3)/2 for all k
    
    # For fixed s, the feasible region for (cx, cy) is intersection of 6 strips (one per edge direction)
    # For each edge direction k, we need:
    #   dot((px - cx, py - cy), n_k) <= s*sqrt(3)/2 for all inner corners (px,py)
    #   => dot((px, py), n_k) - dot((cx,cy), n_k) <= s*sqrt(3)/2
    #   => dot((cx,cy), n_k) >= dot((px,py), n_k) - s*sqrt(3)/2 for all corners
    #   => dot((cx,cy), n_k) >= max_corner dot((px,py), n_k) - s*sqrt(3)/2
    # Similarly:
    #   dot((cx,cy), n_k) <= min_corner dot((px,py), n_k) + s*sqrt(3)/2
    
    # So for each k: L_k(s) <= dot((cx,cy), n_k) <= U_k(s)
    # where L_k(s) = max_p dot(p, n_k) - s*sqrt(3)/2
    # and U_k(s) = min_p dot(p, n_k) + s*sqrt(3)/2
    
    # The feasible region is a hexagon (intersection of 6 strips). We need it to be non-empty.
    # The minimum s is when the feasible region just becomes non-empty.
    
    # Let's collect all corners of all inner hexagons
    all_corners = []
    for x, y, a in inner:
        corners = _hex_corners(x, y, 1.0, a)
        all_corners.extend(corners)
    
    if not all_corners:
        return 0.0
    
    # For the given outer angle, compute the 6 normal directions
    a_rad = math.radians(outer_angle)
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        # Rotate the normal by outer_angle
        # Actually, the normals are fixed relative to the outer hexagon's orientation
        nx = math.cos(theta)
        ny = math.sin(theta)
        normals.append((nx, ny))
    
    # For each normal, compute max and min projection of all corners
    max_proj = []
    min_proj = []
    for nx, ny in normals:
        projs = [px * nx + py * ny for px, py in all_corners]
        max_proj.append(max(projs))
        min_proj.append(min(projs))
    
    # Binary search on s
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        apothem = mid * math.sqrt(3) / 2
        # Check if there exists (cx, cy) satisfying all constraints
        # This is a 2D linear feasibility problem with 12 inequalities
        # We can use a simple approach: the feasible region is a convex polygon
        # Let's check if the intersection of the 6 strips is non-empty
        # Using LP or just checking if the center of the bounding box works... 
        # Better: use the fact that it's a 2D LP with 12 constraints
        # Simple approach: check if the point that maximizes the minimum slack exists
        # Or just: the feasible region is non-empty iff the 6 strips have non-empty intersection
        # We can check this by solving a small LP or by checking all pairs of constraints
        
        # For simplicity, let's use a grid-based check or a proper LP
        # Actually, let's just check if the "center" candidate works, and refine
        # Better: solve the 2D LP exactly using the dual
        
        # The constraints are:
        # For each k: L_k <= nx_k * cx + ny_k * cy <= U_k
        # where L_k = max_proj[k] - apothem, U_k = min_proj[k] + apothem
        
        # This is feasible iff the 6 strips intersect.
        # A necessary and sufficient condition can be checked by solving the LP.
        # Let's use a simple iterative method or just check a few candidate points.
        
        # Candidate 1: average of all corners
        avg_x = sum(px for px, py in all_corners) / len(all_corners)
        avg_y = sum(py for px, py in all_corners) / len(all_corners)
        
        feasible = True
        for k in range(6):
            nx, ny = normals[k]
            val = avg_x * nx + avg_y * ny
            if val < max_proj[k] - apothem - 1e-9 or val > min_proj[k] + apothem + 1e-9:
                # Try another candidate: the point that equalizes max and min
                feasible = False
                break
        
        if feasible:
            hi = mid
        else:
            # Try to find a better center
            # The optimal center for minimizing max projection is related to the Chebyshev center
            # Let's just do a coarse grid search
            found = False
            for _ in range(10):
                # Try shifting
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        tx = avg_x + dx * 0.1
                        ty = avg_y + dy * 0.1
                        ok = True
                        for k in range(6):
                            nx, ny = normals[k]
                            val = tx * nx + ty * ny
                            if val < max_proj[k] - apothem - 1e-9 or val > min_proj[k] + apothem + 1e-9:
                                ok = False
                                break
                        if ok:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                hi = mid
            else:
                lo = mid
    
    return hi

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Start with a reasonable configuration
    # Hexagonal packing: inner hexagons in a hexagonal lattice
    # For N hexagons, arrange in a hexagonal pattern
    
    # Generate candidate configurations
    candidates = []
    
    # Configuration 1: Linear arrangement
    N = 11
    inner1 = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    candidates.append(inner1)
    
    # Configuration 2: Hexagonal packing
    # Center at (0,0), then rings around it
    def hex_packing(n):
        """Generate n hexagons in a hexagonal lattice pattern."""
        pts = [(0, 0)]
        if n == 1:
            return pts
        ring = 1
        while len(pts) < n:
            # Ring r has 6*r hexagons
            for i in range(6 * ring):
                if len(pts) >= n:
                    break
                # Position on ring
                # The ring consists of 6 sides, each of length r
                side = i // ring  # which side (0-5)
                pos = i % ring    # position on that side
                # Direction for each side
                angles = [0, 60, 120, 180, 240, 300]
                base_angle = math.radians(angles[side])
                # Offset along the side
                # The side is in the direction of the next vertex
                next_angle = math.radians(angles[(side + 1) % 6])
                # The position is: ring * (cos(base_angle), sin(base_angle)) + pos * (cos(next_angle) - cos(base_angle), sin(next_angle) - sin(base_angle)) / 1
                # Actually, simpler: the vertices of ring r are at distance r*sqrt(3) from center
                # Let me use a different approach
                
            # Simpler hexagonal packing using axial coordinates
            break
        return pts
    
    # Use axial coordinates for hexagonal grid
    def axial_to_pixel(q, r, size=2.0):
        """Convert axial coordinates to pixel coordinates for pointy-top hexagons."""
        x = size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
        y = size * (3 / 2 * r)
        return x, y
    
    # Generate hexagonal packing
    def hex_grid_positions(n):
        """Generate n positions in a hexagonal grid pattern (centered)."""
        positions = [(0, 0)]
        if n <= 1:
            return positions
        ring = 1
        while len(positions) < n:
            # Ring r has 6*r positions
            for i in range(6 * ring):
                if len(positions) >= n:
                    break
                # Determine which side and position
                side = i // ring
                pos = i % ring
                # The 6 directions for ring expansion
                # Side 0: q from ring to 0, r = 0 (going in +q direction)
                # Actually, let's use the standard hex ring generation
                # For ring r, the 6 corners are at:
                # (r, 0), (r, -r), (0, -r), (-r, 0), (-r, r), (0, r)
                # And we traverse the edges
                corners = [(r, 0), (r, -r), (0, -r), (-r, 0), (-r, r), (0, r)]
                # Side 'side' goes from corners[side] to corners[(side+1)%6]
                c1 = corners[side]
                c2 = corners[(side + 1) % 6]
                dq = (c2[0] - c1[0]) / ring
                dr = (c2[1] - c1[1]) / ring
                q = c1[0] + pos * dq
                rr = c1[1] + pos * dr
                positions.append((q, rr))
        return positions[:n]
    
    # Generate hexagonal packing with proper spacing
    # For unit hexagons in a hexagonal lattice, the distance between centers is 2*sqrt(3) for adjacent hexagons
    # Actually, for regular hexagons of side 1, the distance between centers of adjacent hexagons (sharing an edge) is 2 * apothem = 2 * sqrt(3)/2 * 2 = 2*sqrt(3)/2... 
    # Wait: apothem = sqrt(3)/2, so distance between centers sharing an edge = 2 * apothem = sqrt(3)
    # No: if two unit hexagons share an edge, the distance between their centers is 2 * (distance from center to edge) = 2 * sqrt(3)/2 = sqrt(3)
    # Hmm, let me recalculate. The apothem (center to edge midpoint) is s*sqrt(3)/2 = sqrt(3)/2.
    # If two hexagons share an edge, the distance between centers is 2 * sqrt(3)/2 = sqrt(3).
    # But for non-overlapping hexagons that are just touching, the center distance is sqrt(3).
    
    # Actually, for two regular hexagons of side 1 to be tangent (touching at an edge), 
    # the distance between centers is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    
    # Wait, that's not right either. Let me think again.
    # A regular hexagon with side s has apothem a = s * sqrt(3)/2.
    # Two hexagons sharing an edge: the shared edge is at distance a from each center, so center distance = 2a = s*sqrt(3).
    # For s=1: center distance = sqrt(3) ≈ 1.732.
    
    # So in a hexagonal lattice of touching unit hexagons, adjacent centers are sqrt(3) apart.
    # The lattice vectors are: (sqrt(3), 0) and (sqrt(3)/2, 3/2) ... wait.
    # For pointy-top hexagons, the neighbors are at:
    # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
    # These are at distance sqrt(3) from origin.
    
    hex_positions = hex_grid_positions(N)
    inner2 = []
    for q, r in hex_positions:
        x = math.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        inner2.append([x, y, 0.0])
    candidates.append(inner2)
    
    # Configuration 3: Slightly rotated hexagonal packing
    inner3 = []
    for q, r in hex_positions:
        x = math.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        inner3.append([x, y, 30.0])
    candidates.append(inner3)
    
    # Configuration 4: Compact cluster - 3x3 grid-like
    # For 11 hexagons, try a more compact arrangement
    # Center + 6 around + 4 more
    inner4 = []
    # Center
    inner4.append([0, 0, 0.0])
    # 6 neighbors
    for k in range(6):
        angle = math.radians(60 * k)
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        inner4.append([x, y, 0.0])
    # 4 more in the next ring
    for k in [0, 2, 4, 6]:
        if k == 6:
            k = 0
        angle = math.radians(60 * k)
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        inner4.append([x, y, 0.0])
    candidates.append(inner4)
    
    # Configuration 5: Hexagonal packing with 30-degree rotation of individual hexagons
    inner5 = []
    for q, r in hex_positions:
        x = math.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        inner5.append([x, y, 30.0])
    candidates.append(inner5)
    
    # Evaluate each candidate
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for inner in candidates:
        if time.time() > deadline:
            break
        
        # Try different outer rotations
        for outer_angle in [0.0, 30.0]:
            if time.time() > deadline:
                break
            side = _min_outer_side(inner, outer_angle)
            if side < best_side:
                # Find the center
                # Re-compute to get center
                all_corners = []
                for x, y, a in inner:
                    corners = _hex_corners(x, y, 1.0, a)
                    all_corners.extend(corners)
                
                a_rad = math.radians(outer_angle)
                normals = []
                for k in range(6):
                    theta = math.radians(30 + 60 * k)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    normals.append((nx, ny))
                
                max_proj = []
                min_proj = []
                for nx, ny in normals:
                    projs = [px * nx + py * ny for px, py in all_corners]
                    max_proj.append(max(projs))
                    min_proj.append(min(projs))
                
                apothem = side * math.sqrt(3) / 2
                # Find center: use average of constraints
                # The center should satisfy: max_proj[k] - apothem <= dot((cx,cy), n_k) <= min_proj[k] + apothem
                # Use the midpoint of the feasible region
                cx_sum = 0.0
                cy_sum = 0.0
                for k in range(6):
                    nx, ny = normals[k]
                    mid_val = (max_proj[k] + min_proj[k]) / 2
                    # This gives dot((cx,cy), n_k) = mid_val
                    # Solve for cx, cy using least squares or just average
                    cx_sum += mid_val * nx
                    cy_sum += mid_val * ny
                cx = cx_sum / 3.0  # Since sum of n_k * n_k over 6 directions = 3 * I
                cy = cy_sum / 3.0
                
                best_side = side
                best_inner = [list(x) for x in inner]
                best_center = (cx, cy)
                best_angle = outer_angle
    
    # If no valid configuration found, return a safe default
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Verify and return
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
