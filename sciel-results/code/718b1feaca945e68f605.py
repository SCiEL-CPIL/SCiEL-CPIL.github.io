import math
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Returns vertices of a regular hexagon."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def sat_overlap(v1, v2):
    """
    Separating Axis Theorem for convex polygons.
    Returns True if polygons overlap.
    v1, v2 are lists of (x, y) tuples.
    """
    n1 = len(v1)
    n2 = len(v2)
    
    # Axes to check
    axes = []
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        # Normal
        norm = (-edge[1], edge[0])
        axes.append(norm)
        
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i + 1) % n2]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        # Normal
        norm = (-edge[1], edge[0])
        axes.append(norm)
        
    for ax in axes:
        # Project all vertices onto the axis
        proj1 = [p[0]*ax[0] + p[1]*ax[1] for p in v1]
        proj2 = [p[0]*ax[0] + p[1]*ax[1] for p in v2]
        
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def point_in_polygon(point, polygon):
    """
    Ray casting algorithm to check if a point is inside a polygon.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y
        
    return inside

def is_valid_config(inner_hexes, outer_center, outer_side, outer_angle):
    """
    Checks if the configuration is valid:
    1. Inner hexagons do not overlap each other.
    2. Inner hexagons are inside the outer hexagon.
    """
    # Precompute inner hexagon vertices
    inner_verts = []
    for x, y, ang in inner_hexes:
        verts = hexagon_vertices(x, y, 1.0, ang)
        inner_verts.append(verts)
        
    # Check inner-inner overlaps
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if sat_overlap(inner_verts[i], inner_verts[j]):
                return False
                
    # Precompute outer hexagon vertices
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Check inner-outer containment
    # For each inner hexagon, all its vertices must be inside the outer hexagon
    for verts in inner_verts:
        for v in verts:
            if not point_in_polygon(v, outer_verts):
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = 120.0 - 2.0  # Safety margin
    
    # Generate hexagonal lattice positions
    # Axial coordinates (q, r)
    # Distance in units of side length for hexagons with side 1:
    # d = sqrt(3) * sqrt(q^2 + q*r + r^2)
    # We want to pack as many as possible within a reasonable bound, 
    # but the task says "n unit hexagons" and the seed used N=11 in a line.
    # The prompt says "Place disjoint regular hexagons ... minimizing the outer side length".
    # It doesn't explicitly fix N, but the seed implies N is a parameter.
    # Usually in these tasks, N is fixed by the seed's definition.
    # The seed: N = 11.
    # So we must place 11 unit hexagons.
    
    # Let's try to pack 11 hexagons in a compact cluster.
    # A central hexagon surrounded by 6, and then 4 more?
    # Or just a hexagonal number pattern?
    # 1, 7, 19...
    # 11 is between 7 and 19.
    
    # Let's generate a set of candidate lattice positions and pick the 11 most central ones.
    # Or simply define a specific compact layout for 11 hexagons.
    # Layout: 1 center, 6 ring 1, 4 from ring 2.
    
    # Positions in axial coordinates (q, r):
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 candidates: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # Let's pick 4 from Ring 2 that are close to the center.
    # Actually, any 4 from ring 2 will result in a similar bounding box if symmetric, 
    # but asymmetry might allow a smaller bounding hexagon?
    # No, the outer hexagon is regular and can be rotated.
    # The optimal packing of 11 hexagons into a minimal regular hexagon is a known problem.
    
    # Let's use a simple dense packing:
    # Row 1: 2
    # Row 2: 3
    # Row 3: 3
    # Row 4: 3
    # Total: 11
    
    # Coordinates for this arrangement (side length 1, distance between centers = sqrt(3)):
    # Let's use cartesian coordinates directly for simplicity in the lattice.
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    # Let's generate all hex lattice points within a radius and sort by distance to origin.
    # Then take the first 11.
    
    points = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            x = 1.5 * q
            y = math.sqrt(3) * (r + q / 2.0)
            dist_sq = x*x + y*y
            points.append((dist_sq, q, r, x, y))
            
    points.sort(key=lambda p: p[0])
    
    # Take the first 11 points
    selected = points[:11]
    
    # Convert to cartesian
    inner_hexes = []
    for _, _, _, x, y in selected:
        inner_hexes.append([x, y, 0.0])
        
    # Now, find the minimal outer side length.
    # The outer hexagon is centered at (0,0) initially.
    # We can rotate the outer hexagon to minimize its side length for this fixed cluster.
    # Or we can translate the cluster?
    # The interface allows returning inner positions and outer center.
    # To minimize the outer side length, we should center the cluster optimally.
    # The "center" of the cluster in terms of minimal enclosing hexagon might not be (0,0).
    # However, for a symmetric cluster, (0,0) is likely optimal.
    # The selected 11 points from the hex lattice centered at 0 are symmetric enough.
    
    # Let's try to optimize the outer side length by rotating the outer hexagon.
    # The width of the cluster in direction theta is W(theta).
    # The side length S of a regular hexagon with width W in direction theta (aligned with normal) is:
    # If theta is aligned with a flat side normal, S = W / sqrt(3).
    # If theta is aligned with a vertex normal, S = W / (3/2) = 2W/3.
    # Actually, the width of a regular hexagon of side S in any direction is between S*sqrt(3) and S*3/2?
    # No.
    # Width across flats = S * sqrt(3).
    # Width across corners = S * 2.
    # The width function is periodic.
    
    # A simpler approach:
    # Calculate the maximum distance from the origin to any vertex of the inner hexagons?
    # No, the inner hexagons have size.
    
    # Let's just binary search on the outer side length S.
    # For a given S, check if there exists a rotation angle alpha such that the cluster fits.
    
    # Since the cluster is fixed (relative to origin), we can just check if the cluster fits in a hexagon of side S centered at origin, rotated by alpha.
    
    # The cluster is fixed. Let's find the minimal S.
    # We can sample rotations of the outer hexagon.
    
    best_side = float('inf')
    
    # The cluster is roughly circular. The minimal enclosing hexagon will likely have a vertex pointing towards the "widest" part of the cluster.
    # Let's sample 360 angles.
    
    # To speed up, we can just calculate the required S for a given rotation.
    # For a fixed rotation of the outer hexagon, the constraint is:
    # For every point p in the cluster (actually, every point in the union of inner hexagons),
    # the projection of p onto the 6 normals of the outer hexagon must be <= S * sqrt(3) / 2.
    # Wait, the distance from center to a flat side of a regular hexagon of side S is apothem = S * sqrt(3) / 2.
    # So if we define the 6 normals of the outer hexagon, the max projection of any point in the cluster onto any normal must be <= apothem.
    # So S >= max_projection / (sqrt(3)/2).
    
    # We need to minimize S over the rotation of the outer hexagon.
    
    # The cluster is a set of points (vertices of inner hexagons).
    # Let P be the set of all vertices of all inner hexagons.
    
    cluster_points = []
    for x, y, ang in inner_hexes:
        verts = hexagon_vertices(x, y, 1.0, ang)
        cluster_points.extend(verts)
        
    # Function to calculate minimal S for a given outer rotation
    def calc_min_side(alpha_deg):
        alpha_rad = math.radians(alpha_deg)
        # Normals of the outer hexagon
        # The sides are at angles alpha + k * 60
        # The normals are at angles alpha + k * 60 + 90?
        # Let's define the hexagon such that one flat side is horizontal at the top.
        # Vertices at 30, 90, 150, 210, 270, 330? No.
        # Standard hexagon: vertices at 0, 60, 120, 180, 240, 300.
        # Normals to sides are at 30, 90, 150, 210, 270, 330.
        # So if we rotate the hexagon by alpha, the normals are at alpha + 30 + k * 60.
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(alpha_deg + 30 + k * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            max_p = 0.0
            for px, py in cluster_points:
                proj = px * nx + py * ny
                if proj > max_p:
                    max_p = proj
            if max_p > max_proj:
                max_proj = max_p
                
        # Apothem = S * sqrt(3) / 2
        # max_proj <= Apothem
        # S >= max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3.0)
        
    # Sample rotations
    # The cluster has some symmetry, but let's sample finely.
    # 360 samples might be enough, or we can refine.
    
    best_side = float('inf')
    best_alpha = 0.0
    
    # We can limit the search to 0..60 degrees due to hexagon symmetry?
    # The cluster is not perfectly symmetric, but close.
    # Let's search 0 to 360 in steps of 1 degree first.
    
    for alpha in range(0, 360, 1):
        s = calc_min_side(alpha)
        if s < best_side:
            best_side = s
            best_alpha = alpha
            
    # Refine around best_alpha
    # Step 0.1
    for alpha in [best_alpha - 1, best_alpha - 0.5, best_alpha - 0.1, best_alpha, best_alpha + 0.1, best_alpha + 0.5, best_alpha + 1]:
        if 0 <= alpha < 360:
            s = calc_min_side(alpha)
            if s < best_side:
                best_side = s
                best_alpha = alpha
                
    # We found a candidate side length.
    # Let's verify it with the full SAT check to be sure.
    # The calc_min_side assumes the cluster is centered at (0,0) and the outer hexagon is centered at (0,0).
    # Is (0,0) the optimal center for the outer hexagon?
    # For a convex cluster, the minimal enclosing regular hexagon might be shifted.
    # However, for a roughly symmetric cluster, the center of mass is a good guess.
    # The selected 11 points from the hex lattice are centered at (0,0).
    # Let's assume center (0,0) is optimal for now.
    
    # Verify
    if is_valid_config(inner_hexes, (0.0, 0.0), best_side, best_alpha):
        return inner_hexes, (0.0, 0.0), best_side, best_alpha
    else:
        # If invalid, increase side length slightly
        # This can happen due to floating point or if the center assumption is wrong.
        # Let's try a small increase.
        side = best_side
        for i in range(10):
            side += 0.01
            if is_valid_config(inner_hexes, (0.0, 0.0), side, best_alpha):
                return inner_hexes, (0.0, 0.0), side, best_alpha
        # Fallback
        return inner_hexes, (0.0, 0.0), 10.0, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to keep the interface.
# The seed had N=11 and a linear arrangement.
# This new strategy packs them densely.
# EVOLVE-BLOCK-END
