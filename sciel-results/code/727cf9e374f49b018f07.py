"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # We place N unit regular hexagons inside a larger regular hexagon.
    # Each inner hexagon has side length 1.
    # The outer hexagon is regular with side length L, centered at (cx, cy), rotated by angle_degrees.
    # We want to minimize L.
    
    # Strategy: Arrange the N unit hexagons in a compact cluster.
    # For small N, we can try various arrangements.
    # A unit regular hexagon has "width" (distance between parallel sides) = sqrt(3) and "height" (distance between opposite vertices) = 2.
    # When placed in a hexagonal lattice, the centers are spaced by sqrt(3) in the three lattice directions.
    
    # Let's try to place them in a hexagonal packing pattern.
    # The most compact arrangement for N points in a hexagonal lattice is a hexagonal number pattern.
    # For N=11, we can use a central hexagon surrounded by 6, and then 4 more.
    # Actually, let's think about this more carefully.
    
    # A unit hexagon's vertices (with center at origin, flat-top orientation, i.e., two sides horizontal):
    # The six vertices are at angles 0, 60, 120, 180, 240, 300 degrees from the center, at distance 1.
    # So vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    
    # The "width" (horizontal extent) is 2, and the "height" (vertical extent) is sqrt(3).
    
    # For a regular hexagon of side length L, the distance between opposite sides is L*sqrt(3), and the distance between opposite vertices is 2L.
    
    # If we place unit hexagons in a hexagonal lattice with all in the same orientation, the centers form a triangular lattice with spacing sqrt(3).
    # The minimal enclosing regular hexagon for such a cluster depends on the arrangement.
    
    # Let's try a simple approach: place the N hexagons in a hexagonal cluster, all with the same orientation (0 degrees).
    # Then compute the minimal enclosing regular hexagon (with the same orientation) that contains all of them.
    
    # For a set of points (the vertices of all inner hexagons), we want to find the smallest regular hexagon (with a given orientation) that contains all points.
    # For a regular hexagon with center (cx, cy), side length L, and orientation angle theta, the constraints for a point (x, y) to be inside are:
    # For each of the 6 sides, the point must be on the correct side of the line.
    
    # Actually, let's simplify: if we fix the orientation of the outer hexagon to be 0 degrees (flat-top), then the inner hexagons should also be 0 degrees for optimal packing in a hexagonal lattice.
    
    # Let's generate a hexagonal cluster of N unit hexagons.
    # The hexagonal lattice basis vectors for unit hexagons (flat-top):
    # The center-to-center distance between adjacent hexagons is sqrt(3) in the 3 lattice directions.
    # Lattice vectors: (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2) -- wait, let me recalculate.
    
    # For flat-top hexagons, the 6 neighbors are at:
    # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
    
    # Let's use axial coordinates. In axial coordinates (q, r), the position is:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # Let's generate N points in a hexagonal cluster using axial coordinates.
    # For N=11, we can use a cluster like:
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -- that's 7 (a hexagon of radius 1)
    # Plus (2,0), (1,1), (-1,2), (-2,1) -- 4 more, total 11.
    # Or we can use a more compact shape.
    
    # Let's just use a simple linear arrangement for now and then optimize.
    # Actually, let's try to be smarter. Let's use a hexagonal cluster.
    
    # Generate axial coordinates for a hexagonal cluster of size N.
    # We'll use a spiral or a simple expansion.
    
    def hex_cluster(N):
        # Start with center
        pts = [(0, 0)]
        if N <= 1:
            return pts
        # Add rings
        ring = 1
        while len(pts) < N:
            # Ring ring has 6*ring points
            # Starting from (ring, 0), go around
            q, r = ring, 0
            for i in range(6 * ring):
                if len(pts) >= N:
                    break
                pts.append((q, r))
                # Move to next point in the ring
                if i < ring:
                    q += 1
                elif i < 2 * ring:
                    r += 1
                elif i < 3 * ring:
                    q -= 1
                elif i < 4 * ring:
                    r -= 1
                elif i < 5 * ring:
                    q += 1
                else:
                    r -= 1
            ring += 1
        return pts[:N]
    
    axial_pts = hex_cluster(N)
    
    # Convert to Cartesian
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in axial_pts:
        x = sqrt3 * (q + r / 2)
        y = 1.5 * r
        centers.append((x, y))
    
    # Now, we need to find the minimal enclosing regular hexagon (flat-top, orientation 0) for all vertices of these hexagons.
    # Each inner hexagon has vertices at center + (cos(theta_k), sin(theta_k)) for k=0..5, where theta_k = k*60 degrees.
    
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = np.radians(60 * k)
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    # For a flat-top regular hexagon with center (cx, cy) and side length L, the six sides are:
    # The hexagon can be defined by the constraint that for each of the 6 directions (at angles 0, 60, 120, 180, 240, 300),
    # the projection of (point - center) onto the direction perpendicular to the side is <= L*sqrt(3)/2.
    # Actually, the distance from center to each side is L*sqrt(3)/2.
    # The 6 normal directions (outward) are at angles 30, 90, 150, 210, 270, 330 degrees (for flat-top).
    # Wait, for a flat-top hexagon, the sides are at angles 0, 60, 120, 180, 240, 300 from the horizontal? No.
    # Let me think again. A flat-top hexagon has two horizontal sides (top and bottom). The other 4 sides are at 60 and 120 degrees.
    # The outward normals to the sides are at angles 90, 30, -30, -90, -150, 150 degrees (i.e., 90, 30, 330, 270, 210, 150).
    
    # The constraint is: for each normal direction n_k (unit vector), (p - c) . n_k <= L * sqrt(3) / 2.
    
    # So L >= 2/sqrt(3) * max_k ((p - c) . n_k) for all p in all_vertices.
    # To minimize L, we can choose c. But if we fix c to be the centroid of the cluster, we get a feasible L.
    
    # Let's use the centroid of all vertices as the center.
    c = np.mean(all_vertices, axis=0)
    
    # Normal directions for flat-top hexagon
    normal_angles = [np.radians(a) for a in [90, 30, -30, -90, -150, 150]]
    normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
    
    # For each vertex, compute the max projection onto any normal
    # L_min = 2/sqrt(3) * max over all vertices and all normals of ((p - c) . n)
    
    max_proj = 0
    for p in all_vertices:
        diff = p - c
        for n in normals:
            proj = np.dot(diff, n)
            if proj > max_proj:
                max_proj = proj
    
    L = 2.0 / sqrt3 * max_proj
    
    # Convert centers to the required format
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
    
    return inner, (float(c[0]), float(c[1])), float(L), 0.0
# EVOLVE-BLOCK-END
