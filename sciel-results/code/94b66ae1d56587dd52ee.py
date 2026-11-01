"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N=11 unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Approach:
    - Use a hexagonal lattice arrangement for the inner hexagons.
    - For N=11, a good arrangement is a central hexagon surrounded by 6, plus 4 more
      placed in a compact cluster.
    - Alternatively, use a known optimal packing for small N.
    
    For N=11, a good configuration is to place them in a hexagonal pattern:
    - Center at (0,0)
    - Ring 1: 6 hexagons at distance 2 (touching) in hex directions
    - Ring 2: 4 additional hexagons placed to fill gaps compactly
    
    Actually, let's use a simpler approach: place 11 hexagons in a hexagonal lattice
    with minimal bounding hexagon.
    
    Hexagonal lattice spacing: for unit hexagons (side 1), the distance between centers
    of adjacent hexagons is 2 (since each hex has width 2 in the direction perpendicular
    to a flat side, and the inradius is sqrt(3)/2, so center-to-center for touching is 
    2*inradius = sqrt(3)... wait, let me recalculate.
    
    A regular hexagon with side length s has:
    - Width (flat-to-flat) = 2 * s * sin(60°) = s * sqrt(3)
    - Height (point-to-point) = 2 * s = 2
    
    For two unit hexagons to be disjoint (not overlapping), their centers must be at least
    the sum of their inradii apart in the direction perpendicular to the contact, but for
    hexagons on a hexagonal lattice, the minimum center-to-center distance for non-overlapping
    is 2 * inradius = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732? 
    
    Actually, for regular hexagons of side 1, the distance between centers of two touching
    hexagons (sharing a full edge) is 2 * inradius = sqrt(3). But if they just touch at a
    vertex or partially, the minimum distance can be less. To be safe, let's use center-to-
    center distance of at least sqrt(3) for edge-adjacent, but for a general packing, we need
    to ensure no overlap.
    
    Let me use a hexagonal lattice with spacing d = sqrt(3) + epsilon for safety, but actually
    for exact packing, d = sqrt(3) works for edge-to-edge contact.
    
    Hexagonal lattice basis vectors:
    v1 = (sqrt(3), 0)
    v2 = (sqrt(3)/2, 3/2)
    
    So positions are: a*v1 + b*v2 for integers a, b.
    
    For N=11, let's pick 11 lattice points that form a compact cluster.
    
    The hexagonal numbers are 1, 7, 19, ... so 11 is between 7 and 19.
    A compact arrangement: center + 6 in ring 1 + 4 in ring 2.
    
    Ring 1 (distance 1 in hex lattice): 6 points
    Ring 2 (distance 2 in hex lattice): 12 points, but we only need 4.
    
    Let's place:
    - (0,0)
    - 6 points at distance 1: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) in hex coords
    - 4 points at distance 2: choose 4 from the 12 ring-2 points to minimize bounding box.
    
    Hex coordinates to Cartesian:
    x = sqrt(3) * (a + b/2)
    y = 3/2 * b
    
    Let me compute the 11 points and find the minimal enclosing regular hexagon.
    
    For a regular hexagon centered at (cx, cy) with side length S and rotation theta,
    the distance from center to any point (px, py) in the hexagon's local frame must be
    <= S (the circumradius is S for a regular hexagon... wait, no. The circumradius of a
    regular hexagon with side S is S. The inradius is S*sqrt(3)/2.)
    
    So a point (px,py) is inside a regular hexagon centered at (cx,cy) with side S and
    rotation theta if, after rotating the point by -theta around (cx,cy), the point
    satisfies the hexagon inequality.
    
    For a regular hexagon with flat top (rotation 0), centered at origin, side S:
    The vertices are at angles 0, 60, 120, 180, 240, 300 degrees at distance S.
    The hexagon is the intersection of 6 half-planes.
    
    For a point (x,y) to be inside:
    |x| <= S
    |y| <= S*sqrt(3)/2
    |sqrt(3)*x + y| <= 2*S*sqrt(3)/2 = S*sqrt(3) ... let me just use the support function.
    
    Actually, the distance from origin to the hexagon boundary in direction (cos phi, sin phi)
    is: r(phi) = S * cos(phi mod 60° - 30°) / cos(30°) ... this is getting complex.
    
    Let me just use a simpler approach: for a given set of inner hexagon centers, find the
    minimal side length S of a regular hexagon (with some rotation) that contains all of them,
    where "contains" means the inner hexagons (of side 1) fit inside.
    
    This is equivalent to: the Minkowski sum of the inner hexagons must fit inside the outer
    hexagon. The Minkowski sum of a unit hexagon with a point is the hexagon itself. So we
    need the outer hexagon to contain the union of all inner hexagons.
    
    The distance from the outer hexagon center to the farthest point of any inner hexagon
    must be <= the inradius of the outer hexagon in that direction.
    
    For a regular hexagon with side S, the inradius in direction phi is:
    r_in(phi) = S * sqrt(3)/2 * ... 
    
    Let me just use a numerical optimization approach.
    """
    # Define hexagonal lattice points for N=11
    # Using axial coordinates (q, r)
    hex_points = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2),
    ]
    
    # Convert to Cartesian (flat-top hexagon orientation)
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in hex_points:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Find the center of mass
    center_x = np.mean(centers[:, 0])
    center_y = np.mean(centers[:, 1])
    
    # Try different rotations of the outer hexagon and find minimal S
    best_S = float('inf')
    best_rotation = 0.0
    
    # The inner hexagons are axis-aligned (flat-top), so we need to check if they fit.
    # For a given outer hexagon rotation theta and side S, centered at (cx, cy),
    # each inner hexagon (side 1, flat-top, at center (px, py)) must be fully inside.
    
    # The inner hexagon with side 1 has vertices at:
    # (cos(k*60°), sin(k*60°)) for k=0..5, scaled by 1 (circumradius = side = 1)
    inner_vertices = np.array([[np.cos(np.radians(k * 60)), np.sin(np.radians(k * 60))] for k in range(6)])
    
    # For each inner hexagon, the farthest point from the outer center in any direction
    # must be within the outer hexagon.
    
    # Simplification: check that all 6 vertices of each inner hexagon are inside the outer hexagon.
    
    def is_point_in_hex(px, py, cx, cy, S, theta):
        """Check if point (px, py) is inside regular hexagon centered at (cx, cy) with side S and rotation theta."""
        # Transform to local frame
        dx = px - cx
        dy = py - cy
        cos_t = np.cos(np.radians(theta))
        sin_t = np.sin(np.radians(theta))
        lx = dx * cos_t + dy * sin_t
        ly = -dx * sin_t + dy * cos_t
        
        # For a regular hexagon with flat top, side S, centered at origin:
        # Vertices at (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
        # The hexagon is defined by:
        # |lx| <= S
        # |ly| <= S * sqrt(3) / 2
        # |sqrt(3)*lx + ly| <= 2 * S * sqrt(3) / 2 = S * sqrt(3)
        # |sqrt(3)*lx - ly| <= S * sqrt(3)
        
        s3 = np.sqrt(3)
        if abs(lx) > S + 1e-9:
            return False
        if abs(ly) > S * s3 / 2 + 1e-9:
            return False
        if abs(s3 * lx + ly) > S * s3 + 1e-9:
            return False
        if abs(s3 * lx - ly) > S * s3 + 1e-9:
            return False
        return True
    
    def min_S_for_rotation(theta, cx, cy):
        """Find minimal S for given rotation and center."""
        # Binary search on S
        lo, hi = 0.0, 50.0
        for _ in range(50):
            mid = (lo + hi) / 2
            ok = True
            for px, py in centers:
                for vx, vy in inner_vertices:
                    if not is_point_in_hex(px + vx, py + vy, cx, cy, mid, theta):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                hi = mid
            else:
                lo = mid
        return hi
    
    # Try a few rotations
    for theta_deg in range(0, 360, 15):
        S = min_S_for_rotation(theta_deg, center_x, center_y)
        if S < best_S:
            best_S = S
            best_rotation = theta_deg
    
    # Refine around best rotation
    for theta_deg in np.arange(best_rotation - 15, best_rotation + 15, 1):
        S = min_S_for_rotation(theta_deg, center_x, center_y)
        if S < best_S:
            best_S = S
            best_rotation = theta_deg
    
    # Build inner list
    inner = []
    for px, py in centers:
        inner.append([float(px), float(py), 0.0])
    
    return inner, (float(center_x), float(center_y)), float(best_S), float(best_rotation)
# EVOLVE-BLOCK-END
