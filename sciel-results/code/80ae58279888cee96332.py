"""Hexagonal lattice packing of unit hexagons inside a regular hexagon."""
import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place unit regular hexagons on a hexagonal lattice inside a regular hexagon.
    
    Strategy:
    - Use a hexagonal lattice of points where unit hexagons (side 1) can be placed
      without overlapping.
    - The distance between centers of adjacent non-overlapping unit hexagons on a
      hex lattice is sqrt(3) (for pointy-top or flat-top orientation).
    - We try various sizes of hexagonal clusters (defined by radius r in lattice coords)
      and compute the minimal enclosing regular hexagon that contains all unit hexagons.
    - Return the configuration with the smallest outer side length.
    
    For unit hexagons on a hex lattice:
    - Lattice spacing: d = sqrt(3) (center-to-center for touching hexagons)
    - A hexagonal cluster of radius r has (3*r*(r+1)+1) hexagons
    - The cluster forms a larger hexagon shape
    
    The outer hexagon must contain all unit hexagons. For a hexagonal arrangement of
    unit hexagons with lattice spacing sqrt(3), the overall shape is a hexagon with
    side length proportional to (r+1)*sqrt(3) + 2 (accounting for the extent of each
    unit hexagon).
    
    More precisely, for a hexagonal cluster of radius r (number of hexagons from center 
    to edge in lattice steps), the convex hull of the centers is a hexagon with 
    "radius" (center to vertex) = r * sqrt(3). The unit hexagons extend by 1 (their 
    circumradius) in all directions, so the outer hexagon needs to have inradius at 
    least r*sqrt(3) + 1, meaning side length = (r*sqrt(3) + 1) / (sqrt(3)/2) = 
    2*(r*sqrt(3) + 1)/sqrt(3).
    
    Wait, let me reconsider. A regular hexagon with side length s has inradius 
    s*sqrt(3)/2 and circumradius s. 
    
    For the outer hexagon to contain all unit hexagons:
    - The farthest point from center is at distance r*sqrt(3) + 1 (lattice distance + 
      circumradius of unit hex)
    - So outer hexagon circumradius >= r*sqrt(3) + 1
    - Outer side length = circumradius = r*sqrt(3) + 1
    
    But we also need to check that the hexagons fit in terms of the flat sides.
    Actually, for a hexagonal arrangement, the tightest fit is when the outer hexagon 
    is aligned with the inner lattice.
    
    Let me just compute this numerically by placing hexagons and finding the minimal
    enclosing regular hexagon.
    """
    # Try different hexagonal cluster sizes
    best_config = None
    best_side = float('inf')
    
    # For N=11, we need 11 hexagons. A hexagonal cluster of radius r has:
    # r=0: 1, r=1: 7, r=2: 19
    # So radius 1 gives 7, radius 2 gives 19. We need 11, which is between.
    # Let's try subsets of the radius-2 cluster (19 hexagons) and pick 11 that minimize
    # the enclosing hexagon.
    
    # Generate hexagonal lattice points
    def hex_lattice_points(radius):
        """Generate all lattice points within hexagonal distance <= radius from origin."""
        points = []
        for q in range(-radius, radius + 1):
            for r_ in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
                points.append((q, r_))
        return points
    
    # Convert axial to pixel coordinates for pointy-top hexagons
    # For flat-top: x = size * 3/2 * q, y = size * sqrt(3) * (r + q/2)
    # For pointy-top: x = size * sqrt(3) * (q + r/2), y = size * 3/2 * r
    # 
    # For unit hexagons (side 1), the center-to-center distance for adjacent hexagons
    # in a hex packing is sqrt(3) * side = sqrt(3).
    # 
    # Let's use the standard hex grid: axial coordinates (q, r) map to:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # This gives pointy-top hexagons with side length 1.
    
    def axial_to_xy(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    # Try radius 1 (7 hexagons) - not enough for N=11
    # Try radius 2 (19 hexagons) - pick 11
    
    # For radius 2, generate all points
    lattice_pts = hex_lattice_points(2)
    lattice_xy = [axial_to_xy(q, r) for q, r in lattice_pts]
    
    # We need to pick N=11 points from these 19 that minimize the enclosing hexagon
    # For now, let's just use all points within a certain range and compute the 
    # enclosing hexagon
    
    # Compute the minimal enclosing regular hexagon for a set of points
    def min_enclosing_hexagon(points):
        """
        Find the minimum side length regular hexagon (with any rotation) that contains
        all given points.
        
        A regular hexagon with center c, side length s, and rotation angle theta
        contains a point p if the distance from c to p in each of the 6 normal 
        directions is <= s * sqrt(3) / 2 (the inradius).
        
        Actually, for a regular hexagon with side s:
        - Inradius (center to middle of side) = s * sqrt(3) / 2
        - Circumradius (center to vertex) = s
        
        The hexagon can be described as the intersection of 6 half-planes.
        For a hexagon centered at origin with rotation 0 (flat-top), the constraints are:
        For each of 6 directions at angles 0, 60, 120, 180, 240, 300 degrees:
        |dot(p, n_i)| <= inradius
        
        where n_i are unit vectors in those directions.
        
        To find the minimal enclosing hexagon, we can:
        1. Compute the centroid of the points
        2. For each rotation angle, find the minimal center and side length
        3. The optimal rotation is one of the directions aligned with point pairs
        
        This is complex. Let's use a simpler approach: for a hexagonal lattice of 
        hexagons, the optimal enclosing hexagon is aligned with the lattice.
        """
        pts = np.array(points)
        # Try different rotation angles
        best = float('inf')
        for angle_deg in range(0, 30, 1):  # 30-degree symmetry for hexagons
            angle = np.radians(angle_deg)
            # The 6 normal directions for a flat-top hexagon rotated by angle
            base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
            normals = np.array([np.cos(base_angles + angle), 
                                np.sin(base_angles + angle)]).T
            
            # For a given center c, the constraint is:
            # dot(p - c, n_i) <= inradius for all i
            # dot(c - p, n_i) <= inradius for all i
            # => inradius >= max_i max_p dot(p - c, n_i)
            # => inradius >= max_i (max_p dot(p, n_i) - dot(c, n_i))
            
            # To minimize inradius, we want to maximize min_i (inradius - max_p dot(p,n_i) + dot(c,n_i))
            # This is equivalent to finding the center c that minimizes:
            # max_i (max_p dot(p, n_i) - dot(c, n_i))
            
            # Let M_i = max_p dot(p, n_i)
            # We want to minimize max_i (M_i - dot(c, n_i))
            # This is a linear programming problem, but we can solve it heuristically.
            
            # The optimal c satisfies: for the active constraints, 
            # M_i - dot(c, n_i) = inradius for some set of i.
            
            # Simple approach: try c = centroid
            c = pts.mean(axis=0)
            dots = pts @ normals  # shape (n_points, 6)
            M = dots.max(axis=0)  # max over points for each direction
            inradius = np.max(M - c @ normals)
            
            # Also try adjusting c
            # The optimal c for minimizing max_i(M_i - c.n_i) is where the 
            # maximum is balanced. Let's do a few iterations.
            for _ in range(50):
                vals = M - c @ normals
                i_max = np.argmax(vals)
                # Move c in direction n_i_max to reduce vals[i_max]
                # But this might increase others. Use subgradient method.
                # Gradient of max_i(M_i - c.n_i) w.r.t. c is -n_i_max
                # So move c in direction n_i_max
                c = c + 0.01 * normals[i_max]
                vals = M - c @ normals
            
            inradius = np.max(M - c @ normals)
            side = inradius * 2 / np.sqrt(3)
            
            if side < best:
                best = side
                best_c = c
                best_angle = angle
        
        return best, best_c, best_angle
    
    # For a hexagonal cluster, the optimal enclosing hexagon is aligned with the lattice
    # Let's compute for the full radius-2 cluster and see
    all_xy = [axial_to_xy(q, r) for q, r in lattice_pts]
    
    # For N=11, we need to select 11 points. Let's try the center + all 6 neighbors 
    # + 4 more from the next ring.
    
    # Center: (0,0)
    # Ring 1: 6 points at axial distance 1
    # Ring 2: 12 points at axial distance 2
    
    # Let's just compute the enclosing hexagon for various subsets
    
    # Actually, for a hexagonal lattice packing, the tightest enclosing hexagon for 
    # a hexagonal cluster of radius r is simply a hexagon with side length:
    # s = (r + 1) * sqrt(3) + 2 * (1 - 1/sqrt(3)) ... this is getting complicated.
    
    # Let me just directly compute: for unit hexagons on a hex lattice with spacing 
    # sqrt(3), a cluster of radius r (axial distance) has centers at max distance 
    # r*sqrt(3) from center. The unit hexagons extend 1 unit from their centers.
    # So the farthest point is at distance r*sqrt(3) + 1 from the cluster center.
    # The enclosing regular hexagon (aligned with lattice) needs circumradius >= 
    # r*sqrt(3) + 1, so side = r*sqrt(3) + 1.
    
    # For r=1: side = sqrt(3) + 1 ≈ 2.732, contains 7 hexagons
    # For r=2: side = 2*sqrt(3) + 1 ≈ 4.464, contains 19 hexagons
    
    # We need 11 hexagons. Let's use a subset of the r=2 cluster.
    # The best 11-point subset would be the center, all 6 ring-1 points, and 4 ring-2 points.
    # To minimize the enclosing hexagon, we should pick ring-2 points that are closest
    # to the center in terms of the hexagonal norm.
    
    # All ring-2 points are at axial distance 2, so they're all equidistant.
    # Let's just pick any 4 of them. By symmetry, the enclosing hexagon will be the same.
    
    # Let's compute the actual enclosing hexagon for center + 6 ring-1 + 4 ring-2
    
    # Generate the specific points
    center = (0, 0)
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
             (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -2), (2, -1)]
    
    # Pick 4 from ring2 - let's pick (2,0), (0,2), (-2,2), (-2,0) for symmetry
    selected_axial = [center] + ring1 + [(2, 0), (0, 2), (-2, 2), (-2, 0)]
    selected_xy = [axial_to_xy(q, r) for q, r in selected_axial]
    
    # Compute enclosing hexagon
    side_len, c, ang = min_enclosing_hexagon(selected_xy)
    
    # Create inner list
    inner = []
    for i, (x, y) in enumerate(selected_xy):
        inner.append([x, y, 0.0])
    
    return inner, (c[0], c[1]), side_len, ang
# EVOLVE-BLOCK-END
