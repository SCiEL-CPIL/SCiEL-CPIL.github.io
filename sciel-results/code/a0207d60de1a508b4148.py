"""Improved program for packing 11 unit regular hexagons in a minimal enclosing regular hexagon."""
import numpy as np
from itertools import combinations, product
import math
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Construct placement of 11 unit regular hexagons inside a minimal enclosing regular hexagon.
    
    Strategy:
    1. Use the hexagonal lattice (dual to triangular) where unit hexagons tile the plane.
    2. Enumerate all 11-cell polyhexes (connected sets of 11 hexagons on the hex lattice).
    3. For each polyhex, compute the minimum enclosing regular hexagon side length.
    4. Return the best placement found.
    
    The hex lattice basis vectors for unit hexagons (side length 1):
    - Adjacent hex centers are distance sqrt(3) apart.
    - Basis: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    """
    
    # Hex lattice basis
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3 / 2.0, 1.5])
    
    def lattice_to_cartesian(i, j):
        """Convert hex lattice coordinates (i, j) to Cartesian (x, y)."""
        return i * a1 + j * a2
    
    def cartesian_to_lattice(x, y):
        """Convert Cartesian to hex lattice coordinates (approximate)."""
        # Solve [x, y] = i*a1 + j*a2
        # x = i*sqrt3 + j*sqrt3/2
        # y = j*1.5
        j = y / 1.5
        i = (x - j * sqrt3 / 2) / sqrt3
        return i, j
    
    def hex_neighbors(i, j):
        """Get the 6 neighbors of hex (i, j) in axial coordinates."""
        # In axial coords for hex lattice, neighbors are:
        # (i+1, j), (i-1, j), (i, j+1), (i, j-1), (i+1, j-1), (i-1, j+1)
        return [(i+1, j), (i-1, j), (i, j+1), (i, j-1), (i+1, j-1), (i-1, j+1)]
    
    def compute_enclosing_hex_side(points):
        """
        Compute the minimum side length of a regular hexagon (any rotation) 
        that encloses all given points.
        
        For a regular hexagon with side length s and center c, rotated by angle theta,
        a point p is inside if for all 6 directions, the projection is within bounds.
        
        A regular hexagon can be described as the intersection of 6 half-planes.
        For a hexagon centered at c with "radius" R (distance from center to vertex) 
        and rotation theta, the 6 normal directions are at angles theta + k*pi/3 for k=0..5.
        The half-plane constraint is: dot(p - c, n_k) <= R * cos(pi/6) = R * sqrt(3)/2
        where n_k is the unit normal in direction theta + k*pi/3.
        
        Actually, for a regular hexagon with side length s:
        - The distance from center to each side (apothem) is a = s * sqrt(3)/2
        - The distance from center to each vertex is R = s
        - The 6 outward normals are at angles theta + k*pi/3
        
        A point p is inside if dot(p-c, n_k) <= a for all k=0..5.
        
        To find the minimum enclosing hexagon, we need to find c, theta, s such that
        all points satisfy the constraints, minimizing s.
        
        Simplified approach: For a given rotation theta, the minimum enclosing hexagon
        center and size can be determined. We try several rotations.
        """
        points = np.array(points)
        if len(points) == 0:
            return 0.0, (0.0, 0.0), 0.0
        
        best_s = float('inf')
        best_c = (0.0, 0.0)
        best_theta = 0.0
        
        # Try different rotations
        for theta_deg in range(0, 30, 5):  # 0, 5, 10, 15, 20, 25 degrees
            theta = math.radians(theta_deg)
            # 6 normal directions
            normals = []
            for k in range(6):
                angle = theta + k * math.pi / 3.0
                normals.append(np.array([math.cos(angle), math.sin(angle)]))
            normals = np.array(normals)
            
            # For each point, compute projections onto all 6 normals
            # proj[k] = dot(p, n_k) for each point p
            # The hexagon must contain all points, so:
            # For each direction k, max over points of dot(p-c, n_k) <= a
            # => dot(c, n_k) >= max_p(dot(p, n_k)) - a
            # And dot(c, n_k) <= min_p(dot(p, n_k)) + a
            # So we need: max_p(dot(p, n_k)) - min_p(dot(p, n_k)) <= 2a for all k
            # The width in direction k is w_k = max_p(dot(p, n_k)) - min_p(dot(p, n_k))
            # We need 2a >= max_k(w_k), so a >= max_k(w_k)/2
            # s = 2a / sqrt(3)
            
            projs = points @ normals.T  # shape (n_points, 6)
            widths = np.max(projs, axis=0) - np.min(projs, axis=0)
            max_width = np.max(widths)
            a = max_width / 2.0
            s = 2.0 * a / sqrt3
            
            # Center c: midpoint of the bounding box in each direction
            # c_k = (max_p(dot(p,n_k)) + min_p(dot(p,n_k))) / 2
            # This gives 6 equations for 2 unknowns (c_x, c_y). Use least squares.
            midpoints = (np.max(projs, axis=0) + np.min(projs, axis=0)) / 2.0
            # normals @ c = midpoints
            # Solve least squares
            c = np.linalg.lstsq(normals, midpoints, rcond=None)[0]
            
            # Verify all points are inside
            valid = True
            for p in points:
                for n in normals:
                    if np.dot(p - c, n) > a + 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and s < best_s:
                best_s = s
                best_c = (c[0], c[1])
                best_theta = theta_deg
        
        return best_s, best_c, best_theta
    
    def polyhex_bounding_box(hexes):
        """Compute Cartesian bounding box of a set of hex lattice positions."""
        pts = [lattice_to_cartesian(i, j) for i, j in hexes]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), max(xs), min(ys), max(ys)
    
    def estimate_lower_bound(hexes):
        """
        Lower bound on enclosing hex side length based on cluster diameter.
        The diameter of the cluster is a lower bound on the hexagon's minimum width.
        For a hexagon with side s, the minimum width (distance between parallel sides) is s*sqrt(3).
        So s >= diameter / sqrt(3) is not quite right... 
        Actually, the width of a regular hexagon in any direction ranges from 
        s*sqrt(3) (between opposite sides) to 2*s (between opposite vertices).
        The minimum width over all directions is s*sqrt(3).
        So if the cluster has diameter D, then s*sqrt(3) <= D is not a lower bound on s.
        Rather, the cluster must fit, so the hexagon's minimum width must be >= cluster's max extent.
        This is getting complex. Use a simple bound: s >= max_width / sqrt(3) where max_width
        is the maximum width of the cluster in any direction.
        """
        pts = np.array([lattice_to_cartesian(i, j) for i, j in hexes])
        if len(pts) == 0:
            return 0.0
        
        # Compute diameter
        max_dist = 0.0
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > max_dist:
                    max_dist = d
        
        # The hexagon must have width at least max_dist in the direction of the diameter.
        # The minimum width of a hexagon with side s is s*sqrt(3) (between opposite sides).
        # But the diameter could be in any direction. The width in the direction of the diameter
        # depends on the hexagon's orientation. The minimum possible width in that direction
        # is s*sqrt(3) (if the direction is perpendicular to a pair of opposite sides).
        # So s >= max_dist / sqrt(3) is a valid lower bound.
        return max_dist / sqrt3
    
    # Generate all 11-cell polyhexes using a canonical generation approach
    # This is expensive, so we use a heuristic: generate a few candidate shapes
    
    # Candidate 1: Straight line of 11 hexagons
    # In hex lattice, a straight line: (0,0), (1,0), (2,0), ... or (0,0), (0,1), (0,2), ...
    line_hexes = [(k, 0) for k in range(N)]
    
    # Candidate 2: Compact cluster - hexagonal arrangement
    # A hexagonal number arrangement: center + 6 + 12 = 19, too many.
    # For 11, use a 3x3 block minus corners, or similar.
    # Let's try: rows of 3, 4, 3, 1? No, let's be systematic.
    
    # Generate compact polyhexes via BFS-like growth
    def generate_polyhexes(n_cells, max_candidates=50):
        """Generate candidate n-cell polyhexes using a growth heuristic."""
        candidates = set()
        
        # Start with a single cell at origin
        # Use a canonical form: always add cells to keep the shape "compact"
        
        # Simple approach: try several hand-crafted shapes
        shapes = []
        
        # Shape 1: Line
        shapes.append([(k, 0) for k in range(n_cells)])
        
        # Shape 2: L-shape
        shapes.append([(0, k) for k in range(5)] + [(k, 5) for k in range(1, 7)])
        
        # Shape 3: Compact blob
        # Hexagonal cluster: center (0,0), ring 1: 6 cells, ring 2: partial
        # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # That's 7 cells. Need 4 more.
        # Add from ring 2: (2,0), (1,1), (-2,1), (-1,-1)
        shape3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                  (2,0), (1,1), (-2,1), (-1,-1)]
        shapes.append(shape3)
        
        # Shape 4: 3x3 hexagonal block (9 cells) + 2 more
        # 3x3 in hex lattice: 
        # Row 0: (0,0), (1,0), (2,0)
        # Row 1: (0,1), (1,1), (2,1)
        # Row 2: (0,2), (1,2), (2,2)
        # But this isn't quite right for hex lattice. Let me use axial coords properly.
        # Actually, let's just try a few more variations.
        
        # Shape 5: Two lines offset
        shapes.append([(k, 0) for k in range(6)] + [(k, 1) for k in range(5)])
        
        # Shape 6: Zigzag
        zigzag = []
        for k in range(n_cells):
            if k % 2 == 0:
                zigzag.append((k // 2, 0))
            else:
                zigzag.append((k // 2, 1))
        shapes.append(zigzag)
        
        # For each shape, normalize (translate so min i=0, min j=0) and compute score
        best_score = float('inf')
        best_shape = None
        best_offset = (0, 0)
        
        for shape in shapes:
            if len(shape) != n_cells:
                continue
            if len(set(shape)) != n_cells:
                continue
            
            # Normalize
            min_i = min(i for i, j in shape)
            min_j = min(j for i, j in shape)
            normalized = [(i - min_i, j - min_j) for i, j in shape]
            
            # Compute enclosing hex
            pts = [lattice_to_cartesian(i, j) for i, j in normalized]
            s, c, theta = compute_enclosing_hex_side(pts)
            
            if s < best_score:
                best_score = s
                best_shape = normalized
                best_offset = (min_i, min_j)
        
        return best_shape, best_score
    
    best_shape, best_s = generate_polyhexes(N, max_candidates=100)
    
    # Also try the simple line as a fallback
    line_hexes = [(k, 0) for k in range(N)]
    line_pts = [lattice_to_cartesian(i, j) for i, j in line_hexes]
    line_s, line_c, line_theta = compute_enclosing_hex_side(line_pts)
    
    if line_s < best_s:
        best_shape = line_hexes
        best_s = line_s
    
    # Convert to Cartesian coordinates
    inner = []
    for i, j in best_shape:
        x, y = lattice_to_cartesian(i, j)
        inner.append([float(x), float(y), 0.0])
    
    # Compute the enclosing hexagon parameters
    pts = [lattice_to_cartesian(i, j) for i, j in best_shape]
    s, c, theta = compute_enclosing_hex_side(pts)
    
    # The outer hexagon center and side length
    outer_center = (float(c[0]), float(c[1]))
    outer_side = float(s)
    outer_rotation = float(theta)
    
    return inner, outer_center, outer_side, outer_rotation

# EVOLVE-BLOCK-END
