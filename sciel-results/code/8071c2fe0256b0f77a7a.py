import numpy as np
from scipy.spatial import ConvexHull
import math

def construct():
    # Define 11 unit hexagon centers on a hexagonal lattice
    # Using axial coordinates converted to Cartesian
    # Hex grid: center distance = sqrt(3) for side-1 hexagons
    
    sqrt3 = math.sqrt(3)
    
    # Axial coordinates for a compact 11-hexagon cluster
    # Let's use a "3-4-4" row arrangement
    axial_coords = [
        # Row r=-1: 3 hexagons
        (-1, -1), (0, -1), (1, -1),
        # Row r=0: 4 hexagons
        (-1, 0), (0, 0), (1, 0), (2, 0),
        # Row r=1: 4 hexagons
        (-1, 1), (0, 1), (1, 1), (2, 1),
    ]
    
    # Convert to Cartesian
    centers = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Center the cluster
    center_offset = centers.mean(axis=0)
    centers = centers - center_offset
    
    # For each inner hexagon, we need its 6 vertices
    # A unit hexagon centered at (cx, cy) with rotation angle theta has vertices at:
    # (cx + cos(theta + k*pi/3), cy + sin(theta + k*pi/3)) for k=0..5
    
    # But since all inner hexagons have the same orientation (they're on a lattice),
    # we can compute the convex hull of all vertices of all hexagons.
    
    # Actually, for the enclosing hexagon, we need to consider the full shape.
    # The convex hull of all 11 hexagons' vertices gives the outer boundary.
    
    # Generate all vertices of all 11 hexagons
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = k * math.pi / 3.0
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    # Compute convex hull
    hull = ConvexHull(all_vertices)
    hull_vertices = all_vertices[hull.vertices]
    
    # Now, for a given global rotation angle alpha, rotate the entire cluster
    # and find the minimal enclosing regular hexagon side length.
    
    # A regular hexagon with side length S, centered at origin, with rotation beta,
    # contains a point (x, y) if for all 6 half-planes defined by the hexagon edges,
    # the point is on the correct side.
    
    # The 6 edges of a regular hexagon with side S, rotation beta, are at distances
    # (S * sqrt(3) / 2) from center, at angles beta, beta+60°, ..., beta+300°.
    # Wait, let me think again.
    
    # A regular hexagon with side length S has:
    # - Inradius (apothem) = S * sqrt(3) / 2
    # - Circumradius = S
    # The 6 supporting lines are at distance inradius from center, at angles 30°, 90°, 150°, 210°, 270°, 330° (for a hexagon with a flat top).
    
    # Actually, for a regular hexagon with side S, the distance from center to each edge (apothem) is S * sqrt(3) / 2.
    # The 6 edges are perpendicular to directions at angles 0°, 60°, 120°, 180°, 240°, 300° (if we define it that way).
    
    # For a point to be inside the hexagon, it must satisfy:
    # For each of the 6 directions theta_i (i=0..5), the projection of the point onto that direction must be <= apothem.
    # i.e., x*cos(theta_i) + y*sin(theta_i) <= S * sqrt(3) / 2
    
    # So the minimal S for a given orientation is:
    # S = (2 / sqrt(3)) * max over all points and all 6 directions of (x*cos(theta_i) + y*sin(theta_i))
    
    # But we can also rotate the hexagon, so we need to minimize over the hexagon's rotation.
    
    # Let the hexagon have rotation phi. Then the 6 normal directions are phi, phi+60°, ..., phi+300°.
    # The required side length is:
    # S(phi) = (2 / sqrt(3)) * max_{i=0..5} max_{points p} (p.x * cos(phi + i*60°) + p.y * sin(phi + i*60°))
    
    # We want to minimize S(phi) over phi.
    
    # Note: the inner hexagons can also be rotated, but in this lattice packing, they all have the same orientation.
    # The task says each inner hexagon may translate and rotate, but for a lattice packing, keeping them aligned is natural.
    
    # Let me implement the minimization over phi.
    
    def min_enclosing_hex_side(points, phi):
        """Compute the minimal side length of a regular hexagon with rotation phi that contains all points."""
        apothem_required = 0
        for i in range(6):
            theta = phi + i * math.pi / 3.0
            # Project all points onto direction (cos(theta), sin(theta))
            projections = points[:, 0] * math.cos(theta) + points[:, 1] * math.sin(theta)
            max_proj = np.max(projections)
            apothem_required = max(apothem_required, max_proj)
        # Side length S = 2 * apothem / sqrt(3)
        return 2.0 * apothem_required / sqrt3
    
    # Minimize over phi in [0, pi/3) due to 6-fold symmetry
    # Use ternary search or just sample finely
    
    best_S = float('inf')
    best_phi = 0
    
    # Sample phi finely
    num_samples = 10000
    for i in range(num_samples):
        phi = i * (math.pi / 3.0) / num_samples
        S = min_enclosing_hex_side(hull_vertices, phi)
        if S < best_S:
            best_S = S
            best_phi = phi
    
    # Refine with ternary search
    lo = best_phi - (math.pi / 3.0) / num_samples
    hi = best_phi + (math.pi / 3.0) / num_samples
    for _ in range(50):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if min_enclosing_hex_side(hull_vertices, m1) < min_enclosing_hex_side(hull_vertices, m2):
            hi = m2
        else:
            lo = m1
    
    best_phi = (lo + hi) / 2.0
    best_S = min_enclosing_hex_side(hull_vertices, best_phi)
    
    # Now construct the output
    # Inner hexagons: positions and angles
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])  # angle in degrees
    
    # Outer hexagon: center, side length, rotation
    outer_center = [0.0, 0.0]
    outer_side = best_S
    outer_rotation = math.degrees(best_phi)
    
    return inner, outer_center, outer_side, outer_rotation
