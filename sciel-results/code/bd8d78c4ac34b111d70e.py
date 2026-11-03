"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons inside a minimal regular hexagon.
    Uses a hexagonal lattice arrangement for compact packing.
    """
    # Define hexagonal lattice positions for 11 hexagons
    # Using axial coordinates (q, r) for hex grid
    # Compact arrangement: center + surrounding rings
    
    # For 11 hexagons, use a 4-3-4 row pattern in hex lattice
    # Axial coords: let's define positions
    positions_axial = [
        (0, 0),
        (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    
    # Convert axial to Cartesian for unit hexagons
    # Distance between adjacent centers = sqrt(3)
    # Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in positions_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Center the arrangement at origin
    centroid = centers.mean(axis=0)
    centers -= centroid
    
    # Now find the minimal enclosing regular hexagon
    # A regular hexagon with center c, side s, rotation theta contains a point p
    # if for all 6 directions, the projection is within s*sqrt(3)/2
    # 
    # For a regular hexagon of side s, the distance from center to each side (apothem) is s*sqrt(3)/2
    # The 6 normal directions are at angles theta + k*60 degrees for k=0..5
    #
    # A point p is inside if for all k: dot(p - c, n_k) <= s*sqrt(3)/2
    # where n_k are unit vectors at angles theta + k*pi/3
    #
    # To minimize s, for a given theta, s must be >= max over all points p and all k of dot(p, n_k) / (sqrt(3)/2)
    # i.e., s >= (2/sqrt(3)) * max_k max_p dot(p, n_k)
    #
    # We search over theta to minimize this.
    
    def min_side_for_angle(theta):
        """Compute minimum side length for hexagon with given rotation angle."""
        # 6 normal directions
        angles = theta + np.arange(6) * np.pi / 3
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # For each normal, compute max projection of all centers
        # Project all centers onto each normal
        projections = centers @ normals.T  # shape (N, 6)
        max_proj = projections.max(axis=0)  # shape (6,)
        
        # Also need to consider that hexagons have extent, not just centers
        # Each unit hexagon extends sqrt(3)/2 from its center in any direction (circumradius = 1)
        # Wait, the circumradius of a unit hexagon is 1 (distance from center to vertex)
        # The inradius (apothem) is sqrt(3)/2
        #
        # For a point at distance d from center in direction n, the hexagon extends to d + 1 in that direction
        # But actually, the support function of a unit regular hexagon in direction n is:
        # h(n) = max over vertices v of dot(v, n)
        # For a unit hexagon, the vertices are at angles 0, 60, 120, 180, 240, 300 degrees at radius 1
        # So h(n) = max_k cos(angle(n) - k*pi/3) for k=0..5
        # This equals 1 if n aligns with a vertex, and sqrt(3)/2 if n aligns with an edge normal
        # Actually, the support function of a regular hexagon with circumradius 1 is:
        # h(theta) = max(cos(theta - k*pi/3)) for k=0..5
        # The maximum of this over all theta is 1 (when aligned with vertex)
        # The minimum is sqrt(3)/2 (when aligned with edge)
        
        # For each normal direction n_k, the total extent is max_proj[k] + h(n_k)
        # where h(n_k) is the support function of a unit hexagon in direction n_k
        # The unit hexagon has vertices at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
        # h(n) = max_j cos(phi - j*pi/3) where phi = angle of n
        # Since the hexagon is symmetric, h(n) depends only on the angle modulo pi/3
        
        # Compute support function for each normal
        phi = np.arctan2(normals[:, 1], normals[:, 0])
        # Reduce to [0, pi/3)
        phi_mod = np.mod(phi, np.pi / 3)
        # h(n) = max(cos(phi_mod), cos(pi/3 - phi_mod)) ... actually just cos(min(phi_mod, pi/3 - phi_mod))
        # Wait, for a regular hexagon with vertices at 0, 60, 120, ..., the support function is:
        # h(phi) = cos(phi) for phi in [-30, 30] degrees (relative to vertex at 0)
        # By symmetry, h(phi) = cos(phi_mod) where phi_mod is in [0, 30] degrees
        # cos(30) = sqrt(3)/2, cos(0) = 1
        h_vals = np.cos(phi_mod)  # This gives values in [sqrt(3)/2, 1]
        
        # Total required apothem = max_proj + h_vals
        required_apothem = max_proj + h_vals
        s = (2.0 / sqrt3) * required_apothem.max()
        return s
    
    # Search over rotation angles
    best_s = float('inf')
    best_theta = 0.0
    for theta_deg in np.linspace(0, 60, 121):  # 0 to 60 degrees, 1 degree steps
        theta = np.radians(theta_deg)
        s = min_side_for_angle(theta)
        if s < best_s:
            best_s = s
            best_theta = theta
    
    # Refine with finer search around best
    for theta_deg in np.linspace(best_theta - np.pi/180, best_theta + np.pi/180, 101):
        theta = theta_deg
        s = min_side_for_angle(theta)
        if s < best_s:
            best_s = s
            best_theta = theta
    
    # Build inner hexagons list
    # Each inner hexagon: (x, y, angle_degrees)
    # All hexagons are aligned (0 degrees rotation)
    inner = []
    for x, y in centers:
        inner.append([float(x), float(y), 0.0])
    
    # Outer hexagon: center at origin, side length best_s, rotation best_theta
    outer_center = (0.0, 0.0)
    outer_side = float(best_s)
    outer_angle_deg = float(np.degrees(best_theta))
    
    return inner, outer_center, outer_side, outer_angle_deg
# EVOLVE-BLOCK-END

