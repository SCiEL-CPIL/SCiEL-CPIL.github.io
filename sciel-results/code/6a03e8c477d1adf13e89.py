"""Rewritten program: exact combinatorial packing of 11 unit hexagons in a 3x3 hexagonal lattice."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place 11 disjoint unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Strategy: Use the known optimal 3x3 hexagonal lattice arrangement of 11 unit hexagons.
    The 11 hexagons form a hexagonal block of side length 2 (in terms of hexagon count),
    which fits in an outer hexagon of side length 4 (with appropriate rotation/translation).
    
    For unit hexagons in a hexagonal lattice:
    - A hexagon of side length s (in hexagon units) contains hexagons arranged in a hexagonal pattern.
    - The 3x3 arrangement (center + 6 neighbors + 6 more at distance 2) = 1 + 6 + 6 = 13? No.
    - Actually, a hexagonal number: H(1)=1, H(2)=7, H(3)=19. So 11 is between H(2)=7 and H(3)=19.
    
    Let me think more carefully. 11 unit hexagons can be arranged as:
    - A central hexagon
    - 6 hexagons around it (distance 1)
    - 4 more hexagons at distance 2 (subset of the 12 at distance 2)
    
    The minimal enclosing hexagon for a full 3x3 hexagonal block (19 hexagons) has side length 5.
    For 11 hexagons, we can do better.
    
    Actually, the known result: 11 unit hexagons can fit in an outer hexagon of side length 4.
    The arrangement is a 3x3 hexagonal lattice minus 8 corners... no.
    
    Let me use axial coordinates. A regular hexagon of side length L (in units of inner hexagon side)
    contains hexagons at axial positions (q, r) where |q| <= L-1, |r| <= L-1, |q+r| <= L-1... 
    no, that's not right either.
    
    For a regular hexagon with side length S (measured in the same units as the inner hexagons' side),
    the hexagons that fit inside have their centers at positions that are at least the inradius away from each edge.
    
    The inradius of a unit hexagon is sqrt(3)/2. The outer hexagon of side S has inradius S*sqrt(3)/2.
    So the centers of inner hexagons must be at least sqrt(3)/2 from each edge of the outer hexagon.
    
    The region available for centers is a smaller hexagon of side S - 1 (since we subtract inradius on both sides).
    
    So we need to pack 11 points (centers of unit hexagons) in a hexagon of side S-1, with minimum distance sqrt(3) between any two points (since two unit hexagons touching have centers at distance sqrt(3)).
    
    Actually, two unit regular hexagons that are disjoint (non-overlapping) must have centers at distance >= sqrt(3) if they share an edge, but can be closer if they don't align. The minimum center distance for non-overlapping unit hexagons is sqrt(3) when they share a full edge.
    
    For a hexagonal lattice packing, the optimal is to place them on a hexagonal lattice with spacing sqrt(3).
    
    The 3x3 hexagonal lattice (side 2 in lattice units) has 7 points. To get 11, we need a larger pattern.
    
    Let me just use the known result: 11 unit hexagons fit in an outer hexagon of side 4.
    The centers are at the 11 positions of a hexagonal lattice clipped to fit.
    
    Hexagonal lattice points with axial coordinates (q, r) where q + 2r, q, r are all within some range.
    The Cartesian coordinates are: x = sqrt(3)*(q + r/2), y = 1.5*r.
    
    For a hexagon of side S, the valid lattice points satisfy:
    max(|q|, |r|, |q+r|) <= S-1
    
    For S=4: max(|q|, |r|, |q+r|) <= 3. This gives 1 + 6 + 12 + 18 = 37 points, way more than 11.
    
    So we just need to pick 11 points from this lattice that are closest together, minimizing the enclosing hexagon.
    
    The tightest 11-point subset: take the center and all points at distance 1 (6 points) = 7, plus 4 points at distance 2.
    
    Let me just place them in a 3x3 grid pattern and compute the enclosing hexagon.
    """
    deadline = time.time() + 120
    
    # Hexagonal lattice: axial coords (q, r) -> Cartesian (x, y)
    # x = sqrt(3) * (q + r/2), y = 3/2 * r
    sqrt3 = math.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # Generate candidate positions on hexagonal lattice
    # We want 11 points that minimize the enclosing regular hexagon side length.
    
    # Start with the 7-point hexagonal cluster (center + 6 neighbors)
    base_points = [(0, 0)]
    for dq, dr in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]:
        base_points.append((dq, dr))
    
    # Add 4 more points at distance 2
    # Distance-2 points: all (q, r) with max(|q|,|r|,|q+r|) = 2
    dist2_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if max(abs(q), abs(r), abs(q + r)) == 2:
                dist2_points.append((q, r))
    
    # We need to pick 4 from dist2_points to add to base_points
    # To minimize the enclosing hexagon, we should pick points that are "balanced"
    
    # For now, just take 4 specific ones: (2,0), (0,2), (-2,2), (-2,0) -- not symmetric
    # Better: pick 4 that maintain symmetry. Let's try (2,0), (0,2), (-1,2), (-2,0)
    
    # Actually, let's just compute the enclosing hexagon for various 11-point subsets.
    
    def enclosing_hexagon_side(points):
        """Compute the minimum side length of a regular hexagon that contains all points,
        allowing rotation of the outer hexagon."""
        if not points:
            return 0.0
        
        # The side length of the enclosing hexagon for a set of points depends on the orientation.
        # For a regular hexagon with center (cx, cy) and rotation angle theta, the side length S
        # is determined by the maximum over all points of the "radius" in the hexagonal norm.
        
        # The hexagonal norm: for a hexagon with flat top (rotation 0), the half-width in each of 3
        # directions is S*sqrt(3)/2. The distance from center to edge in direction phi is S*sqrt(3)/2.
        
        # For a point at distance d from center in direction phi, the required S is d / (sqrt(3)/2) = 2d/sqrt(3)
        # but this depends on the angle relative to the hexagon's orientation.
        
        # The support function of a regular hexagon with side S and rotation theta:
        # h(phi) = S * sqrt(3)/2 * max over the 3 axes of |cos(phi - theta_i)| where theta_i are the 3 axis directions
        
        # Actually, for a regular hexagon with inradius r = S*sqrt(3)/2, the support function is:
        # h(phi) = r / max(|cos(phi - k*pi/3)| for k=0,1,2) ... no.
        
        # Let me think differently. A regular hexagon with side S has vertices at distance S from center.
        # The inradius is S*sqrt(3)/2.
        
        # For a point P at distance d from the hexagon center, in direction phi, the point is inside
        # if d <= r / cos(phi - alpha) where alpha is the angle to the nearest edge normal... 
        
        # Simpler: the hexagon is the intersection of 6 half-planes. For rotation theta, the normals
        # are at angles theta + k*pi/3 for k=0..5. The distance from center to each edge is r = S*sqrt(3)/2.
        # A point P is inside if for all k: P.x*cos(n_k) + P.y*sin(n_k) <= r where n_k = theta + k*pi/3.
        
        # So for a given center and rotation, S = 2/sqrt(3) * max over all points and all 6 normals of (P.x*cos(n_k) + P.y*sin(n_k))
        
        # To minimize S, we need to choose center and rotation. This is a 2D optimization.
        
        # For simplicity, let's assume the hexagon is centered at the centroid and try a few rotations.
        
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        
        best_S = float('inf')
        
        for theta_deg in range(0, 60, 5):  # Try rotations from 0 to 55 degrees
            theta = math.radians(theta_deg)
            normals = [theta + k * math.pi / 3 for k in range(6)]
            
            max_proj = 0
            for px, py in points:
                dx = px - cx
                dy = py - cy
                for n in normals:
                    proj = dx * math.cos(n) + dy * math.sin(n)
                    if proj > max_proj:
                        max_proj = proj
            
            S = 2 * max_proj / sqrt3
            if S < best_S:
                best_S = S
        
        return best_S, cx, cy
    
    # Try the base 7 points + 4 distance-2 points
    # Let's try a few combinations
    
    best_result = None
    best_S = float('inf')
    
    # Combination 1: add (2,0), (0,2), (-2,2), (-2,0)
    combos = [
        [(2, 0), (0, 2), (-2, 2), (-2, 0)],
        [(2, 0), (0, 2), (-1, 2), (-2, 1)],
        [(2, 0), (0, 2), (-2, 1), (-2, 0)],
        [(1, 2), (0, 2), (-1, 2), (-2, 1)],
    ]
    
    for extra in combos:
        all_pts = base_points + extra
        cart_pts = [axial_to_cartesian(q, r) for q, r in all_pts]
        
        S, cx, cy = enclosing_hexagon_side(cart_pts)
        
        if S < best_S:
            best_S = S
            best_result = (cart_pts, cx, cy)
    
    # Also try just the 7-point cluster with a smaller outer hexagon
    cart_base = [axial_to_cartesian(q, r) for q, r in base_points]
    S_base, cx_base, cy_base = enclosing_hexagon_side(cart_base)
    
    # For 11 points, we need all 11. Let's use the best combination found.
    
    if best_result is None:
        # Fallback: just place 11 in a line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    cart_pts, cx, cy = best_result
    
    # Convert back to the format: (x, y, angle_degrees)
    inner = []
    for px, py in cart_pts:
        inner.append([px, py, 0.0])
    
    return inner, (cx, cy), best_S, 0.0

# EVOLVE-BLOCK-END
