"""Optimized hexagon packing using concentric rings with radial/angle optimization."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # We need to pack N=11 unit regular hexagons (side length 1) inside a regular hexagon.
    # The inner hexagons can translate and rotate.
    # Score is the outer side length (lower is better).
    #
    # A regular hexagon with side length s has:
    # - Width (flat-to-flat): sqrt(3) * s
    # - Height (vertex-to-vertex): 2 * s
    #
    # For a unit hexagon (side=1):
    # - Flat-to-flat distance: sqrt(3)
    # - Vertex-to-vertex: 2
    #
    # Two unit hexagons are disjoint if the distance between their centers is at least
    # the sum of their "inradius" contributions... actually, for regular hexagons,
    # the minimum distance between centers for non-overlap depends on relative orientation.
    #
    # For two identical regular hexagons of side 1, the minimum center distance for
    # non-overlap is sqrt(3) when they are aligned (flat-to-flat touching).
    # If rotated, the minimum distance can be different.
    #
    # Actually, let's think more carefully. A regular hexagon with side s can be inscribed
    # in a circle of radius s (circumradius = s). The inradius is (sqrt(3)/2)*s.
    #
    # For two convex bodies, they are disjoint if the distance between centers is at
    # least the sum of their support functions in the direction connecting the centers.
    # For regular hexagons, this is complex.
    #
    # Simplification: For axis-aligned regular hexagons (flat top or pointy top),
    # the minimum separation distance between centers is sqrt(3) for flat-to-flat contact.
    #
    # Let's use a simpler collision model: treat each hexagon as a circle of radius r
    # where r is chosen so that the packing is conservative. For a unit hexagon,
    # the circumradius is 1, inradius is sqrt(3)/2 ≈ 0.866.
    #
    # Actually, for two identical regular hexagons with the same orientation, the minimum
    # distance between centers for non-overlap is exactly sqrt(3) (flat-to-flat touching).
    # This is the distance between parallel edges.
    #
    # For simplicity and to ensure validity, I'll use a collision check based on the
    # actual hexagon geometry. But for optimization, I'll use a simplified model.
    #
    # Strategy: Place hexagons on concentric rings.
    # Ring 0: 1 hexagon at center
    # Ring 1: 6 hexagons at distance d1 from center, at angles 0, 60, 120, 180, 240, 300
    # Ring 2: 4 hexagons at distance d2 from center, at some angles
    # Total: 1 + 6 + 4 = 11
    #
    # The outer hexagon must contain all inner hexagons.
    #
    # Let's define the problem:
    # - Inner hexagon centers: (x_i, y_i, angle_i) for i=0..10
    # - Outer hexagon: center (cx, cy), side length S, rotation angle
    #
    # We want to minimize S such that:
    # 1. All inner hexagons are inside the outer hexagon
    # 2. No two inner hexagons overlap
    #
    # For the outer hexagon containment check:
    # A point (px, py) is inside a regular hexagon with center (cx, cy), side S,
    # and rotation θ if it satisfies 6 half-plane inequalities.
    #
    # For a regular hexagon with "flat top" orientation (rotation=0), the vertices are at
    # angles 0°, 60°, 120°, 180°, 240°, 300° from the center, at distance S.
    # The edges are between consecutive vertices.
    #
    # Actually, let me use a standard representation. A regular hexagon with center (cx,cy),
    # side length S, and rotation angle φ has vertices at:
    # (cx + S*cos(φ + k*60°), cy + S*sin(φ + k*60°)) for k=0..5
    #
    # A point is inside if it satisfies all 6 half-plane constraints.
    #
    # For the inner hexagons, each has side 1, so their vertices are at distance 1 from
    # their centers. For a point on an inner hexagon to be inside the outer hexagon,
    # we need all 6 vertices of the inner hexagon to be inside the outer hexagon.
    #
    # This is a complex optimization. Let me use a simpler approach:
    #
    # Use symmetry: Place hexagons symmetrically.
    # Ring 0: 1 at center (0,0), angle 0
    # Ring 1: 6 at distance d1, angles 0,60,120,180,240,300, all with angle 0
    # Ring 2: 4 at distance d2, angles 30,150,210,330 (offset by 30°), all with angle 0
    #
    # Optimize d1, d2 to minimize the required outer hexagon side length.
    #
    # For the outer hexagon, by symmetry, we can assume it's centered at origin with
    # some rotation. By symmetry of the inner arrangement, the optimal outer hexagon
    # should have the same symmetry, so rotation 0 or 30°.
    
    def hex_vertices(cx, cy, s, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        ang = np.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = ang + k * np.pi / 3
            verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
        return np.array(verts)
    
    def point_in_hex(px, py, cx, cy, s, angle_deg):
        """Check if point (px,py) is inside hexagon with center (cx,cy), side s, rotation angle_deg."""
        # Transform point to hexagon's local frame
        dx = px - cx
        dy = py - cy
        ang = np.radians(angle_deg)
        # Rotate point by -angle
        x_local = dx * np.cos(-ang) - dy * np.sin(-ang)
        y_local = dx * np.sin(-ang) + dy * np.cos(-ang)
        
        # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
        # The hexagon is the intersection of 6 half-planes.
        # For a regular hexagon with circumradius s, the inradius is s*sqrt(3)/2
        # The half-planes are: for each edge, the point must be on the correct side.
        
        # Edge normals point inward. For vertex at angle theta, the edge is perpendicular
        # to the direction to the midpoint of the edge.
        # Actually, simpler: the hexagon is defined by |x*cos(60k) + y*sin(60k)| <= s*sqrt(3)/2
        # for k=0,1,2 (three pairs of parallel edges)
        
        inradius = s * np.sqrt(3) / 2
        for k in range(3):
            theta = k * np.pi / 3
            val = x_local * np.cos(theta) + y_local * np.sin(theta)
            if abs(val) > inradius + 1e-9:
                return False
        return True
    
    def hex_in_hex(inner_cx, inner_cy, inner_angle, outer_cx, outer_cy, outer_s, outer_angle):
        """Check if inner hexagon (side 1) is inside outer hexagon."""
        # Check all 6 vertices of inner hexagon
        for vx, vy in hex_vertices(inner_cx, inner_cy, 1.0, inner_angle):
            if not point_in_hex(vx, vy, outer_cx, outer_cy, outer_s, outer_angle):
                return False
        return True
    
    def min_separation(c1, a1, c2, a2):
        """Minimum distance between centers of two unit hexagons with angles a1, a2 for non-overlap.
        For simplicity, use sqrt(3) as the minimum separation (valid for same orientation).
        For different orientations, it could be different, but sqrt(3) is a safe lower bound."""
        # Actually, for two identical regular hexagons, the minimum center distance
        # for non-overlap is sqrt(3) when they are in the same orientation.
        # For different orientations, it can be less, but let's be conservative.
        return np.sqrt(3)
    
    def check_validity(d1, d2):
        """Check if the configuration with given ring distances is valid (no overlaps)."""
        centers = []
        angles = []
        
        # Ring 0
        centers.append((0.0, 0.0))
        angles.append(0.0)
        
        # Ring 1: 6 hexagons
        for k in range(6):
            theta = k * np.pi / 3
            centers.append((d1 * np.cos(theta), d1 * np.sin(theta)))
            angles.append(0.0)
        
        # Ring 2: 4 hexagons at 30, 150, 210, 330 degrees
        for k in [1, 3, 4, 5]:  # 30, 150, 210, 330 degrees
            theta = k * np.pi / 3
            centers.append((d2 * np.cos(theta), d2 * np.sin(theta)))
            angles.append(0.0)
        
        # Check pairwise distances
        n = len(centers)
        min_dist = np.sqrt(3)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist - 1e-9:
                    return False
        return True
    
    def required_outer_side(d1, d2):
        """Compute the minimum outer hexagon side length needed to contain all inner hexagons."""
        centers = []
        angles = []
        
        centers.append((0.0, 0.0))
        angles.append(0.0)
        
        for k in range(6):
            theta = k * np.pi / 3
            centers.append((d1 * np.cos(theta), d1 * np.sin(theta)))
            angles.append(0.0)
        
        for k in [1, 3, 4, 5]:
            theta = k * np.pi / 3
            centers.append((d2 * np.cos(theta), d2 * np.sin(theta)))
            angles.append(0.0)
        
        # Find the maximum distance from origin to any vertex of any inner hexagon
        # The outer hexagon must contain all these vertices.
        # By symmetry, the outer hexagon should be centered at origin.
        # The minimum side length S such that a regular hexagon of side S contains
        # all points at distance r from center is when the inradius of the outer hexagon
        # is at least the maximum distance from center to any vertex.
        
        max_r = 0
        for cx, cy, ang in zip(centers, centers, angles):
            for vx, vy in hex_vertices(cx, cy, 1.0, ang):
                r = np.sqrt(vx * vx + vy * vy)
                max_r = max(max_r, r)
        
        # For a regular hexagon with side S, the inradius is S*sqrt(3)/2
        # The maximum distance from center to a vertex on the boundary (in the direction
        # of an edge normal) is the inradius.
        # But vertices of inner hexagons can be in any direction.
        # The outer hexagon's boundary in direction theta is at distance:
        # d(theta) = inradius / cos(theta mod 60° - 30°) ... this is getting complex.
        
        # Simpler: The outer hexagon must have its circumradius at least max_r
        # if the outer hexagon's vertices align with the farthest points.
        # But actually, the constraint is that all points must be inside.
        # The farthest point from center at distance r in direction theta requires
        # the outer hexagon to have support function >= r in that direction.
        
        # For a regular hexagon with inradius R_in, the support function in direction
        # theta is R_in / cos((theta mod 60°) - 30°) ... no, let me think again.
        
        # Actually, for a regular hexagon with inradius R (distance from center to edge midpoint),
        # the distance from center to boundary in direction theta is:
        # d(theta) = R / cos(delta) where delta is the angle from the nearest edge normal.
        # The edge normals are at 0°, 60°, 120°, 180°, 240°, 300°.
        # So for a direction theta, delta = min over k of |theta - k*60°| mod 60°, clamped to [-30°, 30°]
        
        # We need d(theta) >= r for all vertices.
        # So R >= r * cos(delta) for all vertices.
        # R >= max over all vertices of r * cos(delta)
        
        max_R = 0
        for cx, cy, ang in zip(centers, centers, angles):
            for vx, vy in hex_vertices(cx, cy, 1.0, ang):
                r = np.sqrt(vx * vx + vy * vy)
                if r < 1e-9:
                    continue
                theta = np.arctan2(vy, vx)
                # Find delta: angle from nearest edge normal
                theta_mod = theta % (np.pi / 3)
                if theta_mod > np.pi / 6:
                    theta_mod = np.pi / 3 - theta_mod
                delta = theta_mod
                R_needed = r * np.cos(delta)
                max_R = max(max_R, R_needed)
        
        # Inradius R = S * sqrt(3) / 2, so S = 2*R / sqrt(3)
        S = 2 * max_R / np.sqrt(3)
        return S
    
    def objective(params):
        d1, d2 = params
        if d1 < 0 or d2 < 0:
            return 1e10
        if not check_validity(d1, d2):
            return 1e10
        return required_outer_side(d1, d2)
    
    # Initial guess
    d1_init = np.sqrt(3)
    d2_init = 2 * np.sqrt(3)
    
    result = minimize(objective, [d1_init, d2_init], method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    
    d1, d2 = result.x
    if d1 < 0:
        d1 = d1_init
    if d2 < 0:
        d2 = d2_init
    
    if not check_validity(d1, d2):
        # Fall back to initial guess
        d1 = d1_init
        d2 = d2_init
    
    S = required_outer_side(d1, d2)
    
    # Build inner hexagon positions
    inner = []
    # Ring 0
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1
    for k in range(6):
        theta = k * np.pi / 3
        inner.append([d1 * np.cos(theta), d1 * np.sin(theta), 0.0])
    
    # Ring 2
    for k in [1, 3, 4, 5]:
        theta = k * np.pi / 3
        inner.append([d2 * np.cos(theta), d2 * np.sin(theta), 0.0])
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
