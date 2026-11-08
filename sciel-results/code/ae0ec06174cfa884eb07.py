"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline (10s margin before 150s kill)
    
    # Unit regular hexagon with side length 1
    # A regular hexagon with side s has circumradius s
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees (or offset)
    # For a "flat-top" hexagon (vertex at angle 0):
    #   vertices at (s*cos(k*60°), s*sin(k*60°)) for k=0..5
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        verts = []
        for k in range(6):
            theta = math.radians(angle_deg + k * 60.0)
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts
    
    def polygon_contains_hex(polygon_verts, hex_verts):
        """Check if all hex vertices are inside or on the polygon."""
        # Use point-in-polygon test for each hex vertex
        for hx, hy in hex_verts:
            if not point_in_polygon(hx, hy, polygon_verts):
                return False
        return True
    
    def point_in_polygon(x, y, polygon):
        """Ray casting algorithm for point in polygon."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def compute_enclosing_hexagon(inner_positions, n_inner):
        """
        Compute the minimal enclosing regular hexagon for the given inner hexagons.
        Returns (center_x, center_y, side_length, angle_degrees) of the outer hexagon.
        
        Strategy: The optimal enclosing hexagon will have each of its 6 sides tangent to
        at least one vertex of some inner hexagon. We search over rotations and use
        the support function approach.
        
        For a convex polygon P and a direction θ, the support function h(θ) = max_{p∈P} p·u(θ).
        A regular hexagon with center c, side s, rotation φ has support function:
        h_hex(θ) = c·u(θ) + s * cos(min_k |θ - φ - k*60°|) ... 
        Actually, the support function of a regular hexagon with circumradius R=s is:
        h(θ) = R * cos(δ) where δ is the angle from the nearest vertex direction.
        
        More precisely, for a regular hexagon centered at origin with a vertex at angle 0:
        The vertices are at angles 0°, 60°, ..., 300°.
        The support function is h(θ) = max_k cos(θ - k*60°) for k=0..5, times R.
        Since cos is periodic with 360° and symmetric, h(θ) = cos(min_k |θ - k*60°|) * R
        but we need to be careful: the max of cos(θ - k*60°) over k=0..5.
        
        For the enclosing hexagon to contain all inner hexagons, for every direction θ,
        the support of the inner set in direction θ must be ≤ the support of the outer hexagon in direction θ.
        
        The minimal enclosing regular hexagon can be found by:
        1. Collect all vertices of all inner hexagons.
        2. For a candidate outer hexagon (center, side, rotation), check containment.
        3. Optimize over these parameters.
        
        A better approach: 
        - The center of the enclosing hexagon should be the centroid of the inner set (or optimized).
        - For a given center and rotation, the required side length is determined by the maximum
          support function ratio.
        
        Let's collect all vertices of all inner hexagons. For a given outer hexagon orientation φ,
        the required side length s must satisfy:
          For all vertices v of inner hexagons: v·u(θ) ≤ c·u(θ) + s * h_hex_unit(θ - φ)
          where h_hex_unit(α) = max_k cos(α - k*60°) for k=0..5
        
        Rearranging: s ≥ (v·u(θ) - c·u(θ)) / h_hex_unit(θ - φ) for all v, θ where h_hex_unit > 0
        
        Actually, the support function of the outer hexagon in direction θ is:
          H_outer(θ) = c·u(θ) + s * max_k cos(θ - φ - k*60°)
        
        We need: max_{v∈inner_verts} v·u(θ) ≤ H_outer(θ) for all θ.
        
        So: s ≥ max_θ [max_v (v·u(θ) - c·u(θ))] / max_k cos(θ - φ - k*60°)
        
        This is complex. Let's use a numerical approach:
        1. Compute the convex hull of all inner hexagon vertices.
        2. For a given rotation φ and center c, compute the minimal s such that the hexagon contains the hull.
        3. Optimize over φ and c.
        """
        # Collect all vertices
        all_verts = []
        for (x, y, ang) in inner_positions:
            verts = hex_vertices(x, y, 1.0, ang)
            all_verts.extend(verts)
        
        all_verts = np.array(all_verts)
        
        if len(all_verts) < 3:
            return 0.0, 0.0, 1.0, 0.0
        
        # Compute centroid as initial center
        cx = float(np.mean(all_verts[:, 0]))
        cy = float(np.mean(all_verts[:, 1]))
        
        def min_side_for_rotation(cx, cy, phi_deg, all_verts):
            """Find minimal side length for a hexagon centered at (cx,cy) with rotation phi_deg."""
            phi = math.radians(phi_deg)
            # The 6 vertex directions of the outer hexagon
            # Support function of unit circumradius hexagon: h(θ) = max_k cos(θ - phi - k*60°)
            # For containment, we need for each direction θ:
            #   max_v (v - c)·u(θ) ≤ s * h(θ - phi)
            # So s ≥ max_θ [max_v (v-c)·u(θ)] / h(θ - phi)
            
            # We only need to check θ values where the numerator or denominator is extremal.
            # The max over v of (v-c)·u(θ) is the support function of the translated set.
            # Let's sample θ densely.
            
            # Actually, the critical directions are those where the support of the inner set
            # and the support of the hexagon are both maximal. Let's just sample.
            
            max_ratio = 0.0
            # Sample many angles
            for theta_deg in np.linspace(0, 360, 720, endpoint=False):
                theta = math.radians(theta_deg)
                ux = math.cos(theta)
                uy = math.sin(theta)
                
                # Support of inner set in direction θ
                dots = (all_verts[:, 0] - cx) * ux + (all_verts[:, 1] - cy) * uy
                sup_inner = float(np.max(dots))
                
                if sup_inner <= 0:
                    continue
                
                # Support of unit hexagon in direction (theta - phi)
                # h(α) = max_k cos(α - k*60°) for k=0..5
                alpha = theta - phi
                h_val = 0.0
                for k in range(6):
                    h_val = max(h_val, math.cos(alpha - k * math.pi / 3.0))
                
                if h_val <= 1e-10:
                    # This shouldn't happen for a hexagon
                    continue
                
                ratio = sup_inner / h_val
                if ratio > max_ratio:
                    max_ratio = ratio
            
            return max_ratio
        
        # Optimize over rotation
        best_side = float('inf')
        best_phi = 0.0
        best_cx = cx
        best_cy = cy
        
        # Try different rotations
        for phi_deg in np.linspace(0, 60, 61):  # 0 to 60 degrees in 1° steps
            if time.time() > deadline:
                break
            s = min_side_for_rotation(cx, cy, phi_deg, all_verts)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Refine around best rotation
        if best_phi > 0 and best_phi < 60:
            for phi_deg in np.linspace(best_phi - 1, best_phi + 1, 21):
                if time.time() > deadline:
                    break
                s = min_side_for_rotation(cx, cy, phi_deg, all_verts)
                if s < best_side:
                    best_side = s
                    best_phi = phi_deg
        
        # Try to optimize center as well
        # The optimal center might not be the centroid. Let's try a few shifts.
        for dx in [-0.5, -0.25, 0, 0.25, 0.5]:
            for dy in [-0.5, -0.25, 0, 0.25, 0.5]:
                if time.time() > deadline:
                    break
                test_cx = cx + dx
                test_cy = cy + dy
                s = min_side_for_rotation(test_cx, test_cy, best_phi, all_verts)
                if s < best_side:
                    best_side = s
                    best_cx = test_cx
                    best_cy = test_cy
        
        return best_cx, best_cy, best_side, best_phi
    
    # Generate hexagonal lattice packing
    # Unit hexagons on a honeycomb lattice
    # In a hexagonal tiling, adjacent hexagons are at distance 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    # Wait, for side length 1: apothem = sqrt(3)/2, so center-to-center distance for adjacent = 2*apothem = sqrt(3)
    
    # Hexagonal lattice basis vectors:
    # a1 = (sqrt(3), 0)  -- horizontal
    # a2 = (sqrt(3)/2, 3/2)  -- at 60 degrees
    # Actually, let me reconsider. For flat-top hexagons with side 1:
    # The distance between centers of adjacent hexagons is 2 * apothem = sqrt(3)
    # The lattice vectors can be:
    #   v1 = (sqrt(3), 0)
    #   v2 = (sqrt(3)/2, 3/2)
    
    sqrt3 = math.sqrt(3.0)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2.0, 1.5])
    
    # Generate hexagons in a hexagonal region of the lattice
    # For a hexagonal arrangement with radius R (in lattice units), we get:
    # 1 + 6 + 12 + ... = 1 + 3*R*(R+1) hexagons
    # For N=11: 1 + 3*1*2 = 7, 1+3*2*3=19. So N=11 is between R=1 and R=2.
    # Let's just generate a hexagonal region and take the first N.
    
    # Generate positions in hexagonal lattice order (from center outward)
    positions = [(0.0, 0.0, 0.0)]  # center
    # Ring 1: 6 hexagons
    for k in range(6):
        theta = math.radians(k * 60)
        x = sqrt3 * math.cos(theta)
        y = sqrt3 * math.sin(theta)
        positions.append((x, y, 0.0))
    
    # Ring 2: 12 hexagons (we only need 11 total, so 5 more)
    # Ring 2 positions: 2*(v1 or v2) combinations
    ring2 = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            # Lattice point i*v1 + j*v2
            px = i * v1[0] + j * v2[0]
            py = i * v1[1] + j * v2[1]
            # Check if this is in ring 2 (distance 2 in lattice steps)
            # Ring 1 was at distance 1, ring 2 at distance 2
            # The lattice distance for (i,j) is max(|i|, |j|, |i+j|) for hex coords
            # Actually, let's use a different coordinate system.
            # For axial coordinates (q, r), the hex distance is (|q| + |r| + |q+r|) / 2
            # Our v1, v2 correspond to q=1,r=0 and q=0,r=1 in axial coords
            # So (i,j) in our lattice = (q=i, r=j) in axial
            q, r = i, j
            dist = (abs(q) + abs(r) + abs(q + r)) // 2
            if dist == 2:
                ring2.append((px, py))
    
    # Sort ring 2 by angle for consistent ordering
    ring2.sort(key=lambda p: math.atan2(p[1], p[0]))
    
    for (x, y) in ring2[:5]:  # Need 5 more to reach 11
        positions.append((x, y, 0.0))
    
    # Trim to N
    positions = positions[:N]
    
    # Compute enclosing hexagon
    if time.time() < deadline:
        cx, cy, side, phi = compute_enclosing_hexagon(positions, N)
    else:
        # Fallback: simple bounding
        cx, cy, side, phi = 0.0, 0.0, 10.0, 0.0
    
    return positions, (cx, cy), side, phi
# EVOLVE-BLOCK-END
