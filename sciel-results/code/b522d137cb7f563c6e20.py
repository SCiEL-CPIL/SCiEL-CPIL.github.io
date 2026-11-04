"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, s, ang_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s and rotation ang_deg."""
    a = math.radians(ang_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return np.array(verts)

def _signed_dist_point_to_hex(px, py, cx, cy, s, ang_deg):
    """
    Approximate signed distance from point (px,py) to hexagon centered at (cx,cy) side s rot ang.
    Positive if outside. Uses radial distance in the hexagonal norm.
    For a regular hexagon with side s, the 'radius' to vertex is s, to edge midpoint is s*sqrt(3)/2.
    We use the support function approach: distance = max over 6 edges of (dot(normal, p-c) - offset).
    """
    a = math.radians(ang_deg)
    # The 6 outward normals of a regular hexagon
    normals = []
    for k in range(6):
        theta = a + k * math.pi / 3.0 + math.pi / 2.0  # edge normals are offset by 90 deg from vertex angles? 
        # Actually for a regular hexagon with vertices at angles a + k*pi/3, the edge normals are at a + k*pi/3 + pi/2? 
        # Let's think: vertex at angle 0 is (s,0). The edge between vertex 0 and vertex 1 (at 60 deg) has midpoint at 30 deg direction.
        # The outward normal to that edge points in direction 30 deg.
        # So edge k (between vertex k and k+1) has normal at angle a + (k+0.5)*pi/3.
        theta_normal = a + (k + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta_normal), math.sin(theta_normal)))
    
    # The distance from center to each edge (apothem) is s * sqrt(3) / 2
    apothem = s * math.sqrt(3) / 2.0
    
    dx = px - cx
    dy = py - cy
    max_val = -np.inf
    for nx, ny in normals:
        d = dx * nx + dy * ny - apothem
        if d > max_val:
            max_val = d
    return max_val

def _hex_hex_distance(h1, h2):
    """
    Approximate minimum distance between two hexagons.
    h1, h2 are (cx, cy, s, ang_deg).
    We sample points on the boundary of h1 and check distance to h2, and vice versa.
    For a smoother approximation, we can use the fact that the distance between two convex sets is 
    the minimum over all pairs of points. We approximate by sampling vertices and edge midpoints.
    """
    cx1, cy1, s1, a1 = h1
    cx2, cy2, s2, a2 = h2
    
    # Sample points on h1 boundary
    pts1 = _hex_vertices(cx1, cy1, s1, a1)
    # Also add edge midpoints
    mids1 = []
    for k in range(6):
        v1 = pts1[k]
        v2 = pts1[(k+1)%6]
        mids1.append(((v1[0]+v2[0])/2, (v1[1]+v2[1])/2))
    pts1 = np.vstack([pts1, np.array(mids1)])
    
    # For each point on h1, compute signed distance to h2
    min_dist = np.inf
    for px, py in pts1:
        d = _signed_dist_point_to_hex(px, py, cx2, cy2, s2, a2)
        if d < min_dist:
            min_dist = d
    return min_dist

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Initial configuration: honeycomb-like arrangement
    # 12 hexagons in a roughly hexagonal cluster
    # Let's try a 4-3-3-2 arrangement or similar
    # Actually, let's start with a compact arrangement
    
    # Use a fixed topology: 12 hexagons in a honeycomb cluster
    # The honeycomb cluster of 12 hexagons has a specific shape
    # Let's define relative positions in a hexagonal grid
    
    # Hexagonal grid coordinates to Cartesian
    # For a regular hexagon with side 1, the distance between centers of adjacent hexagons is sqrt(3)
    # (if they share an edge)
    
    # Let's define a honeycomb cluster of 12 hexagons
    # A common arrangement: central hexagon + 6 around + 5 more
    # Or: rows of 3, 4, 3, 2 = 12
    
    # Let's use axial coordinates for hex grid
    # Convert axial (q, r) to Cartesian
    def axial_to_cart(q, r, size=1.0):
        x = size * (3.0/2.0 * q)
        y = size * (math.sqrt(3)/2.0 * q + math.sqrt(3) * r)
        return (x, y)
    
    # Define 12 hexagons in a honeycomb pattern
    # Using axial coordinates
    hex_positions = [
        (0, 0),    # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (-1, 2), (-2, 1), (0, -2),  # partial ring 2
    ]
    
    # Convert to Cartesian
    inner = []
    for q, r in hex_positions:
        x, y = axial_to_cart(q, r)
        inner.append([x, y, 0.0])
    
    # Now optimize: allow translation and rotation of each hexagon
    # Objective: minimize the outer hexagon side length that contains all 12
    
    # For a given configuration of inner hexagons, the minimum enclosing regular hexagon
    # can be found by optimizing over the outer hexagon's center, side length, and rotation.
    
    # This is a complex optimization. Let's use a simpler approach:
    # 1. Fix the outer hexagon rotation to 0 (or optimize it separately)
    # 2. For a given outer hexagon, find the minimum side length that contains all inner hexagons
    
    # Actually, let's use a different approach:
    # The problem is to place 12 hexagons to minimize the enclosing hexagon side.
    # We can use gradient-based optimization on the positions and rotations.
    
    # Let's parameterize:
    # - Outer hexagon: center (ox, oy), side L, rotation alpha
    # - Inner hexagons: positions (x_i, y_i), rotations theta_i
    # - Constraint: all inner hexagons are inside the outer hexagon
    # - Objective: minimize L
    
    # This is a nonlinear optimization problem. Let's use scipy.optimize.
    
    from scipy.optimize import minimize
    
    def enclosing_hexagon_side(inner_hexes, outer_center, outer_rot):
        """
        Find the minimum side length of a regular hexagon centered at outer_center
        with rotation outer_rot that contains all inner_hexes.
        inner_hexes: list of (cx, cy, s, ang_deg)
        """
        ox, oy = outer_center
        alpha = math.radians(outer_rot)
        
        # The outer hexagon has 6 edges. For a point to be inside, it must be on the 
        # correct side of all 6 edges.
        # For a hexagon with side L, center (ox,oy), rotation alpha:
        # The 6 edge normals are at angles alpha + (k+0.5)*pi/3
        # The distance from center to each edge is L * sqrt(3) / 2
        # A point (px, py) is inside if for all k:
        #   (px-ox)*cos(nk) + (py-oy)*sin(nk) <= L*sqrt(3)/2
        # where nk = alpha + (k+0.5)*pi/3
        
        # For a hexagon (inner) to be inside, all its points must satisfy this.
        # The farthest point of the inner hexagon in direction nk is:
        #   center_dot_nk + support_function_inner(nk)
        # where support_function_inner(nk) is the maximum dot product of points on the inner hexagon with nk.
        
        # For a regular hexagon with side s, rotation theta, the support function in direction phi is:
        #   s * max(0, cos(phi - theta - k*pi/3)) for k=0..5? 
        # Actually, the support function of a regular hexagon with circumradius R (R=s for side s? No, R=s)
        # Wait, for a regular hexagon with side length s, the distance from center to vertex is s.
        # The support function in direction phi is s * max_k cos(phi - (theta + k*pi/3))
        # But this is only correct if the max is positive; otherwise it's the apothem in that direction?
        # Actually, the support function is:
        #   h(phi) = s * max(0, cos(phi - theta - k*pi/3)) for k=0..5
        # But this isn't quite right. Let me think again.
        
        # For a regular hexagon with vertices at angles theta + k*pi/3, distance s from center:
        # The support function in direction phi is:
        #   h(phi) = s * max_k cos(phi - (theta + k*pi/3))
        # This is correct for the convex hull.
        
        min_L = np.inf
        for k in range(6):
            nk = alpha + (k + 0.5) * math.pi / 3.0
            max_val = -np.inf
            for cx, cy, s, theta_deg in inner_hexes:
                theta = math.radians(theta_deg)
                # Support function of inner hexagon in direction nk
                h = 0.0
                for j in range(6):
                    val = s * math.cos(nk - (theta + j * math.pi / 3.0))
                    if val > h:
                        h = val
                # The point (cx, cy) projected onto nk
                proj = (cx - ox) * math.cos(nk) + (cy - oy) * math.sin(nk)
                total = proj + h
                if total > max_val:
                    max_val = total
            # We need L * sqrt(3)/2 >= max_val
            L_required = max_val * 2.0 / math.sqrt(3.0)
            if L_required > min_L:
                min_L = L_required
        return max(0.0, min_L)
    
    # This approach is expensive. Let's simplify by fixing the outer rotation to 0
    # and only optimizing the inner positions and the outer center + side.
    
    # Actually, let's use a different strategy:
    # 1. Start with a good initial configuration
    # 2. Use local search to optimize
    
    # Let's try a simple local search:
    # - Perturb positions and rotations
    # - Compute the enclosing hexagon side
    # - Accept if improved
    
    best_inner = [list(x) for x in inner]
    best_outer = (0.0, 0.0)
    best_side = np.inf
    best_rot = 0.0
    
    # Compute initial side
    inner_hexes = [(x, y, 1.0, a) for x, y, a in best_inner]
    best_side = enclosing_hexagon_side(inner_hexes, best_outer, best_rot)
    
    # Local search
    np.random.seed(42)
    step_size = 0.5
    iters = 0
    
    while time.time() < deadline - 5.0:
        iters += 1
        # Perturb a random inner hexagon
        idx = np.random.randint(N)
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dtheta = np.random.uniform(-15, 15)
        
        new_inner = [list(x) for x in best_inner]
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] = (new_inner[idx][2] + dtheta) % 360.0
        
        # Also perturb outer center slightly
        dox = np.random.uniform(-0.2, 0.2)
        doy = np.random.uniform(-0.2, 0.2)
        new_outer = (best_outer[0] + dox, best_outer[1] + doy)
        
        new_hexes = [(x, y, 1.0, a) for x, y, a in new_inner]
        new_side = enclosing_hexagon_side(new_hexes, new_outer, best_rot)
        
        if new_side < best_side - 1e-6:
            best_inner = new_inner
            best_outer = new_outer
            best_side = new_side
            # Reduce step size slightly
            step_size = max(0.05, step_size * 0.98)
        else:
            step_size = min(1.0, step_size * 1.02)
    
    # Also try optimizing outer rotation
    for alpha_deg in [0, 30, 60, 90, 120, 150]:
        inner_hexes = [(x, y, 1.0, a) for x, y, a in best_inner]
        s = enclosing_hexagon_side(inner_hexes, best_outer, alpha_deg)
        if s < best_side:
            best_side = s
            best_rot = alpha_deg
    
    return best_inner, best_outer, best_side, best_rot
# EVOLVE-BLOCK-END
