import math
import time
import numpy as np
from scipy.optimize import minimize
import random

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit

    # Pre-compute hexagon geometry constants
    SQRT3 = math.sqrt(3)
    HEX_VERTS_ANGLES = np.array([0, 60, 120, 180, 240, 300]) * math.pi / 180
    # Unit hexagon vertices (side 1, circumradius 1)
    HEX_VERTS = np.vstack([np.cos(HEX_VERTS_ANGLES), np.sin(HEX_VERTS_ANGLES)]).T
    
    def get_hex_verts(center, angle_rad, side=1.0):
        """
        Returns vertices of a regular hexagon centered at `center` with rotation `angle_rad`.
        Vertices are (6, 2) array.
        """
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return center + (HEX_VERTS * side) @ R.T

    def outer_penalty(params, inner_params):
        """
        Calculate penalty for outer hexagon containing inner hexagons.
        params: (cx, cy, side, angle_rad) for outer hex.
        inner_params: list of (cx_i, cy_i, angle_i) for inner hexes.
        """
        cx, cy, side, o_angle = params
        apothem = side * SQRT3 / 2
        
        # Normals for outer hexagon in local frame
        # The outer hexagon is aligned with angle o_angle.
        # Its normals are at o_angle + 30, o_angle + 90, o_angle + 150 degrees.
        # We check if any vertex of any inner hexagon is outside the outer hexagon.
        # A point (x,y) is inside a hexagon centered at (cx,cy) with rotation phi and apothem A
        # if for all k in 0..2:
        #   proj_k = (x-cx)*cos(n_k) + (y-cy)*sin(n_k) <= A
        #   where n_k = phi + 30 + k*60 degrees.
        
        total_pen = 0.0
        # To speed up, we can vectorize over inner hexes, but N is small.
        # We must check all 6 vertices of each inner hex.
        
        for i in range(N):
            icx, icy, i_angle = inner_params[i]
            # Get vertices of inner hex i
            verts = get_hex_verts(np.array([icx, icy]), i_angle)
            
            # Transform vertices to outer hex local frame
            dx = verts[:, 0] - cx
            dy = verts[:, 1] - cy
            
            c_o, s_o = np.cos(-o_angle), np.sin(-o_angle)
            # Rotate points by -o_angle to align outer hex with standard orientation
            rx = dx * c_o - dy * s_o
            ry = dx * s_o + dy * c_o
            
            # Standard outer hex normals (relative to aligned hex): 30, 90, 150 deg
            # n_0: 30 deg -> (sqrt(3)/2, 1/2)
            # n_1: 90 deg -> (0, 1)
            # n_2: 150 deg -> (-sqrt(3)/2, 1/2)
            
            p0 = rx * (SQRT3/2) + ry * 0.5
            p1 = ry
            p2 = -rx * (SQRT3/2) + ry * 0.5
            
            # Check violations
            # proj <= apothem and proj >= -apothem
            v0 = np.maximum(0, p0 - apothem) + np.maximum(0, -apothem - p0)
            v1 = np.maximum(0, p1 - apothem) + np.maximum(0, -apothem - p1)
            v2 = np.maximum(0, p2 - apothem) + np.maximum(0, -apothem - p2)
            
            total_pen += np.sum(v0**2) + np.sum(v1**2) + np.sum(v2**2)
            
        return total_pen

    def pairwise_penalty(inner_params):
        """
        Calculate penalty for non-overlapping inner hexagons.
        inner_params: list of (cx_i, cy_i, angle_i).
        Uses Minkowski sum approximation or direct vertex-edge checks.
        For differentiability, we use a smooth penalty based on distance between closest points?
        Actually, a common trick is to check if any vertex of hex A is inside hex B or vice versa,
        or if edges intersect. This is non-smooth.
        
        Alternative: Use the fact that two convex polygons do not intersect if there is a separating axis.
        However, for optimization, we can use a "soft" penalty.
        
        Let's use a simpler, robust differentiable penalty:
        For each pair (i, j), check the distance between their boundaries.
        If they overlap, the distance is negative.
        Approximation: Check if the center of one is inside the other? No, they can overlap without centers inside.
        
        Better approach for hexagons:
        The condition for non-intersection of two convex polygons can be formulated via linear inequalities if we fix the relative orientation? No.
        
        Let's use the "point in polygon" penalty for vertices of one against the other.
        If vertex v of A is inside B, penalty += (distance to boundary)^2.
        If vertex v of B is inside A, penalty += (distance to boundary)^2.
        This misses edge-edge intersections where no vertices are inside.
        However, for regular hexagons, if they intersect, either a vertex is inside or edges cross.
        Edge crossing penalty is hard to make smooth.
        
        Let's rely on the fact that if we start with non-overlapping configurations (e.g., lattice),
        and the penalty grows as they get closer, we might maintain separation.
        But we need a penalty that is 0 when separated and >0 when overlapping.
        
        Let's implement a penalty based on the minimum distance between the two hexagons.
        If min_dist < 0 (overlap), penalty = (-min_dist)^2.
        Calculating min_dist between two convex polygons is possible but slightly expensive.
        
        Given N=11, pairs = 55.
        We can use a simplified check:
        If the distance between centers is less than the sum of circumradii (2.0), they *might* overlap.
        If distance > 2.0, they definitely don't.
        If distance < sqrt(3) (sum of apothems), they definitely overlap? No, depends on angle.
        
        Let's use a grid-free, continuous check:
        For each pair, compute the vertices of both.
        Check if any vertex of A is inside B.
        Check if any vertex of B is inside A.
        If yes, add penalty proportional to the depth of penetration.
        
        To handle edge crossings without vertices inside:
        This is the tricky part. However, in practice, for optimization starting from good initial guesses,
        vertex-inside checks often suffice to push them apart if they are close.
        To be safe, we can add a term based on the distance between centers if they are very close.
        
        Let's try:
        Penalty = sum over pairs of (max(0, -min_dist_approx))^2
        
        Approximation for min_dist:
        If centers are far, 0.
        If centers are close, we can check the 6x6 vertex-vertex distances? No, that's not boundary distance.
        
        Let's stick to the vertex-in-polygon penalty. It is differentiable almost everywhere.
        """
        total_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                cx1, cy1, a1 = inner_params[i]
                cx2, cy2, a2 = inner_params[j]
                
                v1 = get_hex_verts(np.array([cx1, cy1]), a1)
                v2 = get_hex_verts(np.array([cx2, cy2]), a2)
                
                # Check v1 inside v2
                # Point in hexagon test:
                # Transform point to local frame of hexagon.
                # Hexagon at origin, angle 0.
                # Normals at 30, 90, 150.
                # Apothem = SQRT3/2.
                
                def point_in_hex_penalty(pts, center, angle):
                    """
                    Returns penalty for points `pts` being inside hexagon at `center` with `angle`.
                    """
                    dx = pts[:, 0] - center[0]
                    dy = pts[:, 1] - center[1]
                    
                    c, s = np.cos(-angle), np.sin(-angle)
                    rx = dx * c - dy * s
                    ry = dx * s + dy * c
                    
                    # Local frame of standard hex (angle 0)
                    # Normals: 30, 90, 150 deg
                    p0 = rx * (SQRT3/2) + ry * 0.5
                    p1 = ry
                    p2 = -rx * (SQRT3/2) + ry * 0.5
                    
                    apothem = SQRT3 / 2
                    
                    # Violation: proj > apothem or proj < -apothem
                    # But we want penalty if INSIDE.
                    # Wait, if a point is INSIDE, it satisfies |proj| <= apothem for all normals.
                    # So "inside" is the feasible region for containment.
                    # For non-overlap, we want NO point of A inside B.
                    # So if a point IS inside, we need a penalty.
                    # How to measure "depth" inside?
                    # The distance to the boundary is min(apothem - |proj_k|).
                    # If min_dist > 0, it is inside.
                    # Penalty = (min_dist)^2?
                    # Actually, if it's inside, min_dist is positive.
                    # We want to push it out.
                    # So penalty = (min_dist)^2 if min_dist > 0.
                    
                    dists = np.minimum(
                        np.minimum(apothem - np.abs(p0), apothem - np.abs(p1)),
                        apothem - np.abs(p2)
                    )
                    
                    # If dists > 0, point is inside.
                    inside_mask = dists > 0
                    if np.any(inside_mask):
                        return np.sum(dists[inside_mask]**2)
                    return 0.0

                pen1 = point_in_hex_penalty(v1, np.array([cx2, cy2]), a2)
                pen2 = point_in_hex_penalty(v2, np.array([cx1, cy1]), a1)
                
                # Add a small penalty if centers are too close to handle edge-edge crossings
                # where no vertices are inside.
                d_centers = math.hypot(cx1-cx2, cy1-cy2)
                if d_centers < 1.8: # Circumradius sum is 2.0, but they can touch at 1.732 (apothem sum)
                    # If they are very close, add penalty based on distance to minimum separation
                    # Minimum separation for aligned hexes is sqrt(3).
                    # For arbitrary angles, it can be less.
                    # Let's just add a soft repulsion if d_centers < 1.5
                    if d_centers < 1.5:
                        total_pen += (1.5 - d_centers)**2 * 10.0 # Strong repulsion

                total_pen += pen1 + pen2
                
        return total_pen

    def objective(x):
        """
        x: [cx1, cy1, a1, ..., cx11, cy11, a11, ocx, ocy, oside, oang]
        """
        inner_params = []
        for i in range(N):
            cx = x[i*3]
            cy = x[i*3+1]
            a = x[i*3+2]
            inner_params.append((cx, cy, a))
            
        ocx = x[N*3]
        ocy = x[N*3+1]
        oside = x[N*3+2]
        oang = x[N*3+3]
        
        p_out = outer_penalty((ocx, ocy, oside, oang), inner_params)
        p_in = pairwise_penalty(inner_params)
        
        # Weighted sum
        # We want to minimize oside, but penalize violations heavily.
        # If there are violations, the objective is large.
        # If no violations, we want to minimize oside.
        
        # Heuristic:
        # If p_out + p_in > 0, we are in violation.
        # Objective = oside + 1000 * (p_out + p_in)
        
        return oside + 1000.0 * (p_out + p_in)

    # Initial guesses
    # Strategy: Start with a hexagonal lattice arrangement.
    # Center at (0,0).
    # Neighbors at distance sqrt(3).
    
    def generate_initial_guess(seed):
        random.seed(seed)
        x0 = []
        
        # Place N hexagons in a compact cluster
        # Use a "spiral" or "lattice" pattern
        # Let's use a lattice with slight random perturbations
        
        # Hex lattice vectors
        v1 = np.array([SQRT3, 0])
        v2 = np.array([SQRT3/2, 1.5])
        
        # Generate lattice points
        points = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i*v1 + j*v2
                points.append(p)
        
        # Sort by distance from origin
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # Take first N
        for k in range(N):
            cx, cy = points[k]
            # Add small random noise
            cx += random.uniform(-0.1, 0.1)
            cy += random.uniform(-0.1, 0.1)
            a = random.uniform(0, 2*math.pi)
            x0.extend([cx, cy, a])
            
        # Outer hexagon initial guess
        # Enclose all points
        max_r = max(math.hypot(p[0], p[1]) for p in points[:N])
        oside = max_r + 1.0
        ocx = 0.0
        ocy = 0.0
        oang = 0.0
        
        x0.extend([ocx, ocy, oside, oang])
        
        return np.array(x0)

    best_result = None
    best_score = float('inf')
    
    # Try multiple restarts with different seeds
    for seed in range(20):
        if time.time() > deadline:
            break
            
        x0 = generate_initial_guess(seed)
        
        # Bounds
        # Inner centers: [-10, 10]
        # Inner angles: [0, 2*pi]
        # Outer center: [-10, 10]
        # Outer side: [1.0, 20.0]
        # Outer angle: [0, pi/3] (symmetry)
        
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))
            bounds.append((-10, 10))
            bounds.append((0, 2*math.pi))
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((1.0, 20.0))
        bounds.append((0, math.pi/3))
        
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6})
            
            if res.success or res.fun < best_score:
                # Extract solution
                inner_rows = []
                for i in range(N):
                    cx = res.x[i*3]
                    cy = res.x[i*3+1]
                    a_rad = res.x[i*3+2]
                    a_deg = math.degrees(a_rad) % 360
                    inner_rows.append([cx, cy, a_deg])
                    
                ocx = res.x[N*3]
                ocy = res.x[N*3+1]
                oside = res.x[N*3+2]
                oang_deg = math.degrees(res.x[N*3+3])
                
                # Verify validity
                # Check for overlaps and containment
                valid = True
                inner_params = [(r[0], r[1], math.radians(r[2])) for r in inner_rows]
                
                # Check pairwise
                p_in = pairwise_penalty(inner_params)
                if p_in > 1e-6:
                    valid = False
                else:
                    # Check outer
                    p_out = outer_penalty((ocx, ocy, oside, oang_deg), inner_params)
                    if p_out > 1e-6:
                        valid = False
                
                if valid:
                    if oside < best_score:
                        best_score = oside
                        best_result = (inner_rows, (ocx, ocy), oside, oang_deg)
                        
        except Exception as e:
            pass # Continue with next seed

    # Fallback if no valid result found
    if best_result is None:
        # Simple linear arrangement
        inner_rows = []
        for i in range(N):
            inner_rows.append([i * SQRT3, 0.0, 0.0])
        
        # Calculate outer side
        max_x = (N-1) * SQRT3
        min_x = 0
        cx = max_x / 2
        cy = 0
        # Width is max_x. Height is SQRT3.
        # Side length needs to cover width.
        # For a hexagon, width = 2 * side * cos(30) = side * SQRT3.
        # So side = width / SQRT3 = max_x.
        oside = max_x + 1.0
        best_result = (inner_rows, (cx, cy), oside, 0.0)

    return best_result

if __name__ == "__main__":
    r = construct()
    print(r)
