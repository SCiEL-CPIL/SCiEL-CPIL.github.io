"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    """Place 12 disjoint unit regular hexagons inside a regular hexagon, minimizing outer side length."""
    deadline = time.time() + 135.0  # safety margin before 150s kill
    
    # A unit regular hexagon (side=1) has vertices at distance 1 from center.
    # Its circumradius = 1, inradius = sqrt(3)/2.
    # For two hexagons to be disjoint, the distance between centers must be at least
    # the sum of their "reach" in the direction of separation.
    
    # For a regular hexagon with side s=1, the support function in direction u is:
    # h(u) = max over vertices of v·u
    # For a unit hexagon centered at origin with angle 0 (flat-top or pointy-top):
    # Let's use pointy-top: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Actually, let's define hexagon with vertices at angles offset by rotation.
    
    # The minimum distance between centers of two non-overlapping unit hexagons
    # depends on their relative orientation. The worst case (minimum separation)
    # is when they're aligned, giving distance = 2 (diameter along flat-to-flat is sqrt(3),
    # but vertex-to-vertex is 2).
    
    # Actually, for two identical regular hexagons of side 1:
    # - If aligned, minimum center distance for no overlap = 2 (vertex to vertex)
    #   Wait, no. The hexagon has circumradius 1. Two hexagons with centers distance d apart
    #   don't overlap if d >= 2 * circumradius = 2? No, that's for circles.
    
    # Let me think more carefully. A regular hexagon with side 1 has:
    # - Circumradius (center to vertex) = 1
    # - Inradius (center to edge midpoint) = sqrt(3)/2 ≈ 0.866
    
    # The support function of a unit hexagon in direction θ (with one vertex at angle 0):
    # h(θ) = cos(θ mod 60° - 30°) * sqrt(3)/2 ... no.
    
    # Let me just compute vertices and use a simpler approach.
    
    # For a unit hexagon centered at origin, rotated by angle α, the vertices are:
    # v_k = (cos(α + k*60°), sin(α + k*60°)) for k=0..5
    
    # Two convex polygons don't overlap iff there exists a separating axis.
    # For optimization, I'll use a smooth approximation of the overlap constraint.
    
    # Strategy: Use L-BFGS-B to optimize positions and angles of 12 hexagons
    # plus the outer hexagon's side length, with a smooth penalty for overlaps.
    
    # Variables: for each inner hex i: x_i, y_i, angle_i (12*3=36 vars)
    # Plus outer hexagon: center_x, center_y, side_length, angle (4 vars)
    # Total: 40 variables
    
    # Objective: minimize outer side length + penalty for inner overlaps + penalty for inner outside outer
    
    # Let me use a simpler approach: fix outer hexagon center at origin, angle at 0.
    # Optimize: 12 inner hexagons (x, y, angle) + outer side length s
    # Total: 36 + 1 = 37 variables
    
    def hex_vertices(cx, cy, s, angle_deg):
        """Return 6 vertices of regular hexagon centered at (cx,cy) with side s, rotated by angle_deg."""
        a = np.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * np.pi / 3
            verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
        return np.array(verts)
    
    def polygon_overlap_distance(v1, v2):
        """Compute minimum distance between two convex polygons (negative if overlapping).
        Returns a scalar: >0 means separated by that distance, <=0 means overlapping or touching."""
        # Use SAT (Separating Axis Theorem) to find minimum overlap
        # For smooth optimization, we need a differentiable version.
        
        # Get all edge normals from both polygons
        axes = []
        for v in [v1, v2]:
            for i in range(len(v)):
                edge = v[(i+1) % len(v)] - v[i]
                # Normal to edge
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-12:
                    axes.append(n / norm)
        
        min_overlap = float('inf')
        for axis in axes:
            # Project both polygons onto this axis
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Overlap in this axis direction
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
        
        # min_overlap > 0 means no separation found (overlapping)
        # min_overlap < 0 means separated by |min_overlap|
        # We want a penalty when min_overlap > 0 (overlap)
        return min_overlap
    
    def smooth_penalty(x):
        """Compute objective: outer side length + smooth penalties."""
        # Unpack variables
        # x[0:36] = inner hexagons (x, y, angle) * 12
        # x[36] = outer side length s
        s_outer = x[36]
        
        total_penalty = 0.0
        
        # Check inner-inner overlaps
        for i in range(N):
            xi, yi, ai = x[3*i], x[3*i+1], x[3*i+2]
            vi = hex_vertices(xi, yi, 1.0, ai)
            for j in range(i+1, N):
                xj, yj, aj = x[3*j], x[3*j+1], x[3*j+2]
                vj = hex_vertices(xj, yj, 1.0, aj)
                overlap = polygon_overlap_distance(vi, vj)
                # Smooth penalty: use softplus-like function
                if overlap > 0:
                    total_penalty += overlap * overlap
        
        # Check inner-outer containment
        # Outer hexagon centered at (0,0) with side s_outer, angle 0
        v_outer = hex_vertices(0.0, 0.0, s_outer, 0.0)
        
        # For containment, each inner hex vertex must be inside outer hex
        # Use distance from each inner vertex to outer hex boundary
        # A point is inside a convex polygon if it's on the correct side of all edges
        for i in range(N):
            xi, yi, ai = x[3*i], x[3*i+1], x[3*i+2]
            vi = hex_vertices(xi, yi, 1.0, ai)
            for v in vi:
                # Check if v is inside v_outer
                # For each edge of outer hex, check if v is on the inside
                max_violation = 0.0
                for k in range(6):
                    e1 = v_outer[k]
                    e2 = v_outer[(k+1) % 6]
                    edge = e2 - e1
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-12:
                        continue
                    n = n / norm
                    # Ensure normal points inward
                    center = np.array([0.0, 0.0])
                    if np.dot(n, center - e1) < 0:
                        n = -n
                    dist = np.dot(n, v - e1)
                    if dist < 0:
                        max_violation = max(max_violation, -dist)
                if max_violation > 0:
                    total_penalty += max_violation * max_violation
        
        return s_outer + 100.0 * total_penalty
    
    # Initial guess: place hexagons in a hexagonal grid pattern
    # A hexagonal packing of 12 unit hexagons...
    # Let's try a rough initial arrangement
    np.random.seed(42)
    x0 = np.zeros(37)
    
    # Place in a roughly hexagonal pattern
    positions = []
    # Center hexagon
    positions.append((0, 0, 0))
    # 6 around it
    for k in range(6):
        angle = k * 60
        r = 2.0  # minimum distance for non-overlap
        positions.append((r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle)), 0))
    # 5 more in outer ring
    for k in range(5):
        angle = 30 + k * 60
        r = 3.5
        positions.append((r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle)), 0))
    
    for i, (px, py, pa) in enumerate(positions):
        x0[3*i] = px
        x0[3*i+1] = py
        x0[3*i+2] = pa
    x0[36] = 5.0  # initial outer side length
    
    # Run L-BFGS-B optimization
    best_x = x0.copy()
    best_val = smooth_penalty(x0)
    
    try:
        result = minimize(
            smooth_penalty,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 50000, 'gtol': 1e-6}
        )
        
        if result.fun < best_val:
            best_x = result.x
            best_val = result.fun
        
        # Check time
        if time.time() < deadline - 30:
            # Try from a few different initializations
            for trial in range(3):
                if time.time() > deadline - 25:
                    break
                x0_trial = best_x + np.random.randn(37) * 0.1
                x0_trial[36] = max(x0_trial[36], 3.0)
                
                result2 = minimize(
                    smooth_penalty,
                    x0_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 20000, 'gtol': 1e-6}
                )
                
                if result2.fun < best_val:
                    best_x = result2.x
                    best_val = result2.fun
    except Exception:
        pass
    
    # Extract best solution
    inner = []
    for i in range(N):
        inner.append([float(best_x[3*i]), float(best_x[3*i+1]), float(best_x[3*i+2])])
    
    outer_side = float(max(best_x[36], 3.0))  # ensure reasonable minimum
    
    # Verify the solution is valid; if not, fall back to a known valid solution
    # Quick check: are all inner hexagons inside the outer?
    v_outer_check = hex_vertices(0.0, 0.0, outer_side, 0.0)
    valid = True
    for i in range(N):
        vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for v in vi:
            inside = True
            for k in range(6):
                e1 = v_outer_check[k]
                e2 = v_outer_check[(k+1) % 6]
                edge = e2 - e1
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12:
                    continue
                n = n / norm
                center = np.array([0.0, 0.0])
                if np.dot(n, center - e1) < 0:
                    n = -n
                dist = np.dot(n, v - e1)
                if dist < -1e-6:
                    inside = False
                    break
            if not inside:
                valid = False
                break
        if not valid:
            break
    
    # Also check inner-inner non-overlap
    if valid:
        for i in range(N):
            vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            for j in range(i+1, N):
                vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                overlap = polygon_overlap_distance(vi, vj)
                if overlap > 1e-4:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fall back to a simple valid solution: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        outer_side = 40.0
    
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
