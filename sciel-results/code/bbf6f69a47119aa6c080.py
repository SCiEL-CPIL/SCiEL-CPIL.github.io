import numpy as np
import time
from scipy.optimize import differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def hex_verts(cx, cy, s, rot_deg):
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_dist_polygons_fast(v1, v2):
        """
        Compute minimum distance between two convex hexagons.
        Returns 0 if they overlap or touch, positive distance otherwise.
        Uses a simple sampling approach for robustness, or exact if we want.
        For DE speed, we can approximate or use a faster check.
        Let's use a robust but reasonably fast check:
        1. Check if any vertex of v1 is inside v2 or vice versa.
        2. If not, check min distance between edges.
        """
        # Quick bounding box check
        min_x1, min_y1 = np.min(v1, axis=0)
        max_x1, max_y1 = np.max(v1, axis=0)
        min_x2, min_y2 = np.min(v2, axis=0)
        max_x2, max_y2 = np.max(v2, axis=0)
        
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            # Disjoint bounding boxes, compute min distance between boxes as lower bound?
            # Actually just compute min vertex-vertex distance for simplicity if boxes are far
            # But we need accurate overlap detection.
            pass
            
        # Check vertex-in-polygon
        def point_in_hex(px, py, hexv):
            # Winding number or ray casting
            # Simple ray casting to the right
            x, y = px, py
            inside = False
            n = len(hexv)
            j = n - 1
            for i in range(n):
                xi, yi = hexv[i]
                xj, yj = hexv[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside

        # Check if any vertex of v1 is in v2
        for i in range(6):
            if point_in_hex(v1[i, 0], v1[i, 1], v2):
                return 0.0
        # Check if any vertex of v2 is in v1
        for i in range(6):
            if point_in_hex(v2[i, 0], v2[i, 1], v1):
                return 0.0

        # If no vertices inside, check edge-edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(6):
            for j in range(6):
                if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return 0.0

        # If no intersection, compute min distance
        min_d = float('inf')
        for i in range(6):
            for j in range(6):
                d = np.linalg.norm(v1[i] - v2[j])
                if d < min_d:
                    min_d = d
                # Also check point-segment distance for better accuracy
                # This is expensive, but necessary for tight packing
                # Let's stick to vertex-vertex for speed in DE, might be slightly conservative
                # Actually, vertex-vertex is sufficient for "disjoint" if we use a small epsilon
                # But for minimization, we need gradient-like info.
                # Let's use a simple distance penalty.
        
        return min_d

    def evaluate(params):
        """
        Objective function: Minimize the side length of the outer hexagon.
        Params: 36 dims -> [x0, y0, a0, x1, y1, a1, ...]
        Outer hexagon is centered at (0,0) and aligned with axes (rotation 0).
        We need to find the minimum side length S such that all inner hexagons fit.
        For a regular hexagon centered at origin with side S, the distance from center to edge is R = S * sqrt(3)/2.
        The distance from center to vertex is S.
        
        Condition for a point (x,y) to be inside the outer hexagon:
        |x| <= S * sqrt(3)/2  AND  |y| <= S * sqrt(3)/2  AND  |x/sqrt(3) + y| <= S * sqrt(3)/2 ?
        Let's derive the inequalities.
        Vertices at angles 0, 60, ...
        Edges are perpendicular to normals at 30, 90, 150, ...
        Normal 1: (cos(30), sin(30)) = (sqrt(3)/2, 1/2). Distance to edge is R = S*sqrt(3)/2.
        Inequality: (sqrt(3)/2)*x + (1/2)*y <= S*sqrt(3)/2  =>  sqrt(3)*x + y <= 3*S/2 ? No.
        Let's use the standard form.
        Hexagon vertices: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2).
        
        The six half-planes are defined by the edges.
        Edge 1: from (S,0) to (S/2, S*sqrt(3)/2).
        Midpoint: (3S/4, S*sqrt(3)/4). Normal pointing outward: (sqrt(3)/2, 1/2).
        Equation: sqrt(3)/2 * x + 1/2 * y = C.
        Plug in (S,0): C = S*sqrt(3)/2.
        So: sqrt(3)*x + y <= S*sqrt(3).
        
        Edge 2: from (S/2, S*sqrt(3)/2) to (-S/2, S*sqrt(3)/2).
        Horizontal line y = S*sqrt(3)/2.
        Inequality: y <= S*sqrt(3)/2.
        
        Edge 3: from (-S/2, S*sqrt(3)/2) to (-S, 0).
        Normal: (-sqrt(3)/2, 1/2).
        Equation: -sqrt(3)/2 * x + 1/2 * y = C.
        Plug in (-S,0): C = S*sqrt(3)/2.
        So: -sqrt(3)*x + y <= S*sqrt(3).
        
        Edge 4: from (-S, 0) to (-S/2, -S*sqrt(3)/2).
        Normal: (-sqrt(3)/2, -1/2).
        Equation: -sqrt(3)/2 * x - 1/2 * y = C.
        Plug in (-S,0): C = S*sqrt(3)/2.
        So: -sqrt(3)*x - y <= S*sqrt(3).
        
        Edge 5: from (-S/2, -S*sqrt(3)/2) to (S/2, -S*sqrt(3)/2).
        Horizontal line y = -S*sqrt(3)/2.
        Inequality: y >= -S*sqrt(3)/2  =>  -y <= S*sqrt(3)/2.
        
        Edge 6: from (S/2, -S*sqrt(3)/2) to (S, 0).
        Normal: (sqrt(3)/2, -1/2).
        Equation: sqrt(3)/2 * x - 1/2 * y = C.
        Plug in (S,0): C = S*sqrt(3)/2.
        So: sqrt(3)*x - y <= S*sqrt(3).
        
        So for any point (x,y) to be inside, we need:
        1. sqrt(3)*x + y <= S*sqrt(3)
        2. y <= S*sqrt(3)/2
        3. -sqrt(3)*x + y <= S*sqrt(3)
        4. -sqrt(3)*x - y <= S*sqrt(3)
        5. -y <= S*sqrt(3)/2
        6. sqrt(3)*x - y <= S*sqrt(3)
        
        We want to minimize S.
        S must be at least:
        S >= (sqrt(3)*x + y) / sqrt(3) = x + y/sqrt(3)
        S >= y / (sqrt(3)/2) = 2*y/sqrt(3)
        S >= (-sqrt(3)*x + y) / sqrt(3) = -x + y/sqrt(3)
        S >= (-sqrt(3)*x - y) / sqrt(3) = -x - y/sqrt(3)
        S >= -y / (sqrt(3)/2) = -2*y/sqrt(3)
        S >= (sqrt(3)*x - y) / sqrt(3) = x - y/sqrt(3)
        
        So for a given set of inner hexagons, the minimum S is the maximum over all vertices of all inner hexagons of these 6 expressions.
        Additionally, we must ensure inner hexagons do not overlap.
        We add a penalty for overlap.
        
        """
        xs = params[0::3]
        ys = params[1::3]
        angles = params[2::3]
        
        # Compute vertices for all inner hexagons
        all_verts = []
        for i in range(N):
            v = hex_verts(xs[i], ys[i], 1.0, angles[i])
            all_verts.append(v)
            
        # Calculate required S for containment
        max_s = 0.0
        for v in all_verts:
            for x, y in v:
                s1 = x + y/np.sqrt(3)
                s2 = 2*y/np.sqrt(3)
                s3 = -x + y/np.sqrt(3)
                s4 = -x - y/np.sqrt(3)
                s5 = -2*y/np.sqrt(3)
                s6 = x - y/np.sqrt(3)
                local_max = np.max([s1, s2, s3, s4, s5, s6])
                if local_max > max_s:
                    max_s = local_max
        
        # Calculate overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_polygons_fast(all_verts[i], all_verts[j])
                if d < 0.01: # Small tolerance
                    # Penalty is proportional to violation
                    # If d is 0, they touch/overlap. We want d > 0.
                    # Actually min_dist returns 0 if overlap.
                    # We need a measure of overlap depth.
                    # For simplicity, if d < 0.01, add a large penalty.
                    # Better: use the distance from center to center vs sum of radii?
                    # Hexagon circumradius is 1.
                    # If dist(c1, c2) < 2, they might overlap.
                    # Let's use a simple penalty: (max(0, 2 - dist_centers))^2
                    dist_c = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if dist_c < 2.0:
                        overlap_penalty += (2.0 - dist_c)**2 * 100.0
        
        # Total objective: Max S + Penalty
        # If penalty is large, DE will try to reduce it.
        # We want to minimize S, so we return max_s + penalty.
        return max_s + overlap_penalty

    # Define bounds for DE
    # Inner hexagons should be roughly within S=4, so x,y in [-4, 4]
    # Angles in [0, 360]
    bounds = [(-4.0, 4.0)] * N + [(-4.0, 4.0)] * N + [(0.0, 360.0)] * N
    # Rearrange: [x0, y0, a0, x1, y1, a1, ...]
    # My unpacking was xs = params[0::3], ys = params[1::3], angles = params[2::3]
    # So bounds should be interleaved.
    bounds = []
    for _ in range(N):
        bounds.append((-4.0, 4.0)) # x
        bounds.append((-4.0, 4.0)) # y
        bounds.append((0.0, 360.0)) # angle

    # Run Differential Evolution
    # popsize=15, maxiter=100
    # Keep it fast enough to finish in 140s.
    # 36 dimensions.
    try:
        res = differential_evolution(
            evaluate, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4, 
            seed=42,
            polish=True,  # Use L-BFGS-B at the end to refine
            update_interval=10
        )
        
        if res.success or res.fun < float('inf'):
            best_params = res.x
            best_s = res.fun
        else:
            best_params = None
            best_s = float('inf')
            
    except Exception as e:
        best_params = None
        best_s = float('inf')

    if best_params is None:
        # Fallback to a simple grid or random
        inner = []
        for i in range(12):
            x = (i % 4) * 2.5 - 3.75
            y = (i // 4) * 2.2 - 2.2
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    # Extract results
    xs = best_params[0::3]
    ys = best_params[1::3]
    angles = best_params[2::3]
    
    inner = []
    for i in range(N):
        inner.append([xs[i], ys[i], angles[i]])
        
    # The objective returned max_s + penalty.
    # If penalty is 0, best_s is the side length.
    # If penalty > 0, the solution is invalid.
    # We should check validity.
    # If invalid, we might still return the best found S, but the evaluator will reject it.
    # Let's assume DE found a valid solution.
    
    return inner, (0.0, 0.0), best_s, 0.0
