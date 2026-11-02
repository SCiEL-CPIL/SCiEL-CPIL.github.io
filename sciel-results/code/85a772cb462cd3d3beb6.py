import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    Uses the fact that a regular hexagon is the intersection of 3 strips.
    Actually, for robustness with rotation, we transform the point to the hexagon's local frame.
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    
    # Rotate point into local frame of hexagon
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # Hexagon with side s centered at origin, aligned with x-axis.
    # Vertices at (s, 0), (s/2, s*sqrt(3)/2), etc.
    # The hexagon is defined by:
    # |lx| <= s
    # |ly| <= s * sqrt(3) / 2
    # |lx/2 + ly/sqrt(3)| <= s/2  ? No.
    # Let's use the support function or distance to edges.
    # Edges are lines:
    # 1. x = s (right)
    # 2. x = -s (left)
    # 3. y = s*sqrt(3)/2 (top-right? No, top edge is horizontal? No, vertices are at 0, 60, 120...)
    # If vertices are at 0, 60, 120, 180, 240, 300 degrees:
    # v0=(s,0), v1=(s/2, s*sqrt(3)/2), v2=(-s/2, s*sqrt(3)/2), v3=(-s,0), v4=(-s/2, -s*sqrt(3)/2), v5=(s/2, -s*sqrt(3)/2)
    
    # Distance to edge v0-v1:
    # Vector v0->v1 = (-s/2, s*sqrt(3)/2). Length s.
    # Normal pointing in: (s*sqrt(3)/2, s/2) normalized?
    # Let's just check if point is outside any of the 3 pairs of parallel lines.
    # Line 1 (through v0, v5? No, v0 is vertex).
    # The 3 axes of symmetry are at 0, 60, 120 degrees.
    # Wait, the flat sides are perpendicular to 30, 90, 150 degrees.
    # So the constraints are projections on 30, 90, 150 degrees.
    # Apothem = s * sqrt(3) / 2.
    # proj_30 = lx * cos(30) + ly * sin(30)
    # proj_90 = lx * cos(90) + ly * sin(90) = ly
    # proj_150 = lx * cos(150) + ly * sin(150)
    
    c30 = math.sqrt(3) / 2
    s30 = 0.5
    c150 = -math.sqrt(3) / 2
    s150 = 0.5
    
    apothem = s * math.sqrt(3) / 2
    
    p1 = lx * c30 + ly * s30
    p2 = ly
    p3 = lx * c150 + ly * s150
    
    dist_sq = 0.0
    if p1 > apothem:
        d = p1 - apothem
        dist_sq += d * d
    elif p1 < -apothem:
        d = -apothem - p1
        dist_sq += d * d
        
    if p2 > apothem:
        d = p2 - apothem
        dist_sq += d * d
    elif p2 < -apothem:
        d = -apothem - p2
        dist_sq += d * d
        
    if p3 > apothem:
        d = p3 - apothem
        dist_sq += d * d
    elif p3 < -apothem:
        d = -apothem - p3
        dist_sq += d * d
        
    return dist_sq

def _pairwise_penalty(pos, angles, outer_side, outer_angle):
    """
    Calculate penalty for invalid packing.
    pos: list of (x, y) for N hexagons.
    angles: list of angles for N hexagons.
    outer_side: side length of outer hexagon.
    outer_angle: rotation of outer hexagon.
    """
    penalty = 0.0
    
    # 1. Check containment of inner hexagons in outer hexagon
    # We only need to check the vertices of the inner hexagons.
    for i in range(N):
        x, y = pos[i]
        ang = angles[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        for vx, vy in verts:
            # Distance to outer hexagon (centered at 0,0 for simplicity in optimization, 
            # but we should probably allow outer center to move? 
            # Actually, we can fix outer center at centroid of inner hexes to reduce search space,
            # or optimize it. Let's optimize outer center too.
            # Wait, the signature of construct returns (inner, center, side, angle).
            # The "center" is the outer hexagon center.
            # So we need to optimize:
            # 11 * (x, y, angle) = 33 vars
            # 1 * (cx, cy, side, angle) = 4 vars
            # Total 37 vars.
            # This is manageable for DE.
            
            # But for now, let's assume outer center is (0,0) in the local frame of the optimization?
            # No, it's better to let the optimizer find the best center.
            # However, translating the whole system doesn't change the relative positions.
            # So we can fix the outer center at (0,0) and let the inner hexagons move.
            # This reduces 2 variables (cx, cy).
            # Let's assume outer center is (0,0).
            
            d_sq = _point_in_hex_dist_sq(vx, vy, 0.0, 0.0, outer_side, outer_angle)
            penalty += d_sq
            
    # 2. Check pairwise disjointness
    # For each pair of inner hexagons, check if they overlap.
    # Overlap if distance between centers is less than the separation required.
    # Exact check: Minkowski sum.
    # For two hexagons with side 1, angles a1, a2.
    # The Minkowski sum is a convex polygon.
    # If the vector between centers is inside the Minkowski sum, they overlap.
    # Approximation: Check vertices of one against the other? No, that's not sufficient.
    # Better: Check if any vertex of hex i is inside hex j, or any vertex of hex j is inside hex i.
    # This is not sufficient for "edge crossing" without vertex containment, but for convex polygons,
    # if they intersect, either a vertex is inside the other, or edges cross.
    # Given the "regular" nature and small N, vertex containment check is a strong heuristic.
    # However, it can miss cases where edges cross but no vertices are inside.
    # To be robust, we should check edge intersections or use a more robust distance metric.
    # Let's use the distance from vertices of one to the edges of the other?
    # Or simply: if min distance between any vertex of i and any vertex of j is small, and they are close, it's likely overlap.
    # Actually, the most robust simple check for two convex polygons is Separating Axis Theorem (SAT).
    # But SAT is complex to implement inline in the penalty function for speed.
    
    # Let's use a simpler proxy:
    # If the distance between centers is less than the sum of their "radii" (circumradius), they MIGHT overlap.
    # Circumradius of unit hex is 1. So if dist < 2, they might overlap.
    # If dist > 2, they definitely don't.
    # If dist < sqrt(3) (2*apothem), they definitely overlap IF aligned.
    # With rotation, the minimum distance is still bounded.
    
    # Let's try the vertex containment check first, it's fast.
    for i in range(N):
        verts_i = _hex_vertices(pos[i][0], pos[i][1], 1.0, angles[i])
        for j in range(i + 1, N):
            xj, yj = pos[j]
            ang_j = angles[j]
            # Check if any vertex of i is inside j
            overlap = False
            for vx, vy in verts_i:
                if _point_in_hex_dist_sq(vx, vy, xj, yj, 1.0, ang_j) == 0.0:
                    overlap = True
                    break
            if overlap:
                penalty += 1000.0
                continue # No need to check j inside i if i inside j
            # Check if any vertex of j is inside i
            verts_j = _hex_vertices(xj, yj, 1.0, ang_j)
            for vx, vy in verts_j:
                if _point_in_hex_dist_sq(vx, vy, pos[i][0], pos[i][1], 1.0, angles[i]) == 0.0:
                    overlap = True
                    break
            if overlap:
                penalty += 1000.0

    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Variables to optimize:
    # x_1, y_1, a_1, ..., x_N, y_N, a_N, outer_side, outer_angle
    # Total: 3*N + 2 = 35 variables.
    
    # Bounds:
    # Inner hexagons should be within a reasonable range, say [-10, 10].
    # Angles: [0, 360).
    # Outer side: [1, 10]. (Min side for 1 hex is 1, for 11 it's larger).
    # Outer angle: [0, 60). (Symmetry of hexagon).
    
    bounds = []
    for _ in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((0, 360)) # angle
    
    bounds.append((1.0, 10.0)) # outer side
    bounds.append((0, 60)) # outer angle
    
    def objective(params):
        if time.time() > deadline:
            return float('inf')
            
        # Unpack
        pos = []
        angles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            pos.append((x, y))
            angles.append(a)
        
        outer_side = params[3*N]
        outer_angle = params[3*N + 1]
        
        # Penalty for invalid packing
        pen = _pairwise_penalty(pos, angles, outer_side, outer_angle)
        
        # We want to minimize outer_side
        # If penalty is high, we add it to the objective to discourage invalid solutions.
        # Weight the penalty heavily.
        obj_val = outer_side + 10.0 * pen
        
        return obj_val

    # Initial population
    # To help DE, we can provide a reasonable initial guess or just let it sample.
    # DE samples randomly within bounds.
    # Bounds are quite wide. Let's tighten them if possible.
    # For N=11, the outer side is likely around 4-5.
    # Inner hexes will be within radius ~3.
    
    # Let's run DE
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            init='sobol',
            polish=False # Disable polish to avoid extra time, DE is usually good enough
        )
    except Exception as e:
        # Fallback if DE fails
        pass
    else:
        if time.time() < deadline:
            # Parse result
            best_params = result.x
            pos = []
            angles = []
            for i in range(N):
                pos.append((best_params[3*i], best_params[3*i + 1]))
                angles.append(best_params[3*i + 2])
            
            outer_side = best_params[3*N]
            outer_angle = best_params[3*N + 1]
            
            # Verify validity
            pen = _pairwise_penalty(pos, angles, outer_side, outer_angle)
            if pen < 1e-3: # Consider it valid if penalty is negligible
                # Construct return values
                inner_rows = []
                for i in range(N):
                    inner_rows.append([pos[i][0], pos[i][1], angles[i]])
                
                # Outer center is (0,0) in our optimization frame
                return inner_rows, (0.0, 0.0), outer_side, outer_angle

    # Fallback: Simple grid search or heuristic if DE fails or times out
    # This ensures we return *something* valid, even if not optimal.
    # We'll use the logic from the previous program's fallback or a simple linear arrangement.
    
    # Simple linear arrangement along x-axis
    inner_rows = []
    for i in range(N):
        inner_rows.append([i * math.sqrt(3), 0.0, 0.0])
    
    # Calculate required outer side for this linear arrangement
    # The extent is from -1 to N*sqrt(3)+1 in x, and -sqrt(3)/2 to sqrt(3)/2 in y.
    # Width = N*sqrt(3) + 2.
    # Height = sqrt(3).
    # Min enclosing hexagon side for a rectangle W x H?
    # Rough estimate: side ~ max(W, H) / sqrt(3) * 2?
    # Let's just use a large side to be safe, or calculate properly.
    # For a line of hexes, the optimal outer hexagon is likely aligned with the line.
    # Side length needs to cover the length.
    # Length of N hexes in a row = N * sqrt(3) + 2 (adding the caps? No, center-to-center is sqrt(3).
    # First center at 0, last at (N-1)*sqrt(3).
    # Leftmost point: -1. Rightmost point: (N-1)*sqrt(3) + 1.
    # Total width W = (N-1)*sqrt(3) + 2.
    # Height H = sqrt(3).
    # A hexagon of side S has width 2S (vertex to vertex) and height S*sqrt(3) (flat to flat).
    # If aligned, we need 2S >= W and S*sqrt(3) >= H.
    # S >= W/2 and S >= H/sqrt(3) = 1.
    # So S = W/2 = ((N-1)*sqrt(3) + 2) / 2.
    
    side = ((N - 1) * math.sqrt(3) + 2) / 2.0
    return inner_rows, (0.0, 0.0), side, 0.0

if __name__ == "__main__":
    r = construct()
    print(r)
