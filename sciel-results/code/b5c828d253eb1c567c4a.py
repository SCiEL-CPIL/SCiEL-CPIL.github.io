import numpy as np
import time
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(p1, p2, eps=1e-5):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        Uses Separating Axis Theorem (SAT).
        """
        for i in range(len(p1)):
            j = (i + 1) % len(p1)
            # Edge normal of p1
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            
            # Project p1 and p2 onto normal
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        for i in range(len(p2)):
            j = (i + 1) % len(p2)
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    def check_validity(positions):
        """
        positions: (12, 2) array of (x, y) for unit hexagons (side=1, angle=0).
        Returns True if all are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint_sat(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def objective(params):
        """
        params: (24,) array of x, y for 12 hexagons.
        Returns outer hexagon side length + penalty if invalid.
        """
        if time.time() > deadline:
            return 1e10
            
        positions = params.reshape((N, 2))
        
        # Check disjointness
        if not check_validity(positions):
            return 1e10
            
        # For a regular hexagon centered at origin with side length S,
        # the distance from center to any vertex is S.
        # The distance from center to the middle of a side (apothem) is S * sqrt(3)/2.
        # To contain a point (x,y) in a regular hexagon (flat-top or pointy-top),
        # the required side length S is determined by the maximum "hexagonal radius".
        
        # A regular hexagon with side S and rotation 0 (flat-top) is defined by:
        # |x| <= S
        # |y| <= S * sqrt(3)/2
        # |x| + |y|/sqrt(3) <= S
        
        # Wait, standard regular hexagon with side S centered at origin:
        # If vertices are at (S, 0), (S/2, S*sqrt(3)/2), etc.
        # The constraints are:
        # 1. x <= S
        # 2. -x <= S
        # 3. sqrt(3)*y <= S*sqrt(3) -> y <= S
        # No, let's derive properly.
        
        # Vertices of regular hexagon side S, angle 0 (vertex at right):
        # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
        
        # The half-width (x-extent) is S.
        # The half-height (y-extent) is S * sqrt(3)/2.
        
        # However, the shape is not a rectangle.
        # The distance from origin to the boundary in direction theta is:
        # r(theta) = S * cos(pi/6) / cos(theta - pi/6) for theta in [-pi/6, pi/6] relative to vertex.
        
        # Simpler: The minimal side length S to contain a set of points is:
        # S = max( |x_i|, |y_i| * 2/sqrt(3), (|x_i| + |y_i|*sqrt(3)/2) * 2/3 ? )
        
        # Let's use the support function or simple projection.
        # For a regular hexagon with side S, the width in the direction of the normal to the side is S*sqrt(3).
        # The distance from center to side is S*sqrt(3)/2.
        
        # Let's check the 3 pairs of parallel sides.
        # Side 1 normal: (1, 0). Distance: |x|. Constraint: |x| <= S*sqrt(3)/2 ? No.
        
        # Let's stick to the vertex definition.
        # If we rotate the outer hexagon optimally, we can reduce S.
        # But let's assume outer hexagon is aligned with inner ones (angle 0) for simplicity first.
        # If inner hexagons are at angle 0, their vertices are at distance 1 from center.
        
        # For a point (x,y), what is the minimum S of a regular hexagon (angle 0) centered at origin containing it?
        # The hexagon is the intersection of 3 strips.
        # Strip 1 (vertical): -S <= x <= S
        # Strip 2 (60 deg): -S*sqrt(3) <= x*sqrt(3) + y <= S*sqrt(3) ? No.
        
        # Let's use the formula for distance from origin to boundary of regular hexagon (side S, angle 0).
        # The apothem (distance to side) is a = S * sqrt(3) / 2.
        # The distance to vertex is R = S.
        
        # The condition for a point (x,y) to be inside a regular hexagon of side S (angle 0) is:
        # max( |x|, |y|*2/sqrt(3), |x + y/sqrt(3)|, |x - y/sqrt(3)| ) <= S ?
        
        # Let's test.
        # Vertex (S, 0): max(S, 0, S+S/sqrt(3)?? No.
        
        # Correct inequalities for regular hexagon (flat top, side S):
        # 1. |x| <= S
        # 2. |y| <= S * sqrt(3) / 2
        # 3. |x| + |y|/sqrt(3) <= S
        
        # Let's verify vertex (S, 0):
        # 1. S <= S (ok)
        # 2. 0 <= S*sqrt(3)/2 (ok)
        # 3. S + 0 <= S (ok)
        
        # Verify side midpoint (S/2, S*sqrt(3)/2):
        # 1. S/2 <= S (ok)
        # 2. S*sqrt(3)/2 <= S*sqrt(3)/2 (ok)
        # 3. S/2 + (S*sqrt(3)/2)/sqrt(3) = S/2 + S/2 = S <= S (ok)
        
        # So for a point (x,y), the required S is:
        # S >= |x|
        # S >= |y| * 2 / sqrt(3)
        # S >= |x| + |y|/sqrt(3)
        
        # Thus S_point = max( |x|, |y| * 2/sqrt(3), |x| + |y|/sqrt(3) )
        
        # We want to minimize S_max = max( S_point_i for all i )
        
        s_vals = []
        for i in range(N):
            x, y = positions[i]
            # The inner hexagons themselves have radius 1.
            # So the points to contain are the vertices of the inner hexagons.
            # Vertices of inner hexagon at (x,y) with side 1:
            # (x+1, y), (x+0.5, y+sqrt(3)/2), ...
            
            # We can just check the 6 vertices of each inner hexagon.
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            for vx, vy in verts:
                s_req = max(abs(vx), abs(vy) * 2.0 / np.sqrt(3.0), abs(vx) + abs(vy) / np.sqrt(3.0))
                s_vals.append(s_req)
                
        if not s_vals:
            return 0.0
            
        return max(s_vals)

    # Initial guess: Honeycomb cluster
    # Center at (0,0)
    # 1 at (0,0)
    # 6 around it at distance sqrt(3) (touching sides)
    # 5 more to make 12? Or just place them densely.
    
    # Distance between centers of touching unit hexagons is 2 * apothem = sqrt(3) * 1 = 1.732.
    # Let's use a grid of hexagons.
    # Axial coordinates (q, r)
    # Position x = sqrt(3) * (q + r/2)
    # Position y = 1.5 * r
    
    hex_centers = []
    # Generate 12 centers in a compact cluster
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 total
    # Add next ring partially: (2,0), (1,1), (0,2), (-1,2), (-2,1) -> 12 total
    
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_positions = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_positions.append([x, y])
        
    init_positions = np.array(init_positions)
    
    # Check if initial is valid
    if check_validity(init_positions):
        init_params = init_positions.flatten()
    else:
        # Fallback: spread them out more
        init_params = np.zeros(2 * N)
        for i in range(N):
            init_params[2*i] = i * 2.0
            init_params[2*i+1] = 0.0

    # Differential Evolution
    # Bounds: -10 to 10
    bounds = [(-10, 10)] * (2 * N)
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            polish=True,
            callback=None
        )
        
        if time.time() > deadline:
            # Return best found so far? DE doesn't easily give intermediate best in this call style without callback.
            # But we can just use the result if it finished, or the initial if it didn't.
            # Actually, if it times out, we should return the initial valid config.
            pass

        best_params = result.x
        best_val = result.fun
        
        if best_val > 1e9:
            # Optimization failed to find valid config, use initial
            best_params = init_params
            best_val = objective(init_params)
            
    except Exception:
        best_params = init_params
        best_val = objective(init_params)

    # Construct final inner list
    inner = []
    for i in range(N):
        x = best_params[2 * i]
        y = best_params[2 * i + 1]
        inner.append([float(x), float(y), 0.0])
        
    # Outer hexagon
    # Center (0,0)
    # Side length best_val
    # Angle 0.0
    
    return inner, (0.0, 0.0), float(best_val), 0.0

# EVOLVE-BLOCK-END
