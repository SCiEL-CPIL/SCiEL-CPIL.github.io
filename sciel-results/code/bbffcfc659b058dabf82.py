"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def hexagon_distance(c1, a1, c2, a2, side=1.0):
        """
        Compute the minimum distance between two regular hexagons.
        c1, c2 are center tuples (x, y); a1, a2 are angles in degrees.
        Returns the minimum distance (0 if overlapping).
        """
        # Get vertices
        v1 = get_hex_vertices(c1[0], c1[1], a1, side)
        v2 = get_hex_vertices(c2[0], c2[1], a2, side)
        
        # Check if any vertex of v1 is inside v2 or vice versa
        # For convex polygons, if no vertex is inside and no edges cross, they are disjoint.
        # But for distance, we need min distance between any point on boundary.
        # Efficient method: check distance between all pairs of edges (segments).
        
        def seg_seg_dist(p1, p2, p3, p4):
            """Minimum distance between segment p1-p2 and p3-p4."""
            # Using vector projection
            d1 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
            d2 = np.array([p4[0]-p3[0], p4[1]-p3[1]])
            r = np.array([p1[0]-p3[0], p1[1]-p3[1]])
            
            a = np.dot(d1, d1)
            e = np.dot(d2, d2)
            f = np.dot(d2, r)
            
            if a < 1e-9 and e < 1e-9:
                return np.linalg.norm(r)
            elif a < 1e-9:
                t2 = max(0.0, min(1.0, f/e))
                dist_vec = r + d2*t2
                return np.linalg.norm(dist_vec)
            elif e < 1e-9:
                t1 = max(0.0, min(1.0, -np.dot(d1, r)/a))
                dist_vec = r - d1*t1
                return np.linalg.norm(dist_vec)
            
            c = np.dot(d1, r)
            b = np.dot(d1, d2)
            denom = a*e - b*b
            
            if denom < 1e-9:
                t1 = 0.0
                t2 = max(0.0, min(1.0, f/e))
                dist_vec = r + d2*t2
                return np.linalg.norm(dist_vec)
            else:
                t1 = max(0.0, min(1.0, (b*f - c*e)/denom))
                t2 = max(0.0, min(1.0, (a*f - b*c)/denom))
                dist_vec = r + d2*t2 - d1*t1
                return np.linalg.norm(dist_vec)

        min_dist = float('inf')
        
        # Check all edge pairs
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1) % 6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1) % 6]
                d = seg_seg_dist(p1, p2, p3, p4)
                if d < min_dist:
                    min_dist = d
        
        # Also check if one hexagon is completely inside the other (distance is negative? No, distance is 0 if overlapping)
        # If min_dist > 0, they are disjoint.
        # If min_dist == 0, they touch or overlap.
        # To be safe, we can also check vertex containment to distinguish touching from overlapping.
        # But for the objective, we want to maximize the minimum distance.
        # So returning min_dist is correct. If they overlap, min_dist will be 0 (or very small).
        
        return min_dist

    def required_L(centers, angles, outer_angle=0.0):
        """Calculate minimum side length of outer hexagon to enclose all inner hexagons."""
        outer_angle_rad = math.radians(outer_angle)
        max_proj = 0.0
        
        for i in range(N):
            verts = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for vx, vy in verts:
                # Rotate point by -outer_angle
                x_rot = vx * math.cos(-outer_angle_rad) - vy * math.sin(-outer_angle_rad)
                y_rot = vx * math.sin(-outer_angle_rad) + vy * math.cos(-outer_angle_rad)
                
                # Project onto 6 edge normals
                for k in range(6):
                    theta = math.radians(30.0 + k * 60.0)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = x_rot * nx + y_rot * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # L * sqrt(3) / 2 >= max_proj  => L >= 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / sqrt3

    def objective(params):
        """
        Objective function: Maximize the minimum distance between hexagons,
        subject to the constraint that they fit in a hexagon of side L=1 (scaled).
        Actually, we minimize L. But L depends on positions.
        
        Let's use a different approach:
        Fix the outer hexagon side length L.
        We want to find positions such that all hexagons are inside and non-overlapping.
        
        Alternatively, we can minimize L directly.
        L = required_L(centers, angles)
        We need to ensure no overlap.
        
        Let's use a penalty method:
        Objective = L + penalty * sum(max(0, 0 - dist_ij)^2)
        
        Or better:
        Objective = L
        Constraint: dist_ij >= 0 for all i<j
        
        Since scipy.minimize doesn't handle constraints well for this,
        let's use a penalty.
        """
        centers = []
        angles = []
        for i in range(N):
            cx = params[2*i]
            cy = params[2*i+1]
            ang = params[2*N + i]
            centers.append((cx, cy))
            angles.append(ang)
            
        L = required_L(centers, angles, 0.0)
        
        # Check overlap
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = hexagon_distance(centers[i], angles[i], centers[j], angles[j])
                if d < 0.0:
                    penalty += (0.0 - d) ** 2
                elif d < 1e-3:
                    # Small gap, add small penalty to encourage separation
                    penalty += (1e-3 - d) ** 2 * 10.0
        
        return L + penalty * 100.0

    # Initial bounds
    # A rough estimate for L is around 4-5.
    # Centers should be within [-L, L] roughly.
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
    for i in range(N):
        bounds.append((0.0, 180.0)) # angle (symmetry)

    best_L = float('inf')
    best_params = None

    # Generate multiple random initial guesses
    num_starts = 10
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
            
        # Random initial configuration
        init_params = []
        for i in range(N):
            init_params.append(np.random.uniform(-3, 3)) # x
            init_params.append(np.random.uniform(-3, 3)) # y
            init_params.append(np.random.uniform(0, 180)) # angle
        
        # Try to optimize
        try:
            res = minimize(
                objective, 
                init_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            if res.fun < best_L:
                best_L = res.fun
                best_params = res.x
        except Exception:
            pass

    # If no valid solution found, return a known valid configuration
    if best_params is None:
        # Simple grid
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    centers = []
    angles = []
    for i in range(N):
        centers.append([best_params[2*i], best_params[2*i+1]])
        angles.append(best_params[2*N + i])
    
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
        
    # Final check and L calculation
    final_L = required_L(centers, angles, 0.0)
    
    # Verify no overlap
    for i in range(N):
        for j in range(i + 1, N):
            d = hexagon_distance(centers[i], angles[i], centers[j], angles[j])
            if d < 1e-4:
                # Overlap or too close, return fallback
                inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
                return inner, (0.0, 0.0), 40.0, 0.0

    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
