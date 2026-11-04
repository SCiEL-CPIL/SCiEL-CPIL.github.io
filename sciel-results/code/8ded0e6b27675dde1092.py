import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        vertices.append((x, y))
    return vertices

def polygon_area(vertices):
    # Shoelace formula
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def polygons_intersect(v1, v2):
    # Simple check: if any vertex of v1 is inside v2 or vice versa, or edges intersect
    # For convex polygons, this is sufficient for non-degenerate cases
    # We'll use a simpler heuristic: check if center of v1 is in v2 or center of v2 is in v1
    # Or check min distance between edges?
    # For speed, we'll just check if the distance between centers is less than a threshold
    # This is not robust, but for optimization it might work as a soft penalty
    
    c1 = np.mean(v1, axis=0)
    c2 = np.mean(v2, axis=0)
    dist = np.linalg.norm(c1 - c2)
    
    # Minimal distance for non-overlap of two unit hexagons
    # If aligned, dist >= sqrt(3)
    # If rotated, dist >= something else
    # We use a soft penalty if dist < 1.7 (sqrt(3) is 1.732)
    if dist < 1.7:
        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial guess: Flower of 7 + 4 random
    # 7 flower
    centers = [(0.0, 0.0)]
    angles = [0.0]
    for k in range(6):
        angle = k * 60.0
        rad = np.deg2rad(angle)
        cx = np.sqrt(3) * np.cos(rad)
        cy = np.sqrt(3) * np.sin(rad)
        centers.append((cx, cy))
        angles.append(0.0)
        
    # 4 extra
    for k in range(4):
        angle = k * 90.0 + 30.0
        rad = np.deg2rad(angle)
        cx = 3.0 * np.cos(rad)
        cy = 3.0 * np.sin(rad)
        centers.append((cx, cy))
        angles.append(0.0)
        
    x0 = []
    for i in range(N):
        x0.append(centers[i][0])
        x0.append(centers[i][1])
        x0.append(angles[i])
        
    def cost(params):
        if time.time() > deadline:
            return 1e6
            
        centers_opt = []
        angles_opt = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            ang = params[3*i+2]
            centers_opt.append((cx, cy))
            angles_opt.append(ang)
            
        # Calculate max distance to boundary of hexagon side L=1 (normalized)
        # We want to minimize L.
        # Let's calculate the required L for this configuration.
        # For each hexagon, find the max distance from origin to its vertices.
        # This is the circumradius R.
        # But we need to fit in a hexagon.
        # The required L is max over all vertices (x,y) of max(|x|, |y|/sqrt(3), |x/2 - y*sqrt(3)/2| ... )
        # Actually, for a hexagon aligned with axes, the condition is:
        # |x| <= L * 0.5 * 2? No.
        # Vertices of container (side L, angle 0):
        # (L, 0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), (-L, 0), (-L/2, -L*sqrt(3)/2), (L/2, -L*sqrt(3)/2)
        # The half-width in x is L.
        # The half-width in y is L*sqrt(3)/2.
        # The condition for a point (x,y) to be inside is:
        # |x| <= L
        # |y| <= L*sqrt(3)/2
        # |x/2 + y*sqrt(3)/2| <= L  (for the slanted sides? No)
        
        # Let's use the support function.
        # The width of the container in direction theta is L * (cos(theta) + cos(theta - pi/3) ... )
        # Easier: Check if all vertices of all inner hexagons are inside the container of side L.
        # We can estimate L by taking the max over all vertices of the "hexagonal norm".
        # Hexagonal norm of (x,y) is max( |x|, |y|/sqrt(3) + |x|/3, ... )
        # Actually, the distance from origin to the boundary of the hexagon in direction (x,y) is:
        # d = 1 / max( |x|/L, |y|/(L*sqrt(3)/2), |(x/2 + y*sqrt(3)/2)|/L, ... )
        # This is getting messy.
        
        # Simple approximation:
        # L_req = max( max(|x|), max(|y|)/sqrt(3)*2, max(|x/2 - y*sqrt(3)/2|)*2 )
        # Let's just use the max distance to the 6 sides of the unit hexagon (side 1).
        # Unit hexagon sides equations:
        # x <= 1
        # -x <= 1
        # sqrt(3)y/2 + x/2 <= 1
        # -sqrt(3)y/2 - x/2 <= 1
        # sqrt(3)y/2 - x/2 <= 1
        # -sqrt(3)y/2 + x/2 <= 1
        
        max_norm = 0.0
        for i in range(N):
            verts = get_hex_vertices(centers_opt[i][0], centers_opt[i][1], angles_opt[i])
            for v in verts:
                x, y = v
                # Distance to sides of unit hexagon (normalized to side 1)
                # The "radius" in this metric is the L required.
                n1 = abs(x)
                n2 = abs(y) / np.sqrt(3) * 2 # Wait, y max is L*sqrt(3)/2, so y / (L*sqrt(3)/2) = 2y/(L*sqrt(3))
                # Let's define the hexagonal distance d(x,y) such that the hexagon is d <= 1.
                # d(x,y) = max( |x|, |y|/sqrt(3) + |x|/3, |-y|/sqrt(3) + |x|/3 )?
                # Actually, for a regular hexagon with vertices at (1,0), etc:
                # The equation of the side connecting (1,0) and (0.5, sqrt(3)/2) is:
                # y - 0 = (sqrt(3)/2 - 0) / (0.5 - 1) * (x - 1)
                # y = -sqrt(3) (x - 1)
                # sqrt(3)x + y = sqrt(3)
                # So |sqrt(3)x + y| <= sqrt(3)
                # Similarly for other sides.
                
                val1 = abs(x)
                val2 = abs(np.sqrt(3)*x + y) / np.sqrt(3)
                val3 = abs(np.sqrt(3)*x - y) / np.sqrt(3)
                # The hexagon is the intersection of these 6 strips.
                # So d = max(val1, val2, val3)
                # If d <= 1, it's inside.
                # So L_required = max(d)
                d = max(val1, val2, val3)
                if d > max_norm:
                    max_norm = d
                    
        # Penalty for overlaps
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                c1 = np.array(centers_opt[i])
                c2 = np.array(centers_opt[j])
                dist = np.linalg.norm(c1 - c2)
                if dist < 1.7:
                    overlap_pen += (1.7 - dist) ** 2
                    
        return max_norm + 10.0 * overlap_pen

    res = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    # Extract best L
    best_L = cost(res.x)
    
    # Reconstruct positions
    inner = []
    for i in range(N):
        cx = res.x[3*i]
        cy = res.x[3*i+1]
        ang = res.x[3*i+2]
        inner.append([cx, cy, ang])
        
    # Ensure L is valid
    # The cost function returned max_norm + penalty.
    # If penalty is 0, L = max_norm.
    # We should add a small epsilon.
    final_L = best_L + 0.01
    
    return inner, (0.0, 0.0), final_L, 0.0

# EVOLVE-BLOCK-END
