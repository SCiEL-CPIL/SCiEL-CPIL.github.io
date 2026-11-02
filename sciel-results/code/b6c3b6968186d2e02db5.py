import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Normals for a hexagon with orientation 0 (flat side at angle 0? No, vertex at 0?)
    # Standard hexagon: vertices at angles 0, 60, 120...
    # Normals (apothem directions) at 30, 90, 150...
    # Let's define normals for the constraint check.
    # If orientation is 0, the sides are perpendicular to these normals.
    normals = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])
    
    # Minkowski difference apothem for two unit hexagons with same orientation
    # Side length of Minkowski sum of two unit hexagons is 2.
    # Apothem of Minkowski sum is 2 * APOTHEM = sqrt(3).
    MINKOWSKI_APOTHEM = math.sqrt(3)
    
    def get_vertices(cx, cy, angle=0.0):
        # Vertices of a unit hexagon
        # If angle=0, let's say a vertex is at (1,0).
        # Then the flat side is at angle 30 deg.
        # Normals are at 30, 90, ...
        # This matches the normals array above.
        verts = []
        for k in range(6):
            ang = angle + k * math.pi / 3.0
            verts.append((cx + math.cos(ang), cy + math.sin(ang)))
        return np.array(verts)

    def calc_side_length(center, points):
        # points: (N, 2) array of centers
        # We need the max projection of any vertex of any hexagon onto the normals
        # relative to the outer hexagon center.
        # Outer hexagon normals are the same as inner if orientation is 0.
        max_proj = 0.0
        for i in range(N):
            cx, cy = points[i]
            verts = get_vertices(cx, cy, 0.0)
            rel = verts - np.array(center)
            projs = np.abs(rel @ normals.T)
            max_proj = max(max_proj, np.max(projs))
        
        # S = (D + APOTHEM) / APOTHEM
        # D is the max projection from center.
        # The outer hexagon must contain the vertices.
        # Distance from center to side is S * APOTHEM.
        # So S * APOTHEM >= max_proj.
        # S = max_proj / APOTHEM.
        return max_proj / APOTHEM

    def overlap_penalty(points):
        # Sum of squared penetration depths
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                diff = points[i] - points[j]
                projs = np.abs(diff @ normals.T)
                max_proj = np.max(projs)
                if max_proj < MINKOWSKI_APOTHEM:
                    pen = MINKOWSKI_APOTHEM - max_proj
                    penalty += pen * pen
        return penalty

    def objective(params):
        cx, cy = params[0], params[1]
        pts = params[2:].reshape(N, 2)
        S = calc_side_length([cx, cy], pts)
        pen = overlap_penalty(pts)
        # We want to minimize S, but must satisfy pen == 0.
        # If pen > 0, add a large penalty.
        if pen > 1e-6:
            return S + 1000.0 * pen
        return S

    def gradient(params):
        # Numerical gradient if needed, but SLSQP can use finite diff
        # For speed, we'll let scipy handle it.
        pass

    # Initial configurations
    configs = []
    
    # Config 1: Flower + 4 in gaps
    flower = [
        (0, 0),
        (math.sqrt(3), 0),
        (math.sqrt(3)/2, 1.5),
        (-math.sqrt(3)/2, 1.5),
        (-math.sqrt(3), 0),
        (-math.sqrt(3)/2, -1.5),
        (math.sqrt(3)/2, -1.5)
    ]
    # Gaps at 30, 90, 150, 210, 270, 330 deg
    # Distance to gap center:
    # The gap between two tangent hexagons in the flower.
    # The "valley" is at distance 2 * 1.5 * cos(30) ? No.
    # The vertices of the inner ring are at distance 2 from center.
    # The gap is between two adjacent vertices of the inner ring.
    # Let's just place them at distance 3.5 in the gap directions.
    gap_angles = [math.pi/6, math.pi/2, 5*math.pi/6, 7*math.pi/6, 3*math.pi/2, 11*math.pi/6]
    extra = []
    for k in range(4):
        ang = gap_angles[k]
        r = 3.5
        extra.append((r * math.cos(ang), r * math.sin(ang)))
    
    pts1 = np.array(flower + extra)
    configs.append(pts1)
    
    # Config 2: Grid-like
    # 4x3 grid roughly
    pts2 = []
    for r in range(3):
        for c in range(4):
            x = c * math.sqrt(3)
            y = r * 1.5
            if r % 2 == 1:
                x += math.sqrt(3)/2
            pts2.append((x, y))
    # We have 12 points, remove one
    pts2 = np.array(pts2[:11])
    configs.append(pts2)
    
    # Config 3: Random
    np.random.seed(42)
    pts3 = np.random.uniform(-3, 3, (11, 2))
    configs.append(pts3)
    
    best_S = float('inf')
    best_params = None
    
    for init_pts in configs:
        # Center of mass
        cx, cy = np.mean(init_pts[:, 0]), np.mean(init_pts[:, 1])
        params = np.concatenate(([cx, cy], init_pts.ravel()))
        
        try:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            res = minimize(objective, params, method='SLSQP', options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    cx, cy = best_params[0], best_params[1]
    pts = best_params[2:].reshape(N, 2)
    
    # Final check and cleanup
    # Ensure no overlap by slightly scaling if necessary?
    # The optimizer should have handled it.
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), 0.0])
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = 0.0
    
    return inner, center, side, rotation
