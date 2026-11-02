import math
import numpy as np
import time
import random

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Parameters: 11 hexagons * (x, y, angle) + 1 outer angle
    # Total 34 parameters
    
    def get_vertices(params):
        verts = []
        for i in range(N):
            x, y, ang = params[i*3], params[i*3+1], params[i*3+2]
            # Unit hexagon vertices relative to center
            # Side length 1. Vertices at distance 1 from center.
            # Angles: 0, 60, 120, 180, 240, 300 degrees
            for k in range(6):
                va = ang + k * math.pi / 3.0
                vx = x + math.cos(va)
                vy = y + math.sin(va)
                verts.append((vx, vy))
        return np.array(verts)

    def evaluate(params):
        outer_ang = params[-1]
        verts = get_vertices(params)
        
        # Normals for outer hexagon
        # A regular hexagon with "flat top" or "pointy top"?
        # The normals are perpendicular to the sides.
        # If we define the hexagon by its circumradius R=1 (side 1),
        # the distance from center to side is APOTHEM.
        # The normals are at angles outer_ang + pi/6 + k*pi/3
        widths = []
        for k in range(6):
            ang = outer_ang + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            projs = verts[:, 0] * nx + verts[:, 1] * ny
            w = np.max(projs) - np.min(projs)
            widths.append(w)
        
        max_width = max(widths)
        # Side length S is related to max_width by S = max_width / APOTHEM
        # Because width = 2 * D, and D = S * APOTHEM / 1? 
        # Wait. For a hexagon of side S, the distance between opposite sides is 2 * APOTHEM * S = sqrt(3) * S.
        # So max_width = sqrt(3) * S_min.
        # S_min = max_width / sqrt(3).
        # APOTHEM = sqrt(3)/2.
        # So S_min = max_width / (2 * APOTHEM).
        return max_width / (2.0 * APOTHEM)

    # Initial state: compact grid
    # 4x3 grid of hexagons
    # Hexagon side 1. Distance between centers in grid is 2 * APOTHEM = sqrt(3) for horizontal, 
    # and for vertical rows, offset by sqrt(3)/2 and height 1.5?
    # Standard hex packing:
    # Row 0: x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3)
    # Row 1: x = sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2, 7*sqrt(3)/2, y = 1.5
    # Row 2: x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3), y = 3.0
    # Total 12 spots. We need 11.
    
    init_params = []
    spots = []
    for r in range(3):
        for c in range(4):
            x = c * math.sqrt(3) + (r % 2) * math.sqrt(3) / 2.0
            y = r * 1.5
            spots.append((x, y))
    
    # Use first 11 spots
    for i in range(N):
        x, y = spots[i]
        init_params.extend([x, y, 0.0])
    
    # Add outer angle
    init_params.append(0.0)
    
    current_params = init_params
    current_score = evaluate(current_params)
    
    best_params = current_params[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.995
    
    start_time = time.time()
    
    while time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        # Perturb
        new_params = current_params[:]
        
        # Randomly choose which hexagon to perturb
        idx = random.randint(0, N-1)
        
        # Perturb x, y, angle
        # Step size depends on T
        step = 0.5 * T
        
        new_params[idx*3] += random.uniform(-step, step)
        new_params[idx*3+1] += random.uniform(-step, step)
        new_params[idx*3+2] += random.uniform(-step, step)
        
        # Occasionally perturb outer angle
        if random.random() < 0.1:
            new_params[-1] += random.uniform(-0.1, 0.1)
            
        new_score = evaluate(new_params)
        
        # Acceptance criterion
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params[:]
        else:
            delta = new_score - current_score
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_params = new_params
                    current_score = new_score
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final refinement: Local search on best_params
    # Optimize outer angle and center? 
    # The evaluate function assumes the outer hexagon is centered such that the widths are minimized?
    # No, evaluate calculates the width of the point set. The minimal enclosing hexagon for a fixed rotation 
    # has side length S = max_width / (2 * APOTHEM) IF we can choose the center freely.
    # Can we? 
    # For a fixed rotation, the minimal enclosing regular hexagon of a set of points P 
    # has side length S = max_k (max(p.n_k) - min(p.n_k)) / (2 * APOTHEM).
    # The center is chosen such that it is the midpoint of the projections for the critical directions.
    # So yes, the score calculated is the minimal possible side length for that rotation.
    
    # We should refine the outer angle of the best solution to ensure it's optimal for the fixed inner positions.
    best_inner = []
    for i in range(N):
        best_inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
    
    # Refine outer angle
    best_outer_ang = best_params[-1]
    verts = get_vertices(best_params[:-1])
    
    def score_for_angle(ang):
        widths = []
        for k in range(6):
            a = ang + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(a)
            ny = math.sin(a)
            projs = verts[:, 0] * nx + verts[:, 1] * ny
            widths.append(np.max(projs) - np.min(projs))
        return max(widths) / (2.0 * APOTHEM)
    
    # Coarse grid for outer angle
    best_ang = best_outer_ang
    best_val = best_score
    for i in range(36):
        ang = i * math.pi / 18.0 # 10 degree steps
        val = score_for_angle(ang)
        if val < best_val:
            best_val = val
            best_ang = ang
            
    # Fine grid
    for i in range(36):
        ang = best_ang + (i - 18) * math.pi / 180.0 # 1 degree steps around best
        val = score_for_angle(ang)
        if val < best_val:
            best_val = val
            best_ang = ang

    # Construct output
    # The inner hexagons are at best_params[i*3...i*3+2]
    # The outer hexagon center:
    # For the optimal angle best_ang, we need to find the center C.
    # C is the center of the bounding box in each normal direction?
    # For a regular hexagon, if we minimize the side length, the center is the intersection of the midlines of the critical faces.
    # Simple approximation: Centroid of the vertices?
    # Better: Solve for C such that max(|(v-C).n_k|) is minimized.
    # This is a linear programming problem or can be approximated by averaging the midpoints of the critical extents.
    
    # Let's compute the center as the average of the midpoints of the 6 projections.
    # For each k, midpoint m_k = (max(v.n_k) + min(v.n_k)) / 2.
    # We want C.n_k = m_k for all k. This is an overdetermined system.
    # Least squares solution:
    # A C = b, where A is 6x2 matrix of normals, b is 6x1 vector of midpoints.
    
    normals = []
    midpoints = []
    for k in range(6):
        a = best_ang + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(a)
        ny = math.sin(a)
        normals.append([nx, ny])
        projs = verts[:, 0] * nx + verts[:, 1] * ny
        mid = (np.max(projs) + np.min(projs)) / 2.0
        midpoints.append(mid)
        
    A = np.array(normals)
    b = np.array(midpoints)
    
    # Least squares
    C, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = C[0], C[1]
    
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        ang_deg = math.degrees(best_params[i*3+2])
        inner.append([float(x), float(y), float(ang_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_val)
    rotation = float(math.degrees(best_ang))
    
    return inner, center, side, rotation
