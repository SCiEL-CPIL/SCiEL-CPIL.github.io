import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def hex_vertices(cx, cy, angle_deg=0.0, side=1.0):
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            ang = a + k * math.pi / 3.0
            verts.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
        return verts

    def support_function(pts, normals):
        """Max projection of points onto each normal direction."""
        projs = np.array([[p[0] * n[0] + p[1] * n[1] for p in pts] for n in normals])
        return np.max(projs, axis=1)

    def outer_hex_side_length(pts, cx, cy, beta):
        """Compute the minimum side length of outer hexagon (rotation beta) containing all pts."""
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        centered = [(x - cx, y - cy) for x, y in pts]
        max_proj = 0
        for nx, ny in normals:
            for px, py in centered:
                proj = abs(px * nx + py * ny)
                max_proj = max(max_proj, proj)
        
        return (max_proj + APOTHEM) / APOTHEM

    def overlap_penalty(pts, threshold=0.5, soft=0.1):
        """Smooth penalty for pairs of hexagons that overlap."""
        penalty = 0.0
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[j][0] - pts[i][0]
                dy = pts[j][1] - pts[i][1]
                dist = math.sqrt(dx * dx + dy * dy)
                # Two unit hexagons don't overlap if center distance >= 2*APOTHEM in some sense
                # Actually, the minimum distance between centers for non-overlapping unit hexagons
                # depends on relative orientation. For same orientation, min distance is 2*APOTHEM = sqrt(3)
                # For 30-degree rotation, it can be closer. Use a conservative threshold.
                min_dist = 1.732  # sqrt(3), the distance for parallel hexagons touching
                if dist < min_dist:
                    diff = min_dist - dist
                    penalty += diff * diff / (2 * soft)
        return penalty

    def objective(params):
        """Objective: outer hex side length + overlap penalty."""
        n = N
        # params: [x0, y0, ..., x_{n-1}, y_{n-1}, cx, cy, beta]
        pts = [(params[i * 2], params[i * 2 + 1]) for i in range(n)]
        cx = params[2 * n]
        cy = params[2 * n + 1]
        beta = params[2 * n + 2]
        
        # Keep beta in [0, pi/6)
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        
        S = outer_hex_side_length(pts, cx, cy, beta)
        pen = overlap_penalty(pts)
        
        return S + 1000.0 * pen

    def objective_smooth(params, temp=0.5):
        """Smaller temperature for smoother optimization."""
        n = N
        pts = [(params[i * 2], params[i * 2 + 1]) for i in range(n)]
        cx = params[2 * n]
        cy = params[2 * n + 1]
        beta = params[2 * n + 2]
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        
        S = outer_hex_side_length(pts, cx, cy, beta)
        pen = overlap_penalty(pts, threshold=0.5, soft=temp)
        
        return S + 500.0 * pen

    # Generate initial configurations using axial coordinates
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return pts

    # Axial coordinate candidates
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]

    candidates_axial = [axial_1, axial_2, axial_3, axial_4, axial_5, axial_6, axial_7, axial_8, axial_9, axial_10]

    best_S = float('inf')
    best_params = None

    for axial in candidates_axial:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts_init = axial_to_cartesian(axial)
        
        # Compute centroid
        cx0 = sum(p[0] for p in pts_init) / N
        cy0 = sum(p[1] for p in pts_init) / N
        
        # Build initial parameter vector
        params0 = []
        for x, y in pts_init:
            params0.extend([x - cx0, y - cy0])
        params0.extend([0.0, 0.0, 0.0])  # cx=0, cy=0, beta=0
        
        # Multi-stage optimization with decreasing penalty weight
        current_params = params0[:]
        
        for stage in range(4):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Decreasing temperature for overlap penalty
            temps = [1.0, 0.5, 0.2, 0.1]
            temp = temps[min(stage, len(temps) - 1)]
            
            def obj_stage(p, t=temp):
                n = N
                pts = [(p[i * 2], p[i * 2 + 1]) for i in range(n)]
                cx = p[2 * n]
                cy = p[2 * n + 1]
                beta = ((p[2 * n + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                
                S = outer_hex_side_length(pts, cx, cy, beta)
                pen = overlap_penalty(pts, threshold=0.5, soft=t)
                
                weight = 500.0 / (stage + 1)
                return S + weight * pen
            
            try:
                res = minimize(obj_stage, current_params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < float('inf'):
                    current_params = res.x
            except Exception:
                pass
        
        # Evaluate final result
        try:
            pts_final = [(current_params[i * 2], current_params[i * 2 + 1]) for i in range(N)]
            cx_f = current_params[2 * N]
            cy_f = current_params[2 * N + 1]
            beta_f = ((current_params[2 * N + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            
            S_final = outer_hex_side_length(pts_final, cx_f, cy_f, beta_f)
            pen_final = overlap_penalty(pts_final, threshold=0.0, soft=0.01)
            
            if pen_final < 1e-6 and S_final < best_S:
                best_S = S_final
                best_params = current_params[:]
        except Exception:
            pass

    # If no valid solution found, fall back to a simple configuration
    if best_params is None:
        # Use the first axial configuration with minimal outer hex
        pts_fallback = axial_to_cartesian(axial_1)
        cx_fb = sum(p[0] for p in pts_fallback) / N
        cy_fb = sum(p[1] for p in pts_fallback) / N
        
        best_S = float('inf')
        for beta_test in [0.0, math.pi/12, math.pi/6 * 0.5]:
            S_test = outer_hex_side_length(pts_fallback, 0.0, 0.0, beta_test)
            if S_test < best_S:
                best_S = S_test
                best_beta = beta_test
        
        # Build fallback params
        best_params = []
        for x, y in pts_fallback:
            best_params.extend([x - cx_fb, y - cy_fb])
        best_params.extend([0.0, 0.0, best_beta])
    
    # Extract final result
    pts_result = [(best_params[i * 2], best_params[i * 2 + 1]) for i in range(N)]
    cx_res = best_params[2 * N]
    cy_res = best_params[2 * N + 1]
    beta_res = ((best_params[2 * N + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
    
    # Scale up slightly to avoid numerical issues
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in pts_result:
        sx = cx_res + (x - cx_res) * SCALE
        sy = cy_res + (y - cy_res) * SCALE
        pts_scaled.append((sx, sy))
    
    # Recompute S with scaled points
    S_scaled = outer_hex_side_length(pts_scaled, cx_res, cy_res, beta_res)
    
    inner = []
    angle_deg = math.degrees(beta_res)
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx_res), float(cy_res)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
