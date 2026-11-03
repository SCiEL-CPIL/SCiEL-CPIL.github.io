import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave some margin for 120s limit

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback to a simple valid configuration
        # 11 hexagons in a line? Or a grid?
        # Let's just return the seed-like result or a simple valid one
        # A simple valid one: 11 hexagons in a hexagonal grid pattern might be too big.
        # Let's use the axial grid from the seed as a fallback.
        axial = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (1, 3), (2, 3)
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        pts = np.array(pts)
        cx, cy = np.mean(pts, axis=0)
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(cx), float(cy)], float(S), 0.0

    # Precompute local vertices of a unit hexagon (centered at 0,0, orientation 0)
    # Vertices are at angles 0, 60, ..., 300
    # x = cos(theta), y = sin(theta)
    local_vertices = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)
    ])

    # Normals of the outer hexagon (fixed orientation 0)
    # Normals are at 30, 90, ..., 330
    outer_normals = np.array([
        [math.cos(math.pi / 6 + k * math.pi / 3), math.sin(math.pi / 6 + k * math.pi / 3)] for k in range(6)
    ])

    def get_inner_vertices(x, y, alpha):
        """
        Returns the 6 vertices of a unit hexagon centered at (x, y) with rotation alpha.
        """
        c = math.cos(alpha)
        s = math.sin(alpha)
        R = np.array([[c, -s], [s, c]])
        # local_vertices shape (6, 2)
        # R @ local_vertices.T shape (2, 6) -> T -> (6, 2)
        rotated = (R @ local_vertices.T).T
        return rotated + np.array([x, y])

    def calc_S_and_penalty(params, lambda_pen):
        """
        params: [x1, y1, ..., x11, y11, alpha]
        Returns: (S, penalty)
        """
        xs = params[:2 * N:2]
        ys = params[1:2 * N:2]
        alpha = params[-1]

        # 1. Calculate S (size of outer hexagon)
        max_proj = 0.0
        # Compute all vertices
        all_vertices = []
        for i in range(N):
            verts = get_inner_vertices(xs[i], ys[i], alpha)
            all_vertices.append(verts)
        
        # Project all vertices onto outer normals
        # We can do this efficiently
        for verts in all_vertices:
            # verts shape (6, 2)
            # outer_normals shape (6, 2)
            # projs = verts @ outer_normals.T -> (6, 6)
            projs = verts @ outer_normals.T
            # We need max of |projs|
            max_abs = np.max(np.abs(projs))
            if max_abs > max_proj:
                max_proj = max_abs
        
        S = (max_proj + APOTHEM) / APOTHEM

        # 2. Calculate Overlap Penalty
        # Assuming all inner hexagons have the same orientation alpha
        # Non-overlap condition for two hexagons with same orientation:
        # max_k | (ci - cj) . n_k(alpha) | > APOTHEM_SUM
        # APOTHEM_SUM = sqrt(3) (apothem of Minkowski sum)
        APOTHEM_SUM = math.sqrt(3)
        
        # Normals of inner hexagons (rotated by alpha)
        c = math.cos(alpha)
        s = math.sin(alpha)
        R = np.array([[c, -s], [s, c]])
        inner_normals = (R @ local_vertices.T).T # Wait, normals are perpendicular to edges.
        # The normals of a hexagon with vertices at angles 0, 60.. are at 30, 90..
        # So inner_normals should be rotated by alpha + pi/6?
        # Let's define inner_normals explicitly.
        inner_normals = np.array([
            [math.cos(alpha + math.pi / 6 + k * math.pi / 3), math.sin(alpha + math.pi / 6 + k * math.pi / 3)] for k in range(6)
        ])

        penalty = 0.0
        centers = np.column_stack((xs, ys))
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                # Project diff onto inner normals
                projs = diff @ inner_normals.T
                max_abs_proj = np.max(np.abs(projs))
                
                # If max_abs_proj <= APOTHEM_SUM, they overlap
                if max_abs_proj < APOTHEM_SUM:
                    # Penalty is square of penetration depth
                    pen = (APOTHEM_SUM - max_abs_proj) ** 2
                    penalty += pen

        return S + lambda_pen * penalty, S, penalty

    def objective(params, lambda_pen):
        val, _, _ = calc_S_and_penalty(params, lambda_pen)
        return val

    # Initial guess: Hexagonal grid
    # 1 center, 6 around, 4 more
    # Let's generate a reasonable initial layout
    init_params = []
    # Center
    init_params.extend([0.0, 0.0])
    # 6 around
    for k in range(6):
        ang = k * math.pi / 3
        x = math.sqrt(3) * math.cos(ang)
        y = math.sqrt(3) * math.sin(ang)
        init_params.extend([x, y])
    # 4 more: place them in the gaps of the outer ring?
    # The gaps are at angles 30, 90, 150, 210...
    # Distance from center to gap is 2 * apothem = sqrt(3) * sqrt(3) = 3?
    # No, the vertices of the outer ring are at distance 2 from center?
    # Center (0,0). Neighbor (sqrt(3), 0). Distance sqrt(3).
    # The "gap" between (sqrt(3), 0) and (sqrt(3)/2, 1.5) is at (1.5*sqrt(3)/2 + sqrt(3)/2, ...)?
    # Let's just place them at distance 2*sqrt(3) from center?
    # Or use a simple grid.
    # Let's use a 3x3 grid minus 2 corners?
    # Or just randomize slightly around the 7-hexagon cluster.
    # For the 4 extra, put them at the "corners" of the cluster.
    # Cluster radius is approx 2*sqrt(3) = 3.46.
    # Let's place them at distance 2*sqrt(3) * 0.8?
    
    # Better initial guess:
    # 11 hexagons can be arranged in a "rhombus" or "strip".
    # Let's try a 2x2x3 block?
    # Axial:
    # (0,0), (1,0), (2,0)
    # (0,1), (1,1), (2,1)
    # (0,2), (1,2), (2,2)
    # (1,3), (2,3)
    
    axial_init = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    
    init_params = []
    for q, r in axial_init:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_params.extend([x, y])
    init_params.append(0.0) # alpha
    
    # Optimization
    best_S = float('inf')
    best_params = init_params
    
    # Use Differential Evolution to find a good basin
    # Bounds for x, y: -10 to 10
    # Bounds for alpha: 0 to pi/3
    bounds = [(-10, 10) for _ in range(2 * N)] + [(0, math.pi / 3)]
    
    try:
        # DE is stochastic and might be slow, but robust.
        # We limit maxiter.
        res_de = differential_evolution(
            lambda p: objective(p, 1.0), 
            bounds, 
            maxiter=100, 
            tol=1e-4,
            popsize=15,
            seed=42,
            polish=False
        )
        if res_de.fun < best_S:
            best_S = res_de.fun
            best_params = res_de.x
    except Exception:
        pass

    # Refine with Nelder-Mead
    # Increase penalty weight to ensure non-overlap
    lambda_pen = 10.0
    
    try:
        res_nm = minimize(
            lambda p: objective(p, lambda_pen),
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res_nm.fun < best_S:
            best_S = res_nm.fun
            best_params = res_nm.x
    except Exception:
        pass

    # Final check and extraction
    final_val, S_val, pen_val = calc_S_and_penalty(best_params, 1.0)
    
    # If there is still penalty, try to push them apart?
    # For now, we assume the optimization resolved it.
    # If pen_val is significant, the solution might be invalid.
    # We should check validity.
    
    xs = best_params[:2 * N:2]
    ys = best_params[1:2 * N:2]
    alpha = best_params[-1]
    
    # Verify non-overlap
    c = math.cos(alpha)
    s = math.sin(alpha)
    inner_normals = np.array([
        [math.cos(alpha + math.pi / 6 + k * math.pi / 3), math.sin(alpha + math.pi / 6 + k * math.pi / 3)] for k in range(6)
    ])
    centers = np.column_stack((xs, ys))
    APOTHEM_SUM = math.sqrt(3)
    is_valid = True
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            projs = diff @ inner_normals.T
            max_abs_proj = np.max(np.abs(projs))
            if max_abs_proj < APOTHEM_SUM - 1e-9:
                is_valid = False
                break
        if not is_valid:
            break
    
    if not is_valid:
        # Fallback to axial grid solution
        return construct()

    inner = []
    for i in range(N):
        inner.append([float(xs[i]), float(ys[i]), float(math.degrees(alpha))])
    
    center = [0.0, 0.0]
    side = float(S_val)
    rotation = 0.0
    
    return inner, center, side, rotation

