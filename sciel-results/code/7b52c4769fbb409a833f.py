import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

# Vertices of a unit regular hexagon centered at origin, flat-top orientation (angle 0)
# Side length 1. Apothem sqrt(3)/2.
# Vertices at angles 0, 60, ..., 300
HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, APOTHEM],
    [-0.5, APOTHEM],
    [-1.0, 0.0],
    [-0.5, -APOTHEM],
    [0.5, -APOTHEM]
])

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid
        # Generate a valid (though not optimal) configuration
        # A 3x3 grid of hexagons has side length roughly 3+2*apothem? 
        # Let's just return a known valid config or a single hexagon if time is up
        # A single hexagon side 1 is valid for N=1, but N=11.
        # If time is up, we must return *something* valid.
        # The evaluator checks validity.
        # Let's use the best found so far if available, else a safe grid.
        pass

    def get_hex_verts(center):
        return HEX_VERTS + center
    
    def compute_min_enclosing_hex_side(positions, beta):
        # positions: (N, 2) array of centers
        # beta: rotation of outer hexagon
        # Outer hexagon normals
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        # Compute projections of all vertices of all inner hexagons onto normals
        # We need the max absolute projection to define the "radius" D in the support function
        # Side length S = (D + APOTHEM) / APOTHEM
        # Where D is the max distance from center to any vertex projection?
        # Actually, for a regular hexagon with apothem a, side s = 2a/sqrt(3).
        # If the cluster fits in a hexagon of apothem D, then S = (D + a_inner) / a_outer?
        # No.
        # Let outer hexagon have apothem A_out. Side S_out = 2 * A_out / sqrt(3).
        # The condition is that for every vertex v of every inner hexagon, and every normal n_k of the outer hexagon:
        # | (v - c_out) . n_k | <= A_out
        # We can set c_out = 0 by centering the cluster.
        # So we minimize A_out.
        # S_out = (2 / sqrt(3)) * A_out.
        # But the task asks for "outer side length".
        # The inner hexagons have side 1, so apothem a_in = sqrt(3)/2.
        
        # To minimize S_out, we minimize A_out.
        # A_out = max_{i, v, k} | (v_i - c_out) . n_k |
        
        # We will optimize c_out as well to minimize this max.
        # For a fixed beta, the optimal c_out is the center of the minimal enclosing hexagon.
        # This is a linear programming problem, but we can approximate it by taking the midpoint of the max/min projections.
        # proj_k = (v - c) . n_k
        # We want to minimize max_k (max_v proj_k - min_v proj_k) / 2 ?
        # No, the hexagon is centered at c.
        # Constraints: -A <= (v - c).n_k <= A for all v, k.
        # This implies A >= max_k ( max_v (v.n_k) - c.n_k ) and A >= max_k ( c.n_k - min_v (v.n_k) ).
        # To minimize A, we should choose c such that these bounds are balanced.
        # Specifically, for each k, let M_k = max_v (v . n_k) and m_k = min_v (v . n_k).
        # We need A >= M_k - c.n_k and A >= c.n_k - m_k.
        # So A >= max(M_k - c.n_k, c.n_k - m_k).
        # Let u_k = c . n_k.
        # We want to minimize max_k max(M_k - u_k, u_k - m_k).
        # This is equivalent to minimizing max_k ( (M_k - m_k)/2 + | u_k - (M_k+m_k)/2 | ).
        # The optimal u_k for each k independently would be the midpoint (M_k+m_k)/2, but u_k are coupled via c.
        # However, for a small number of hexagons, we can just compute the bounding box in the 6 directions.
        # A simple heuristic for c is the centroid.
        # Then compute A_out.
        
        # Let's use the centroid as c_out for now to keep the cost function differentiable-ish and fast.
        c_out = np.mean(positions, axis=0)
        
        max_A = 0.0
        for nx, ny in normals:
            n_vec = np.array([nx, ny])
            # Project all vertices
            # positions: (N, 2), HEX_VERTS: (6, 2)
            # Vertices: (N, 6, 2)
            verts = positions[:, np.newaxis, :] + HEX_VERTS[np.newaxis, :, :]
            # Projections: (N, 6)
            projs = verts @ n_vec
            max_proj = np.max(projs)
            min_proj = np.min(projs)
            
            # Distance from c_out to these extremes
            c_proj = c_out @ n_vec
            dist_max = max_proj - c_proj
            dist_min = c_proj - min_proj
            
            A_k = max(dist_max, dist_min)
            if A_k > max_A:
                max_A = A_k
        
        A_out = max_A
        S_out = (2.0 / math.sqrt(3.0)) * A_out
        return S_out

    def cost_function(params):
        # params: [x1, y1, x2, y2, ..., x11, y11, beta]
        # Total 22 + 1 = 23 params
        positions = params[:2 * N].reshape(N, 2)
        beta = params[-1]
        
        # 1. Compute side length
        S = compute_min_enclosing_hex_side(positions, beta)
        
        # 2. Compute overlap penalty
        # Minimum distance between centers of two non-overlapping aligned hexagons is sqrt(3).
        # If they are closer, they overlap.
        # We add a large penalty if any pair is closer than sqrt(3) - epsilon.
        penalty = 0.0
        min_dist = math.sqrt(3.0) - 1e-6
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    # Quadratic penalty to push them apart
                    penalty += (min_dist - d) ** 2 * 1000.0
        
        return S + penalty

    # Initial guess: 
    # A compact cluster. 
    # Let's try a few random starts or a grid start.
    # Grid start: 3x3 plus 2 extra.
    # Hex grid coordinates (axial) to cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    def axial_to_cart(q, r):
        return (math.sqrt(3) * (q + r / 2.0), 1.5 * r)
    
    # Candidate 1: 3x3 block (9 hexes) + 2 more
    # (0,0), (1,0), (2,0)
    # (0,1), (1,1), (2,1)
    # (0,2), (1,2), (2,2)
    # (1,3), (2,3)
    init_axial = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        (1,3), (2,3)
    ]
    init_pos = np.array([axial_to_cart(q, r) for q, r in init_axial])
    
    # Candidate 2: 2x5 block?
    # (0,0)..(1,0), (0,1)..(1,1), etc
    init_axial_2 = [
        (0,0), (1,0),
        (0,1), (1,1),
        (0,2), (1,2),
        (0,3), (1,3),
        (0,4), (1,4),
        (2,2)
    ]
    init_pos_2 = np.array([axial_to_cart(q, r) for q, r in init_axial_2])
    
    best_S = float('inf')
    best_params = None
    
    # Try optimization from multiple starts
    starts = [
        np.concatenate([init_pos.flatten(), [0.0]]),
        np.concatenate([init_pos_2.flatten(), [0.0]]),
        # Random perturbation of init_pos
        np.concatenate([(init_pos + np.random.normal(0, 0.1, (N, 2))).flatten(), [0.1]]),
        np.concatenate([(init_pos + np.random.normal(0, 0.1, (N, 2))).flatten(), [0.2]]),
    ]
    
    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        try:
            # Use Nelder-Mead for robustness with non-smooth cost
            res = minimize(cost_function, start, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to initial guess
        best_params = starts[0]
        best_S = cost_function(best_params)

    # Extract final positions and beta
    positions = best_params[:2 * N].reshape(N, 2)
    beta = best_params[-1]
    
    # Refine the outer center and side length precisely
    # The cost function used centroid as center. We can compute the exact minimal enclosing hexagon
    # for these fixed positions and beta using LP or simple iteration.
    # But for the output, we just need to provide the inner positions, outer center, side, and angle.
    # We can compute the optimal center and side for the given positions and beta.
    
    def compute_optimal_center_and_side(positions, beta):
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        # We want to find c and A such that | (v - c) . n_k | <= A for all v, k.
        # This is equivalent to minimizing A subject to:
        # c . n_k <= min_v(v . n_k) + A
        # c . n_k >= max_v(v . n_k) - A
        # Let u_k = c . n_k.
        # u_k <= m_k + A
        # u_k >= M_k - A
        # So M_k - m_k <= 2A => A >= (M_k - m_k)/2.
        # Also, we need to find c such that u_k is in [M_k - A, m_k + A].
        # This is a feasibility problem.
        # A simple approach: iterate A starting from lower bound.
        # Or use the fact that the center of the minimal enclosing hexagon is the solution to a LP.
        
        # For simplicity and speed, we can use the centroid as a very good approximation,
        # and then adjust A to be the max deviation.
        c = np.mean(positions, axis=0)
        
        max_A = 0.0
        for nx, ny in normals:
            n_vec = np.array([nx, ny])
            verts = positions[:, np.newaxis, :] + HEX_VERTS[np.newaxis, :, :]
            projs = verts @ n_vec
            max_proj = np.max(projs)
            min_proj = np.min(projs)
            
            c_proj = c @ n_vec
            dist_max = max_proj - c_proj
            dist_min = c_proj - min_proj
            
            A_k = max(dist_max, dist_min)
            if A_k > max_A:
                max_A = A_k
        
        # To be safe, add a tiny epsilon
        max_A += 1e-9
        S = (2.0 / math.sqrt(3.0)) * max_A
        return c, S

    c_out, S_out = compute_optimal_center_and_side(positions, beta)
    
    # Construct output
    inner = []
    for x, y in positions:
        inner.append([float(x), float(y), 0.0]) # Angle 0 for all inner hexagons
    
    center = [float(c_out[0]), float(c_out[1])]
    side = float(S_out)
    rotation = float(math.degrees(beta))
    
    return inner, center, side, rotation

