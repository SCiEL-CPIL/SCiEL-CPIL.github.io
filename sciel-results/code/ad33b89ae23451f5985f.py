import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    projections = all_verts_flat @ normals.T
    max_projections = np.max(projections, axis=0)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    return min_s

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            separated = False
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Initial guess: Hexagonal lattice arrangement
    h = np.sqrt(3)/2
    init_pos = np.array([
        [0, 0],
        [2*h, 0],
        [h, 1.5],
        [-h, 1.5],
        [-2*h, 0],
        [-h, -1.5],
        [h, -1.5],
        [3*h, 1.5],
        [0, 3],
        [-3*h, 1.5],
        [-3*h, -1.5],
        [3*h, -1.5]
    ])
    
    def objective(x):
        positions = x[:24].reshape(12, 2)
        angle_deg = x[24]
        s = _min_outer_side(positions, angle_deg)
        return s

    def make_constraint(i, j):
        """
        Returns a function that calculates the minimum separation distance
        between hexagon i and hexagon j.
        We require this to be >= 0 (with a small epsilon for safety).
        """
        def constraint(x):
            positions = x[:24].reshape(12, 2)
            pi = positions[i]
            pj = positions[j]
            
            vi = _hex_vertices(pi[0], pi[1], 1, 0)
            vj = _hex_vertices(pj[0], pj[1], 1, 0)
            
            min_sep = float('inf')
            
            # Check axes from i
            for k in range(6):
                v1 = vi[k]
                v2 = vi[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                
                # Separation is the gap between intervals
                # max_i - min_j or max_j - min_i
                # If positive, they are separated. If negative, they overlap.
                # We want the *minimum* separation (closest approach) to be positive.
                # Actually, SAT says if ANY axis has a gap, they are separated.
                # The constraint is that they must NOT overlap.
                # Overlap happens if for ALL axes, the intervals intersect.
                # So we need: Exists axis k such that gap_k > 0.
                # This is a disjunctive constraint, hard for SLSQP.
                # However, for convex polygons, if they are separated, the maximum gap
                # is the distance. If they are touching, max gap is 0.
                # If they overlap, max gap is < 0.
                # So we can constrain: max_k (gap_k) >= 0.
                # gap_k = max(min_i - max_j, min_j - max_i)
                
                gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                if gap > min_sep:
                    min_sep = gap
            
            # Check axes from j
            for k in range(6):
                v1 = vj[k]
                v2 = vj[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                
                gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                if gap > min_sep:
                    min_sep = gap
            
            # We want min_sep >= -epsilon (allowing tiny overlap for numerical stability, 
            # but ideally >= 0). 
            # Using -1e-5 to help SLSQP converge.
            return min_sep - (-1e-5)
        
        return constraint

    constraints = []
    for i in range(12):
        for j in range(i+1, 12):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
    # Multi-start SLSQP
    num_restarts = 5
    np.random.seed(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            x0 = np.concatenate([init_pos.flatten(), [0.0]])
        else:
            noise_pos = np.random.randn(12, 2) * 0.3
            noise_angle = np.random.uniform(-5, 5)
            x0 = np.concatenate([(init_pos + noise_pos).flatten(), [noise_angle]])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
        except Exception:
            continue
            
        if res.success:
            x_opt = res.x
            positions = x_opt[:24].reshape(12, 2)
            angle_deg = x_opt[24]
            
            inner_candidate = np.zeros((12, 3))
            inner_candidate[:, :2] = positions
            inner_candidate[:, 2] = 0.0
            
            outer_candidate = np.array([0, 0, angle_deg])
            s_candidate = _min_outer_side(positions, angle_deg)
            
            if _check_validity(inner_candidate, outer_candidate, s_candidate):
                score_candidate = 1.0 / s_candidate
                if score_candidate > best_score:
                    best_score = score_candidate
                    best_inner = inner_candidate
                    best_outer = outer_candidate
                    best_s = s_candidate
        else:
            # Even if SLSQP fails, check if the result is valid (it might have found a local min)
            x_opt = res.x
            positions = x_opt[:24].reshape(12, 2)
            angle_deg = x_opt[24]
            
            inner_candidate = np.zeros((12, 3))
            inner_candidate[:, :2] = positions
            inner_candidate[:, 2] = 0.0
            outer_candidate = np.array([0, 0, angle_deg])
            s_candidate = _min_outer_side(positions, angle_deg)
            
            if _check_validity(inner_candidate, outer_candidate, s_candidate):
                score_candidate = 1.0 / s_candidate
                if score_candidate > best_score:
                    best_score = score_candidate
                    best_inner = inner_candidate
                    best_outer = outer_candidate
                    best_s = s_candidate

    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback
        inner_fallback = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.165, 0],
            [-1.25, 2.165, 0],
            [1.25, -2.165, 0],
            [-1.25, -2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, -2.165, 0],
            [-3.75, -2.165, 0],
            [0, 4.33, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END
