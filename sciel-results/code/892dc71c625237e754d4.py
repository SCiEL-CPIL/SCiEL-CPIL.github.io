import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

def _hex_verts(cx, cy, angle, S=1.0):
    # Regular hexagon vertices
    base = np.pi / 6 + np.arange(6) * np.pi / 3
    ang = base + angle
    return np.column_stack((cx + S * np.cos(ang), cy + S * np.sin(ang)))

def _min_dist_hex_hex(v1, v2):
    # Minimum distance between two convex hexagons (6 vertices each)
    # Check all vertices of v1 against all edges of v2, and vice versa
    min_d = np.inf
    for i in range(6):
        p = v1[i]
        # Against edges of v2
        for j in range(6):
            a = v2[j]
            b = v2[(j + 1) % 6]
            ab = b - a
            ap = p - a
            len_sq = np.dot(ab, ab)
            if len_sq < 1e-12:
                d = np.linalg.norm(ap)
            else:
                t = np.clip(np.dot(ap, ab) / len_sq, 0.0, 1.0)
                proj = a + t * ab
                d = np.linalg.norm(p - proj)
            if d < min_d:
                min_d = d
    for i in range(6):
        p = v2[i]
        for j in range(6):
            a = v1[j]
            b = v1[(j + 1) % 6]
            ab = b - a
            ap = p - a
            len_sq = np.dot(ab, ab)
            if len_sq < 1e-12:
                d = np.linalg.norm(ap)
            else:
                t = np.clip(np.dot(ap, ab) / len_sq, 0.0, 1.0)
                proj = a + t * ab
                d = np.linalg.norm(p - proj)
            if d < min_d:
                min_d = d
    return min_d

def _min_dist_to_boundary(v, outer_verts):
    # Minimum distance from inner hexagon vertices to outer boundary edges
    min_d = np.inf
    for i in range(6):
        p = v[i]
        for j in range(6):
            a = outer_verts[j]
            b = outer_verts[(j + 1) % 6]
            ab = b - a
            ap = p - a
            len_sq = np.dot(ab, ab)
            if len_sq < 1e-12:
                d = np.linalg.norm(ap)
            else:
                t = np.clip(np.dot(ap, ab) / len_sq, 0.0, 1.0)
                proj = a + t * ab
                d = np.linalg.norm(p - proj)
            if d < min_d:
                min_d = d
    return min_d

def construct():
    # We'll use a two-phase approach:
    # Phase 1: Use differential evolution to find a good initial configuration
    # Phase 2: Refine with L-BFGS-B
    
    def make_objective(outer_angle):
        def objective(params):
            S = params[-1]
            if S < 1.5:
                S = 1.5
            S = max(S, 1.5)
            
            centers_x = params[0:2*N:3]
            centers_y = params[1:2*N+1:3]
            angles = params[2:2*N+2:3]
            
            inner_verts = []
            for i in range(N):
                inner_verts.append(_hex_verts(centers_x[i], centers_y[i], angles[i]))
            
            outer_verts = _hex_verts(0.0, 0.0, outer_angle, S=S)
            
            # Compute minimum clearance
            min_clearance = np.inf
            
            # Check inner-inner distances
            for i in range(N):
                for j in range(i + 1, N):
                    d = _min_dist_hex_hex(inner_verts[i], inner_verts[j])
                    if d < min_clearance:
                        min_clearance = d
            
            # Check inner-outer distances
            for i in range(N):
                d = _min_dist_to_boundary(inner_verts[i], outer_verts)
                if d < min_clearance:
                    min_clearance = d
            
            # We want to minimize S, but with a penalty if min_clearance < 0
            # Actually, min_clearance should be >= 0 for valid packing
            # If min_clearance < 0, there's overlap
            penalty = 0.0
            if min_clearance < 0:
                penalty = 1000.0 * min_clearance ** 2
            
            return S + penalty
        
        return objective

    # Phase 1: Differential Evolution for global search
    # Variables: x1, y1, a1, ..., xN, yN, aN, S
    # Bounds:
    #   x, y in [-10, 10]
    #   angle in [0, 2*pi]
    #   S in [2.0, 10.0]
    
    bounds = []
    for _ in range(N):
        bounds.append((-10.0, 10.0))  # x
        bounds.append((-10.0, 10.0))  # y
        bounds.append((0.0, 2 * np.pi))  # angle
    bounds.append((2.0, 10.0))  # S
    
    # Try a few different outer angles
    best_S = np.inf
    best_params = None
    
    outer_angles_to_try = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]
    
    for oa in outer_angles_to_try:
        obj = make_objective(oa)
        
        try:
            # Use differential evolution
            de_result = differential_evolution(
                obj,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            
            if de_result.fun < best_S:
                best_S = de_result.fun
                best_params = de_result.x.copy()
                best_oa = oa
                
        except Exception as e:
            pass
    
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Phase 2: Refine with L-BFGS-B
    obj = make_objective(best_oa)
    
    try:
        res = minimize(
            obj,
            best_params,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x.copy()
    except Exception as e:
        pass
    
    # Extract results
    S_final = best_params[-1]
    S_final = max(S_final, 1.5)
    
    inner = []
    for i in range(N):
        x = best_params[3 * i]
        y = best_params[3 * i + 1]
        ang = best_params[3 * i + 2]
        # Normalize angle to [0, 360)
        ang_deg = np.degrees(ang) % 360.0
        inner.append([x, y, ang_deg])
    
    outer_center = (0.0, 0.0)
    outer_side = S_final
    outer_angle = best_oa
    
    return inner, outer_center, outer_side, outer_angle
