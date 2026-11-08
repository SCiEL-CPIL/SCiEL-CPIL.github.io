import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute unit hexagon vertices (flat-top, side=1, centered at origin)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(6) * 60)
    unit_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normals for the 6 faces of the outer hexagon (flat-top)
    # Outer hexagon with side S has "inradius" (distance to face) R = S * sqrt(3) / 2
    normal_angles = np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    def get_vertices(params):
        """
        params: (12, 3) array of [x, y, angle_rad]
        Returns: (12, 6, 2) array of vertices for each hexagon
        """
        xs = params[:, 0]
        ys = params[:, 1]
        rots = params[:, 2]
        
        cos_r = np.cos(rots)
        sin_r = np.sin(rots)
        
        # unit_verts is (6, 2)
        # We need to rotate each hexagon's vertices and translate.
        # v_rot = R @ unit_verts.T -> (2, 6)
        # v_trans = v_rot + center -> (2, 6)
        # We want (12, 6, 2)
        
        # cos_r, sin_r are (12,)
        # unit_verts is (6, 2)
        
        # Rotation matrix for each hexagon: (12, 2, 2)
        # R_i = [[cos_i, -sin_i], [sin_i, cos_i]]
        
        # v_rot[i] = R_i @ unit_verts.T -> (2, 6)
        # v_rot[i].T = unit_verts @ R_i.T -> (6, 2)
        
        # Let's compute:
        # unit_verts @ [[cos_i, sin_i], [-sin_i, cos_i]]
        
        # R_i.T = [[cos_i, sin_i], [-sin_i, cos_i]]
        
        # (6, 2) @ (2, 12) is not quite right for batch.
        # Let's do it explicitly for clarity and correctness, then vectorize if needed.
        # Since N=12 is small, a loop is fine and less error-prone.
        
        verts = np.zeros((12, 6, 2))
        for i in range(12):
            c = np.array([xs[i], ys[i]])
            r = rots[i]
            # Rotate unit vertices
            cos_r_i = np.cos(r)
            sin_r_i = np.sin(r)
            rot_mat = np.array([[cos_r_i, -sin_r_i], [sin_r_i, cos_r_i]])
            v = unit_verts @ rot_mat.T + c
            verts[i] = v
            
        return verts

    def min_dist_polygons(v1, v2):
        """
        Compute minimum distance between two convex polygons (hexagons).
        Returns 0 if they intersect.
        v1, v2: (6, 2) arrays of vertices.
        """
        min_d = float('inf')
        # Check vertex of v1 to edges of v2
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        # Check vertex of v2 to edges of v1
        for i in range(6):
            p1 = v2[i]
            for j in range(6):
                p2 = v1[j]
                p3 = v1[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        return min_d

    def evaluate(params):
        """
        Returns (outer_side_length, overlap_penalty)
        """
        verts = get_vertices(params)
        
        # 1. Calculate required outer side length
        max_support = 0.0
        for i in range(12):
            v = verts[i]
            # v is (6, 2). normals is (6, 2).
            # Compute dot products for all vertices and all normals
            # (6, 2) @ (2, 6) -> (6, 6)
            dots = v @ normals.T
            m = np.max(dots)
            if m > max_support:
                max_support = m
                
        if max_support < 1e-6:
             max_support = 1e-6
             
        outer_S = 2.0 * max_support / np.sqrt(3)
        
        # 2. Check overlap constraints
        eps = 1e-3
        overlap_penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                cx1, cy1 = params[i, 0], params[i, 1]
                cx2, cy2 = params[j, 0], params[j, 1]
                c_dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if c_dist < 2.0:
                    d = min_dist_polygons(verts[i], verts[j])
                    if d < eps:
                        overlap_penalty += (eps - d) ** 2
        
        return outer_S, overlap_penalty

    def objective_flat(params_flat):
        """
        Objective function for L-BFGS-B.
        params_flat: (36,)
        """
        params = params_flat.reshape(12, 3)
        S, pen = evaluate(params)
        return S + 1000.0 * pen

    def gradient_flat(params_flat):
        """
        Numerical gradient for L-BFGS-B.
        """
        params = params_flat.reshape(12, 3)
        grad = np.zeros_like(params_flat)
        h = 1e-6
        for i in range(36):
            params_plus = params.copy()
            params_flat_plus = params_flat.copy()
            params_flat_plus[i] += h
            params_plus = params_flat_plus.reshape(12, 3)
            
            # Quick check: if this change causes huge overlap, gradient might be noisy
            # But L-BFGS-B can handle it if we have a good initial guess
            
            f_plus = objective_flat(params_flat_plus)
            
            # For efficiency, we could compute gradient analytically, but numerical is safer for complex constraints
            # However, 36 * 2 evaluations per step is expensive.
            # Let's try a semi-analytic approach or just accept the cost if steps are few.
            
            # Actually, let's use a simpler gradient:
            # dS/dparams + 1000 * dPen/dparams
            # dPen is 0 if no overlap.
            # dS is smooth.
            
            # Let's just use finite difference for now, it's robust.
            # But 36*2 = 72 evaluations per gradient step.
            # Each evaluation is ~0.1ms? 72 * 0.1ms = 7.2ms per step.
            # 100 steps = 0.7s. Acceptable.
            
            # We need f_minus too for central difference?
            # Forward difference is cheaper.
            # Let's use forward difference.
            # grad[i] = (f_plus - f_current) / h
            # We need f_current.
            
            # To save time, let's compute f_current once outside the loop?
            # No, f_current changes if we modify params.
            # But we are evaluating f_plus for each i.
            
            # This is getting complicated. Let's just use scipy.optimize.minimize with method='L-BFGS-B' 
            # and let it compute the gradient numerically? 
            # L-BFGS-B does not support numerical gradient by default in all versions, 
            # or it's slow.
            # Better: provide the gradient.
            
            # Let's stick to a simpler approach: use 'Nelder-Mead' or 'Powell' which don't need gradients?
            # Nelder-Mead is robust but slow in high dims.
            # Powell is good for bounded problems.
            
            # Let's try Powell's method. It doesn't require gradients.
            # It works well for smooth-ish functions.
            
            # So we don't need gradient_flat.
            pass
        return grad

    def generate_initial_guesses():
        guesses = []
        
        # 1. Hexagonal Lattice
        # 12 hexagons. 
        # A hexagonal cluster of 12? 
        # 1 center, 6 around, 5 more? 
        # Or 3 rows of 4?
        # Let's try a grid that fits well.
        # Spacing: dx = 2.0, dy = sqrt(3) * 1.0 = 1.732
        # 4 cols, 3 rows.
        # Total width: 3 * 2.0 = 6.0. Plus 1 for radius = 7.0?
        # Outer side S ~ 4.
        
        xs = []
        ys = []
        rots = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 - 3.0
                y = r * 1.732 - 1.732
                xs.append(x)
                ys.append(y)
                rots.append(0.0)
        
        params1 = np.column_stack([xs, ys, rots]).flatten()
        guesses.append(params1)
        
        # 2. Radial Layout
        # 1 in center, 11 around?
        # 11 is not a multiple of 6.
        # 1 center, 6 inner ring, 5 outer ring?
        
        xs2 = [0.0]
        ys2 = [0.0]
        rots2 = [0.0]
        
        # Inner ring: 6 hexagons
        # Distance from center: sqrt(3) ~ 1.732
        for i in range(6):
            ang = np.deg2rad(i * 60)
            x = 1.732 * np.cos(ang)
            y = 1.732 * np.sin(ang)
            xs2.append(x)
            ys2.append(y)
            rots2.append(0.0)
            
        # Outer ring: 5 hexagons?
        # Just add 5 more in a larger circle
        for i in range(5):
            ang = np.deg2rad(i * 72 + 30)
            x = 3.5 * np.cos(ang)
            y = 3.5 * np.sin(ang)
            xs2.append(x)
            ys2.append(y)
            rots2.append(0.0)
            
        params2 = np.column_stack([xs2, ys2, rots2]).flatten()
        guesses.append(params2)
        
        # 3. Random Jittered Grid
        np.random.seed(42)
        xs3 = []
        ys3 = []
        rots3 = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 - 3.0 + np.random.uniform(-0.1, 0.1)
                y = r * 1.732 - 1.732 + np.random.uniform(-0.1, 0.1)
                xs3.append(x)
                ys3.append(y)
                rots3.append(np.random.uniform(0, 2*np.pi))
                
        params3 = np.column_stack([xs3, ys3, rots3]).flatten()
        guesses.append(params3)
        
        # 4. Another random
        np.random.seed(123)
        xs4 = []
        ys4 = []
        rots4 = []
        for i in range(12):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            xs4.append(x)
            ys4.append(y)
            rots4.append(np.random.uniform(0, 2*np.pi))
            
        params4 = np.column_stack([xs4, ys4, rots4]).flatten()
        guesses.append(params4)
        
        return guesses

    best_S = float('inf')
    best_params = None
    best_pen = float('inf')
    
    guesses = generate_initial_guesses()
    
    for idx, guess in enumerate(guesses):
        if time.time() > deadline - 10.0:
            break
            
        # Use Powell's method
        # Bounds: x, y in [-5, 5], rot in [0, 2*pi]
        bounds = [(-5, 5)] * 12 + [(-5, 5)] * 12 + [(0, 2*np.pi)] * 12
        
        try:
            res = minimize(
                objective_flat, 
                guess, 
                method='Powell', 
                bounds=bounds,
                options={'maxiter': 200, 'maxfev': 5000}
            )
            
            if res.fun < best_S:
                best_params = res.x
                best_S = res.fun
                # Check penalty
                S_val, pen_val = evaluate(res.x.reshape(12, 3))
                best_pen = pen_val
                
        except Exception as e:
            pass
            
    # If no good result, use the first guess
    if best_params is None:
        best_params = guesses[0]
        best_S, best_pen = evaluate(best_params.reshape(12, 3))
    else:
        # Re-evaluate to get true S and penalty
        best_S, best_pen = evaluate(best_params.reshape(12, 3))
        
    # If the best solution has significant overlap, try to repair or fallback
    if best_pen > 1e-4:
        # Try a simple relaxation: push apart overlapping hexagons
        params = best_params.reshape(12, 3)
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            verts = get_vertices(params)
            moved = False
            for i in range(12):
                for j in range(i+1, 12):
                    d = min_dist_polygons(verts[i], verts[j])
                    if d < 1e-3:
                        # Push apart
                        dx = params[i, 0] - params[j, 0]
                        dy = params[i, 1] - params[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 1e-6:
                            dx, dy = 1, 0
                            dist = 1
                        else:
                            dx /= dist
                            dy /= dist
                            
                        push = 0.1
                        params[i, 0] += dx * push
                        params[i, 1] += dy * push
                        params[j, 0] -= dx * push
                        params[j, 1] -= dy * push
                        moved = True
                        
            if not moved:
                break
                
        # Re-evaluate
        new_S, new_pen = evaluate(params)
        if new_pen < best_pen:
            best_params = params.flatten()
            best_S = new_S
            best_pen = new_pen
            
    # Final check: if still invalid, use safe grid
    if best_pen > 1e-4:
        xs_safe = []
        ys_safe = []
        rots_safe = []
        for r in range(3):
            for c in range(4):
                x = c * 2.1 - 3.0
                y = r * 1.8 - 1.5
                xs_safe.append(x)
                ys_safe.append(y)
                rots_safe.append(0.0)
        safe_params = np.column_stack([xs_safe, ys_safe, rots_safe]).flatten()
        safe_S, safe_pen = evaluate(safe_params.reshape(12, 3))
        if safe_pen < best_pen:
            best_params = safe_params
            best_S = safe_S
            best_pen = safe_pen

    # Extract final solution
    params = best_params.reshape(12, 3)
    xs = params[:, 0]
    ys = params[:, 1]
    rots = params[:, 2]
    rots_deg = np.rad2deg(rots)
    
    inner = []
    for i in range(12):
        inner.append([xs[i], ys[i], rots_deg[i]])
        
    # Calculate true outer side length
    true_S, _ = evaluate(params)
    
    return inner, (0.0, 0.0), true_S, 0.0

if __name__ == "__main__":
    construct()
