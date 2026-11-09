import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute unit hexagon vertices (flat-top, side=1, centered at origin)
    angles = np.deg2rad(np.arange(6) * 60)
    unit_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normals for the 6 faces of the outer hexagon (flat-top)
    normal_angles = np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    def _hex_verts(cx, cy, side, angle_deg):
        a = np.radians(angle_deg)
        return np.array([
            [cx + side * np.cos(a + np.radians(60 * i)),
             cy + side * np.sin(a + np.radians(60 * i))]
            for i in range(6)
        ])

    def _polygon_overlap_depth(A, B):
        """SAT overlap depth between convex polygons A and B (arrays of 2D verts). 0 if disjoint."""
        min_depth = float('inf')
        axes = []
        for poly in (A, B):
            n = len(poly)
            for i in range(n):
                edge = poly[(i + 1) % n] - poly[i]
                length = np.linalg.norm(edge)
                if length > 1e-12:
                    axes.append(np.array([-edge[1], edge[0]]) / length)
        for ax in axes:
            pa = np.dot(A, ax)
            pb = np.dot(B, ax)
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return 0.0
            ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            if ov < min_depth:
                min_depth = ov
        return min_depth

    def _containment_violation(inner_verts, outer_verts):
        """Sum of distances by which inner verts fall outside outer (edge-wise)."""
        n_out = len(outer_verts)
        pen = 0.0
        for iv in inner_verts:
            for i in range(n_out):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % n_out]
                edge = v2 - v1
                el = np.linalg.norm(edge)
                if el < 1e-12:
                    continue
                nrm = np.array([-edge[1], edge[0]]) / el
                sd = np.dot(iv - v1, nrm)
                if sd < 0:
                    pen += -sd
        return pen

    def _total_penalty(params, S, center, outer_angle):
        """Total penalty: sum of squares of containment + overlap. 0 means valid."""
        inner_list = []
        for i in range(N):
            inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
        outer = _hex_verts(center[0], center[1], S, outer_angle)
        
        total = 0.0
        for iv in inner_list:
            pen = _containment_violation(iv, outer)
            if pen > 0:
                total += pen * pen
        for i in range(N):
            for j in range(i+1, N):
                d = _polygon_overlap_depth(inner_list[i], inner_list[j])
                if d > 0:
                    total += d * d
        return total

    def _check_validity(params, S, center, outer_angle, tol=1e-6):
        """Check if packing is valid (no overlaps, all contained)."""
        inner_list = []
        for i in range(N):
            inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
        outer = _hex_verts(center[0], center[1], S, outer_angle)
        
        for iv in inner_list:
            if _containment_violation(iv, outer) > tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                if _polygon_overlap_depth(inner_list[i], inner_list[j]) > tol:
                    return False
        return True

    def _physics_refine(params, S, center, outer_angle, deadline):
        """Physics-based refinement."""
        params = params.copy()
        step_size = 0.1
        max_iters = 200
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Calculate forces
            forces = np.zeros(3 * N)
            inner_list = []
            for i in range(N):
                inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
            outer = _hex_verts(center[0], center[1], S, outer_angle)
            
            # Containment forces
            for i in range(N):
                iv = inner_list[i]
                # For each vertex, if outside, push inward
                for iv_pt in iv:
                    for k in range(6):
                        v1 = outer[k]
                        v2 = outer[(k + 1) % 6]
                        edge = v2 - v1
                        el = np.linalg.norm(edge)
                        if el < 1e-12: continue
                        nrm = np.array([-edge[1], edge[0]]) / el
                        sd = np.dot(iv_pt - v1, nrm)
                        if sd < 0:
                            # Force on center is opposite to normal (push in)
                            # Scale by violation
                            forces[3*i:3*i+2] += -nrm * (-sd)
            
            # Overlap forces
            for i in range(N):
                for j in range(i+1, N):
                    d = _polygon_overlap_depth(inner_list[i], inner_list[j])
                    if d > 0:
                        # Calculate push direction: from center of j to center of i
                        c_i = np.array([params[3*i], params[3*i+1]])
                        c_j = np.array([params[3*j], params[3*j+1]])
                        diff = c_i - c_j
                        dist = np.linalg.norm(diff)
                        if dist < 1e-6:
                            diff = np.array([1.0, 0.0])
                            dist = 1.0
                        dir_vec = diff / dist
                        # Apply force
                        forces[3*i:3*i+2] += dir_vec * d
                        forces[3*j:3*j+2] -= dir_vec * d
            
            # Normalize and apply
            f_norm = np.linalg.norm(forces)
            if f_norm < 1e-8:
                break
            forces = forces / f_norm
            
            # Adaptive step size
            new_params = params + step_size * forces
            new_pen = _total_penalty(new_params, S, center, outer_angle)
            old_pen = _total_penalty(params, S, center, outer_angle)
            
            if new_pen < old_pen:
                params = new_params
                # Can increase step slightly if improving fast
                step_size = min(step_size * 1.05, 0.5)
            else:
                step_size *= 0.7
            
            if step_size < 1e-6:
                break
                
        return params

    def _refine_local(params, S, center, outer_angle, deadline):
        """Hybrid refinement: Physics then L-BFGS."""
        # First try physics-based
        p_phys = _physics_refine(params, S, center, outer_angle, deadline)
        
        # Then try L-BFGS from the physics result
        def obj(p):
            return _total_penalty(p, S, center, outer_angle)
        
        try:
            res = minimize(obj, p_phys, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 5000,
                                    'ftol': 1e-14, 'gtol': 1e-10})
            if time.time() > deadline:
                return p_phys
            # Check if L-BFGS result is better
            if res.fun < _total_penalty(p_phys, S, center, outer_angle):
                return res.x
        except:
            pass
        return p_phys

    def _solve_for_S(params, center, outer_angle, deadline):
        """Binary search for the minimum S such that the configuration is valid."""
        # Lower bound: based on area or simple geometric constraint
        # Upper bound: large enough
        lo = 1.0
        hi = 10.0
        
        # Ensure hi is valid
        while not _check_validity(params, hi, center, outer_angle, tol=1e-4):
            hi *= 1.5
            if hi > 100:
                return hi # Should not happen
        
        # Binary search
        for _ in range(20):
            if time.time() > deadline:
                break
            mid = (lo + hi) / 2
            if _check_validity(params, mid, center, outer_angle, tol=1e-4):
                hi = mid
            else:
                lo = mid
        
        return hi

    # Strategy:
    # 1. Generate diverse initial configurations (lattices, rings, etc.)
    # 2. For each, run a global optimizer (DE) to find a good topology/positions
    # 3. Refine with physics + L-BFGS
    # 4. Binary search for minimal S
    # 5. Keep the best result

    best_params = None
    best_S = float('inf')
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Initial configurations
    # 1. 4x3 Grid
    xs = np.zeros(N)
    ys = np.zeros(N)
    rots = np.zeros(N)
    for i in range(N):
        row = i // 4
        col = i % 4
        xs[i] = col * 2.1 - 3.0
        ys[i] = row * 1.8 - 1.5
        rots[i] = 0.0
    init_params_1 = np.concatenate([xs, ys, rots])

    # 2. Hexagonal Lattice (approx)
    xs = np.zeros(N)
    ys = np.zeros(N)
    rots = np.zeros(N)
    # Center at 0
    # 1 in center
    xs[0] = 0; ys[0] = 0
    # 6 around
    for i in range(1, 7):
        ang = i * np.pi / 3
        xs[i] = 1.732 * np.cos(ang) # sqrt(3)
        ys[i] = 1.732 * np.sin(ang)
    # 5 more?
    # Let's just use a ring of 11 around center? Or 2 rings?
    # 12 hexes: 1 center, 6 ring 1, 5 from ring 2?
    # Let's try a ring of 12
    for i in range(1, 12):
        ang = (i-1) * 2 * np.pi / 11
        xs[i] = 2.0 * np.cos(ang)
        ys[i] = 2.0 * np.sin(ang)
    init_params_2 = np.concatenate([xs, ys, rots])

    # 3. Random
    np.random.seed(42)
    xs = np.random.uniform(-3, 3, N)
    ys = np.random.uniform(-3, 3, N)
    rots = np.random.uniform(0, 360, N)
    init_params_3 = np.concatenate([xs, ys, rots])

    initial_configs = [init_params_1, init_params_2, init_params_3]

    # Try a few outer angles
    outer_angles = [0.0, 30.0, 15.0]

    for outer_angle in outer_angles:
        if time.time() > deadline - 10:
            break
            
        for init_params in initial_configs:
            if time.time() > deadline - 10:
                break
                
            # Run DE to optimize positions and rotations for a fixed S?
            # No, better to optimize positions to minimize S directly?
            # The previous program minimized S + penalty.
            # Let's stick to that but with better penalty (SAT).
            
            def objective(params):
                # params: 36 values
                # Estimate S from params
                # Max support function
                inner_list = []
                for i in range(N):
                    inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
                
                # Calculate required S
                # For each vertex, max dot with normals
                # Normals for outer hex at origin (center fixed at 0,0 for now in objective)
                # Wait, center is fixed at 0,0 in this formulation?
                # Yes, let's fix center at 0,0 to reduce variables.
                # Outer hex is centered at 0,0 with angle outer_angle.
                
                max_support = 0.0
                for iv in inner_list:
                    for p in iv:
                        # Rotate p by -outer_angle to align with standard normals?
                        # Or rotate normals by outer_angle?
                        # Standard normals are at 30, 90, ...
                        # If outer hex is rotated by outer_angle, normals are rotated too.
                        # Let's just compute containment violation for a hypothetical S=1?
                        # No, we need S.
                        # S = 2 * max_support / sqrt(3)
                        # max_support is max distance from origin to any vertex in direction of normals.
                        # For a point p, support in direction n is dot(p, n).
                        # Normals n_k = (cos(30+60k), sin(30+60k))
                        # If outer hex is rotated by theta, normals are rotated by theta.
                        # So n_k' = (cos(30+60k+theta), sin(30+60k+theta))
                        
                        dots = np.array([np.dot(p, np.array([np.cos(np.radians(30+60*k+outer_angle)), np.sin(np.radians(30+60*k+outer_angle))])) for k in range(6)])
                        m = np.max(dots)
                        if m > max_support:
                            max_support = m
                
                outer_S = 2.0 * max_support / np.sqrt(3)
                
                # Penalty for overlaps
                # We don't check containment against a fixed S here, we derive S.
                # But we must ensure no overlaps.
                pen = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        d = _polygon_overlap_depth(inner_list[i], inner_list[j])
                        if d > 0:
                            pen += d * d
                            
                return outer_S + 1000.0 * pen

            # DE bounds
            bounds = [(-5, 5)] * 12 + [(-5, 5)] * 12 + [(0, 360)] * 12
            
            # Run DE
            try:
                res = differential_evolution(
                    objective, 
                    bounds, 
                    maxiter=50, 
                    popsize=15, 
                    tol=1e-4, 
                    seed=42,
                    x0=init_params
                )
                
                best_params_candidate = res.x
                # Get true S
                # We need to solve for S with this topology
                S_candidate = _solve_for_S(best_params_candidate, (0.0, 0.0), outer_angle, deadline)
                
                if S_candidate < best_S:
                    # Refine
                    # Create params for refinement (same as best_params_candidate)
                    refined_params = _refine_local(best_params_candidate, S_candidate, (0.0, 0.0), outer_angle, deadline)
                    S_refined = _solve_for_S(refined_params, (0.0, 0.0), outer_angle, deadline)
                    
                    if S_refined < best_S:
                        best_S = S_refined
                        best_params = refined_params
                        best_outer_angle = outer_angle
                        best_center = (0.0, 0.0)
                        
            except Exception as e:
                pass

    # If no valid solution found, fallback
    if best_params is None:
        # Use the first initial config and solve for S
        best_params = init_params_1
        best_S = _solve_for_S(best_params, (0.0, 0.0), 0.0, deadline)
        best_outer_angle = 0.0
        best_center = (0.0, 0.0)

    # Final extraction
    xs = best_params[0:N]
    ys = best_params[N:2*N]
    rots = best_params[2*N:3*N]
    
    inner = []
    for i in range(N):
        inner.append([xs[i], ys[i], rots[i]])
        
    return inner, best_center, best_S, best_outer_angle
