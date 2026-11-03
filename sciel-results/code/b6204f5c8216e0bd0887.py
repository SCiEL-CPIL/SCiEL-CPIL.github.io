"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import random

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = np.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()

    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS

    def get_all_vertices(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        verts_list = []
        for i in range(N):
            x, y, a = params[i*3], params[i*3+1], params[i*3+2]
            rad = np.deg2rad(a)
            rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            verts = unit_hex_verts @ rot.T + np.array([x, y])
            verts_list.append(verts)
        return verts_list

    def hexagons_disjoint_fast(verts_list, eps=1e-7):
        for i in range(N):
            for j in range(i+1, N):
                v1, v2 = verts_list[i], verts_list[j]
                # Check all 12 normals (6 from each)
                # Actually, for convex polygons, checking the 12 edge normals is sufficient for SAT.
                # But we can optimize by just checking distance between centers if they are far apart?
                # No, let's stick to SAT but make it fast.
                
                # Quick reject: if distance between centers > 2*radius (which is 2 for unit hex), they are disjoint.
                # Wait, radius of unit hex is 1. So if dist > 2, disjoint.
                # Actually, the max distance from center to vertex is 1.
                # So if dist(center1, center2) > 2, they are definitely disjoint.
                c1 = np.mean(v1, axis=0)
                c2 = np.mean(v2, axis=0)
                if np.linalg.norm(c1 - c2) > 2.0 + eps:
                    continue

                # SAT check
                overlap = True
                # Normals from v1
                for k in range(6):
                    n = edge_normals[k]
                    proj1 = v1 @ n
                    proj2 = v2 @ n
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 <= min2 + eps or max2 <= min1 + eps:
                        overlap = False
                        break
                if not overlap:
                    continue

                # Normals from v2 (rotated)
                # We need the normals of v2 in global frame.
                # v2 is unit_hex_verts @ rot.T + c2.
                # The edges are also rotated.
                # Instead of recalculating, we can just use the same logic:
                # The normals of v2 are the rotated edge_normals.
                # But we can just check the 6 normals of v2 directly.
                # Actually, SAT requires checking normals of both polygons.
                # Since v2 is rotated, its normals are rotated.
                # Let's compute the normals for v2 explicitly.
                for k in range(6):
                    # Edge from v2[k] to v2[k+1]
                    e = v2[(k+1)%6] - v2[k]
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-9: continue
                    n /= norm
                    proj1 = v1 @ n
                    proj2 = v2 @ n
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 <= min2 + eps or max2 <= min1 + eps:
                        overlap = False
                        break
                if overlap:
                    return False
        return True

    def compute_min_outer_side(verts_list):
        # Minimize S such that all vertices are inside hexagon of side S centered at origin.
        # Hexagon inequalities (flat-top, centered at 0):
        # 1. |x| <= S
        # 2. |sqrt(3)/2 * x + 1/2 * y| <= sqrt(3)/2 * S  => |x + y/sqrt(3)| <= S
        # 3. |sqrt(3)/2 * x - 1/2 * y| <= sqrt(3)/2 * S  => |x - y/sqrt(3)| <= S
        
        # Actually, for a regular hexagon with side S centered at origin:
        # The support function h(theta) = S * cos(theta - theta_i) for theta in [theta_i - pi/3, theta_i + pi/3]
        # where theta_i are the normal angles of the sides.
        # For flat-top hexagon, side normals are at 0, 60, 120, 180, 240, 300.
        # Wait, the inequality |x| <= S corresponds to normal (1,0).
        # The other two are at 60 and 120 degrees?
        # Let's verify.
        # Vertices of outer hexagon with side S:
        # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
        # Normal to edge from (S,0) to (S/2, S*sqrt(3)/2):
        # Edge vector: (-S/2, S*sqrt(3)/2). Normal: (S*sqrt(3)/2, S/2) -> (sqrt(3)/2, 1/2). Angle 30 deg.
        # So normals are at 30, 90, 150, 210, 270, 330 degrees.
        
        # Let's use the 3 pairs of opposite sides.
        # 1. Normal (1, 0). Inequality: |x| <= S.
        # 2. Normal (sqrt(3)/2, 1/2). Inequality: |sqrt(3)/2 * x + 1/2 * y| <= S.
        #    Wait, if the normal is unit vector n, the inequality is n.p <= S.
        #    For the side with normal (sqrt(3)/2, 1/2), the max dot product with vertices of the outer hex is S.
        #    Let's check vertex (S, 0). Dot = S*sqrt(3)/2. This is not S.
        #    Ah, the distance from center to side is the apothem, which is S * sqrt(3)/2.
        #    So the inequality is |n.p| <= S * sqrt(3)/2.
        #    Therefore, S >= |n.p| / (sqrt(3)/2) = 2/3 * sqrt(3) * |n.p|.
        
        # Let's re-derive.
        # Outer hex side S. Apothem a = S * sqrt(3)/2.
        # The constraints are |n_i . p| <= a for i=0..5.
        # We want to minimize S, which is equivalent to minimizing a.
        # So we find the minimum a such that all points satisfy the constraints.
        # a = max over all points p and all normals n_i of (n_i . p).
        # Then S = 2 * a / sqrt(3).
        
        # Normals:
        # n0 = (1, 0)
        # n1 = (sqrt(3)/2, 1/2)
        # n2 = (-sqrt(3)/2, 1/2)
        # n3 = (-1, 0)
        # n4 = (-sqrt(3)/2, -1/2)
        # n5 = (sqrt(3)/2, -1/2)
        
        # We can just check the 3 unique directions (since n and -n give same absolute value).
        # dirs = [(1, 0), (sqrt(3)/2, 1/2), (-sqrt(3)/2, 1/2)]
        # a = max(abs(d . p)) for all p, all d.
        
        sqrt3 = np.sqrt(3)
        dirs = np.array([
            [1.0, 0.0],
            [sqrt3/2.0, 0.5],
            [-sqrt3/2.0, 0.5]
        ])
        
        max_a = 0.0
        for verts in verts_list:
            # verts: (6, 2)
            # dots: (6, 3)
            dots = verts @ dirs.T
            abs_dots = np.abs(dots)
            max_a = max(max_a, np.max(abs_dots))
            
        S = 2.0 * max_a / sqrt3
        return S

    def objective(params):
        verts_list = get_all_vertices(params)
        
        # Check disjointness
        if not hexagons_disjoint_fast(verts_list, eps=1e-7):
            # Penalty for overlap
            # Calculate total overlap depth?
            # For now, just return a large value.
            # To make it differentiable-ish, we could sum the penetration depths.
            # But simple penalty is fine for DE.
            return 1000.0 + compute_min_outer_side(verts_list)
        
        return compute_min_outer_side(verts_list)

    # Phase 1: Differential Evolution
    # Bounds: x, y in [-5, 5], angle in [0, 60] (due to symmetry)
    bounds = [(-5, 5), (-5, 5), (0, 60)] * N
    
    # Initial population
    # Use hex lattice as a good starting point for some individuals
    # But DE needs a population.
    
    # Let's run DE for a while.
    def time_check():
        return (time.time() - t0) < DEADLINE

    best_result = None
    best_score = float('inf')
    
    # Run DE
    try:
        # maxiter and popsize need to be tuned for time.
        # Each eval is fast.
        # Let's try popsize=20, maxiter=50.
        # 20 * 50 * 36 (3*12) = 36000 evals?
        # DE evaluates popsize * maxiter * (something) times.
        # Actually, DE generates popsize new individuals per iteration.
        # So total evals ~ popsize * maxiter.
        # 20 * 50 = 1000 evals.
        # Each eval is ~1ms. So 1 second.
        # We can do more.
        
        # Let's do multiple runs with different settings or just one big run.
        # Let's use a tighter time budget for DE and then polish.
        
        de_result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-4, 
            seed=42,
            polish=False,  # We will polish later
            update_interval=10,
            callback=lambda xk, convergence: False if not time_check() else None
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
            
    except Exception as e:
        pass

    # Phase 2: Polish with Nelder-Mead
    if best_result is not None and time_check():
        try:
            nm_result = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_result = nm_result.x
        except:
            pass

    # Phase 3: Local Search / Random Restarts
    # If we have time, try to improve further.
    # Or just ensure we have a valid result.
    
    if best_result is None:
        # Fallback: Hex lattice
        points = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                x = 1.5 * q
                y = np.sqrt(3) * (r + 0.5 * q)
                points.append((x, y))
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_points = points[:N]
        scale = 1.01
        best_result = []
        for x, y in init_points:
            best_result.extend([x * scale, y * scale, 0.0])
        best_result = np.array(best_result)
        best_score = objective(best_result)

    # Final Check
    verts_list = get_all_vertices(best_result)
    if not hexagons_disjoint_fast(verts_list, eps=1e-6):
        # If invalid, try to fix or fallback
        # For now, just return it, hoping it's close enough or the evaluator is lenient?
        # No, better to return a valid one.
        # Let's use the fallback lattice which is guaranteed valid (with scale 1.01).
        points = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                x = 1.5 * q
                y = np.sqrt(3) * (r + 0.5 * q)
                points.append((x, y))
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_points = points[:N]
        scale = 1.02
        best_result = []
        for x, y in init_points:
            best_result.extend([x * scale, y * scale, 0.0])
        best_result = np.array(best_result)
        best_score = objective(best_result)

    inner = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        a = best_result[i*3+2]
        inner.append([float(x), float(y), float(a)])
        
    S_final = compute_min_outer_side(get_all_vertices(best_result))
    
    center = (0.0, 0.0)
    side = float(S_final)
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
