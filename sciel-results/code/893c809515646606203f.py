"""Improved packing of 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12
DEADLINE = 138.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices (pointy-top, angle 0 means vertex at 0 deg)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    sqrt3 = np.sqrt(3.0)
    
    def time_ok():
        return (time.time() - t0) < DEADLINE

    def get_vertices(params):
        """Get all vertices for N hexagons from params [x1,y1,a1, x2,y2,a2, ...]."""
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            x, y, a = params[i*3], params[i*3+1], params[i*3+2]
            rad = np.deg2rad(a)
            c, s = np.cos(rad), np.sin(rad)
            rot = np.array([[c, -s], [s, c]])
            verts[i] = _UNIT_HEX_VERTS @ rot.T + np.array([x, y])
        return verts

    def sat_check_fast(v1, v2):
        """Check if two convex hexagons are disjoint using SAT. Returns True if disjoint."""
        c1 = np.mean(v1, axis=0)
        c2 = np.mean(v2, axis=0)
        dx, dy = c1[0]-c2[0], c1[1]-c2[1]
        if dx*dx + dy*dy > 4.001:
            return True
        
        # 6 normals from v1
        for k in range(6):
            e = v1[(k+1)%6] - v1[k]
            nx, ny = -e[1], e[0]
            n = np.sqrt(nx*nx + ny*ny)
            if n < 1e-12:
                continue
            nx, ny = nx/n, ny/n
            p1 = v1 @ np.array([nx, ny])
            p2 = v2 @ np.array([nx, ny])
            if np.max(p1) <= np.min(p2) + 1e-9 or np.max(p2) <= np.min(p1) + 1e-9:
                return True
        
        # 6 normals from v2
        for k in range(6):
            e = v2[(k+1)%6] - v2[k]
            nx, ny = -e[1], e[0]
            n = np.sqrt(nx*nx + ny*ny)
            if n < 1e-12:
                continue
            nx, ny = nx/n, ny/n
            p1 = v1 @ np.array([nx, ny])
            p2 = v2 @ np.array([nx, ny])
            if np.max(p1) <= np.min(p2) + 1e-9 or np.max(p2) <= np.min(p1) + 1e-9:
                return True
        
        return False

    def all_disjoint(verts, eps=1e-9):
        for i in range(N):
            for j in range(i+1, N):
                if not sat_check_fast(verts[i], verts[j]):
                    return False
        return True

    def min_outer_side(verts, outer_angle_deg=0.0):
        """Compute minimum side length S of outer hexagon (at given rotation) containing all inner hexagons."""
        # Outer hexagon with side S has apothem a = S*sqrt(3)/2
        # 6 edge normals at angles: outer_angle + 30 + 60*k
        base = np.radians(outer_angle_deg + 30)
        # 3 unique directions (opposite pairs)
        dirs = np.array([
            [np.cos(base + np.radians(60*k)), np.sin(base + np.radians(60*k))]
            for k in range(3)
        ])
        
        max_apothem = 0.0
        for i in range(N):
            dots = verts[i] @ dirs.T
            max_apothem = max(max_apothem, np.max(np.abs(dots)))
        
        return 2.0 * max_apothem / sqrt3

    def objective(params):
        verts = get_vertices(params)
        if not all_disjoint(verts):
            return 50.0 + min_outer_side(verts)
        return min_outer_side(verts)

    def objective_with_angle(params):
        """params: [x1,y1,a1,...,x12,y12,a12, outer_angle]"""
        inner_params = params[:N*3]
        outer_angle = params[N*3]
        verts = get_vertices(inner_params)
        if not all_disjoint(verts):
            return 50.0 + min_outer_side(verts, outer_angle)
        return min_outer_side(verts, outer_angle)

    # Generate initial hexagonal lattice configuration
    def hex_lattice(scale=1.0):
        points = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = 1.5 * q * scale
                y = sqrt3 * (r + 0.5 * q) * scale
                points.append((x, y))
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init = points[:N]
        params = []
        for x, y in init:
            params.extend([x, y, 0.0])
        return np.array(params)

    # Best result tracking
    best_params = None
    best_score = float('inf')

    def update_best(params, score):
        nonlocal best_params, best_score
        if score < best_score:
            best_score = score
            best_params = params.copy()

    # Phase 1: Multiple DE runs from different initial configs
    bounds_inner = [(-5, 5), (-5, 5), (0, 60)] * N
    bounds_full = bounds_inner + [(0, 60)]  # Add outer angle

    initial_configs = [hex_lattice(s) for s in [1.0, 1.02, 1.05, 1.1]]
    
    random.seed(12345)
    for _ in range(4):
        params = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            params.extend([x, y, a])
        initial_configs.append(np.array(params))

    # DE with outer angle
    from scipy.optimize import differential_evolution
    
    for idx, init in enumerate(initial_configs):
        if not time_ok():
            break
        full_init = np.concatenate([init, [0.0]])
        try:
            de_res = differential_evolution(
                objective_with_angle,
                bounds_full,
                x0=full_init,
                maxiter=40,
                popsize=12,
                tol=1e-5,
                seed=100+idx,
                polish=False,
                update_interval=3,
                callback=lambda xk, conv: False if not time_ok() else None
            )
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_params = de_res.x.copy()
        except:
            pass

    # Phase 2: Nelder-Mead polish
    if best_params is not None and time_ok():
        try:
            nm_res = minimize(
                objective_with_angle,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'disp': False}
            )
            if nm_res.fun < best_score:
                best_score = nm_res.fun
                best_params = nm_res.x.copy()
        except:
            pass

    # Phase 3: If still no valid result, use fallback
    if best_params is None or best_score > 100.0:
        for scale in [1.02, 1.05, 1.08, 1.12, 1.15]:
            p = hex_lattice(scale)
            full_p = np.concatenate([p, [0.0]])
            s = objective_with_angle(full_p)
            if s < best_score:
                best_score = s
                best_params = full_p.copy()
            if s < 50.0:
                break

    # Phase 4: Final high-precision polish with tighter tolerances
    if best_params is not None and time_ok():
        try:
            nm_res2 = minimize(
                objective_with_angle,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
            )
            if nm_res2.fun < best_score:
                best_score = nm_res2.fun
                best_params = nm_res2.x.copy()
        except:
            pass

    # Extract results
    if best_params is None:
        best_params = np.concatenate([hex_lattice(1.1), [0.0]])
        best_score = objective_with_angle(best_params)

    inner_params = best_params[:N*3]
    outer_angle = best_params[N*3]
    
    # Compute final side length
    verts = get_vertices(inner_params)
    
    # Validate
    if not all_disjoint(verts):
        # Fallback to valid config
        for scale in [1.05, 1.1, 1.15, 1.2]:
            p = hex_lattice(scale)
            v = get_vertices(p)
            if all_disjoint(v):
                inner_params = p
                outer_angle = 0.0
                best_score = min_outer_side(v, 0.0)
                break
    
    inner = []
    for i in range(N):
        inner.append([float(inner_params[i*3]), float(inner_params[i*3+1]), float(inner_params[i*3+2])])
    
    verts_final = get_vertices(inner_params)
    S_final = min_outer_side(verts_final, outer_angle)
    
    center = (0.0, 0.0)
    side = float(S_final)
    angle = float(outer_angle)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
