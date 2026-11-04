"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = math.sqrt(3)

    # Unit hexagon vertices at 30, 90, 150, 210, 270, 330
    unit_verts = np.array([
        [math.cos(math.radians(30)), math.sin(math.radians(30))],
        [math.cos(math.radians(90)), math.sin(math.radians(90))],
        [math.cos(math.radians(150)), math.sin(math.radians(150))],
        [math.cos(math.radians(210)), math.sin(math.radians(210))],
        [math.cos(math.radians(270)), math.sin(math.radians(270))],
        [math.cos(math.radians(330)), math.sin(math.radians(330))],
    ])

    # Support function for a unit hexagon in direction u
    def support_hex(u):
        # u is a 2D vector
        dots = unit_verts @ u
        return float(np.max(dots))

    # Minimum enclosing regular hexagon side length
    # centers: (N,2) array, alpha: rotation of inner hexes (radians), beta: rotation of outer (radians)
    def calc_S(centers, alpha, beta):
        # For each of 6 directions of outer hexagon normals:
        # direction of normal k: theta_k = beta + pi/6 + k*pi/3
        # constraint: max_i (centers[i] . n_k + h_unit(R_{-alpha} n_k)) <= side * sqrt(3)/2
        # so side >= (2/sqrt(3)) * max over k of max_i (centers[i] . n_k + h_unit(R_{-alpha} n_k))

        max_val = 0.0
        for k in range(6):
            theta = beta + math.pi/6 + k * math.pi/3
            n = np.array([math.cos(theta), math.sin(theta)])
            # Rotate n by -alpha
            c = math.cos(-alpha)
            s = math.sin(-alpha)
            R = np.array([[c, -s], [s, c]])
            n_rot = R @ n
            h = support_hex(n_rot)
            # Project all centers
            projs = centers @ n + h
            m = float(np.max(projs))
            if m > max_val:
                max_val = m
        return 2.0 * max_val / sqrt3

    # Precompute for a given alpha: the 6 rotated normals and their support values
    # Then calc_S becomes: max over k of max_i (centers . n_k_rot + h_k)
    # = max over k of (max_i (centers . n_k_rot)) + h_k  ... wait no
    # Actually: max_i (centers[i] . n_k + h_k) = (max_i centers[i].n_k) + h_k
    # So calc_S = (2/sqrt3) * max_k [(max_i centers[i].n_k) + h_k]
    
    # This is a convex function of centers (max of linear functions).
    
    # Topology generation: axial to cartesian
    def axial_to_cartesian(axial, scale=2.0):
        pts = []
        for q, r in axial:
            x = scale * (q + 0.5 * r)
            y = scale * (sqrt3 / 2.0) * r
            pts.append((x, y))
        return np.array(pts)

    # Generate several topologies
    def gen_topologies():
        tops = []
        # Parallelogram m x n
        for m in range(2, 7):
            for n in range(2, 7):
                if m * n == 12:
                    top = [(q, r) for q in range(m) for r in range(n)]
                    if len(set(top)) == 12:
                        tops.append(top)
        # Hexagonal cluster: center + 6 ring + 5 from ring 2
        # Ring 2 positions: (2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        center = (0,0)
        # Pick subsets of ring2 of size 5
        import itertools
        for combo in itertools.combinations(range(10), 5):
            top = [center] + ring1 + [ring2[i] for i in combo]
            if len(set(top)) == 12:
                tops.append(top)
        # Remove duplicates
        seen = set()
        unique = []
        for t in tops:
            key = frozenset(t)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    tops = gen_topologies()
    
    best_S = float('inf')
    best_centers = None
    best_alpha = 0.0
    best_beta = 0.0

    # For each topology, optimize alpha, beta, and center
    for t_idx, top in enumerate(tops):
        if time.time() > deadline - 5:
            break
        
        base_centers = axial_to_cartesian(top, scale=2.0)
        # Center it
        cx = np.mean(base_centers[:, 0])
        cy = np.mean(base_centers[:, 1])
        base_centers = base_centers - np.array([cx, cy])

        # Grid search over alpha, beta
        # Period is 60 degrees for both
        alpha_grid = np.arange(0, 60, 3.0)
        beta_grid = np.arange(0, 60, 3.0)
        
        best_local_S = float('inf')
        best_local_a = 0.0
        best_local_b = 0.0
        best_local_centers = base_centers.copy()
        
        for a_deg in alpha_grid:
            if time.time() > deadline - 5:
                break
            a = math.radians(a_deg)
            for b_deg in beta_grid:
                if time.time() > deadline - 5:
                    break
                b = math.radians(b_deg)
                S = calc_S(base_centers, a, b)
                if S < best_local_S:
                    best_local_S = S
                    best_local_a = a
                    best_local_b = b
                    best_local_centers = base_centers.copy()

        # Local refinement using scipy
        def objective(params):
            cx_p, cy_p, a_p, b_p = params
            shifted = base_centers + np.array([cx_p, cy_p])
            return calc_S(shifted, a_p, b_p)
        
        x0 = [0.0, 0.0, best_local_a, best_local_b]
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_local_S:
                best_local_S = res.fun
                cx_p, cy_p, a_p, b_p = res.x
                best_local_centers = base_centers + np.array([cx_p, cy_p])
                best_local_a = a_p
                best_local_b = b_p
        except:
            pass

        if best_local_S < best_S:
            best_S = best_local_S
            best_centers = best_local_centers.copy()
            best_alpha = best_local_a
            best_beta = best_local_b

    # Final polish
    if time.time() < deadline - 2 and best_centers is not None:
        def obj_final(params):
            cx_p, cy_p, a_p, b_p = params
            shifted = best_centers + np.array([cx_p, cy_p])
            return calc_S(shifted, a_p, b_p)
        
        x0 = [0.0, 0.0, best_alpha, best_beta]
        try:
            res = minimize(obj_final, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                cx_p, cy_p, a_p, b_p = res.x
                best_centers = best_centers + np.array([cx_p, cy_p])
                best_alpha = a_p
                best_beta = b_p
        except:
            pass

    if best_centers is None:
        # Fallback: simple 3x4
        top = [(q, r) for q in range(3) for r in range(4)]
        best_centers = axial_to_cartesian(top, scale=2.0)
        best_alpha = 0.0
        best_beta = 0.0
        best_S = calc_S(best_centers, 0.0, 0.0)

    inner = []
    for i in range(N):
        x = float(best_centers[i, 0])
        y = float(best_centers[i, 1])
        angle_deg = math.degrees(best_alpha) % 360
        inner.append([x, y, angle_deg])

    return inner, (0.0, 0.0), float(best_S), float(math.degrees(best_beta) % 360)

# EVOLVE-BLOCK-END
