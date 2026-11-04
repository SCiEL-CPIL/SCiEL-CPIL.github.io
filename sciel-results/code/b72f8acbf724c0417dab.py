"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S_for_container(vertices_list, ocx, ocy, oangle):
        """Compute the minimum side length S of the outer hexagon (centered at ocx, ocy, rot oangle) to contain all vertices."""
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                lx = v[0] - ocx
                ly = v[1] - ocy
                rx = lx * cos_a + ly * sin_a
                ry = -lx * sin_a + ly * cos_a
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = rx * np.cos(theta_n) + ry * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem

    def sat_overlap(v1, v2):
        """Check if two convex polygons overlap using Separating Axis Theorem."""
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True

    def objective(params):
        """
        params: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, oangle]
        Returns S + penalty.
        """
        n = N
        inner_params = params[:3 * n]
        container_params = params[3 * n:]
        ocx, ocy, oangle = container_params
        
        vertices_list = []
        for i in range(n):
            cx = inner_params[3 * i]
            cy = inner_params[3 * i + 1]
            angle = inner_params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S_for_container(vertices_list, ocx, ocy, oangle)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = vertices_list[i].mean(axis=0)
                c2 = vertices_list[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0:
                    if sat_overlap(vertices_list[i], vertices_list[j]):
                        penalty += 100.0 * (2.0 - dist)**2
        
        return S + penalty

    def extract_and_compute(params):
        """Extract inner, center, side, angle from params and compute S."""
        n = N
        inner = []
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            inner.append([cx, cy, np.degrees(angle)])
        
        ocx = params[3 * N]
        ocy = params[3 * N + 1]
        oangle = params[3 * N + 2]
        
        vertices_list = [hex_vertices(inner[i][0], inner[i][1], np.radians(inner[i][2])) for i in range(N)]
        S = compute_S_for_container(vertices_list, ocx, ocy, oangle)
        
        return inner, (ocx, ocy), S, np.degrees(oangle)

    # Initial guess: hexagonal lattice centered at 0,0
    # 1 center, 6 ring 1, 4 ring 2 (symmetric selection)
    params = np.zeros(3 * N + 3)
    
    # Center
    params[0:3] = [0, 0, 0]
    
    # Ring 1 (6 hexagons)
    idx = 1
    for k in range(6):
        theta = k * np.pi / 3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1
        
    # Ring 2 (4 hexagons)
    for k in range(4):
        theta = k * np.pi / 2
        cx = 2 * np.sqrt(3) * np.cos(theta)
        cy = 2 * np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1

    # Container initial guess
    params[3 * N] = 0.0
    params[3 * N + 1] = 0.0
    params[3 * N + 2] = 0.0

    best_params = params.copy()
    best_score = objective(params)
    best_result = extract_and_compute(best_params)

    # Phase 1: Local refinement with Powell
    if time.time() < DEADLINE - 30:
        res = minimize(objective, best_params, method='Powell', options={'maxiter': 2000, 'maxfev': 20000})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            best_result = extract_and_compute(best_params)

    # Phase 2: Global search with Differential Evolution
    # Define bounds for DE
    # Centers can be roughly within [-5, 5], angles [0, 2pi]
    # Outer center [-2, 2], outer angle [0, 2pi]
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))  # cx
        bounds.append((-5, 5))  # cy
        bounds.append((0, 2 * np.pi))  # angle
    bounds.append((-2, 2))  # ocx
    bounds.append((-2, 2))  # ocy
    bounds.append((0, 2 * np.pi))  # oangle

    # DE objective wrapper
    def de_objective(x):
        return objective(x)

    if time.time() < DEADLINE - 10:
        # Use DE to explore the space
        # popsize=15, maxiter=50, tol=1e-6
        # We need to be careful with the time budget
        de_start = time.time()
        try:
            res_de = differential_evolution(
                de_objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                tol=1e-6, 
                popsize=15,
                polish=True,
                init='latinhypercube'
            )
            if res_de.fun < best_score:
                best_params = res_de.x
                best_score = res_de.fun
                best_result = extract_and_compute(best_params)
        except Exception:
            pass

    # Final local polish on the best candidate
    if time.time() < DEADLINE - 5:
        try:
            res_final = minimize(objective, best_params, method='Powell', options={'maxiter': 500, 'maxfev': 5000})
            if res_final.fun < best_score:
                best_params = res_final.x
                best_score = res_final.fun
                best_result = extract_and_compute(best_params)
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
