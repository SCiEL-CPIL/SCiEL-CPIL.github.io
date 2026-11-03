"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle


def _precompute_triples(n):
    """Precompute all index triples (i, j, k) with i < j < k."""
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    return np.array(triples, dtype=np.int32)

TRIPLES = _precompute_triples(N)
N_TRIPLES = len(TRIPLES)


def _min_triangle_area(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    pi = points[TRIPLES[:, 0]]
    pj = points[TRIPLES[:, 1]]
    pk = points[TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _objective_minmax(x):
    """
    Max-min formulation: x = [x0..x10, y0..y10, t] where t is the minimum area.
    We minimize -t subject to all triangle areas >= t and points in triangle.
    For simplicity, we use a penalty-free approach: minimize -t + penalty for violations.
    """
    xy = x[:2 * N]
    t = x[2 * N]
    points = xy.reshape(N, 2)
    
    # Compute all triangle areas
    pi = points[TRIPLES[:, 0]]
    pj = points[TRIPLES[:, 1]]
    pk = points[TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Penalty: sum of max(0, t - area_i) for all triples
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations)
    
    # Objective: minimize -t + penalty
    return -t + penalty


def _objective_minmax_grad(x):
    """Gradient of the max-min objective."""
    xy = x[:2 * N].copy()
    t = x[2 * N]
    points = xy.reshape(N, 2)
    
    pi = points[TRIPLES[:, 0]]
    pj = points[TRIPLES[:, 1]]
    pk = points[TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Gradient with respect to t: -1 + sum of (1 if t > area_i else 0)
    grad_t = -1.0 + np.sum((t > areas).astype(float))
    
    # Gradient with respect to points: for each triple where t > area_i,
    # the gradient of -area_i w.r.t. points contributes
    grad_xy = np.zeros(2 * N)
    active = (t > areas)
    if np.any(active):
        for idx in np.where(active)[0]:
            i, j, k = TRIPLES[idx]
            pi_p = points[i]
            pj_p = points[j]
            pk_p = points[k]
            # Area = 0.5 * |cross(pj-pi, pk-pi)|
            # d(area)/d(pi) = -0.5 * cross(pj-pi, pk-pi) / |cross| * ... 
            # Actually, for signed area A = 0.5*((xj-xi)*(yk-yi)-(yj-yi)*(xk-xi))
            # dA/dxi = -0.5*(yk-yi) + 0.5*(yj-yi)*(xk-xi)*0 ... let me compute properly
            # A = 0.5 * (xj-xi)*(yk-yi) - 0.5*(yj-yi)*(xk-xi)
            # dA/dxi = -0.5*(yk-yi) + 0.5*(yj-yi)*(xk-xi)  [wait, no]
            # dA/dxi = -0.5*(yk-yi) - 0.5*(yj-yi)*(-1) = -0.5*(yk-yi) + 0.5*(yj-yi)
            # Hmm, let me be more careful
            # A = 0.5 * [(xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)]
            # dA/dxi = 0.5 * [-(yk-yi) - (yj-yi)*(-1)] = 0.5 * [-(yk-yi) + (yj-yi)]
            # dA/dyi = 0.5 * [-(xk-xi) + (xj-xi)]
            # dA/dxj = 0.5 * [(yk-yi) - 0] = 0.5*(yk-yi)
            # dA/dyj = 0.5 * [0 - (xk-xi)] = -0.5*(xk-xi)
            # dA/dxk = 0.5 * [0 - (yj-yi)*1] = -0.5*(yj-yi)
            # dA/dyk = 0.5 * [(xj-xi) - 0] = 0.5*(xj-xi)
            
            # But we have |A|, so sign matters
            sign = np.sign(cross[idx])
            if sign == 0:
                sign = 1.0
            
            dA_dpi_x = 0.5 * sign * (-(pk_p[1] - pi_p[1]) + (pj_p[1] - pi_p[1]))
            dA_dpi_y = 0.5 * sign * (-(pk_p[0] - pi_p[0]) + (pj_p[0] - pi_p[0]))
            dA_dpj_x = 0.5 * sign * (pk_p[1] - pi_p[1])
            dA_dpj_y = 0.5 * sign * (-(pk_p[0] - pi_p[0]))
            dA_dpk_x = 0.5 * sign * (-(pj_p[1] - pi_p[1]))
            dA_dpk_y = 0.5 * sign * (pj_p[0] - pi_p[0])
            
            # We're minimizing -t + sum(max(0, t-area)), so gradient of -area part
            # For active constraints, we add -d(area)/d(points)
            grad_xy[2*i] += -dA_dpi_x
            grad_xy[2*i+1] += -dA_dpi_y
            grad_xy[2*j] += -dA_dpj_x
            grad_xy[2*j+1] += -dA_dpj_y
            grad_xy[2*k] += -dA_dpk_x
            grad_xy[2*k+1] += -dA_dpk_y
    
    grad = np.concatenate([grad_xy, [grad_t]])
    return grad


def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])


def _clamp_bary(bary):
    """Clamp barycentric coords to valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary


def _make_initial_bary(seed, strategy='structured'):
    """Generate initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    if strategy == 'structured':
        # Use vertices, edge midpoints, and centroid
        bary_list = [
            [0.0, 0.0],      # V0
            [1.0, 0.0],      # V1
            [0.0, 1.0],      # V2
            [0.5, 0.0],      # midpoint V0-V1
            [0.5, 0.5],      # midpoint V1-V2
            [0.0, 0.5],      # midpoint V2-V0
            [1/3, 1/3],      # centroid
        ]
        # Add more structured interior points
        for r in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for theta in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
                u = r * np.cos(theta) + 1/3
                v = r * np.sin(theta) * np.sqrt(3)/2 / (np.sqrt(3)/2) + 1/3
                # Simpler: just add random interior points
                break
        while len(bary_list) < N:
            u = rng.uniform(0.1, 0.9)
            v = rng.uniform(0.1, 0.9)
            if u + v < 0.85:
                bary_list.append([u, v])
        return np.array(bary_list[:N])
    else:
        # Random spread
        bary_list = []
        for _ in range(200):
            u = rng.uniform(0.05, 0.95)
            v = rng.uniform(0.05, 0.95)
            if u + v < 0.9:
                bary_list.append([u, v])
                if len(bary_list) == N:
                    break
        while len(bary_list) < N:
            u = rng.uniform(0.1, 0.8)
            v = rng.uniform(0.1, 0.8)
            if u + v < 0.85:
                bary_list.append([u, v])
        return np.array(bary_list[:N])


def _polish_l_bfgs(bary0, maxiter=100):
    """Polish a barycentric solution using L-BFGS-B on the min-max objective."""
    points0 = _bary_to_cart(bary0)
    t0 = _min_triangle_area(points0)
    x0 = np.concatenate([points0.flatten(), [t0]])
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, TRI_AREA)]
    
    try:
        res = minimize(
            _objective_minmax, x0, method='L-BFGS-B',
            bounds=bounds,
            jac=_objective_minmax_grad,
            options={'maxiter': maxiter, 'gtol': 1e-12}
        )
        points_opt = res.x[:2*N].reshape(N, 2)
        # Clamp to triangle
        bary_opt = _cart_to_bary(points_opt)
        bary_opt = _clamp_bary(bary_opt)
        points_final = _bary_to_cart(bary_opt)
        score = _min_triangle_area(points_final)
        return points_final, score
    except Exception:
        return points0, t0


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    best_points = None
    best_score = -np.inf

    def time_left():
        return deadline - time.time()

    # Phase 1: Differential Evolution for global search
    # Work in barycentric space, objective is negative min area
    def _obj_bary(x):
        bary = _clamp_bary(x.reshape(-1, 2))
        points = _bary_to_cart(bary)
        return -_min_triangle_area(points)

    bounds_bary = [(0.0, 1.0)] * (2 * N)
    
    try:
        de_res = differential_evolution(
            _obj_bary,
            bounds=bounds_bary,
            seed=42,
            maxiter=50,
            popsize=12,
            tol=1e-5,
            polish=False,
            init='latinhypercube'
        )
        bary_de = _clamp_bary(de_res.x.reshape(-1, 2))
        points_de = _bary_to_cart(bary_de)
        score_de = _min_triangle_area(points_de)
        if score_de > best_score:
            best_score = score_de
            best_points = points_de.copy()
    except Exception:
        pass

    # Phase 2: Multi-start local optimization from DE result + structured inits
    if time_left() > 10.0:
        starts = []
        
        # Start 1: DE result
        if best_points is not None:
            starts.append(_cart_to_bary(best_points))
        
        # Start 2-4: Structured inits
        for seed in [1, 2, 3]:
            starts.append(_make_initial_bary(seed, 'structured'))
        
        # Start 5: Random init
        starts.append(_make_initial_bary(42, 'random'))
        
        for bary0 in starts:
            if time_left() < 5.0:
                break
            points_pol, score_pol = _polish_l_bfgs(bary0, maxiter=80)
            if score_pol > best_score:
                best_score = score_pol
                best_points = points_pol.copy()

    # Phase 3: Fine-tuning with Nelder-Mead on the best solution
    if best_points is not None and time_left() > 8.0:
        def _obj_cart(x):
            pts = x.reshape(N, 2)
            # Clamp to triangle
            bary = _cart_to_bary(pts)
            bary = _clamp_bary(bary)
            pts = _bary_to_cart(bary)
            return -_min_triangle_area(pts)
        
        x0 = best_points.flatten()
        try:
            res_nm = minimize(
                _obj_cart, x0, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            points_nm = res_nm.x.reshape(N, 2)
            bary_nm = _clamp_bary(_cart_to_bary(points_nm))
            points_nm = _bary_to_cart(bary_nm)
            score_nm = _min_triangle_area(points_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_points = points_nm.copy()
        except Exception:
            pass

    # Phase 4: If time remains, try more random perturbations of best
    if best_points is not None and time_left() > 5.0:
        rng = np.random.RandomState(99)
        for _ in range(3):
            if time_left() < 3.0:
                break
            bary_best = _cart_to_bary(best_points)
            bary_pert = bary_best + rng.normal(0, 0.03, size=bary_best.shape)
            bary_pert = _clamp_bary(bary_pert)
            points_pol, score_pol = _polish_l_bfgs(bary_pert, maxiter=60)
            if score_pol > best_score:
                best_score = score_pol
                best_points = points_pol.copy()

    # Final fallback
    if best_points is None:
        bary0 = _make_initial_bary(0, 'structured')
        best_points = _bary_to_cart(bary0)

    return np.array(best_points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
