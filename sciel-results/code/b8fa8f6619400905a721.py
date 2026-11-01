import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def soft_min_score(x, tau):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    areas = np.maximum(areas, 1e-12)
        
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def heptagon_7_7_configuration(params):
    """
    Two concentric heptagons: 7 outer points at radius 1, 7 inner at radius r,
    with angular offset th_off.
    """
    r_in, th_off = params
    angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    angles_in = th_off + np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_in = r_in * np.column_stack((np.cos(angles_in), np.sin(angles_in)))
    return np.vstack((pts_out, pts_in))

def neg_heptagon_7_7_score(params):
    try:
        points = heptagon_7_7_configuration(params)
        return -compute_score(points)
    except Exception:
        return 1e10

def two_heptagons_soft(params, tau):
    """
    Soft-min score for the 7+7 heptagon parameterization.
    params: [r_in, th_off]
    """
    r_in, th_off = params
    angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    angles_in = th_off + np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_in = r_in * np.column_stack((np.cos(angles_in), np.sin(angles_in)))
    points = np.vstack((pts_out, pts_in))
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    areas = np.maximum(areas, 1e-12)
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def two_heptagons_grad(params, tau):
    """
    Gradient of the soft-min score for 7+7 heptagons.
    Returns (value, gradient)
    """
    r_in, th_off = params
    angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    angles_in = th_off + np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_in = r_in * np.column_stack((np.cos(angles_in), np.sin(angles_in)))
    points = np.vstack((pts_out, pts_in))
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0, np.zeros(2)
    except Exception:
        return 0.0, np.zeros(2)
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    areas = np.maximum(areas, 1e-12)
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    exp_vals = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # weights sum to 1
    
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    value = -soft_min_val / hull_area
    
    # Gradient via automatic differentiation would be ideal, but let's use numerical
    # Actually, let's just return value and let L-BFGS-B do numerical gradients
    # Or we can compute the gradient analytically for the soft-min part
    # d(soft_min)/d(area_k) = -weights[k]
    # So d(value)/d(params) = -d(soft_min)/d(params) / hull_area
    
    # d(soft_min)/d(params) = sum_k weights[k] * d(area_k)/d(params)
    # This is complex to compute analytically, so let's use finite differences
    # for the gradient in the 2D parameter space
    
    eps = 1e-6
    grad = np.zeros(2)
    for i in range(2):
        params_plus = params.copy()
        params_plus[i] += eps
        val_plus = two_heptagons_soft(params_plus, tau)
        params_minus = params.copy()
        params_minus[i] -= eps
        val_minus = two_heptagons_soft(params_minus, tau)
        grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return value, grad

def ring_configuration(params):
    r1, r2, r3, th1, th2, th3, n1, n2, n3 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    
    while n1 + n2 + n3 > 14:
        if n3 >= 1: n3 -= 1
        elif n2 >= 1: n2 -= 1
        else: n1 -= 1
    while n1 + n2 + n3 < 14:
        n3 += 1
    
    points = []
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    if n3 > 0:
        angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    return result

def ring_score(params):
    try:
        points = ring_configuration(params)
        return compute_score(points)
    except Exception:
        return 0.0

def neg_ring_score(params):
    return -ring_score(params)

def three_ring_configuration(params):
    r1, r2, r3, th1, th2, th3, n1, n2, n3 = params
    n1 = int(round(n1)); n2 = int(round(n2)); n3 = int(round(n3))
    while n1 + n2 + n3 > 14:
        if n3 >= 1: n3 -= 1
        elif n2 >= 1: n2 -= 1
        else: n1 -= 1
    while n1 + n2 + n3 < 14:
        n3 += 1
    pts = []
    for r, th, n in [(r1, th1, n1), (r2, th2, n2), (r3, th3, n3)]:
        if n > 0:
            a = th + np.linspace(0, 2*np.pi, n, endpoint=False)
            pts.append(np.column_stack((r*np.cos(a), r*np.sin(a))))
    if not pts:
        return np.zeros((N, 2))
    res = np.vstack(pts)
    if len(res) < N:
        res = np.vstack([res, np.zeros((N - len(res), 2))])
    return res[:N]

def neg_three_ring_score(params):
    try:
        return -compute_score(three_ring_configuration(params))
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # Phase 0: Fine grid search + local optimization for 7+7 heptagon structure
    if time.time() - start_time < deadline - 55:
        # Refined grid: r in [0.3, 0.85] step 0.01, theta in [0, pi/7] step pi/140
        r_vals = np.arange(0.3, 0.86, 0.01)
        th_vals = np.arange(0, np.pi/7 + 1e-9, np.pi/140)
        
        best_77_params = None
        best_77_score = -1e10
        
        for r in r_vals:
            for th in th_vals:
                s = compute_score(heptagon_7_7_configuration([r, th]))
                if s > best_77_score:
                    best_77_score = s
                    best_77_params = [r, th]
        
        if best_77_params is not None:
            # Use soft-min with gradient for more precise optimization
            x0 = np.array(best_77_params)
            taus = [0.05, 0.02, 0.005, 0.001, 0.0003, 0.0001]
            current_x = x0.copy()
            
            for tau in taus:
                if time.time() - start_time > deadline - 50:
                    break
                try:
                    res = minimize(
                        two_heptagons_soft, current_x, args=(tau,),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8}
                    )
                    current_x = res.x
                    p = heptagon_7_7_configuration(current_x)
                    try_update(p)
                except Exception:
                    pass
            
            # Final exact polish
            try:
                res = minimize(neg_heptagon_7_7_score, current_x, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
                p = heptagon_7_7_configuration(res.x)
                try_update(p)
            except Exception:
                pass
            
            # Multiple perturbed restarts around the best
            for _ in range(15):
                if time.time() - start_time > deadline - 45:
                    break
                perturbed = current_x.copy()
                perturbed[0] += rng.normal(0, 0.02)
                perturbed[1] += rng.normal(0, 0.05)
                perturbed[0] = np.clip(perturbed[0], 0.15, 1.0)
                perturbed[1] = perturbed[1] % (2*np.pi)
                try:
                    res = minimize(neg_heptagon_7_7_score, perturbed, method='Nelder-Mead',
                                   options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                    p = heptagon_7_7_configuration(res.x)
                    try_update(p)
                except Exception:
                    pass

    # Phase 1: Differential Evolution on 3-ring parameters
    bounds = [
        (0.1, 1.5), (0.05, 1.2), (0.0, 0.8),
        (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi),
        (1, 14), (1, 14), (1, 14)
    ]
    
    for restart in range(3):
        if time.time() - start_time > deadline - 40:
            break
        try:
            res = differential_evolution(
                neg_ring_score, bounds,
                seed=rng.integers(0, 2**31),
                maxiter=100, tol=1e-8, popsize=15,
                mutation=(0.3, 1.2), recombination=0.9,
                polish=True, init='latinhypercube'
            )
            if res.fun < 0:
                points = ring_configuration(res.x)
                try_update(points)
        except Exception:
            pass

    # Phase 2: Multi-start L-BFGS-B with soft-min annealing from diverse starts
    if best_points is not None and time.time() - start_time < deadline - 35:
        starts = [best_points.copy()]
        
        # Jittered versions
        for _ in range(10):
            jittered = best_points + rng.normal(0, 0.1, (N, 2))
            starts.append(jittered)
        
        # Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        starts.append(np.column_stack((np.cos(angles), np.sin(angles))))
        
        # 7+7 standard
        a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a_in = a_out + np.pi/7
        starts.append(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
        )))
        
        # 5+4+5 ring
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 4, endpoint=False)
        a3 = np.linspace(np.pi/10, 2*np.pi + np.pi/10, 5, endpoint=False)
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.6*np.cos(a2), 0.6*np.sin(a2))),
            np.column_stack((0.3*np.cos(a3), 0.3*np.sin(a3)))
        )))
        
        # 8+6
        a1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        a2 = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 6, endpoint=False)
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.55*np.cos(a2), 0.55*np.sin(a2)))
        )))
        
        # 6+8
        a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        a2 = np.linspace(np.pi/12, 2*np.pi + np.pi/12, 8, endpoint=False)
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.5*np.cos(a2), 0.5*np.sin(a2)))
        )))
        
        for start in starts:
            if time.time() - start_time > deadline - 25:
                break
            
            x0 = start.flatten()
            taus = [0.05, 0.02, 0.005, 0.001, 0.0003, 0.0001]
            
            current_x = x0
            for tau in taus:
                if time.time() - start_time > deadline - 20:
                    break
                try:
                    res = minimize(
                        soft_min_score, current_x, args=(tau,),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-8}
                    )
                    current_x = res.x
                    new_p = current_x.reshape((N, 2))
                    try_update(new_p)
                except Exception:
                    pass

    # Phase 3: Basin hopping
    if best_points is not None and time.time() - start_time < deadline - 20:
        x0 = best_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11},
                minsteps=5, T=1.0, stepsize=0.5,
                interval=10, disp=False
            )
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass

    # Phase 4: Powell refinement
    if best_points is not None and time.time() - start_time < deadline - 15:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                          options={'maxiter': 1000, 'xtol': 1e-11, 'ftol': 1e-13})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 5: Final Nelder-Mead polish with very tight tolerances
    if best_points is not None and time.time() - start_time < deadline - 8:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 6: Coordinate-wise refinement
    if best_points is not None and time.time() - start_time < deadline - 4:
        x0 = best_points.flatten()
        for _ in range(5):
            if time.time() - start_time > deadline - 4:
                break
            try:
                res = minimize(neg_score, x0, method='Powell',
                              options={'maxiter': 3000, 'xtol': 1e-13, 'ftol': 1e-15})
                x0 = res.x
                try_update(x0.reshape((N, 2)))
            except Exception:
                pass

    return best_points
