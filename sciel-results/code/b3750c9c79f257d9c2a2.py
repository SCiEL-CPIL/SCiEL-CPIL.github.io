import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

# Precompute triangle indices
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
    This is differentiable and approximates -min(A) as tau -> 0.
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 1e10
    except Exception:
        return 1e10
    
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
    Generate points from two nested heptagons (7 outer, 7 inner).
    params: [r_inner, theta_offset]
    Outer ring is fixed at radius 1.0 and angle 0.
    """
    r_in, th_off = params
    
    # Outer 7 points
    angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    
    # Inner 7 points
    angles_in = th_off + np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_in = r_in * np.column_stack((np.cos(angles_in), np.sin(angles_in)))
    
    return np.vstack((pts_out, pts_in))

def neg_heptagon_7_7_score(params):
    try:
        points = heptagon_7_7_configuration(params)
        return -compute_score(points)
    except Exception:
        return 1e10

def ring_configuration_3(params):
    """
    Generate points from 3 rings.
    params: [r1, r2, r3, th1, th2, th3, n1, n2, n3]
    """
    r1, r2, r3, th1, th2, th3, n1, n2, n3 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    
    # Ensure sum is 14
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

def neg_ring_score_3(params):
    try:
        points = ring_configuration_3(params)
        return -compute_score(points)
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
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

    # Phase 1: Optimize the 7+7 Nested Heptagon Structure
    # This is the strongest candidate based on symmetry arguments.
    # Search space: r_inner in [0.1, 0.9], theta_offset in [0, pi/7]
    
    # 1a. Coarse Grid Search to find basin
    r_vals = np.linspace(0.2, 0.8, 15)
    th_vals = np.linspace(0, np.pi/7, 15)
    
    best_77_params = None
    best_77_score = -1e10
    
    for r in r_vals:
        for th in th_vals:
            s = compute_score(heptagon_7_7_configuration([r, th]))
            if s > best_77_score:
                best_77_score = s
                best_77_params = [r, th]
    
    if best_77_params is not None:
        # 1b. Refine with Nelder-Mead
        x0 = np.array(best_77_params)
        try:
            res = minimize(neg_heptagon_7_7_score, x0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            best_77_points = heptagon_7_7_configuration(res.x)
            try_update(best_77_points)
        except Exception:
            pass
        
        # 1c. Try Differential Evolution on the 2D space for global assurance
        if time.time() - start_time < deadline - 60:
            bounds_77 = [(0.1, 0.9), (0, np.pi/7)]
            try:
                res_de = differential_evolution(
                    neg_heptagon_7_7_score, bounds_77,
                    seed=rng.integers(0, 2**31),
                    maxiter=100, tol=1e-8, popsize=20,
                    polish=True, init='latinhypercube'
                )
                if res_de.fun < -best_score:
                    pts_de = heptagon_7_7_configuration(res_de.x)
                    try_update(pts_de)
            except Exception:
                pass

    # Phase 2: Explore 3-ring asymmetric structures (5+5+4, 7+4+3, etc.)
    # This helps if the optimal solution breaks 7-fold symmetry.
    if time.time() - start_time < deadline - 40:
        bounds_3ring = [
            (0.5, 1.5),  # r1
            (0.2, 1.0),  # r2
            (0.0, 0.6),  # r3
            (0, 2*np.pi),  # th1
            (0, 2*np.pi),  # th2
            (0, 2*np.pi),  # th3
            (3, 7),  # n1
            (3, 7),  # n2
            (3, 7),  # n3
        ]
        try:
            res_de3 = differential_evolution(
                neg_ring_score_3, bounds_3ring,
                seed=rng.integers(0, 2**31),
                maxiter=50, tol=1e-6, popsize=15,
                polish=True, init='latinhypercube'
            )
            if res_de3.fun < -best_score:
                pts_de3 = ring_configuration_3(res_de3.x)
                try_update(pts_de3)
        except Exception:
            pass

    # Phase 3: Full 28D Refinement from best candidates
    # We take the best points found so far and refine them in the full space
    # using a soft-min surrogate to handle the non-differentiability of 'min'.
    
    if best_points is not None and time.time() - start_time < deadline - 30:
        # Create a few diverse starting points for the refinement
        starts = [best_points]
        
        # Jittered versions to escape potential local minima in the symmetric space
        for _ in range(3):
            jittered = best_points + rng.normal(0, 0.05, (N, 2))
            starts.append(jittered)
            
        # Add a regular 14-gon as a fallback reference
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        starts.append(np.column_stack((np.cos(angles), np.sin(angles))))

        for start in starts:
            if time.time() - start_time > deadline - 15:
                break
            
            current_x = start.flatten()
            
            # Annealing schedule for tau
            # Start with larger tau to navigate landscape, decrease to focus on min
            taus = [0.1, 0.05, 0.01, 0.005, 0.001]
            
            for tau in taus:
                if time.time() - start_time > deadline - 10:
                    break
                try:
                    res = minimize(
                        soft_min_score,
                        current_x,
                        args=(tau,),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    current_x = res.x
                    new_p = current_x.reshape((N, 2))
                    try_update(new_p)
                except Exception:
                    pass

    # Phase 4: High-Precision Polish on the best point found
    if best_points is not None and time.time() - start_time < deadline - 5:
        x0 = best_points.flatten()
        # Use Powell which is robust for non-smooth functions
        try:
            res_powell = minimize(neg_score, x0, method='Powell',
                                  options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12})
            try_update(res_powell.x.reshape((N, 2)))
        except Exception:
            pass
        
        # Final Nelder-Mead for extra precision
        if time.time() - start_time < deadline - 2:
            x0 = best_points.flatten()
            try:
                res_nm = minimize(neg_score, x0, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
                try_update(res_nm.x.reshape((N, 2)))
            except Exception:
                pass

    return best_points
