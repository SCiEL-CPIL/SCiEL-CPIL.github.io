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

    # Phase 0: Optimize the 7+7 Nested Heptagon Structure
    # This is a highly symmetric candidate for N=14
    if time.time() - start_time < deadline - 40:
        # Grid search + Local Optimization for [r_inner, theta_offset]
        # r_inner in [0.2, 0.9], theta_offset in [0, pi/7] (symmetry period)
        
        # Initial coarse grid to find good basin
        r_vals = np.linspace(0.3, 0.8, 10)
        th_vals = np.linspace(0, np.pi/7, 10)
        
        best_77_params = None
        best_77_score = -1e10
        
        for r in r_vals:
            for th in th_vals:
                s = compute_score(heptagon_7_7_configuration([r, th]))
                if s > best_77_score:
                    best_77_score = s
                    best_77_params = [r, th]
        
        if best_77_params is not None:
            # Refine with Nelder-Mead
            x0 = np.array(best_77_params)
            try:
                res = minimize(neg_heptagon_7_7_score, x0, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                best_77_points = heptagon_7_7_configuration(res.x)
                try_update(best_77_points)
            except Exception:
                pass
            
            # Also try a few local perturbations of the best grid point
            for _ in range(5):
                perturbed = best_77_params + rng.normal(0, 0.05, 2)
                perturbed[0] = np.clip(perturbed[0], 0.1, 1.0)
                perturbed[1] = np.clip(perturbed[1], 0, 2*np.pi)
                try:
                    res = minimize(neg_heptagon_7_7_score, perturbed, method='Nelder-Mead',
                                   options={'maxiter': 200})
                    p = heptagon_7_7_configuration(res.x)
                    try_update(p)
                except Exception:
                    pass

    # Phase 1: Multi-start L-BFGS-B on full 28D space from good ring configs
    if best_points is not None and time.time() - start_time < deadline - 30:
        # Generate diverse starting points
        starts = [best_points]
        
        # Add jittered versions
        for _ in range(5):
            jittered = best_points + rng.normal(0, 0.1, (N, 2))
            starts.append(jittered)
        
        # Add some structured starts
        # Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        starts.append(np.column_stack((np.cos(angles), np.sin(angles))))
        
        # Two rings 7+7 (Standard symmetric guess)
        a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a_in = a_out + np.pi/7
        starts.append(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
        )))
        
        # Three rings 5+5+4
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = a1 + np.pi/5
        a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.6*np.cos(a2), 0.6*np.sin(a2))),
            np.column_stack((0.25*np.cos(a3), 0.25*np.sin(a3)))
        )))
        
        # Four rings 4+4+3+3
        a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        a2 = a1 + np.pi/4
        a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
        a4 = a3 + np.pi/3
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.65*np.cos(a2), 0.65*np.sin(a2))),
            np.column_stack((0.35*np.cos(a3), 0.35*np.sin(a3))),
            np.column_stack((0.1*np.cos(a4), 0.1*np.sin(a4)))
        )))
        
        for start in starts:
            if time.time() - start_time > deadline - 15:
                break
            
            x0 = start.flatten()
            
            # Annealing schedule for tau
            taus = [0.05, 0.01, 0.001, 0.0001]
            
            current_x = x0
            for tau in taus:
                if time.time() - start_time > deadline - 10:
                    break
                
                try:
                    res = minimize(
                        soft_min_score,
                        current_x,
                        args=(tau,),
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                    current_x = res.x
                    new_p = current_x.reshape((N, 2))
                    try_update(new_p)
                except Exception:
                    pass
    
    # Phase 2: Basin hopping from best
    if best_points is not None and time.time() - start_time < deadline - 15:
        x0 = best_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9},
                minsteps=3, T=0.6, stepsize=0.3,
                interval=5, disp=False
            )
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass
    
    # Phase 3: Powell refinement
    if best_points is not None and time.time() - start_time < deadline - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                          options={'maxiter': 400, 'xtol': 1e-8, 'ftol': 1e-10})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish
    if best_points is not None and time.time() - start_time < deadline - 5:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    return best_points
