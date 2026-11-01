import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    # Precompute all triangle index triples
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    def get_points(x):
        """
        x has 2*(N-2) elements: coordinates of points 2..N-1
        Points 0 and 1 are fixed at (0,0) and (1,0)
        """
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_areas_vectorized(pts):
        """Compute all triangle areas efficiently using vectorization"""
        areas = np.zeros(n_triples)
        for idx, (i, j, k) in enumerate(triples):
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            areas[idx] = 0.5 * abs(det)
        return areas

    def compute_grad_logsumexp(x, beta):
        """
        Compute the gradient of the log-sum-exp term: 
        f(x) = -1/beta * log(sum(exp(-beta * A_i)))
        grad f = -1/beta * (1/U) * sum(exp(-beta A_i) * (-beta * grad A_i))
               = sum(w_i * grad A_i) where w_i = exp(-beta A_i) / U
        Note: This ignores the dependence on Hull Area H(x) for the gradient calculation,
        assuming H is locally constant or that the normalization is handled externally.
        """
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        
        # Numerically stable log-sum-exp weights
        a_max = np.max(areas)
        exp_terms = np.exp(-beta * (areas - a_max))
        U = np.sum(exp_terms)
        
        if U < 1e-15:
            return np.zeros_like(x)
        
        weights = exp_terms / U # w_i
        
        grad_x = np.zeros_like(x)
        
        # Accumulate gradients for each point
        # grad A_ijk w.r.t p_i, p_j, p_k
        # A = 0.5 * |det(v1, v2)| where v1 = p_j - p_i, v2 = p_k - p_i
        # Let D = det(v1, v2). A = 0.5 * |D|.
        # grad A w.r.t p is 0.5 * sign(D) * grad D.
        
        # We need to map point indices in x to their coordinates
        # x[2*i] = p_i[0], x[2*i+1] = p_i[1] for i in 2..N-1
        # Points 0 and 1 are fixed.
        
        for idx, (i, j, k) in enumerate(triples):
            w = weights[idx]
            if w < 1e-12:
                continue
            
            pi = pts[i]
            pj = pts[j]
            pk = pts[k]
            
            v1 = pj - pi
            v2 = pk - pi
            det = v1[0] * v2[1] - v1[1] * v2[0]
            
            if abs(det) < 1e-12:
                continue
                
            sign_det = 1.0 if det > 0 else -1.0
            
            # Gradient of det w.r.t points
            # det = (pjx - pix)(pky - piy) - (pjy - piy)(pkx - pix)
            # d/dpix = - (pky - piy) + (pkx - pix)
            # d/dpiy = - (pjx - pix) + (pjy - piy) ... wait
            # Let's use standard cross product gradient.
            # D = v1_x v2_y - v1_y v2_x
            # dD/dv1 = [v2_y, -v2_x]
            # dD/dv2 = [-v1_y, v1_x]
            
            # v1 = pj - pi => dD/dpj = dD/dv1, dD/dpi = -dD/dv1
            # v2 = pk - pi => dD/dpk = dD/dv2, dD/dpi = dD/dpi - dD/dv2
            
            dD_dv1 = np.array([v2[1], -v2[0]])
            dD_dv2 = np.array([-v1[1], v1[0]])
            
            grad_D_pi = -dD_dv1 - dD_dv2
            grad_D_pj = dD_dv1
            grad_D_pk = dD_dv2
            
            # grad A = 0.5 * sign(D) * grad D
            factor = 0.5 * sign_det * w
            
            # Add to grad_x
            # Point i
            if i >= 2:
                grad_x[2*i] += factor * grad_D_pi[0]
                grad_x[2*i+1] += factor * grad_D_pi[1]
            # Point j
            if j >= 2:
                grad_x[2*j] += factor * grad_D_pj[0]
                grad_x[2*j+1] += factor * grad_D_pj[1]
            # Point k
            if k >= 2:
                grad_x[2*k] += factor * grad_D_pk[0]
                grad_x[2*k+1] += factor * grad_D_pk[1]
                
        return grad_x
    
    def compute_hull_area(pts):
        """Compute convex hull area"""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0
    
    def compute_min_area(pts):
        """Compute minimum triangle area and hull area"""
        areas = compute_areas_vectorized(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 0.0, 1.0
        return min_area, hull_area
    
    def logsumexp_min_objective(x, beta=500.0):
        """
        Smoothed objective: -min_area / hull_area using log-sum-exp.
        min(a_i) ≈ -1/beta * log(sum(exp(-beta * a_i)))
        """
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        
        # Numerically stable log-sum-exp
        a_max = np.max(areas)
        if a_max < 1e-15:
            return 1.0
        
        # log(sum(exp(-beta * a))) = -beta * a_max + log(sum(exp(-beta * (a - a_max))))
        exp_terms = np.exp(-beta * (areas - a_max))
        log_sum = -beta * a_max + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area)

    def logsumexp_min_jacobian(x, beta=500.0):
        """
        Jacobian of the smoothed objective.
        Note: This approximates the gradient by ignoring the derivative of the hull area H(x).
        This is valid if H(x) is locally constant (hull vertices don't change).
        """
        pts = get_points(x)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return np.zeros_like(x)
        
        grad_lse = compute_grad_logsumexp(x, beta)
        # f = - (1/H) * LSE(x)
        # df/dx = -(1/H) * dLSE/dx
        return -grad_lse / hull_area

    def raw_objective(x):
        """Raw objective: -min_area / hull_area"""
        pts = get_points(x)
        min_area, hull_area = compute_min_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)
    
    def initial_circle(seed_offset=0.0, radius=1.0):
        """Generate initial points on a circle"""
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = radius * np.cos(angles[idx])
            x[2 * i + 1] = radius * np.sin(angles[idx])
        return x
    
    def initial_elliptic(ratio=0.7, seed_offset=0.0, radius=1.0):
        """Generate initial points on an ellipse"""
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = radius * np.cos(angles[idx])
            x[2 * i + 1] = radius * ratio * np.sin(angles[idx])
        return x

    # Time management
    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    # Phase 1: Global search with differential evolution
    # We use a wider bounds to allow the hull to stretch
    bounds = [(-2.0, 2.0)] * (2 * (N - 2))
    
    try:
        # Increase popsize and maxiter for better global coverage
        result_de = differential_evolution(
            raw_objective,
            bounds,
            seed=42,
            maxiter=30, # Reduced to allow time for local optimization
            popsize=20,
            tol=1e-5,
            init='sobol',
            polish=False, # We will do local optimization separately
            mutation=(0.5, 1.5),
            recombination=0.7
        )
        if result_de.fun < -best_score:
            best_x = result_de.x.copy()
            best_score = -result_de.fun
    except Exception:
        pass
    
    # Phase 2: Multi-start local optimization with analytic gradients
    # We use a fixed beta for the main optimization, as the analytic gradient
    # approximation is most stable when the landscape is smooth but not too flat.
    
    beta_opt = 200.0
    
    # Multiple diverse starting points
    start_points = []
    # Regular polygon based
    for angle_offset in [0.0, 0.2, 0.5, 0.8, 1.1, 1.5]:
        start_points.append(initial_circle(angle_offset, radius=1.0))
    # Elliptic
    for ratio in [0.5, 0.7, 0.85, 0.95]:
        for angle_offset in [0.0, 0.5, 1.0]:
            start_points.append(initial_elliptic(ratio, angle_offset, radius=1.0))
    
    # Add the DE result if available
    if best_x is not None:
        start_points.append(best_x.copy())
        # Add jittered versions of the best result
        for _ in range(5):
            jitter = np.random.normal(0, 0.1, size=2*(N-2))
            start_points.append(best_x + jitter)
            
    # Iterate through start points
    for x0 in start_points:
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Try L-BFGS-B with analytic gradient
        try:
            res = minimize(
                logsumexp_min_objective,
                x0,
                args=(beta_opt,),
                jac=lambda x, b: logsumexp_min_jacobian(x, b),
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (N - 2),
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            if res.fun < -best_score:
                best_x = res.x.copy()
                best_score = -res.fun
        except Exception:
            pass
        
    # Phase 3: Basin Hopping with the analytic gradient local optimizer
    # This helps escape local minima by randomly perturbing the best solution
    # and re-optimizing.
    
    if best_x is not None and time.time() - start_time < time_limit - 15.0:
        
        def bh_objective(x):
            return logsumexp_min_objective(x, beta_opt)
        
        def bh_minimizer(x):
            res = minimize(
                bh_objective,
                x,
                jac=lambda x, b: logsumexp_min_jacobian(x, b),
                args=(beta_opt,),
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (N - 2),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            return res.x, res.fun, {}
            
        try:
            bh_res = basinhopping(
                bh_objective,
                best_x,
                minimizer=bh_minimizer,
                niter=50,
                stepsize=0.5,
                minimizer_kwargs={},
                seed=42
            )
            
            if bh_res.fun < -best_score:
                best_x = bh_res.x.copy()
                best_score = -bh_res.fun
        except Exception:
            pass

    # Phase 4: Final high-precision refinement with very high beta
    # Using analytic gradient again, but with higher beta
    if time.time() - start_time < time_limit - 5.0 and best_x is not None:
        for beta_final in [500.0, 1000.0, 2000.0]:
            if time.time() - start_time > time_limit - 3.0:
                break
            
            try:
                res_final = minimize(
                    logsumexp_min_objective,
                    best_x,
                    args=(beta_final,),
                    jac=lambda x, b: logsumexp_min_jacobian(x, b),
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (N - 2),
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if res_final.fun < -best_score:
                    best_x = res_final.x.copy()
                    best_score = -res_final.fun
            except Exception:
                pass
    
    # Phase 5: Final Nelder-Mead on raw objective for last-mile improvement
    if time.time() - start_time < time_limit - 2.0 and best_x is not None:
        try:
            res_nm = minimize(
                raw_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < -best_score:
                best_x = res_nm.x.copy()
                best_score = -res_nm.fun
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: return circle configuration
        best_x = initial_circle(0.0)
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
