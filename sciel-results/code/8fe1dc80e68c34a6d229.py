import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Objective function: Minimize max distance squared, given min distance is 1
    # This is equivalent to minimizing (max/min)^2
    def exact_objective(params):
        pts = params.reshape(-1, 2)
        # Compute pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
        
        # Scale so that min distance is 1
        scale = 1.0 / np.sqrt(min_d_sq)
        pts_scaled = pts * scale
        diff_scaled = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
        dists_sq_scaled = np.sum(diff_scaled**2, axis=2)
        d_sq_scaled = dists_sq_scaled[iu]
        
        max_d_sq_scaled = np.max(d_sq_scaled)
        
        return max_d_sq_scaled

    # Smooth objective for gradient-based optimization
    def smooth_objective(params, beta=50.0):
        pts = params.reshape(-1, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Soft min and max
        # Prevent overflow
        max_d_sq_raw = np.max(d_sq)
        min_d_sq_raw = np.min(d_sq)
        
        if min_d_sq_raw < 1e-12:
            return 1e10
            
        # Normalize by soft min
        # We want (soft_max / soft_min)
        # soft_max = log(sum(exp(beta * d_sq))) / beta
        # soft_min = -log(sum(exp(-beta * d_sq))) / beta
        
        # To avoid overflow in exp, shift by max/min
        m = np.max(d_sq)
        s = np.sum(np.exp(beta * (d_sq - m)))
        soft_max = (m + np.log(s) / beta)
        
        m_min = np.min(d_sq)
        s_min = np.sum(np.exp(-beta * (d_sq - m_min)))
        soft_min = (m_min - np.log(s_min) / beta)
        
        if soft_min < 1e-12:
            return 1e10
            
        return soft_max / soft_min

    # Low-dimensional parameterization: 4x4 grid with sinusoidal deformation
    def param_to_points(params):
        # params: [a1, a2, b1, b2, c1, c2, d1, d2]
        # a1, a2: amplitude and frequency of x-deformation
        # b1, b2: amplitude and frequency of y-deformation
        # c1, c2: amplitude and frequency of diagonal deformation
        # d1, d2: amplitude and frequency of anti-diagonal deformation
        
        a1, a2, b1, b2, c1, c2, d1, d2 = params
        
        # Base 4x4 grid
        i = np.arange(4)
        j = np.arange(4)
        I, J = np.meshgrid(i, j, indexing='ij')
        
        # Flatten to 16 points
        x_base = I.flatten()
        y_base = J.flatten()
        
        # Add sinusoidal deformations
        x_deform = a1 * np.sin(a2 * x_base) + c1 * np.sin(c2 * (x_base + y_base)) + d1 * np.sin(d2 * (x_base - y_base))
        y_deform = b1 * np.sin(b2 * y_base) + c1 * np.cos(c2 * (x_base + y_base)) + d1 * np.cos(d2 * (x_base - y_base))
        
        x = x_base + x_deform
        y = y_base + y_deform
        
        return np.column_stack([x, y]).flatten()

    def low_dim_objective(params):
        pts = param_to_points(params)
        return exact_objective(pts)

    # Generate initial configurations
    def get_regular_polygon(radius=1.0):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        return pts.flatten()

    def get_hexagonal_lattice():
        # Generate a hexagonal lattice and pick 16 closest to center
        R = 8
        pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                pts.append((x, y))
        pts = np.array(pts)
        dists_from_origin = np.sum(pts**2, axis=1)
        sorted_indices = np.argsort(dists_from_origin)
        initial_subset = sorted_indices[:N]
        return pts[initial_subset].flatten()

    def get_random_circle():
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.random.uniform(-0.1, 0.1, N)
        radii = np.ones(N) * 0.8 + np.random.uniform(-0.1, 0.1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts.flatten()

    # Try low-dimensional sinusoidal deformation approach
    best_params = None
    best_score = np.inf
    
    # Phase 0: Differential Evolution on low-dimensional parameter space
    if time.time() < deadline - 30:
        try:
            bounds = [(-0.5, 0.5), (0.1, 3.0), (-0.5, 0.5), (0.1, 3.0), 
                      (-0.5, 0.5), (0.1, 3.0), (-0.5, 0.5), (0.1, 3.0)]
            
            de_result = differential_evolution(low_dim_objective, bounds, 
                                               maxiter=50, popsize=15, 
                                               tol=1e-8, seed=42)
            
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = param_to_points(de_result.x)
        except Exception:
            pass

    # Generate initial configurations for full-space optimization
    initial_configs = []
    
    # Add low-dim result if we have one
    if best_params is not None:
        initial_configs.append(best_params)
    
    # Add other initial configurations
    initial_configs.extend([
        get_regular_polygon(1.0),
        get_hexagonal_lattice(),
        get_random_circle(),
        get_regular_polygon(1.0) + np.random.normal(0, 0.1, 2*N),
        get_random_circle(),
    ])

    for i, init_params in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
            
        current_score = exact_objective(init_params)
        if current_score < best_score:
            best_score = current_score
            best_params = init_params.copy()
        
        # Phase 1: Smooth optimization with L-BFGS-B
        if time.time() < deadline - 10:
            try:
                # Use a moderate beta for stability
                res = minimize(lambda x: smooth_objective(x, beta=20.0), init_params, 
                               method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
                if res.fun < 1e9:
                    smooth_score = res.fun
                    # Evaluate exact score at the smooth optimum
                    exact_score_res = exact_objective(res.x)
                    if exact_score_res < current_score:
                        current_score = exact_score_res
                        init_params = res.x
            except Exception:
                pass

        # Phase 2: Exact optimization with Nelder-Mead (derivative free, good for non-smooth)
        if time.time() < deadline - 5:
            try:
                # Nelder-Mead can be sensitive to initial simplex size
                # Use a small step
                res = minimize(exact_objective, init_params, 
                               method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < current_score:
                    current_score = res.fun
                    init_params = res.x
            except Exception:
                pass

        if current_score < best_score:
            best_score = current_score
            best_params = init_params.copy()

    if best_params is None:
        # Fallback to regular polygon
        best_params = get_regular_polygon(1.0)
        best_score = exact_objective(best_params)

    # Final polish with high precision Nelder-Mead if time permits
    if time.time() < deadline - 2:
        try:
            res = minimize(exact_objective, best_params, 
                           method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    return best_params.reshape(N, 2)
