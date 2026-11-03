import numpy as np
import time
import scipy.optimize as so

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # 1. Initialization
    # Start with a grid-like structure to avoid terrible local minima
    # 7x7 grid has 49 points. We need 50.
    # Let's try a 7x8 grid cropped, or just random + grid mix.
    # A good init is often a perturbed grid.
    
    def make_grid_init():
        m = 7
        # Create 7x7 grid
        x = np.linspace(-0.45, 0.45, m)
        y = np.linspace(-0.45, 0.45, m)
        grid_x, grid_y = np.meshgrid(x, y)
        centers = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        # We have 49 points. Add one more.
        # Place the 50th point in the center, shifting others slightly?
        # Or just use a 5x10 grid? 5*10=50.
        # 5x10 grid:
        x5 = np.linspace(-0.45, 0.45, 5)
        y10 = np.linspace(-0.45, 0.45, 10)
        g5, g10 = np.meshgrid(x5, y10)
        return np.column_stack((g5.ravel(), g10.ravel()))
    
    init_centers = make_grid_init()
    
    # Add small noise to break symmetry
    np.random.seed(42)
    init_centers += np.random.normal(0, 0.01, init_centers.shape)
    
    # Clip to bounds
    init_centers = np.clip(init_centers, -0.5, 0.5)
    
    best_centers = init_centers
    best_radius = _compute_radius(best_centers)

    # 2. Soft-Min Optimization (L-BFGS-B)
    # Objective: Minimize -soft_min_radius
    # soft_min_radius = (1/beta) * log( sum( exp(-beta * d_k) ) )
    # where d_k are the constraints: d_ij/2 and dist_to_bound
    
    def soft_min_radius(centers, beta):
        # centers: (N, 2)
        # Calculate pairwise distances / 2
        # and distances to boundary
        c = centers
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Upper triangle
        iu = np.triu_indices(N, 1)
        d_pairs = dists[iu] / 2.0
        
        # Boundary
        # dist to left/right: 0.5 - |x|
        # dist to top/bottom: 0.5 - |y|
        d_bounds_x = 0.5 - np.abs(c[:, 0])
        d_bounds_y = 0.5 - np.abs(c[:, 1])
        
        all_dists = np.concatenate([d_pairs, d_bounds_x, d_bounds_y])
        
        # Stable log-sum-exp
        # log(sum(exp(-beta * d))) = -beta * min(d) + log(sum(exp(-beta * (d - min(d)))))
        min_d = np.min(all_dists)
        if min_d < 0: # Should not happen if we clip, but just in case
            min_d = 0
            
        # To avoid overflow in exp, we shift by min_d
        # exp(-beta * (d - min_d)) is <= 1
        shifted = all_dists - min_d
        # If beta is large, beta*shifted might still be large for non-min elements?
        # Actually, we want sum(exp(-beta * d)).
        # log(sum(exp(-beta * d))) = -beta * min_d + log(sum(exp(-beta * (d - min_d))))
        
        # Calculate log_sum_exp
        # terms = np.exp(-beta * shifted)
        # log_sum = np.log(np.sum(terms))
        
        # Simpler: just use np.logaddexp.reduce if available, or manual
        # For stability:
        max_val = np.max(-beta * shifted)
        if max_val > 700: # Overflow risk
             max_val = 700
        terms = np.exp(-beta * shifted - max_val)
        log_sum = max_val + np.log(np.sum(terms))
        
        soft_min = min_d + log_sum / beta
        return soft_min

    def objective(x):
        centers = x.reshape(-1, 2)
        beta = 20.0 # Start with moderate sharpness
        r = soft_min_radius(centers, beta)
        return -r

    def objective_grad(x):
        # Numerical gradient is safer and likely fast enough for 100 vars
        # L-BFGS-B can use numerical gradients
        return None

    if time.time() < deadline - 20:
        try:
            res = so.minimize(
                objective, 
                init_centers.ravel(), 
                method='L-BFGS-B', 
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            centers_lbgfs = res.x.reshape(-1, 2)
            r_lbgfs = _compute_radius(centers_lbgfs)
            if r_lbgfs > best_radius:
                best_radius = r_lbgfs
                best_centers = centers_lbgfs
        except Exception as e:
            pass

    # 3. SLSQP Polish
    # Maximize r subject to constraints
    # Variables: [x1, y1, ..., xN, yN, r]
    
    def slsqp_objective(x):
        # x = [centers, r]
        r = x[-1]
        return -r

    def slsqp_jacobian(x):
        # Gradient of -r is 0 for centers, -1 for r
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        return grad

    def make_constraints():
        cons = []
        # Pairwise constraints: ||ci - cj||^2 - (2r)^2 >= 0
        for i in range(N):
            for j in range(i + 1, N):
                def con_func(x, i=i, j=j):
                    ci = x[2*i:2*i+2]
                    cj = x[2*j:2*j+2]
                    r = x[-1]
                    return np.sum((ci - cj)**2) - (2*r)**2
                cons.append({'type': 'ineq', 'fun': con_func})
        
        # Boundary constraints: 0.5 - |c_ik| - r >= 0
        for i in range(N):
            # x-axis
            def con_x_pos(x, i=i):
                ci = x[2*i:2*i+2]
                r = x[-1]
                return 0.5 - ci[0] - r
            def con_x_neg(x, i=i):
                ci = x[2*i:2*i+2]
                r = x[-1]
                return 0.5 + ci[0] - r
            # y-axis
            def con_y_pos(x, i=i):
                ci = x[2*i:2*i+2]
                r = x[-1]
                return 0.5 - ci[1] - r
            def con_y_neg(x, i=i):
                ci = x[2*i:2*i+2]
                r = x[-1]
                return 0.5 + ci[1] - r
            cons.append({'type': 'ineq', 'fun': con_x_pos})
            cons.append({'type': 'ineq', 'fun': con_x_neg})
            cons.append({'type': 'ineq', 'fun': con_y_pos})
            cons.append({'type': 'ineq', 'fun': con_y_neg})
            
        return cons

    if time.time() < deadline - 5:
        # Initial guess for SLSQP
        x0 = np.concatenate([best_centers.ravel(), [best_radius]])
        
        bounds = [(-0.5, 0.5)] * (2 * N) + [(0, 0.5)]
        
        try:
            cons = make_constraints()
            res_slsqp = so.minimize(
                slsqp_objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res_slsqp.success:
                x_opt = res_slsqp.x
                centers_opt = x_opt[:-1].reshape(-1, 2)
                r_opt = x_opt[-1]
                
                # Verify validity
                if np.all(np.isfinite(centers_opt)) and np.all(centers_opt >= -0.5) and np.all(centers_opt <= 0.5):
                    r_verified = _compute_radius(centers_opt)
                    if r_verified > best_radius:
                        best_radius = r_verified
                        best_centers = centers_opt
        except Exception as e:
            pass

    return best_centers

def _compute_radius(centers):
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    min_dist = 1.0
    # Pairwise
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            if d < min_dist:
                min_dist = d
    # Boundary
    for i in range(N):
        for k in range(2):
            d = 0.5 - np.abs(centers[i, k])
            if d < min_dist:
                min_dist = d
    return min_dist
