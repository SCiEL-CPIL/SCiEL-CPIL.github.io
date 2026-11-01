import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    best_score = -np.inf
    best_centers = None
    
    # Define the smooth objective and gradient
    def soft_min_obj_and_grad(params, beta, gamma_wall=10.0):
        centers = params.reshape(N, 3)
        
        # Pairwise distances
        # d_ij
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        # We only need upper triangle to avoid double counting, but for soft-min sum, 
        # double counting just adds a constant factor to the sum if we are careful, 
        # or we can just sum all i<j.
        # Let's sum i<j.
        iu = np.triu_indices(N, k=1)
        d_pairs = dists[iu]
        
        # Wall distances (smoothed)
        # w_i = 0.5 - softmax_gamma(|x|, |y|, |z|)
        abs_centers = np.abs(centers)
        # To avoid overflow in exp, subtract max
        max_abs = np.max(abs_centers, axis=1, keepdims=True)
        exp_terms = np.exp(gamma_wall * (abs_centers - max_abs))
        sum_exp = np.sum(exp_terms, axis=1)
        # softmax = (1/gamma) * ln(sum_exp) + max_abs
        # Wait, softmax(a,b,c) = (1/g) ln(exp(g a) + exp(g b) + exp(g c))
        # = (1/g) ln(exp(g(a-m)) * exp(gm) + ...) + m  -- no
        # ln(exp(g a) + ...) = ln(exp(g(a-m)) sum + exp(gm)) = gm + ln(sum(exp(g(a-m))))
        # So softmax = a_max + (1/g) ln(sum(exp(g(a - a_max))))
        soft_max_val = max_abs.flatten() + (1.0 / gamma_wall) * np.log(sum_exp.flatten())
        w = 0.5 - soft_max_val
        
        # Combine all distances into one list
        # d_all = concat(d_pairs, w)
        # To avoid memory issues, compute log-sum-exp directly
        # log(S) = log( sum_pairs exp(-beta d) + sum_walls exp(-beta w) )
        
        # Gradient of L = -1/beta * log(S)
        # dL/dd = exp(-beta d) / S
        
        # Calculate S components
        exp_pairs = np.exp(-beta * d_pairs)
        exp_walls = np.exp(-beta * w)
        
        S = np.sum(exp_pairs) + np.sum(exp_walls)
        
        # Objective
        val = -1.0 / beta * np.log(S)
        
        # Gradient with respect to d
        grad_d_pairs = exp_pairs / S
        grad_d_walls = exp_walls / S
        
        # Gradient with respect to params
        grad_params = np.zeros_like(centers)
        
        # Pairwise part
        # grad_d_pairs[iu] corresponds to distance between i and j
        # d d_ij / d x_i = (x_i - x_j) / d_ij
        # d d_ij / d x_j = (x_j - x_i) / d_ij
        
        # We need to accumulate gradients
        # This is sparse, so looping or using index addition is needed
        # For N=50, i<j pairs are 1225. Loop is fine.
        
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            d = d_pairs[idx]
            if d < 1e-6:
                continue
            weight = grad_d_pairs[idx]
            diff_ij = centers[i] - centers[j]
            grad_params[i] += weight * (diff_ij / d)
            grad_params[j] -= weight * (diff_ij / d)
            
        # Wall part
        # w_i = 0.5 - soft_max(|x_i|, |y_i|, |z_i|)
        # dw/dx_i = - d(soft_max)/dx_i
        # d(soft_max)/dx_i = (1/g) * d/dx_i ln(sum exp(g |x_k|))
        # = (1/g) * (g sign(x_i) exp(g |x_i|)) / sum exp(g |x_k|)
        # = sign(x_i) exp(g |x_i|) / sum exp(g |x_k|)
        
        for i in range(N):
            weight = grad_d_walls[i]
            if weight == 0: continue
            abs_c = abs_centers[i]
            # Reuse exp_terms and sum_exp from above? 
            # They are for the whole batch. 
            # exp_terms[i] is exp(g * (abs_c - max_abs_i))
            # sum_exp[i] is sum of those.
            # The term in the derivative is exp(g * abs_c_k) / sum exp(g * abs_c_k)
            # This is equal to exp(g * (abs_c_k - max)) / sum exp(g * (abs_c_k - max))
            # which is exp_terms[i, k] / sum_exp[i]
            
            probs = exp_terms[i] / sum_exp[i]
            # grad w.r.t x_i is - weight * sign(x_i) * probs_k
            # centers[i] is x_i
            signs = np.sign(centers[i])
            grad_params[i] -= weight * signs * probs
            
        return val, grad_params.flatten()

    def optimize_start(centers_init, verbose=False):
        centers = centers_init.copy()
        # Annealing schedule
        betas = [5.0, 15.0, 40.0, 100.0]
        
        for beta in betas:
            res = minimize(
                lambda p: soft_min_obj_and_grad(p, beta)[0],
                centers.flatten(),
                jac=lambda p: soft_min_obj_and_grad(p, beta)[1],
                method='BFGS',
                options={'maxiter': 50, 'gtol': 1e-5}
            )
            centers = res.x.reshape(N, 3)
            if verbose:
                print(f"Beta {beta}: Val {res.fun}")
                
        return centers

    # Start 1: Random
    np.random.seed(42)
    c1 = np.random.uniform(-0.4, 0.4, (N, 3))
    
    # Start 2: FCC Lattice
    # FCC lattice points in [0,1]^3
    # Basis vectors: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # Generate points
    pts = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k])
                pts.append([i+0.5, j, k+0.5])
                pts.append([i, j+0.5, k+0.5])
    pts = np.array(pts)
    # Scale to fit in [-0.5, 0.5] roughly? 
    # FCC packing density is pi/sqrt(18) ~ 0.74.
    # Volume of 50 spheres radius r: 50 * 4/3 pi r^3.
    # r ~ 0.136. Vol ~ 50 * 1.06 * 0.0025 ~ 0.13.
    # Cube vol 1.
    # Just take first 50 points from a fine grid of FCC and center them.
    # Simpler: Random is often good enough with multiple restarts.
    # Let's do 5 random restarts.
    
    best_c = c1
    best_r = -1
    
    for k in range(5):
        np.random.seed(100 + k)
        c_init = np.random.uniform(-0.45, 0.45, (N, 3))
        c_opt = optimize_start(c_init)
        
        # Calculate true min radius
        # Pairwise
        diff = c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall
        max_c = np.max(np.abs(c_opt), axis=1)
        min_wall = 0.5 - np.min(max_c)
        
        r = min(min_pair, min_wall)
        
        if r > best_r:
            best_r = r
            best_c = c_opt
            
    return best_c
