import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    time_limit = 620.0  # Safety margin before 630s kill

    # Precompute bin indices for the 4 variables
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    V_m1 = X - Y  # x - y
    V_0 = X       # x
    V_1 = X + Y   # x + y
    V_2 = X + 2*Y # x + 2y

    # Flatten for bincount
    idx_m1 = (V_m1 + 9).flatten() # 0..18
    idx_0 = V_0.flatten()         # 0..9
    idx_1 = V_1.flatten()         # 0..18
    idx_2 = V_2.flatten()         # 0..27

    num_bins_m1 = 19
    num_bins_0 = 10
    num_bins_1 = 19
    num_bins_2 = 28

    # Precompute diagonal structures for faster P construction
    # diagonal d corresponds to x - y = d
    # d ranges from -9 to 9
    diag_data = []
    for d in range(-9, 10):
        y_min = max(0, -d)
        y_max = min(9, 9 - d)
        if y_min > y_max:
            continue
        ys = np.arange(y_min, y_max + 1)
        xs = ys + d
        # Flatten indices y*10 + x
        flat_indices = ys * 10 + xs
        diag_data.append((d, ys, flat_indices))

    def get_probabilities(params):
        """
        Reconstruct P_flat from params.
        params: (38,) array
        First 19: logits for diagonal weights q_d
        Next 19: tilt parameters alpha_d
        """
        q_logits = params[:19]
        alphas = params[19:]
        
        # 1. Diagonal weights q_d
        # Softmax
        q = np.exp(q_logits - np.max(q_logits))
        q_sum = np.sum(q)
        if q_sum < 1e-15:
            return None
        q = q / q_sum
        
        # 2. Build P_flat
        P_flat = np.zeros(100)
        
        # Map cells to diagonals
        # d = x - y
        for d, ys, flat_indices in diag_data:
            d_idx = d + 9
            weight_d = q[d_idx]
            alpha_d = alphas[d_idx]
            
            if weight_d < 1e-15:
                continue
                
            # Compute local probabilities along the diagonal
            # p_y proportional to exp(alpha_d * y)
            # Centering y helps with numerical stability
            ys_centered = ys - 4.5
            logits = alpha_d * ys_centered
            p_local = np.exp(logits - np.max(logits))
            p_local_sum = np.sum(p_local)
            if p_local_sum < 1e-15:
                p_local = np.ones_like(ys) / len(ys)
            else:
                p_local = p_local / p_local_sum
            
            # Assign to P_flat
            P_flat[flat_indices] = weight_d * p_local
                
        # Normalize
        total_sum = np.sum(P_flat)
        if total_sum < 1e-9:
            return None
        P_flat = P_flat / total_sum
        return P_flat

    def compute_entropies(P_flat):
        counts_m1 = np.bincount(idx_m1, weights=P_flat, minlength=num_bins_m1)
        counts_0 = np.bincount(idx_0, weights=P_flat, minlength=num_bins_0)
        counts_1 = np.bincount(idx_1, weights=P_flat, minlength=num_bins_1)
        counts_2 = np.bincount(idx_2, weights=P_flat, minlength=num_bins_2)

        def entropy(counts):
            c = counts[counts > 1e-15]
            if len(c) == 0:
                return 0.0
            return -np.sum(c * np.log2(c))

        H_m1 = entropy(counts_m1)
        H_0 = entropy(counts_0)
        H_1 = entropy(counts_1)
        H_2 = entropy(counts_2)
        return H_m1, H_0, H_1, H_2

    def compute_ratio_exact(params):
        P_flat = get_probabilities(params)
        if P_flat is None:
            return 0.0
        H_m1, H_0, H_1, H_2 = compute_entropies(P_flat)
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        return H_m1 / M

    def compute_ratio_smooth(params, beta):
        P_flat = get_probabilities(params)
        if P_flat is None:
            return 0.0
        H_m1, H_0, H_1, H_2 = compute_entropies(P_flat)
        
        # Soft max of H_0, H_1, H_2
        Hs = np.array([H_0, H_1, H_2])
        if beta < 1e-6:
            M = np.max(Hs)
        else:
            # Stable log-sum-exp
            max_H = np.max(Hs)
            log_sum_exp = max_H + np.log(np.sum(np.exp(beta * (Hs - max_H))))
            M = log_sum_exp / beta
            
        if M < 1e-9:
            return 0.0
        return H_m1 / M

    def epigraph_objective(params, t, beta):
        """
        Epigraph formulation: minimize -H_m1 + t subject to H_i <= t
        We use a penalty for constraint violations to make it smooth.
        params: (38,) array for P
        t: scalar upper bound on max(H_0, H_1, H_2)
        beta: penalty strength for constraint violations
        """
        P_flat = get_probabilities(params)
        if P_flat is None:
            return 1e10
        H_m1, H_0, H_1, H_2 = compute_entropies(P_flat)
        
        # Penalty for violating H_i <= t
        violations = np.maximum(0, H_0 - t) + np.maximum(0, H_1 - t) + np.maximum(0, H_2 - t)
        penalty = beta * violations
        
        # Objective: -H_m1 + t + penalty
        return -H_m1 + t + penalty

    def objective_smooth(params, beta):
        r = compute_ratio_smooth(params, beta)
        return -r

    def objective_exact(params):
        r = compute_ratio_exact(params)
        return -r

    # Initial guess
    init_params = np.zeros(38)
    
    best_score = compute_ratio_exact(init_params)
    best_params = init_params.copy()
    
    rng = np.random.default_rng(42)
    
    # Generate start points
    start_points = [init_params]
    
    # Random starts with varying scales
    for scale in [1.0, 2.0, 4.0]:
        for _ in range(8):
            p = np.random.normal(0, scale, 38)
            start_points.append(p)
        
    # Structured starts:
    # 1. Uniform q, random alpha
    for _ in range(8):
        p = np.zeros(38)
        p[19:] = np.random.normal(0, 3.0, 19)
        start_points.append(p)
        
    # 2. Random q, zero alpha
    for _ in range(8):
        p = np.zeros(38)
        p[:19] = np.random.normal(0, 3.0, 19)
        start_points.append(p)
    
    # 3. Concentrate on middle diagonals (x-y near 0), spread alpha
    for _ in range(4):
        p = np.zeros(38)
        # Make middle diagonals (d=0,±1,±2) have higher logits
        p[9] = 2.0  # d=0
        p[8] = 1.5  # d=-1
        p[10] = 1.5  # d=1
        p[7] = 1.0   # d=-2
        p[11] = 1.0  # d=2
        p[19:] = np.random.normal(0, 2.0, 19)
        start_points.append(p)
    
    # 4. Concentrate on edge diagonals, spread alpha
    for _ in range(4):
        p = np.zeros(38)
        p[0] = 2.0   # d=-9
        p[18] = 2.0  # d=9
        p[1] = 1.5   # d=-8
        p[17] = 1.5  # d=8
        p[19:] = np.random.normal(0, 2.0, 19)
        start_points.append(p)
    
    # 5. Alternating diagonal weights
    for _ in range(4):
        p = np.zeros(38)
        for i in range(19):
            p[i] = 1.5 if i % 2 == 0 else -1.5
        p[19:] = np.random.normal(0, 2.0, 19)
        start_points.append(p)
    
    # 6. Bimodal: concentrate on two far-apart diagonal groups
    for _ in range(4):
        p = np.zeros(38)
        # Group 1: d in {-4,-3,-2}, Group 2: d in {2,3,4}
        for d_idx in [5,6,7,11,12,13]:  # d=-4,-3,-2,2,3,4
            p[d_idx] = 2.5
        p[19:] = np.random.normal(0, 2.5, 19)
        start_points.append(p)
    
    # 7. Triangular weight profile
    for _ in range(3):
        p = np.zeros(38)
        for i in range(19):
            d = i - 9
            p[i] = max(0, 3.0 - abs(d) * 0.5)
        p[19:] = np.random.normal(0, 2.0, 19)
        start_points.append(p)

    # Optimization Loop
    for sp in start_points:
        if time.time() - start_time > time_limit - 50:
            break
            
        current_params = sp.copy()
        
        # Annealing schedule for beta - more steps for smoother transition
        betas = [0.3, 1.0, 3.0, 7.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0]
        
        for beta in betas:
            if time.time() - start_time > time_limit - 60:
                break
                
            try:
                def obj_wrapper(p):
                    return objective_smooth(p, beta)
                    
                res = minimize(obj_wrapper, current_params, method='L-BFGS-B', 
                               options={'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-11})
                
                if res.success or res.fun < objective_smooth(current_params, beta):
                    current_params = res.x
            except Exception:
                pass
        
        # Also try epigraph formulation for this starting point
        if time.time() - start_time < time_limit - 50:
            try:
                # Initialize t to a reasonable value
                P_flat = get_probabilities(current_params)
                if P_flat is not None:
                    H_m1, H_0, H_1, H_2 = compute_entropies(P_flat)
                    t_init = max(H_0, H_1, H_2) + 0.3
                else:
                    t_init = 3.0
                
                # Optimize over params and t jointly with multiple penalty strengths
                for beta_epi in [50.0, 100.0, 200.0]:
                    def epigraph_wrapper(x, b=beta_epi):
                        params = x[:38]
                        t = x[38]
                        return epigraph_objective(params, t, b)
                    
                    x0 = np.concatenate([current_params, [t_init]])
                    res_epi = minimize(epigraph_wrapper, x0, method='L-BFGS-B',
                                       options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11})
                    
                    if res_epi.success or res_epi.fun < 1e9:
                        new_params = res_epi.x[:38]
                        score = compute_ratio_exact(new_params)
                        if score > best_score:
                            best_score = score
                            best_params = new_params.copy()
                            current_params = new_params.copy()
            except Exception:
                pass
        
        # Evaluate exact score for this candidate
        score = compute_ratio_exact(current_params)
        if score > best_score:
            best_score = score
            best_params = current_params.copy()
            
    # Final polish with exact objective - multiple methods
    if time.time() - start_time < time_limit - 30:
        try:
            res_final = minimize(objective_exact, best_params, method='L-BFGS-B', 
                                 options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res_final.fun < -best_score:
                best_score = -res_final.fun
                best_params = res_final.x
        except Exception:
            pass
    
    # SLSQP polish - can handle the kink in max better with careful setup
    if time.time() - start_time < time_limit - 15:
        try:
            res_slsqp = minimize(objective_exact, best_params, method='SLSQP',
                              options={'maxiter': 5000, 'ftol': 1e-15, 'maxiter': 5000})
            if res_slsqp.fun < -best_score:
                best_score = -res_slsqp.fun
                best_params = res_slsqp.x
        except Exception:
            pass
            
    # COBYLA polish - good for non-smooth, derivative-free
    if time.time() - start_time < time_limit - 8:
        try:
            res_cobyla = minimize(objective_exact, best_params, method='COBYLA',
                              options={'maxiter': 3000, 'rhobeg': 1e-4, 'catol': 1e-12})
            if res_cobyla.fun < -best_score:
                best_score = -res_cobyla.fun
                best_params = res_cobyla.x
        except Exception:
            pass
    
    # Nelder-Mead polish (good for non-smooth)
    if time.time() - start_time < time_limit - 4:
        try:
            res_nm = minimize(objective_exact, best_params, method='Nelder-Mead',
                              options={'maxiter': 8000, 'fatol': 1e-14, 'xatol': 1e-12})
            if res_nm.fun < -best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass
    
    # One more L-BFGS-B pass after Nelder-Mead to refine
    if time.time() - start_time < time_limit - 1:
        try:
            res_refine = minimize(objective_exact, best_params, method='L-BFGS-B',
                                  options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res_refine.fun < -best_score:
                best_score = -res_refine.fun
                best_params = res_refine.x
        except Exception:
            pass

    # Reconstruct P from best_params
    P_flat = get_probabilities(best_params)
    if P_flat is None:
        P = np.ones((10, 10)) / 100.0
    else:
        P = P_flat.reshape(10, 10)
        
    # Normalize and validate
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-9:
        P = np.ones((10, 10)) / 100.0
    else:
        P = P / s
        
    return P

if __name__ == "__main__":
    pass
