import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

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
    diag_data = []
    for d in range(-9, 10):
        y_min = max(0, -d)
        y_max = min(9, 9 - d)
        if y_min > y_max:
            continue
        ys = np.arange(y_min, y_max + 1)
        xs = ys + d
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
        q_max = np.max(q_logits)
        q = np.exp(q_logits - q_max)
        q_sum = np.sum(q)
        if q_sum < 1e-15:
            return None
        q = q / q_sum
        
        # 2. Build P_flat
        P_flat = np.zeros(100)
        
        for d, ys, flat_indices in diag_data:
            d_idx = d + 9
            weight_d = q[d_idx]
            alpha_d = alphas[d_idx]
            
            if weight_d < 1e-15:
                continue
                
            ys_centered = ys - 4.5
            logits = alpha_d * ys_centered
            logits_max = np.max(logits)
            p_local = np.exp(logits - logits_max)
            p_local_sum = np.sum(p_local)
            if p_local_sum < 1e-15:
                p_local = np.ones_like(ys) / len(ys)
            else:
                p_local = p_local / p_local_sum
            
            P_flat[flat_indices] = weight_d * p_local
                
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
        
        Hs = np.array([H_0, H_1, H_2])
        if beta < 1e-6:
            M = np.max(Hs)
        else:
            max_H = np.max(Hs)
            log_sum_exp = max_H + np.log(np.sum(np.exp(beta * (Hs - max_H))))
            M = log_sum_exp / beta
            
        if M < 1e-9:
            return 0.0
        return H_m1 / M

    def objective_smooth(params, beta):
        r = compute_ratio_smooth(params, beta)
        return -r

    def objective_exact(params):
        r = compute_ratio_exact(params)
        return -r

    # Phase 1: Differential Evolution for global exploration
    # Bounded parameters to keep search space manageable
    bounds = [(-5.0, 5.0)] * 38
    
    best_score = -np.inf
    best_params = np.zeros(38)
    
    # Initial evaluation
    init_score = compute_ratio_exact(best_params)
    if init_score > best_score:
        best_score = init_score
    
    # Check time before starting DE
    if time.time() - start_time < time_limit - 300:
        try:
            # Use a smooth objective for DE (beta=10 is a good compromise)
            def de_objective(params):
                return objective_smooth(params, 10.0)
            
            # Differential evolution with moderate population
            de_result = differential_evolution(
                de_objective, 
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,  # We'll polish separately
                update_interval=1,
                callback=None
            )
            
            if de_result.fun < -best_score:
                best_score = -de_result.fun
                best_params = de_result.x.copy()
                
            # Also check the initial point again
            init_score = compute_ratio_exact(best_params)
            if init_score > best_score:
                best_score = init_score
                
        except Exception:
            pass
    
    # Phase 2: Multi-start local optimization with L-BFGS-B
    rng = np.random.default_rng(123)
    
    # Generate diverse start points
    start_points = [best_params.copy()]
    
    # Random starts with varying scales
    for scale in [1.0, 2.0, 3.0]:
        for _ in range(8):
            p = rng.normal(0, scale, 38)
            start_points.append(p)
    
    # Structured starts
    # 1. Uniform q, random alpha
    for _ in range(8):
        p = np.zeros(38)
        p[19:] = rng.normal(0, 3.0, 19)
        start_points.append(p)
        
    # 2. Random q, zero alpha
    for _ in range(8):
        p = np.zeros(38)
        p[:19] = rng.normal(0, 3.0, 19)
        start_points.append(p)
    
    # 3. Concentrate on middle diagonals
    for _ in range(5):
        p = np.zeros(38)
        # Make middle diagonals (d=-2,-1,0,1,2) have higher logits
        for d_idx in [7, 8, 9, 10, 11]:  # d=-2,-1,0,1,2 correspond to indices 7-11
            p[d_idx] = rng.uniform(1.0, 3.0)
        p[19:] = rng.normal(0, 2.0, 19)
        start_points.append(p)
    
    # 4. Concentrate on corners (extreme diagonals)
    for _ in range(5):
        p = np.zeros(38)
        # Make extreme diagonals have higher logits
        p[0] = rng.uniform(1.0, 3.0)  # d=-9
        p[18] = rng.uniform(1.0, 3.0) # d=9
        p[19:] = rng.normal(0, 2.0, 19)
        start_points.append(p)
    
    # Annealing schedule for beta
    betas = [5.0, 20.0, 100.0, 500.0]
    
    for sp in start_points:
        if time.time() - start_time > time_limit - 120:
            break
            
        current_params = sp.copy()
        
        for beta in betas:
            if time.time() - start_time > time_limit - 120:
                break
                
            try:
                def obj_wrapper(p):
                    return objective_smooth(p, beta)
                    
                res = minimize(obj_wrapper, current_params, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < objective_smooth(current_params, beta):
                    current_params = res.x
            except Exception:
                pass
        
        # Evaluate exact score for this candidate
        score = compute_ratio_exact(current_params)
        if score > best_score:
            best_score = score
            best_params = current_params.copy()
            
    # Phase 3: Final polish with exact objective
    if time.time() - start_time < time_limit - 60:
        try:
            res_final = minimize(objective_exact, best_params, method='L-BFGS-B', 
                                 options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res_final.fun < -best_score:
                best_score = -res_final.fun
                best_params = res_final.x
        except Exception:
            pass
            
    # Phase 4: Nelder-Mead polish for non-smooth regions
    if time.time() - start_time < time_limit - 30:
        try:
            res_nm = minimize(objective_exact, best_params, method='Nelder-Mead',
                              options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-8})
            if res_nm.fun < -best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
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
