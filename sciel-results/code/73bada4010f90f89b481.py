"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    def autoconv(h):
        return np.convolve(h, h, mode='full')
    
    def score_fn(h):
        """Return the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf). Higher is better."""
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h * h)
        if norm2_sq < 1e-30:
            return 0.0
        norm1 = np.sum(h)
        if norm1 < 1e-30:
            return 0.0
        ac = autoconv(h)
        ac_max = np.max(np.abs(ac))
        if ac_max < 1e-30:
            return 0.0
        return norm2_sq / (norm1 * ac_max)
    
    def make_lobes(centers, widths, heights):
        """Create a signal from Gaussian lobes at given centers with given widths and heights."""
        x = np.linspace(-0.25, 0.25, N)
        h = np.zeros(N)
        for c, w, ht in zip(centers, widths, heights):
            h += ht * np.exp(-((x - c) ** 2) / (2 * w ** 2))
        return h
    
    def log_param_score(log_h):
        """Score with log-parameterization. h = exp(log_h)."""
        h = np.exp(log_h)
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h * h)
        if norm2_sq < 1e-30:
            return 0.0
        norm1 = np.sum(h)
        if norm1 < 1e-30:
            return 0.0
        ac = autoconv(h)
        ac_max = np.max(np.abs(ac))
        if ac_max < 1e-30:
            return 0.0
        return norm2_sq / (norm1 * ac_max)
    
    best_h = None
    best_score = 0.0
    
    # Phase 1: Parametric geometric search over lobe configurations
    # Try various numbers of lobes, positions, widths
    
    rng = np.random.RandomState(123)
    
    # Try 2, 3, 4 lobes with different configurations
    for n_lobe in [2, 3, 4, 5]:
        if time.time() > deadline - 30:
            break
        
        for trial in range(30):
            if time.time() > deadline - 30:
                break
            
            if trial == 0:
                # Symmetric configuration
                if n_lobe % 2 == 0:
                    centers = np.linspace(-0.2, 0.2, n_lobe)
                else:
                    centers = np.linspace(-0.2, 0.2, n_lobe)
                widths = np.full(n_lobe, 0.02 + 0.01 * (trial % 3))
                heights = np.ones(n_lobe)
            else:
                # Random configuration
                centers = np.sort(rng.uniform(-0.22, 0.22, n_lobe))
                widths = rng.uniform(0.01, 0.04, n_lobe)
                heights = rng.uniform(0.5, 1.5, n_lobe)
            
            h0 = make_lobes(centers, widths, heights)
            h0 = np.maximum(h0, 0.0)
            if np.sum(h0 * h0) < 1e-30:
                continue
            
            # Normalize
            norm2 = np.sqrt(np.sum(h0 * h0))
            h0 = h0 / norm2
            
            # Local refinement via coordinate descent
            h_curr = h0.copy()
            curr_score = score_fn(h_curr)
            
            # Quick coordinate descent with adaptive steps
            for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
                if time.time() > deadline - 30:
                    break
                improved = True
                while improved and time.time() < deadline - 30:
                    improved = False
                    for i in range(N):
                        # Try perturbation
                        h_trial = h_curr.copy()
                        h_trial[i] += step
                        h_trial = np.maximum(h_trial, 0.0)
                        n2 = np.sqrt(np.sum(h_trial * h_trial))
                        if n2 > 1e-30:
                            h_trial = h_trial / n2
                        s = score_fn(h_trial)
                        if s > curr_score + 1e-12:
                            h_curr = h_trial
                            curr_score = s
                            improved = True
                            continue
                        h_trial = h_curr.copy()
                        h_trial[i] -= step
                        h_trial = np.maximum(h_trial, 0.0)
                        n2 = np.sqrt(np.sum(h_trial * h_trial))
                        if n2 > 1e-30:
                            h_trial = h_trial / n2
                        s = score_fn(h_trial)
                        if s > curr_score + 1e-12:
                            h_curr = h_trial
                            curr_score = s
                            improved = True
            
            if curr_score > best_score:
                best_score = curr_score
                best_h = h_curr.copy()
    
    # Phase 2: Use scipy SLSQP for more sophisticated optimization
    try:
        from scipy.optimize import minimize
        
        def neg_score(h):
            return -score_fn(h)
        
        # Use the best from phase 1 as starting point, plus some structured inits
        starts = []
        if best_h is not None:
            starts.append(best_h.copy())
        
        # Add some structured starts
        x = np.linspace(-0.25, 0.25, N)
        
        # Sparse center
        h_sp = np.zeros(N)
        h_sp[N//2-2:N//2+3] = 1.0
        h_sp = h_sp / np.sqrt(np.sum(h_sp*h_sp))
        starts.append(h_sp)
        
        # Two narrow lobes
        h_2l = make_lobes([-0.1, 0.1], [0.02, 0.02], [1.0, 1.0])
        h_2l = h_2l / np.sqrt(np.sum(h_2l*h_2l))
        starts.append(h_2l)
        
        # Three lobes
        h_3l = make_lobes([-0.15, 0.0, 0.15], [0.02, 0.02, 0.02], [1.0, 1.0, 1.0])
        h_3l = h_3l / np.sqrt(np.sum(h_3l*h_3l))
        starts.append(h_3l)
        
        # Four narrow lobes
        h_4l = make_lobes([-0.18, -0.06, 0.06, 0.18], [0.015, 0.015, 0.015, 0.015], [1.0, 1.0, 1.0, 1.0])
        h_4l = h_4l / np.sqrt(np.sum(h_4l*h_4l))
        starts.append(h_4l)
        
        # Random
        for _ in range(8):
            h_r = np.maximum(rng.random(N), 0.01)
            h_r = h_r / np.sqrt(np.sum(h_r*h_r))
            starts.append(h_r)
        
        for h0 in starts:
            if time.time() > deadline - 15:
                break
            
            h0 = np.maximum(h0, 0.0)
            n2 = np.sqrt(np.sum(h0 * h0))
            if n2 < 1e-30:
                continue
            h0 = h0 / n2
            
            # SLSQP with constraints: sum(h^2) = 1, h >= 0
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x**2) - 1.0,
                     'jac': lambda x: 2 * x}]
            bounds = [(0.0, None)] * N
            
            res = minimize(
                neg_score,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            
            h_opt = np.maximum(res.x, 0.0)
            n2 = np.sqrt(np.sum(h_opt * h_opt))
            if n2 > 1e-30:
                h_opt = h_opt / n2
            s = score_fn(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
    
    except (ImportError, Exception):
        pass
    
    # Phase 3: Fine polish on best solution
    if best_h is not None and time.time() < deadline - 8:
        h_curr = best_h.copy()
        curr_score = score_fn(h_curr)
        
        for step in [0.003, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]:
            if time.time() > deadline - 6:
                break
            improved = True
            while improved and time.time() < deadline - 6:
                improved = False
                for i in range(N):
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = np.maximum(h_trial, 0.0)
                    n2 = np.sqrt(np.sum(h_trial * h_trial))
                    if n2 > 1e-30:
                        h_trial = h_trial / n2
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-12:
                        h_curr = h_trial
                        curr_score = s
                        improved = True
                        continue
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = np.maximum(h_trial, 0.0)
                    n2 = np.sqrt(np.sum(h_trial * h_trial))
                    if n2 > 1e-30:
                        h_trial = h_trial / n2
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-12:
                        h_curr = h_trial
                        curr_score = s
                        improved = True
        
        if curr_score > best_score:
            best_score = curr_score
            best_h = h_curr.copy()
    
    # Phase 4: Smooth surrogate optimization with L-BFGS-B
    # Use log-sum-exp for a smooth approximation of max(|ac|)
    if best_h is not None and time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            
            # Smooth surrogate: use log-sum-exp to approximate max(|ac|)
            # This makes the objective differentiable
            def smooth_ac_max(ac, beta=50.0):
                """Smooth approximation of max(|ac|) using log-sum-exp."""
                abs_ac = np.abs(ac)
                # Subtract max for numerical stability
                m = np.max(abs_ac)
                if m < 1e-30:
                    return m
                shifted = abs_ac / m
                log_sum_exp = np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta
                return m * np.exp(log_sum_exp)
            
            def neg_smooth_score(h):
                """Score with smooth surrogate for the max in autoconvolution."""
                h = np.maximum(h, 0.0)
                norm2_sq = np.sum(h * h)
                if norm2_sq < 1e-30:
                    return 1e10
                norm1 = np.sum(h)
                if norm1 < 1e-30:
                    return 1e10
                ac = autoconv(h)
                ac_max_smooth = smooth_ac_max(ac, beta=50.0)
                if ac_max_smooth < 1e-30:
                    return 1e10
                return -(norm2_sq / (norm1 * ac_max_smooth))
            
            # Convert to log parameterization for positivity
            h_base = best_h.copy()
            h_base = np.maximum(h_base, 1e-10)
            log_h0 = np.log(h_base)
            
            def neg_log_smooth_score(log_h):
                h = np.exp(log_h)
                return neg_smooth_score(h)
            
            # First pass: coarse optimization with smooth surrogate
            res = minimize(
                neg_log_smooth_score,
                log_h0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = np.exp(res.x)
            h_opt = np.maximum(h_opt, 0.0)
            n2 = np.sqrt(np.sum(h_opt * h_opt))
            if n2 > 1e-30:
                h_opt = h_opt / n2
            s = score_fn(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
            
            # Second pass: fine polish with higher beta (closer to true max)
            if time.time() < deadline - 3:
                h_base2 = best_h.copy()
                h_base2 = np.maximum(h_base2, 1e-10)
                log_h0_2 = np.log(h_base2)
                
                def neg_log_smooth_score_fine(log_h):
                    h = np.exp(log_h)
                    h = np.maximum(h, 0.0)
                    norm2_sq = np.sum(h * h)
                    if norm2_sq < 1e-30:
                        return 1e10
                    norm1 = np.sum(h)
                    if norm1 < 1e-30:
                        return 1e10
                    ac = autoconv(h)
                    ac_max_smooth = smooth_ac_max(ac, beta=200.0)
                    if ac_max_smooth < 1e-30:
                        return 1e10
                    return -(norm2_sq / (norm1 * ac_max_smooth))
                
                res2 = minimize(
                    neg_log_smooth_score_fine,
                    log_h0_2,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt2 = np.exp(res2.x)
                h_opt2 = np.maximum(h_opt2, 0.0)
                n2_2 = np.sqrt(np.sum(h_opt2 * h_opt2))
                if n2_2 > 1e-30:
                    h_opt2 = h_opt2 / n2_2
                s2 = score_fn(h_opt2)
                if s2 > best_score:
                    best_score = s2
                    best_h = h_opt2.copy()
                    
        except (ImportError, Exception):
            pass
    
    # Final high-precision polish: alternating small perturbations
    if best_h is not None and time.time() < deadline - 1:
        h_curr = best_h.copy()
        curr_score = score_fn(h_curr)
        
        # Very fine steps to squeeze out last bits
        for step in [0.00002, 0.00001, 0.000005]:
            if time.time() > deadline - 0.5:
                break
            for _ in range(2):  # two passes per step size
                if time.time() > deadline - 0.5:
                    break
                for i in range(N):
                    if time.time() > deadline - 0.5:
                        break
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = np.maximum(h_trial, 0.0)
                    n2 = np.sqrt(np.sum(h_trial * h_trial))
                    if n2 > 1e-30:
                        h_trial = h_trial / n2
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-13:
                        h_curr = h_trial
                        curr_score = s
                        continue
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = np.maximum(h_trial, 0.0)
                    n2 = np.sqrt(np.sum(h_trial * h_trial))
                    if n2 > 1e-30:
                        h_trial = h_trial / n2
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-13:
                        h_curr = h_trial
                        curr_score = s
        
        if curr_score > best_score:
            best_score = curr_score
            best_h = h_curr.copy()
    
    if best_h is None:
        return [1.0] * N
    
    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
