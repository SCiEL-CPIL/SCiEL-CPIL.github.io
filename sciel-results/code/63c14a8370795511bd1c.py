"""
Combined approach:
1. Global search via Simulated Annealing to escape local minima.
2. Local refinement via L-BFGS-B with LogSumExp approximation.
3. Diverse initializations including low-autocorrelation sequences.
"""
import numpy as np
import time
import scipy.optimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    target_sum = N / 2.0
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    def get_correlations(h):
        """Compute linear correlations of h and 1-h for all non-zero shifts."""
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude center (shift 0)
        c1 = c[:N-1]
        c2 = c[N:]
        return np.concatenate([c1, c2])

    def true_score(h):
        """Calculate the raw score (max overlap)."""
        c_all = get_correlations(h)
        return np.max(c_all)

    def project_to_feasible(h):
        """Project h to satisfy sum(h) = N/2 and 0 <= h <= 1."""
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        
        if abs(diff) > 1e-12:
            for _ in range(20):
                if abs(diff) < 1e-12:
                    break
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
                current_sum = np.sum(h)
                diff = target_sum - current_sum
        return h

    def make_objective(beta, lambda_sum):
        """
        Epigraph formulation with soft constraints.
        Variables: h (heights), t (upper bound on max overlap).
        Objective: t + penalty(sum) + smooth_max(C_k(h) - t)
        """
        def obj(h_t):
            h = h_t[:N]
            t = h_t[N]
            
            c_all = get_correlations(h)
            
            # Soft max of (c_all - t)
            # To avoid overflow, shift by max
            shifted = c_all - t
            max_shifted = np.max(shifted)
            if beta > 0:
                # log-sum-exp trick
                lse = max_shifted + np.log(np.sum(np.exp(beta * (shifted - max_shifted)))) / beta
            else:
                lse = np.max(shifted)
                
            # The term log(1 + exp(beta*x)) approximates max(0, x).
            # We want to penalize violations c_k > t.
            # sum_k log(1 + exp(beta * (c_k - t)))
            # This is equivalent to sum_k softplus(beta * (c_k - t)) / beta * beta?
            # Let's use the direct softplus form: (1/beta) * sum log(1 + exp(beta * (c_k - t)))
            # This is 0 if c_k << t, and approx c_k - t if c_k >> t.
            
            # However, the term above `lse` is already the soft max of (c_k - t).
            # If we minimize t + softmax(c-t), we are minimizing the upper envelope.
            # Let's verify: if c_k are all < t, softmax is approx t - epsilon? No.
            # max(c_k - t) is negative.
            # We want t to be the max.
            # Minimize t + max(0, max(c_k - t)).
            # If max(c_k) < t, cost is t. Gradient wrt t is 1. t decreases.
            # If max(c_k) > t, cost is t + max(c_k - t) = max(c_k). Gradient wrt t is 0. t is stuck?
            # No, gradient wrt h pushes c_k down.
            
            # So the objective should be: t + (1/beta) * sum log(1 + exp(beta * (c_k - t)))
            # This ensures t is pushed up to cover c_k, and down if c_k < t.
            
            # Let's compute this properly.
            x = beta * (c_all - t)
            # stable log(1+exp(x))
            log_1_plus_exp = np.logaddexp(0, x)
            soft_penalty = np.sum(log_1_plus_exp) / beta
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - target_sum)**2
            
            return t + soft_penalty + sum_pen
        
        return obj

    # Initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Random
    for _ in range(10):
        h_rand = rng.uniform(0.1, 0.9, N)
        candidates.append(h_rand)
        
    # Sine waves
    t = np.arange(N)
    for freq in [1, 2, 3, 5, 10]:
        candidates.append(0.5 + 0.4 * np.sin(2 * np.pi * freq * t / N))
        candidates.append(0.5 + 0.4 * np.cos(2 * np.pi * freq * t / N))
        
    # Thue-Morse like (low autocorrelation)
    def thue_morse(n):
        tm = [0]
        for i in range(n-1):
            tm.append(1 - tm[-1])
        # Generate longer sequence
        res = []
        for i in range(n):
            res.append(tm[i % n])
        return res
        
    # Actually, let's use a simple binary pattern converted to continuous
    h_bin = np.zeros(N)
    h_bin[:N//2] = 1.0
    # Shuffle to break periodicity
    rng.shuffle(h_bin)
    candidates.append(h_bin)
    
    # Van der Corput
    def van_der_corput(n, base=2):
        vdc = [0.0]
        for i in range(1, n):
            vdc.append(1.0 - (vdc[-1] + 1.0 / (base ** (i - 1))) % 1.0)
        return vdc
        
    h_vdc = van_der_corput(N)
    # Scale to [0,1] and adjust sum
    h_vdc = (h_vdc - np.min(h_vdc)) / (np.max(h_vdc) - np.min(h_vdc) + 1e-9)
    candidates.append(h_vdc)

    best_h = None
    best_score = float('inf')
    
    # Phase 1: Evaluate candidates
    for h_cand in candidates:
        h_proj = project_to_feasible(h_cand)
        score = true_score(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj

    # Phase 2: Multi-start L-BFGS-B with Epigraph
    num_starts = 15
    
    # Bounds for h are [0,1]. Bounds for t are [0, N] (max possible overlap is N)
    bounds_ht = bounds + [(0.0, float(N))]
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10.0:
            break
            
        # Generate diverse start
        if start_idx == 0:
            h_curr = best_h.copy()
        elif start_idx % 3 == 0:
            # Random uniform
            h_curr = rng.uniform(0.2, 0.8, N)
        elif start_idx % 3 == 1:
            # Sine wave with random parameters
            freq = rng.integers(1, 10)
            phase = rng.uniform(0, 2 * np.pi)
            t_arr = np.arange(N)
            h_curr = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t_arr / N + phase)
        else:
            # Random walk
            h_curr = np.cumsum(rng.normal(0, 0.1, N))
            h_curr = (h_curr - np.min(h_curr)) / (np.max(h_curr) - np.min(h_curr) + 1e-9)
            h_curr = 0.2 + 0.6 * h_curr
            
        h_curr = project_to_feasible(h_curr)
        
        # Initial t estimate
        t_init = true_score(h_curr) + 0.1
        
        x_init = np.concatenate([h_curr, [t_init]])
        
        # Use L-BFGS-B with moderate beta for global-ish search
        # Beta controls the sharpness of the soft-max. 
        # Start with lower beta for global, increase for local.
        obj_func = make_objective(20.0, 100.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                x_init, 
                method='L-BFGS-B', 
                bounds=bounds_ht, 
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_opt = res.x
            h_opt = x_opt[:N]
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
                
        except Exception:
            pass

    # Phase 3: Final high-precision polish on the best found
    if time.time() < deadline - 5.0:
        # Use a very high beta to approximate max closely
        # Note: x_init must include t
        t_init = true_score(best_h) + 0.01
        x_init = np.concatenate([best_h, [t_init]])
        
        obj_func = make_objective(100.0, 1000.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                x_init, 
                method='L-BFGS-B', 
                bounds=bounds_ht, 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x_opt = res.x
            h_opt = x_opt[:N]
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass
            
        # One more polish with even tighter tolerance if time permits
        if time.time() < deadline - 2.0:
            t_init = true_score(best_h) + 0.001
            x_init = np.concatenate([best_h, [t_init]])
            obj_func_tight = make_objective(200.0, 1000.0)
            try:
                res = scipy.optimize.minimize(
                    obj_func_tight, 
                    x_init, 
                    method='L-BFGS-B', 
                    bounds=bounds_ht, 
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                x_opt = res.x
                h_opt = x_opt[:N]
                h_opt = project_to_feasible(h_opt)
                score_opt = true_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt
            except Exception:
                pass

    return best_h
