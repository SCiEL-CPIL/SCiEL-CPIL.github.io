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
        def obj(h):
            c_all = get_correlations(h)
            max_c = np.max(c_all)
            if beta > 0:
                exp_c = np.exp(beta * (c_all - max_c))
                lse = max_c + np.log(np.sum(exp_c)) / beta
            else:
                lse = max_c
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - target_sum)**2
            
            return lse + sum_pen
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

    # Phase 2: Simulated Annealing + Local Polish
    # We will run a few SA loops, each ending with a polish.
    num_sa_loops = 5
    sa_steps = 2000
    
    for loop_idx in range(num_sa_loops):
        if time.time() > deadline - 10.0:
            break
            
        # Start from a random point or best found
        if loop_idx == 0:
            h_curr = best_h.copy()
        else:
            h_curr = rng.uniform(0.2, 0.8, N)
            h_curr = project_to_feasible(h_curr)
            
        T_start = 0.5
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / sa_steps)
        
        h_sa = h_curr.copy()
        score_sa = true_score(h_sa)
        
        for step in range(sa_steps):
            if time.time() > deadline - 5.0:
                break
                
            # Generate neighbor: perturb one or two indices
            idx1 = rng.integers(0, N)
            idx2 = rng.integers(0, N)
            
            delta1 = rng.normal(0, 0.1)
            delta2 = rng.normal(0, 0.1)
            
            h_new = h_sa.copy()
            h_new[idx1] += delta1
            h_new[idx2] += delta2
            
            h_new = project_to_feasible(h_new)
            
            score_new = true_score(h_new)
            
            # Acceptance criterion
            if score_new < score_sa:
                h_sa = h_new
                score_sa = score_new
            else:
                delta_e = score_new - score_sa
                if delta_e < 0:
                    h_sa = h_new
                    score_sa = score_new
                else:
                    T = T_start * (cooling_rate ** step)
                    if rng.random() < np.exp(-delta_e / (T + 1e-9)):
                        h_sa = h_new
                        score_sa = score_new
                        
        # Polish the result of SA
        h_polished = h_sa.copy()
        
        # Use L-BFGS-B with high beta for polish
        obj_func = make_objective(50.0, 100.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                h_polished, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = res.x
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
                
        except Exception:
            pass

    # Phase 3: Final high-precision polish on the best found
    if time.time() < deadline - 5.0:
        obj_func = make_objective(100.0, 1000.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                best_h, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = res.x
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass

    return best_h
