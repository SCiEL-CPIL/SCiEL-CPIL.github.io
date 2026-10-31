import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def smooth_max(x, temp=0.01):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + temp * np.log(np.sum(np.exp((x - x_max) / temp)))

def epigraph_objective(params, temp=0.01):
    """
    Epigraph formulation: minimize t + penalty - h2_sq/t
    params = [log(h_0), ..., log(h_{N-1}), t]
    """
    log_h = params[:N]
    t = params[N]
    
    h = np.exp(log_h)
    if not np.all(np.isfinite(h)):
        return 1e10
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_soft_max = smooth_max(conv, temp)
    
    # Penalty for t being below the smooth max
    penalty = max(0.0, conv_soft_max - t)**2 * 100.0
    
    # We want to minimize t while keeping t >= conv_soft_max
    # and maximize h2_sq / t
    return t + penalty - h2_sq / (t + 1e-10)

def find_sidon_set_greedy(n, start_idx=0, step=1):
    """Greedy construction of a Sidon set."""
    A = []
    used_sums = set()
    for x in range(start_idx, n, step):
        valid = True
        new_sums = []
        for a in A:
            s = a + x
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if 2 * x in used_sums:
            valid = False
        if valid:
            A.append(x)
            used_sums.update(new_sums)
            used_sums.add(2 * x)
    return A

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    def elapsed():
        return time.time() - t0
    
    # Initial best: single spike
    best_h = np.zeros(N, dtype=np.float64)
    best_h[N // 2] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []
    
    def add_candidate(h):
        nonlocal best_h, best_score
        h = np.maximum(h, 0.0)
        if np.sum(h) < 1e-15:
            return
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))
    
    # --- Phase 1: Generate diverse structured candidates ---
    
    # 1. Single spike
    add_candidate(best_h)
    
    # 2. Two spikes (sample)
    for i in range(0, N, 3):
        for j in range(i+3, N, 3):
            h = np.zeros(N)
            h[i] = 0.5
            h[j] = 0.5
            add_candidate(h)
            if elapsed() > deadline - 10:
                return list(best_h)
    
    # 3. Sidon sets with various parameters
    for step in [1, 2, 3]:
        for start in range(0, 5, step):
            A = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(A) >= 2:
                h = np.zeros(N)
                for idx in A:
                    h[idx] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 10:
                    return list(best_h)
    
    # 4. Random sparse sets with varied sparsity
    rng = np.random.default_rng(42)
    for _ in range(60):
        k = rng.integers(2, 25)
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        add_candidate(h)
        if elapsed() > deadline - 10:
            return list(best_h)
    
    # 5. Arithmetic progressions
    for start in range(0, N, 5):
        for step in range(2, 15):
            indices = list(range(start, N, step))
            if len(indices) >= 2:
                h = np.zeros(N)
                h[indices] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 10:
                    return list(best_h)
    
    # 6. Uniform
    add_candidate(np.ones(N))
    
    # 7. Gaussian profiles
    x = np.arange(N)
    for sigma in [2, 5, 10, 20]:
        h = np.exp(-0.5 * ((x - N/2) / sigma)**2)
        add_candidate(h)
    
    # 8. Random phase spectral designs
    for trial in range(15):
        rng_trial = np.random.default_rng(42 + trial)
        phases = rng_trial.uniform(0, 2*np.pi, N)
        for k in range(1, N):
            phases[N-k] = -phases[k]
        H = np.exp(1j * phases)
        h = np.fft.ifft(H).real
        h = np.maximum(h, 0.0)
        if np.sum(h) > 1e-15:
            add_candidate(h)
    
    # Sort by score
    candidates.sort(key=lambda x: -x[0])
    
    # --- Phase 2: Epigraph optimization from top candidates ---
    
    top_candidates = candidates[:25]
    
    for sc, h_init in top_candidates:
        if elapsed() > deadline - 5:
            break
        
        h_norm_init = h_init / np.sum(h_init)
        h_clipped = np.maximum(h_norm_init, 1e-8)
        log_h_init = np.log(h_clipped)
        
        # Initial t: slightly above current conv peak
        conv_init = np.convolve(h_norm_init, h_norm_init, mode='full')
        t_init = np.max(conv_init) * 1.05
        
        params_init = np.concatenate([log_h_init, [t_init]])
        
        bounds = [(-15, 15)] * N + [(1e-6, 1.0)]
        
        try:
            res = minimize(epigraph_objective, params_init, method='L-BFGS-B',
                          bounds=bounds, 
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8})
            
            if elapsed() > deadline - 3:
                break
                
            log_h_opt = res.x[:N]
            h_opt = np.exp(log_h_opt)
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # --- Phase 3: Refine best solution with tighter optimization ---
    
    if elapsed() < deadline - 5:
        h_norm_best = best_h / np.sum(best_h)
        h_clipped = np.maximum(h_norm_best, 1e-8)
        log_h_init = np.log(h_clipped)
        
        conv_best = np.convolve(h_norm_best, h_norm_best, mode='full')
        t_init = np.max(conv_best) * 1.01
        
        params_init = np.concatenate([log_h_init, [t_init]])
        
        bounds = [(-20, 20)] * N + [(1e-6, 1.0)]
        
        try:
            # Use smaller temperature for sharper max approximation
            res = minimize(lambda p: epigraph_objective(p, temp=0.001), 
                          params_init, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
            
            if elapsed() > deadline - 2:
                break
                
            log_h_opt = res.x[:N]
            h_opt = np.exp(log_h_opt)
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # --- Phase 4: Final Nelder-Mead polish ---
    
    if elapsed() < deadline - 1:
        h_norm_best = best_h / np.sum(best_h)
        
        def neg_ratio(x):
            h = np.maximum(x, 0)
            s = np.sum(h)
            if s < 1e-15:
                return 1e10
            return -compute_ratio(h)
        
        try:
            res = minimize(neg_ratio, h_norm_best, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if elapsed() > deadline - 0.5:
                break
                
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
