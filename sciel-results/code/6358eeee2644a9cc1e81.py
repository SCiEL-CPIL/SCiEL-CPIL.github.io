import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def _optimize_weights_fast(support, rng, n_restarts=3, maxiter=100):
    """
    Fast weight optimization for a given support using Nelder-Mead on log-softmax parameterization.
    """
    K = len(support)
    if K == 0:
        return np.zeros(0), 0.0
    
    def neg_ratio_logw(log_w):
        w = np.exp(log_w - np.max(log_w))
        s = np.sum(w)
        if s < 1e-300:
            return 0.0
        w = w / s
        h_temp = np.zeros(N)
        h_temp[support] = w
        return -compute_ratio(h_temp)
    
    best_w = None
    best_score = -np.inf
    
    for r in range(n_restarts):
        if r == 0:
            log_w_init = np.zeros(K)
        else:
            log_w_init = rng.normal(0, 0.5, size=K)
        
        res = minimize(
            neg_ratio_logw,
            log_w_init,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-7}
        )
        
        log_w_opt = res.x
        w_opt = np.exp(log_w_opt - np.max(log_w_opt))
        s = np.sum(w_opt)
        if s < 1e-300:
            continue
        w_opt = w_opt / s
        
        h_temp = np.zeros(N)
        h_temp[support] = w_opt
        score = compute_ratio(h_temp)
        
        if score > best_score:
            best_score = score
            best_w = w_opt.copy()
    
    if best_w is None:
        w_uniform = np.ones(K) / K
        h_temp = np.zeros(N)
        h_temp[support] = w_uniform
        best_score = compute_ratio(h_temp)
        best_w = w_uniform
    
    return best_w, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = -np.inf
    
    # Initialize
    initial_K = 8
    init_indices = rng.choice(N, size=initial_K, replace=False)
    current_support = set(init_indices)
    
    current_w, current_score = _optimize_weights_fast(current_support, rng)
    current_h = np.zeros(N)
    current_h[list(current_support)] = current_w
    
    if current_score > best_score:
        best_score = current_score
        best_h = current_h.copy()
    
    # Simulated Annealing on support
    T_start = 0.15
    T_end = 1e-5
    max_steps = 3000
    
    for step in range(max_steps):
        if time.time() > deadline - 15.0:
            break
            
        progress = step / max_steps
        T = T_start * (T_end / T_start) ** progress
        
        new_support = set(current_support)
        K_curr = len(current_support)
        
        r = rng.random()
        if r < 0.35:
            if K_curr > 0:
                old_elem = rng.choice(list(new_support))
                new_elem = rng.integers(0, N)
                new_support.remove(old_elem)
                new_support.add(new_elem)
        elif r < 0.55:
            new_elem = rng.integers(0, N)
            new_support.add(new_elem)
        elif r < 0.75:
            if K_curr > 3:
                old_elem = rng.choice(list(new_support))
                new_support.remove(old_elem)
        else:
            new_K = max(3, min(18, K_curr + rng.integers(-2, 3)))
            old_list = list(current_support)
            keep_prob = min(1.0, len(old_list) / max(1, new_K))
            new_candidates = []
            for idx in old_list:
                if rng.random() < keep_prob:
                    new_candidates.append(idx)
            while len(new_candidates) < new_K:
                cand = rng.integers(0, N)
                if cand not in new_candidates:
                    new_candidates.append(cand)
            new_support = set(new_candidates[:new_K])
        
        if len(new_support) < 3:
            continue
            
        w_opt, score_new = _optimize_weights_fast(new_support, rng, n_restarts=2, maxiter=80)
        
        if score_new > best_score:
            best_score = score_new
            best_h = np.zeros(N)
            best_h[list(new_support)] = w_opt
        
        if score_new >= current_score:
            current_support = new_support
            current_h = np.zeros(N)
            current_h[list(new_support)] = w_opt
            current_score = score_new
        else:
            delta = score_new - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_support = new_support
                    current_h = np.zeros(N)
                    current_h[list(new_support)] = w_opt
                    current_score = score_new
    
    # Final Polish
    if best_h is not None and time.time() < deadline - 5.0:
        support = np.where(best_h > 1e-8)[0]
        if len(support) > 0:
            w_best, score_best = _optimize_weights_fast(support, rng, n_restarts=5, maxiter=200)
            if score_best > best_score:
                best_score = score_best
                best_h = np.zeros(N)
                best_h[support] = w_best
            
            # Try nearby supports
            for trial in range(30):
                if time.time() > deadline - 2.0:
                    break
                trial_support = set(support)
                n_mut = rng.integers(1, 3)
                for _ in range(n_mut):
                    op = rng.integers(0, 3)
                    if op == 0 and len(trial_support) > 3:
                        elem = rng.choice(list(trial_support))
                        trial_support.remove(elem)
                    elif op == 1:
                        elem = rng.integers(0, N)
                        trial_support.add(elem)
                    else:
                        if len(trial_support) > 0:
                            old = rng.choice(list(trial_support))
                            new = rng.integers(0, N)
                            trial_support.remove(old)
                            trial_support.add(new)
                
                if len(trial_support) < 3:
                    continue
                
                w_trial, score_trial = _optimize_weights_fast(trial_support, rng, n_restarts=2, maxiter=100)
                if score_trial > best_score:
                    best_score = score_trial
                    best_h = np.zeros(N)
                    best_h[list(trial_support)] = w_trial
    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
