import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _score_normalized(h):
    """Score assuming h is non-negative. Returns 0 if invalid."""
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _objective_log_inf(x):
    """
    Objective to minimize: log(||h*h||_inf) subject to ||h||_2 = 1.
    We enforce ||h||_2 = 1 by normalizing inside the objective if needed, 
    but L-BFGS-B works on unconstrained variables. 
    Actually, it's better to use a penalty or just normalize h in the objective.
    However, standard approach for scale invariant problems:
    Minimize f(h) = log(||h*h||_inf) with constraint ||h||_2 = 1.
    We can use the method of Lagrange multipliers or just normalize h in the objective.
    Let's normalize h in the objective to ensure scale invariance is handled.
    """
    h = np.abs(x)  # Ensure non-negativity
    l2 = np.linalg.norm(h)
    if l2 < 1e-12:
        return 1e10, np.zeros_like(x)
    
    # Normalize to unit L2 for the objective calculation
    h_norm = h / l2
    
    l1 = np.sum(h_norm)
    conv = _autoconvolution(h_norm)
    l_inf = np.max(np.abs(conv))
    
    if l1 < 1e-12 or l_inf < 1e-12:
        return 1e10, np.zeros_like(x)
    
    val = np.log(l1) + np.log(l_inf)
    
    # Gradient w.r.t h_norm
    # d/dh log(l1) = 1/l1
    grad_l1 = np.ones_like(h) / l1
    
    # d/dh log(l_inf)
    # max indices
    max_idx = np.where(np.abs(conv) == l_inf)[0]
    grad_linf = np.zeros_like(h)
    for k in max_idx:
        # d/dh_i (h*h)[k] = 2 * h[k-i] (with zero padding)
        # We need to add 2*h[k-i] to grad_linf[i] for all i.
        # This is a convolution of a delta at k with h, scaled by 2.
        # Specifically, grad_linf[i] += 2 * h[k-i] if k-i is in [0, N-1].
        # Let's construct this efficiently.
        # For a fixed k, the vector v_i = h[k-i] is a shifted/reversed h.
        # We can use np.roll or slicing.
        # Let's just loop i for simplicity since N=50 is small.
        for i in range(N):
            j = k - i
            if 0 <= j < N:
                grad_linf[i] += 2 * h_norm[j]
            # Note: autoconvolution is symmetric, so (h*h)[k] = (h*h)[2N-2-k].
            # The derivative formula 2*h[k-i] assumes standard convolution.
            # Let's verify with a small example.
            # h = [1, 0]. h*h = [1, 0, 0].
            # k=0: d/dh_0 = 2*h[0] = 2. d/dh_1 = 2*h[-1]=0.
            # k=1: d/dh_0 = 2*h[1] = 0. d/dh_1 = 2*h[0] = 2.
            # k=2: d/dh_0 = 2*h[2] = 0. d/dh_1 = 2*h[1] = 0.
            # Correct.
            
    grad_linf /= l_inf
    
    # Total gradient w.r.t h_norm
    grad_hnorm = grad_l1 + grad_linf
    
    # Chain rule: h_norm = h / l2.
    # We are optimizing x (which is h in this context, but we passed x).
    # Actually, we should optimize h directly, not x.
    # The function _objective_log_inf takes x, which we treat as h.
    # But we normalized inside.
    # This is problematic for gradient.
    # Let's change strategy: Optimize h directly, and add a penalty for ||h||_2 != 1.
    
    # Penalty term: mu * (||h||_2^2 - 1)^2
    mu = 1000.0
    penalty = mu * (np.sum(h**2) - 1.0)**2
    val += penalty
    
    # Gradient of penalty: 2 * mu * (||h||_2^2 - 1) * 2 * h
    grad_penalty = 4 * mu * (np.sum(h**2) - 1.0) * h
    
    # Gradient of log terms w.r.t h (unnormalized)
    # We calculated grad w.r.t h_norm.
    # h_norm = h / l2.
    # d h_norm_i / d h_j = (delta_ij * l2 - h_i * 2 * h_j / l2) / l2^2 = (delta_ij / l2) - (2 * h_i * h_j) / l2^3
    # This is getting messy.
    
    # Let's restart the gradient calculation for unnormalized h.
    # Objective: f(h) = log(sum(h)) + log(max(conv(h))) + mu * (sum(h^2)-1)^2
    
    # d/dh log(sum(h)) = 1/sum(h)
    g1 = np.ones_like(h) / np.sum(h)
    
    # d/dh log(max(conv(h)))
    # Let M = max(conv(h)).
    # d/dh_i M = sum_{k in argmax} d/dh_i conv_k
    # d/dh_i conv_k = 2 * h[k-i] (zero padded)
    g2 = np.zeros_like(h)
    max_idx = np.where(np.abs(conv) == l_inf)[0]
    for k in max_idx:
        for i in range(N):
            j = k - i
            if 0 <= j < N:
                g2[i] += 2 * h[j]
    g2 /= l_inf
    
    # Total gradient
    grad = g1 + g2 + grad_penalty
    
    return val, grad

def _objective_log_inf_simple(x):
    """Simpler objective for Nelder-Mead: just the score, negated."""
    h = np.abs(x)
    l2 = np.linalg.norm(h)
    if l2 < 1e-12:
        return 1e10
    h_norm = h / l2
    l1 = np.sum(h_norm)
    conv = _autoconvolution(h_norm)
    l_inf = np.max(np.abs(conv))
    if l1 < 1e-12 or l_inf < 1e-12:
        return 1e10
    score = 1.0 / (l1 * l_inf)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # Spikes
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        candidates.append(h)
        
    # Adjacent pairs
    for i in range(N - 1):
        h = np.zeros(N)
        h[i] = 1.0
        h[i+1] = 1.0
        candidates.append(h)
        
    # Small Gaussians
    for center in range(0, N, 2):
        for width in [1.0, 1.5, 2.0, 3.0]:
            x = np.arange(N)
            h = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            if np.max(h) > 0:
                h = h / np.linalg.norm(h)
                candidates.append(h)
                
    # Small Box functions
    for center in range(0, N, 2):
        for half_width in [1, 2, 3]:
            h = np.zeros(N)
            start = max(0, center - half_width)
            end = min(N, center + half_width + 1)
            h[start:end] = 1.0
            if np.sum(h) > 0:
                h = h / np.linalg.norm(h)
                candidates.append(h)
                
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 30:
            break
        s = _score_normalized(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # 2. Polish the top candidates with Nelder-Mead
    # We need to track the top few to polish them.
    # Let's just polish the best one found so far, and maybe a few others if time permits.
    # To be safe, let's collect the top 5 from the candidates.
    
    scored_candidates = []
    for h in candidates:
        s = _score_normalized(h)
        scored_candidates.append((s, h))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    top_k = 5
    for s, h in scored_candidates[:top_k]:
        if time.time() > deadline - 10:
            break
            
        # Normalize to unit L2 for optimization
        h_opt = h.copy()
        l2 = np.linalg.norm(h_opt)
        if l2 > 0:
            h_opt = h_opt / l2
        else:
            continue
            
        # Nelder-Mead on the negative score
        # We use a simple objective that doesn't require gradient
        def nm_obj(x):
            return _objective_log_inf_simple(x)
            
        try:
            res = minimize(nm_obj, h_opt, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            h_res = np.abs(res.x)
            l2_res = np.linalg.norm(h_res)
            if l2_res > 0:
                h_res = h_res / l2_res
                s_res = _score_normalized(h_res)
                if s_res > best_score:
                    best_score = s_res
                    best_h = h_res.copy()
        except Exception:
            pass
            
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    uniform = np.ones(N)
    print("Uniform score:", _score(uniform))
    
    spike = np.zeros(N)
    spike[N//2] = 1.0
    print("Spike score:", _score(spike))
    
    result = construct()
    print("Best score:", _score(np.array(result)))
    print("Result sum:", sum(result))
