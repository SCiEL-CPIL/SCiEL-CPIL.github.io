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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def _time_left():
        return deadline - time.time()
    
    def _nm_obj(x):
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

    def _nm_obj_grad(x):
        h = np.abs(x)
        l2 = np.linalg.norm(h)
        if l2 < 1e-12:
            return 1e10, np.zeros_like(x)
        
        h_norm = h / l2
        l1 = np.sum(h_norm)
        conv = _autoconvolution(h_norm)
        l_inf = np.max(np.abs(conv))
        
        if l1 < 1e-12 or l_inf < 1e-12:
            return 1e10, np.zeros_like(x)
        
        val = -(1.0 / (l1 * l_inf))
        
        # Gradient of -1/(l1 * l_inf) = 1/(l1^2 * l_inf) * dl1/dh + 1/(l1 * l_inf^2) * dl_inf/dh
        # Wait, f = -1/(l1*l_inf). f' = 1/(l1*l_inf)^2 * (l_inf * l1' + l1 * l_inf')
        
        # dl1/dh = 1 (for h_norm, but we need chain rule)
        # Let's compute gradient w.r.t h_norm first.
        # g = -1/(l1*l_inf)
        # dg/dh_norm = 1/(l1*l_inf)^2 * (l_inf * 1 + l1 * d(l_inf)/dh_norm)
        
        # d(l_inf)/dh_norm_i = sum_{k in argmax} 2 * h_norm[k-i] / l_inf
        
        max_idx = np.where(np.abs(conv) == l_inf)[0]
        grad_linf = np.zeros_like(h_norm)
        for k in max_idx:
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    grad_linf[i] += 2 * h_norm[j]
        grad_linf /= l_inf
        
        grad_hnorm = (1.0 / (l1 * l_inf)**2) * (l_inf * np.ones_like(h_norm) + l1 * grad_linf)
        
        # Chain rule: h_norm = h / l2
        # h = l2 * h_norm
        # We are optimizing x, where h = |x|.
        # This is getting complicated with the abs and normalization.
        # Let's just use L-BFGS-B on x directly with the penalty method or just L-BFGS-B on h >= 0 with penalty.
        
        # Actually, for L-BFGS-B, it's better to optimize h >= 0 directly.
        # Objective: -score(h) + penalty(||h||2 - 1)
        # Let's just return value and zero gradient for now and rely on L-BFGS-B's numerical diff?
        # No, L-BFGS-B is fast if we have gradient.
        
        # Let's try a different approach for gradient.
        # Optimize y = log(h) ? No.
        
        # Let's stick to the robust Nelder-Mead for the final polish, 
        # but use a smarter initialization.
        
        # For now, return val and zero grad to make it work with L-BFGS-B if needed, 
        # but I will primarily use Nelder-Mead.
        return val, np.zeros_like(x)

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
                
    # Power law profiles (new family)
    for center in range(0, N, 2):
        for power in [0.5, 1.0, 1.5, 2.0]:
            h = np.zeros(N)
            for i in range(N):
                dist = abs(i - center)
                h[i] = (1.0 + dist) ** (-power)
            if np.sum(h) > 0:
                h = h / np.linalg.norm(h)
                candidates.append(h)
                
    # Two-peak profiles (new family)
    for c1 in range(0, N, 5):
        for c2 in range(c1 + 5, N, 5):
            for w in [1.0, 2.0]:
                x = np.arange(N)
                h = np.exp(-((x - c1) ** 2) / (2 * w ** 2)) + np.exp(-((x - c2) ** 2) / (2 * w ** 2))
                if np.sum(h) > 0:
                    h = h / np.linalg.norm(h)
                    candidates.append(h)

    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        if time.time() > deadline - 20:
            break
        s = _score_normalized(h)
        scored_candidates.append((s, h))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 10 for polishing
    top_k = 10
    for s, h in scored_candidates[:top_k]:
        if time.time() > deadline - 5:
            break
            
        # Normalize to unit L2 for optimization
        h_opt = h.copy()
        l2 = np.linalg.norm(h_opt)
        if l2 > 0:
            h_opt = h_opt / l2
        else:
            continue
            
        # Nelder-Mead on the negative score
        try:
            res = minimize(_nm_obj, h_opt, method='Nelder-Mead', 
                           options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
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
            
    # If we have time, do a second round of Nelder-Mead on the best found, with different options
    if time.time() < deadline - 10 and best_score > 0:
        try:
            res = minimize(_nm_obj, best_h, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
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
