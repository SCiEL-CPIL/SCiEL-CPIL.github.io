"""Optimized program for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_value(h):
    """Compute the objective value for a given h."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))

def _soft_max(x, beta=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    shifted = beta * (x - x_max)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    return x_max + np.log(np.sum(np.exp(shifted))) / beta

def _objective_soft(x, beta=100.0):
    """Differentiable objective using soft-max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    sm = _soft_max(conv, beta)
    return 2.0 * N * sm / (s ** 2)

def _objective_hard(x):
    """Exact objective with hard max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s ** 2))

def _grad_soft(x, beta=100.0):
    """Gradient of the soft-max objective (without abs, assuming positive)."""
    h = x
    n = len(h)
    s = np.sum(h)
    conv = np.convolve(h, h, mode='full')
    x_max = np.max(conv)
    shifted = beta * (conv - x_max)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # softmax weights
    
    # d(sm)/d(conv_k) = weights[k]
    # sm = x_max + log(sum_exp)/beta
    # d(sm)/dh = sum_k weights[k] * d(conv_k)/dh
    
    # conv_k = sum_i h_i * h_{k-i} for k in 0..2n-2
    # d(conv_k)/dh_j = h_{k-j} + h_{k-j} (if k-j valid, considering both terms)
    # Actually conv is full: conv[k] = sum_{i} h[i] * h[k-i] where valid indices
    # d(conv[k])/dh[j] = h[k-j] if k-j in [0,n-1], plus h[j-k] if j-k in [0,n-1]... 
    # More carefully: conv[k] = sum_i h[i]*h[k-i] (with h[i]=0 outside [0,n-1])
    # d(conv[k])/dh[j] = h[k-j] + (if j != k-j) ... no, it's just:
    # d/dh[j] sum_i h[i]*h[k-i] = h[k-j] (from i=j term) + sum_{i != j} h[i] * d(h[k-i])/dh[j]
    # = h[k-j] + h[j-k] (from i such that k-i=j, i.e., i=k-j, if k-j != j)
    # Wait, if k-i = j then i = k-j. So the term is h[k-j] (from i=j) and if k-j != j, 
    # there's also h[k-j] from the other factor? No.
    # conv[k] = sum_i h[i]*h[k-i]
    # d/dh[j]: term i=j gives h[j]*h[k-j] derivative is h[k-j]
    # term i=k-j gives h[k-j]*h[j] derivative is h[j]
    # So if j != k-j: d(conv[k])/dh[j] = h[k-j] + h[j]
    # if j == k-j (i.e., k=2j): d(conv[k])/dh[j] = 2*h[j]
    
    # This is getting complex. Let me use a simpler approach: 
    # The gradient of sum_k weights[k]*conv[k] with respect to h
    # = sum_k weights[k] * d(conv[k])/dh
    # 
    # Using the fact that conv = h * h (linear convolution),
    # d(conv)/dh is a 2n-1 by n matrix. 
    # An efficient way: grad_h = convolve(weights, h) + convolve(h, weights) 
    # but with proper alignment.
    
    # Actually, let's just compute it directly but efficiently.
    # grad_h[j] = sum_k weights[k] * (h[k-j] + h[j-k]) where indices are valid
    # = sum_k weights[k]*h[k-j] + sum_k weights[k]*h[j-k]
    # = (weights * h)[j] reversed appropriately + (weights * h)[j]
    
    # Let me use: the linear convolution c = conv(h, h)
    # dc/dh is a matrix. grad = (dc/dh).T @ weights
    # 
    # Alternative: use the identity that if c = conv(h, h), then
    # grad_h = conv(weights, h) + conv(h, weights) but we need to be careful.
    # Actually for full convolution:
    # c[k] = sum_i h[i]*h[k-i]
    # grad[j] = sum_k w[k] * (h[k-j] + h[j-k])  [with zero padding]
    # = sum_k w[k]*h[k-j] + sum_k w[k]*h[j-k]
    # First term: let m = k-j, k = m+j: sum_m w[m+j]*h[m] = (w_shifted * h) where w_shifted[m] = w[m+j]
    # This is cross-correlation.
    
    # Simpler: just compute numerically or use the fact that
    # sum_k w[k]*conv[k] = sum_k w[k] * sum_i h[i]*h[k-i]
    # = sum_i h[i] * sum_k w[k]*h[k-i]
    # = sum_i h[i] * (w * h)[i]  where (w*h) is cross-correlation
    # = h dot correlate(w, h)
    # So grad_h = correlate(w, h)
    
    # np.correlate(w, h, 'full') gives the cross-correlation
    # But we need to align it properly.
    
    # Let me just do a direct computation which is O(n^2) but n=400 so it's fine.
    grad_h = np.zeros(n)
    for k in range(2*n - 1):
        wk = weights[k]
        if wk < 1e-15:
            continue
        for j in range(n):
            # d(conv[k])/dh[j]
            # conv[k] = sum_i h[i]*h[k-i]
            # term from i=j: h[k-j] if 0 <= k-j < n
            # term from i=k-j: h[j] if 0 <= k-j < n and k-j != j
            kj = k - j
            if 0 <= kj < n:
                if kj == j:
                    grad_h[j] += wk * 2.0 * h[j]
                else:
                    grad_h[j] += wk * (h[kj] + h[j])
    
    # Now the full objective is 2*N*sm/s^2
    # d/dh = 2*N * (d(sm)/dh * s^2 - sm * 2*s * ds/dh) / s^4
    # = 2*N * (d(sm)/dh * s - 2*sm*ds/dh) / s^3
    # ds/dh = 1 (vector of ones)
    sm = _soft_max(conv, beta)
    factor = 2.0 * N / (s ** 3)
    grad_obj = factor * (s * grad_h - 2.0 * sm * np.ones(n))
    
    return grad_obj

def construct():
    """Construct step heights to minimize the autocorrelation inequality constant."""
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # Should be 2.0
    
    def try_candidate(h):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=np.float64)
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-15:
            return
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Phase 1: Try structured initializations
    for frac_pos in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        n_pos = max(1, int(frac_pos * N))
        for pos_val in [1.0, 2.0, 5.0, 10.0]:
            for neg_val in [-0.1, -0.5, -1.0, -2.0]:
                h = np.full(N, neg_val)
                h[:n_pos] = pos_val
                try_candidate(h)
    
    # Try: h = a * ones + b * random_signs
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        a = np.random.uniform(0.5, 5.0)
        b = np.random.uniform(-3.0, 3.0)
        signs = np.random.choice([-1, 1], N)
        h = a + b * signs
        try_candidate(h)
    
    # Phase 2: Gradient-based optimization with soft-max
    def run_optimization(start_h, max_iter=3000, beta=100.0):
        nonlocal best_h, best_val
        start_h = np.asarray(start_h, dtype=np.float64).copy()
        s = np.sum(start_h)
        if abs(s) < 1e-10:
            start_h[0] += 1.0
        
        def obj(x):
            return _objective_soft(x, beta)
        
        def grad(x):
            return _grad_soft(x, beta)
        
        try:
            res = minimize(obj, start_h, jac=grad, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'gtol': 1e-9, 'ftol': 1e-12,
                                  'maxls': 50})
            hard_val = _objective_hard(res.x)
            if hard_val < best_val:
                best_val = hard_val
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Run optimization from multiple starting points
    starting_points = [
        np.ones(N),
        np.ones(N) * 0.5,
        np.linspace(1, -1, N),
        np.cos(2 * np.pi * np.arange(N) / N),
        np.sin(2 * np.pi * np.arange(N) / N),
        np.ones(N) - 0.5 * np.cos(2 * np.pi * np.arange(N) / N),
        np.ones(N) + 0.3 * np.sin(4 * np.pi * np.arange(N) / N),
    ]
    
    np.random.seed(123)
    for _ in range(15):
        if time.time() > deadline - 30:
            break
        # Random smooth signals
        freqs = np.random.uniform(0.1, 3.0, 3)
        amps = np.random.uniform(0.1, 1.0, 3)
        phases = np.random.uniform(0, 2*np.pi, 3)
        h = np.ones(N) * 0.5
        for f, a, p in zip(freqs, amps, phases):
            h += a * np.sin(2 * np.pi * f * np.arange(N) / N + p)
        starting_points.append(h)
    
    for sp in starting_points:
        if time.time() > deadline - 10:
            break
        run_optimization(sp, max_iter=2000, beta=100.0)
    
    # Phase 3: Local refinement with higher beta
    if time.time() < deadline - 15:
        for beta in [200.0, 500.0]:
            if time.time() > deadline - 10:
                break
            run_optimization(best_h, max_iter=1000, beta=beta)
    
    # Phase 4: Nelder-Mead polish
    if time.time() < deadline - 5:
        def obj_nm(x):
            return _objective_hard(x)
        
        try:
            res = minimize(obj_nm, best_h, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
            hard_val = _objective_hard(res.x)
            if hard_val < best_val:
                best_val = hard_val
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Try more aggressive structured patterns
    if time.time() < deadline - 5:
        # Two-block with optimized ratio
        for k in range(1, N, 10):
            if time.time() > deadline - 3:
                break
            for ratio in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                h = np.zeros(N)
                h[:k] = ratio
                h[k:] = -1.0
                try_candidate(h)
        
        # Three-block
        for k1 in range(1, N//3, 20):
            if time.time() > deadline - 2:
                break
            for k2 in range(k1+1, 2*N//3, 20):
                if time.time() > deadline - 1:
                    break
                for v1 in [1.0, 2.0]:
                    for v2 in [-0.5, -1.0, -2.0]:
                        for v3 in [0.5, 1.0]:
                            h = np.zeros(N)
                            h[:k1] = v1
                            h[k1:k2] = v2
                            h[k2:] = v3
                            try_candidate(h)
    
    return best_h
