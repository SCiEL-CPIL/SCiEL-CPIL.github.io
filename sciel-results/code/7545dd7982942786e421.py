"""Rewritten program for second autocorrelation inequality task.

Uses a smooth log-sum-exp surrogate for norm_inf with analytic gradients,
L-BFGS-B optimization, diverse structured starts, and Nelder-Mead polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # len(g) = 99
W = 1.0 / (M + 1)  # 1/100
DEADLINE = 130.0  # safety margin before 150s kill


def _build_y(g):
    """Build y = [0, g[0], ..., g[M-1], 0]."""
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=np.float64)
    if len(h) != N:
        return 0.0
    if not np.all(np.isfinite(h)):
        return 0.0
    if np.any(h < 0):
        return 0.0

    g = np.convolve(h, h)
    m = len(g)

    y = _build_y(g)

    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (W / 3.0) * np.sum(terms)

    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))

    if norm_1 * norm_inf == 0:
        return 0.0

    value = l2 / (norm_1 * norm_inf)
    return value


def _smooth_value_and_grad(h):
    """
    Compute smooth surrogate value and gradient.
    Uses log-sum-exp for smooth max, and smooth L1 approximation.
    h is assumed non-negative and we work on the simplex (sum(h)=1).
    """
    h = np.asarray(h, dtype=np.float64)
    if len(h) != N or not np.all(np.isfinite(h)) or np.any(h < 0):
        return 0.0, np.zeros(N)

    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)

    # Smooth L2 numerator: l2 = (w/3) * sum(y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # This is smooth in h, so we can compute exact gradient.
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (W / 3.0) * np.sum(terms)

    # For norm_1: use soft L1 with temperature
    # For norm_inf: use log-sum-exp with temperature
    # We'll use a combined smooth surrogate
    # 
    # Actually, let's use a different approach:
    # value = l2 / (norm_1 * norm_inf)
    # Since h >= 0, g >= 0, so abs() is identity.
    # norm_1 = sum(g) / (m+1) = (sum(h))^2 / (m+1)
    # norm_inf = max(g)
    #
    # If we fix sum(h) = 1, then norm_1 = 1/(m+1) is constant!
    # So value = l2 * (m+1) / norm_inf
    #
    # This simplifies things. Let's optimize over the simplex sum(h)=1.
    # Then we just need to maximize l2 / norm_inf.
    
    # For smooth max, use log-sum-exp: max(g) ≈ (1/t) * log(sum(exp(t*g)))
    # For better conditioning, use: max(g) ≈ (1/t) * log(sum(exp(t*(g - max_g)))) + max_g
    # But we want differentiability, so use the standard log-sum-exp.
    
    t = 50.0  # temperature for log-sum-exp
    
    g_max = np.max(g)
    # Smooth max using log-sum-exp
    # log_sum_exp(t*g) = t*max(g) + log(sum(exp(t*(g - max(g)))))
    exp_shifted = np.exp(t * (g - g_max))
    log_sum_exp_val = t * g_max + np.log(np.sum(exp_shifted))
    smooth_max = log_sum_exp_val / t
    
    # Gradient of smooth_max w.r.t. g: softmax(t*g)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Now compute gradient of l2 w.r.t. h
    # l2 = (W/3) * sum_{i=0}^{m} (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # y = [0, g[0], ..., g[m-1], 0]
    # g = h * h (convolution)
    # 
    # Let's compute d(l2)/d(g) first, then chain rule to h.
    
    # d(l2)/d(y[i]) for i in 0..m+1
    dy_l2 = np.zeros(m + 2)
    for i in range(m + 1):
        dy_l2[i] += (W / 3.0) * (2 * y[i] + y[i+1])
        dy_l2[i+1] += (W / 3.0) * (y[i] + 2 * y[i+1])
    
    # y[i] = g[i-1] for i=1..m, y[0]=y[m+1]=0
    # So d(l2)/d(g[j]) = dy_l2[j+1] for j=0..m-1
    dg_l2 = dy_l2[1:m+1]
    
    # g = h * h, so g[k] = sum_{i+j=k} h[i]*h[j]
    # d(g[k])/d(h[a]) = h[k-a] + h[k-a] if k-a is valid index, accounting for symmetry
    # More precisely: g[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i] * h[k-i]
    # d(g[k])/d(h[a]) = h[k-a] + (h[k-a] if a != k-a else h[k-a]) 
    # Actually: d(g[k])/d(h[a]) = h[k-a] + h[k-a] if a != k-a, else 2*h[k-a]
    # But since convolution: g[k] = sum_i h[i]*h[k-i]
    # d(g[k])/d(h[a]) = h[k-a] + h[k-a] if both terms exist and a != k-a
    # Let me just compute it directly:
    # g[k] = sum_{i=0}^{N-1} h[i] * h[k-i] (with h[i]=0 if i out of [0,N-1])
    # d(g[k])/d(h[a]) = h[k-a] + h[k-a] if a != k-a, else 2*h[k-a]
    # Wait, no: d/dh[a] of sum_i h[i]*h[k-i] = h[k-a] + h[k-a] if a appears in both terms
    # The term i=a gives h[a]*h[k-a], derivative w.r.t h[a] is h[k-a]
    # The term i=k-a gives h[k-a]*h[a], derivative w.r.t h[a] is h[k-a]
    # So if a != k-a, total is 2*h[k-a]
    # If a == k-a (i.e., 2a=k), then it's just one term h[a]^2, derivative is 2*h[a]
    # So in all cases: d(g[k])/d(h[a]) = 2*h[k-a] if 0 <= k-a <= N-1, else h[k-a] if... 
    # Actually simpler: it's always 2*h[k-a] if k-a is in [0,N-1], else 0
    # Wait, let me re-derive:
    # g[k] = sum_{i=0}^{N-1} h[i] * h[k-i]
    # d(g[k])/d(h[a]) = sum_i [delta_{i,a}*h[k-i] + h[i]*delta_{i,k-a}]
    # = h[k-a] + h[k-a] if both a and k-a are in [0,N-1]
    # = h[k-a] if only one is in range
    # Since h[i]=0 for i not in [0,N-1], we can just say:
    # d(g[k])/d(h[a]) = h[k-a] + h[k-a] if 0<=k-a<=N-1, but if a==k-a then it's 2*h[a]
    # Actually the formula h[k-a] + h[k-a] works even when a==k-a, giving 2*h[a]
    # So: d(g[k])/d(h[a]) = 2*h[k-a] if 0 <= k-a <= N-1, else 0
    
    # d(value)/d(h[a]) where value = l2 / smooth_max (since norm_1 is constant on simplex)
    # = (d(l2)/d(h[a]) * smooth_max - l2 * d(smooth_max)/d(h[a])) / smooth_max^2
    
    # d(l2)/d(h[a]) = sum_k dg_l2[k] * d(g[k])/d(h[a])
    # d(smooth_max)/d(h[a]) = sum_k weights[k] * d(g[k])/d(h[a])
    
    dh_l2 = np.zeros(N)
    dh_smooth_max = np.zeros(N)
    
    for k in range(m):
        if k < 0 or k >= m:
            continue
        # d(g[k])/d(h[a]) = 2*h[k-a] if 0 <= k-a <= N-1
        for a in range(N):
            j = k - a
            if 0 <= j < N:
                dh_l2[a] += 2.0 * h[j] * dg_l2[k]
                dh_smooth_max[a] += 2.0 * h[j] * weights[k]
    
    # value = l2 / smooth_max (on simplex)
    # d(value)/d(h[a]) = (dh_l2[a] * smooth_max - l2 * dh_smooth_max[a]) / smooth_max^2
    
    if smooth_max <= 0:
        return 0.0, np.zeros(N)
    
    value = l2 / smooth_max
    grad = (dh_l2 * smooth_max - l2 * dh_smooth_max) / (smooth_max * smooth_max)
    
    return value, grad


def _generate_starts():
    """Generate diverse starting points on the simplex."""
    starts = []
    rng = np.random.RandomState(42)
    
    # Uniform
    starts.append(np.ones(N) / N)
    
    # Triangular
    h = np.linspace(0, 1, N)
    starts.append(h / np.sum(h))
    
    # Inverted triangular
    h = np.linspace(1, 0, N)
    starts.append(h / np.sum(h))
    
    # Gaussian centered
    x = np.arange(N)
    h = np.exp(-((x - N/2.0)**2) / (2 * (N/6.0)**2))
    starts.append(h / np.sum(h))
    
    # Concentrated in middle
    h = np.zeros(N)
    h[N//2 - 5:N//2 + 5] = 1.0
    starts.append(h / np.sum(h))
    
    # Concentrated at edges
    h = np.zeros(N)
    h[:5] = 1.0
    h[-5:] = 1.0
    starts.append(h / np.sum(h))
    
    # Bimodal
    h = np.zeros(N)
    h[N//4 - 3:N//4 + 3] = 1.0
    h[3*N//4 - 3:3*N//4 + 3] = 1.0
    starts.append(h / np.sum(h))
    
    # Power law
    h = np.power(np.linspace(0.1, 1.0, N), 2)
    starts.append(h / np.sum(h))
    
    # Sparse spike
    h = np.zeros(N)
    h[N//2] = 1.0
    starts.append(h / np.sum(h))
    
    # Two spikes
    h = np.zeros(N)
    h[N//4] = 1.0
    h[3*N//4] = 1.0
    starts.append(h / np.sum(h))
    
    # Random Dirichlet starts
    for _ in range(20):
        h = rng.dirichlet(np.ones(N))
        starts.append(h)
    
    # More structured: alternating
    h = np.ones(N)
    h[::2] = 0.1
    starts.append(h / np.sum(h))
    
    # Smooth bump
    x = np.arange(N)
    h = np.exp(-((x - N*0.3)**2) / (2 * 3.0**2)) + np.exp(-((x - N*0.7)**2) / (2 * 3.0**2))
    starts.append(h / np.sum(h))
    
    # Narrow Gaussian
    h = np.exp(-((x - N/2.0)**2) / (2 * 2.0**2))
    starts.append(h / np.sum(h))
    
    # Wide Gaussian
    h = np.exp(-((x - N/2.0)**2) / (2 * 8.0**2))
    starts.append(h / np.sum(h))
    
    # Edge-heavy with decay
    h = np.exp(-np.abs(x - N/2.0) / 3.0)
    starts.append(h / np.sum(h))
    
    # Cosine taper
    h = np.cos(np.pi * x / N) ** 2
    h = np.clip(h, 0, None)
    starts.append(h / np.sum(h))
    
    # Sine squared
    h = np.sin(np.pi * x / N) ** 2
    starts.append(h / np.sum(h))
    
    return starts


def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Seed value
    seed_h = np.ones(N)
    seed_value = compute_value(seed_h)
    
    best_h = seed_h.copy()
    best_value = seed_value
    
    starts = _generate_starts()
    
    for i, h0 in enumerate(starts):
        if time_left() < 8.0:
            break
        
        h0 = np.asarray(h0, dtype=np.float64)
        h0 = np.clip(h0, 0, None)
        s = np.sum(h0)
        if s <= 0:
            continue
        h0 = h0 / s
        
        # Use L-BFGS-B with analytic gradient
        # Value is scale-invariant, so we optimize on the simplex via penalty
        
        def objective_and_grad(h):
            val, grad = _smooth_value_and_grad(h, t=50.0)
            # Add penalty for leaving simplex
            penalty = 0.01 * (np.sum(h) - 1.0)**2
            pen_grad = 0.02 * (np.sum(h) - 1.0) * np.ones(N)
            return -val + penalty, -grad + pen_grad
        
        try:
            res = minimize(
                objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                bounds=[(1e-10, 1.0)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_opt = np.clip(res.x, 0, None)
            s = np.sum(h_opt)
            if s > 0:
                h_opt = h_opt / s
            
            val_opt = compute_value(h_opt)
            
            if val_opt > best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # Nelder-Mead polish on exact objective
    if time_left() > 5.0:
        try:
            def exact_neg_value(h):
                return -compute_value(h)
            
            res_final = minimize(
                exact_neg_value,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            h_final = np.clip(res_final.x, 0, None)
            s = np.sum(h_final)
            if s > 0:
                h_final = h_final / s
            
            val_final = compute_value(h_final)
            
            if val_final > best_value:
                best_value = val_final
                best_h = h_final.copy()
                
        except Exception:
            pass
    
    # Second Nelder-Mead polish with different initial simplex
    if time_left() > 3.0:
        try:
            def exact_neg_value2(h):
                return -compute_value(h)
            
            # Perturb best_h slightly to escape potential local minimum
            rng2 = np.random.RandomState(123)
            h_perturbed = best_h + rng2.randn(N) * 0.01
            h_perturbed = np.clip(h_perturbed, 0, None)
            s = np.sum(h_perturbed)
            if s > 0:
                h_perturbed = h_perturbed / s
            
            res_final2 = minimize(
                exact_neg_value2,
                h_perturbed,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            h_final2 = np.clip(res_final2.x, 0, None)
            s = np.sum(h_final2)
            if s > 0:
                h_final2 = h_final2 / s
            
            val_final2 = compute_value(h_final2)
            
            if val_final2 > best_value:
                best_value = val_final2
                best_h = h_final2.copy()
                
        except Exception:
            pass
    
    # Ensure non-negative and finite
    best_h = np.clip(best_h, 0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = seed_h
    
    return list(best_h)
