"""
Structural change: Replace Newton's method with L-BFGS-B using a smooth
log-sum-exp surrogate for the non-smooth norm_inf (peak autocorrelation).
This decouples the optimization from the kink at the peak, allowing
gradient-based convergence to sharper optima.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _smooth_value(h, beta=50.0):
    """
    Smooth surrogate for the value function.
    Uses Log-Sum-Exp for norm_inf to provide differentiability.
    Returns negative value for minimization.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1) # g >= 0
    
    # Smooth max: (1/beta) * log(sum(exp(beta * g)))
    # Shift g to prevent overflow
    g_max = np.max(g)
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0
        
    val = l2 / (norm_1 * lse)
    return -val

def _smooth_value_grad(h, beta=50.0):
    """
    Compute value and gradient of the smooth surrogate.
    Returns (value, gradient) where value is negative (for minimization).
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1) # g >= 0
    
    # Smooth max via log-sum-exp
    g_max = np.max(g)
    exp_terms = np.exp(beta * (g - g_max))
    lse = g_max + (1.0 / beta) * np.log(np.sum(exp_terms))
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0, np.zeros(N)
    
    val = l2 / (norm_1 * lse)
    
    # Gradient computation
    # d(l2)/dh: l2 = (W/3) * sum(a^2 + a*b + b^2)
    # a[i] = y[i] for i in 0..m, b[i] = y[i+1] for i in 0..m
    # y[1:-1] = g, so y[j] = g[j-1] for j in 1..m
    # a[i] = y[i], b[i] = y[i+1]
    # For i=0: a[0]=y[0]=0, b[0]=y[1]=g[0]
    # For i=m: a[m]=y[m]=g[m-1], b[m]=y[m+1]=0
    # For 0<i<m: a[i]=g[i-1], b[i]=g[i]
    
    # dl2/dy[j] for j in 0..m+1:
    # y[j] appears in:
    #   term (j-1): b[j-1]^2 + a[j-1]*b[j-1] -> if j-1 >= 0 and j-1 <= m
    #   term j: a[j]^2 + a[j]*b[j] -> if j <= m
    # So dl2/dy[j] = (W/3) * [2*b[j-1] + a[j-1] (from term j-1, if j>=1) + 2*a[j] + b[j] (from term j, if j<=m)]
    
    dl2_dy = np.zeros(m + 2)
    for i in range(m + 1):
        # Contribution to dl2/dy[i] from term i: 2*a[i] + b[i]
        dl2_dy[i] += 2 * a[i] + b[i]
        # Contribution to dl2/dy[i+1] from term i: 2*b[i] + a[i]
        dl2_dy[i+1] += 2 * b[i] + a[i]
    
    dl2_dy *= (W / 3.0)
    
    # d(norm_1)/dh: norm_1 = sum(g)/(m+1), dg/dh is convolution structure
    # g = convolve(h, h), so dg[0]/dh[0] = h[0], etc.
    # Actually, for g = h * h (convolution), dg[k]/dh[i] = h[k-i] if 0<=k-i<N, else 0
    # More precisely: g[k] = sum_j h[j]*h[k-j], so dg[k]/dh[i] = h[k-i] (if valid) + h[i] (from the other term)
    # Wait, g[k] = sum_j h[j] h[k-j]. dg[k]/dh[i] = h[k-i] + h[i] (both terms where i appears)
    # Actually: dg[k]/dh[i] = h[k-i] (from j=i in first factor) + h[i] (from k-j=i, i.e., j=k-i in second factor)
    # But we need to be careful about bounds.
    
    # For efficiency, let's compute dg/dh as a matrix-free operation.
    # norm_1 = sum(g)/(m+1) = (sum(h)^2)/(m+1) since sum(convolve(h,h)) = sum(h)^2
    # So d(norm_1)/dh[i] = 2*sum(h)*h[i] / (m+1)
    
    sum_h = np.sum(h)
    dnorm1_dh = 2 * sum_h * h / (m + 1)
    
    # d(lse)/dh: lse = g_max + (1/beta)*log(sum(exp(beta*(g-g_max))))
    # d(lse)/dg[k] = exp(beta*(g[k]-g_max)) / sum(exp(beta*(g-g_max))) = exp_terms[k] / sum(exp_terms)
    # d(lse)/dh[i] = sum_k (d(lse)/dg[k]) * (dg[k]/dh[i])
    
    weights = exp_terms / np.sum(exp_terms)  # d(lse)/dg[k]
    
    # dg[k]/dh[i] for g = convolve(h,h):
    # g[k] = sum_j h[j]*h[k-j]
    # dg[k]/dh[i] = h[k-i] (if 0 <= k-i < N) + h[i] (if 0 <= k-i < N, from the symmetric term)
    # Wait, let me think more carefully.
    # g[k] = sum_{j=0}^{N-1} h[j] * h[k-j] where h[k-j] is 0 if k-j not in [0,N-1]
    # dg[k]/dh[i] = h[k-i] (from j=i term, derivative of h[j] w.r.t. h[i]) + h[i] (from j=k-i term, derivative of h[k-j] w.r.t. h[i])
    # But these are the same term if k-i = i, i.e., k=2i.
    # Actually no: when j=i, we get h[i]*h[k-i], derivative w.r.t. h[i] is h[k-i]
    # When j=k-i, we get h[k-i]*h[i], derivative w.r.t. h[i] is h[k-i]
    # So if k-i != i, dg[k]/dh[i] = 2*h[k-i] (if 0 <= k-i < N)
    # If k-i == i (k=2i), dg[k]/dh[i] = h[i] (only one term)
    
    # For efficiency, let's compute d(lse)/dh[i] = sum_k weights[k] * dg[k]/dh[i]
    # = sum_k weights[k] * (h[k-i] + h[i]) for valid k, with adjustment for k=2i
    
    # Alternative: use the fact that d(lse)/dh = convolve(weights, h) + convolve(h, weights) - diagonal correction
    # Actually, let's just do it directly for N=50, M=99 which is small enough.
    
    dlse_dh = np.zeros(N)
    for i in range(N):
        s = 0.0
        for k in range(m):
            # dg[k]/dh[i]:
            # Term 1: j=i in g[k] = sum_j h[j]h[k-j], derivative is h[k-i] if 0<=k-i<N
            # Term 2: j=k-i in g[k], derivative is h[i] if 0<=k-i<N
            # But if k-i == i, both terms are the same, so we only count once
            if 0 <= k - i < N:
                if k - i == i:
                    s += weights[k] * h[i]
                else:
                    s += weights[k] * (h[k-i] + h[i])
        dlse_dh[i] = s
    
    # val = l2 / (norm_1 * lse)
    # d(val)/dh = (dl2/dh * norm_1 * lse - l2 * (dnorm1/dh * lse + norm_1 * dlse/dh)) / (norm_1 * lse)^2
    # = (dl2/dh / (norm_1 * lse)) - l2 * (dnorm1/dh / (norm_1 * lse) + dlse/dh / (norm_1 * lse)) / (norm_1 * lse)
    # Let me recompute:
    # val = l2 / (norm_1 * lse)
    # dval/dh = (dl2/dh * (norm_1 * lse) - l2 * (dnorm1/dh * lse + norm_1 * dlse/dh)) / (norm_1 * lse)^2
    # = dl2/dh / (norm_1 * lse) - l2 * (dnorm1/dh / (norm_1 * lse) + dlse/dh / (norm_1 * lse)) / (norm_1 * lse)
    # Hmm, let me just use the quotient rule properly.
    
    # val = l2 / (norm_1 * lse)
    # Let u = l2, v = norm_1 * lse
    # dval/dh = (du/dh * v - u * dv/dh) / v^2
    # dv/dh = dnorm1/dh * lse + norm_1 * dlse/dh
    
    v = norm_1 * lse
    du_dh = dl2_dy[1:-1]  # dl2/dg, and dg/dh... wait, dl2_dy is dl2/dy, and y[1:-1]=g
    # Actually, dl2_dy[1:-1] gives dl2/dg[k] for k=0..m-1
    # Then du/dh[i] = sum_k (dl2/dg[k]) * (dg[k]/dh[i])
    
    dl2_dg = dl2_dy[1:-1]  # length m
    
    # Compute du/dh
    du_dh = np.zeros(N)
    for i in range(N):
        s = 0.0
        for k in range(m):
            if 0 <= k - i < N:
                if k - i == i:
                    s += dl2_dg[k] * h[i]
                else:
                    s += dl2_dg[k] * (h[k-i] + h[i])
        du_dh[i] = s
    
    dv_dh = dnorm1_dh * lse + norm_1 * dlse_dh
    
    dval_dh = (du_dh * v - val * dv_dh) / (v * v)
    
    # We're minimizing -val, so gradient of objective is -dval_dh
    grad = -dval_dh
    
    return -val, grad

def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    # Uniform
    starts.append(np.ones(N))
    # Gaussian
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(p1 + p2)
    # Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    # Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    # Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    # Random smooth profiles
    for _ in range(20):
        freqs = RNG.randint(1, 8, size=6)
        phases = RNG.uniform(0, 2 * np.pi, size=6)
        amps = RNG.uniform(0.5, 1.0, size=6)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.1, prof))
        
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 15.0) # 15s safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    starts = _generate_starts()
    
    # Initial evaluation
    for s in starts:
        track(s)

    # Optimization Phase
    bounds = [(0, None)] * N
    
    # Use analytic gradient for speed
    # Start with moderate beta, then refine
    betas = [50.0, 100.0, 200.0]
    
    # Reduce number of starts to save time for more iterations
    # Use top 10 starts by initial value
    start_values = [(compute_value(np.maximum(0, s)), s) for s in starts]
    start_values.sort(key=lambda x: -x[0])
    top_starts = [s for _, s in start_values[:10]]
    
    for s in top_starts:
        if time.time() > deadline:
            break
        
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            try:
                def obj(x):
                    return _smooth_value(x, beta)
                
                def obj_grad(x):
                    val, grad = _smooth_value_grad(x, beta)
                    return val, grad
                
                res = opt.minimize(
                    obj,
                    x0,
                    jac=obj_grad,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < 0:
                    track(res.x)
            except Exception:
                pass

    # Polish phase: Use best_h and try local refinements
    for _ in range(40):
        if time.time() > deadline:
            break
        
        # Try different perturbation scales
        scale = 0.05 + 0.1 * RNG.random()
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        
        try:
            def obj(x):
                return _smooth_value(x, 200.0)
            
            def obj_grad(x):
                val, grad = _smooth_value_grad(x, 200.0)
                return val, grad
            
            res = opt.minimize(
                obj,
                pert,
                jac=obj_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            track(res.x)
        except Exception:
            pass

    # Final polish: Try to improve with exact (non-smooth) local search
    # Use scipy's minimize with COBYLA or SLSQP on the exact objective
    if time.time() < deadline - 5.0:
        try:
            def exact_obj(x):
                return -compute_value(np.maximum(0, x))
            
            res = opt.minimize(
                exact_obj,
                best_h,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            track(res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
