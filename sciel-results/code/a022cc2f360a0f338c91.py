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
    Smooth surrogate with analytic gradient for faster convergence.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    
    g_max = np.max(g)
    exp_terms = np.exp(beta * (g - g_max))
    sum_exp = np.sum(exp_terms)
    lse = g_max + (1.0 / beta) * np.log(sum_exp)
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0, np.zeros_like(h)
        
    val = l2 / (norm_1 * lse)
    
    # Gradient via chain rule
    # dval/dg_i = (1/(norm_1*lse)) * (dl2/dg_i - l2 * d(log(norm_1*lse))/dg_i)
    # dl2/dg_i from y = [0, g, 0], y[i]^2 + y[i]*y[i+1] + y[i+1]^2
    # g[i] appears in y[i+1] and y[i+2]
    dl2_dg = np.zeros(m)
    for i in range(m):
        # y[i+1] = g[i], contributes to terms (i-1,i) and (i,i+1) in the sum
        # Term (i-1): y[i-1]^2 + y[i-1]*y[i] + y[i]^2 -> d/dg[i] = y[i-1] + 2*g[i]
        # Term (i): y[i]^2 + y[i]*y[i+1] + y[i+1]^2 -> d/dg[i] = 2*g[i] + y[i+1]
        if i > 0:
            dl2_dg[i] += (a[i-1] + 2*b[i-1])  # from term i-1: y[i-1] + 2*g[i]
        if i < m - 1:
            dl2_dg[i] += (2*a[i] + b[i])       # from term i: 2*g[i] + y[i+1]
    dl2_dg *= W / 3.0
    
    # d(norm_1)/dg_i = 1/(m+1)
    # d(lse)/dg_i = (1/beta) * exp(beta*(g_i - g_max)) / sum_exp
    dlse_dg = (1.0 / beta) * exp_terms / sum_exp
    dnorm1_dg = 1.0 / (m + 1)
    
    # dval/dg_i = (1/(norm_1*lse)) * (dl2_dg_i - val * (dlse_dg_i + dnorm1_dg_i))
    dval_dg = (1.0 / (norm_1 * lse)) * (dl2_dg - val * (dlse_dg + dnorm1_dg))
    
    # dval/dh_j = sum_i dval/dg_i * dg_i/dh_j
    # g = convolve(h, h), so dg_i/dh_j = h[i-j] if 0<=i-j<N, else 0
    # Use FFT or direct
    grad_h = np.zeros(N)
    for j in range(N):
        for i in range(m):
            k = i - j
            if 0 <= k < N:
                grad_h[j] += dval_dg[i] * h[k]
    
    return -val, -grad_h

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
    # Plateau with smooth edges
    plateau = np.ones(N)
    edge = 8
    for i in range(edge):
        plateau[i] = 0.5 * (1 + np.cos(np.pi * i / edge))
        plateau[N-1-i] = 0.5 * (1 + np.cos(np.pi * i / edge))
    starts.append(plateau)
    # Bimodal with different widths
    w1, w2 = N/6, N/10
    p1 = np.exp(-0.5 * ((x - N * 0.3) / w1) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.7) / w2) ** 2)
    starts.append(p1 + 0.8 * p2)
    # Random smooth profiles
    for _ in range(30):
        freqs = RNG.randint(1, 10, size=8)
        phases = RNG.uniform(0, 2 * np.pi, size=8)
        amps = RNG.uniform(0.3, 1.2, size=8)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.05, prof))
    # Near-uniform with small noise
    for _ in range(10):
        noise = RNG.normal(0, 0.1, N)
        starts.append(np.maximum(0.1, 1.0 + noise))
        
    return starts

def _local_search(h0, beta, maxiter=300, ftol=1e-14, gtol=1e-12):
    """Single L-BFGS-B optimization with gradient."""
    bounds = [(0, None)] * N
    try:
        res = opt.minimize(
            lambda x: _smooth_value(x, beta),
            h0,
            jac=lambda x: _smooth_value_grad(x, beta)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        return res.x
    except Exception:
        return h0

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 25.0)  # 25s safety margin

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

    # Multi-stage beta annealing from each start
    # Stage 1: coarse (beta=10), Stage 2: medium (beta=50), Stage 3: fine (beta=200)
    betas = [10.0, 50.0, 200.0]
    
    for s in starts:
        if time.time() > deadline:
            break
        
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        current = x0.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
            current = _local_search(current, beta)
            track(current)

    # Polish phase: adaptive perturbations around best
    # Use multiple perturbation scales
    for scale in [0.05, 0.1, 0.2, 0.02]:
        for _ in range(15):
            if time.time() > deadline:
                break
            pert = best_h + RNG.normal(0.0, scale, N)
            pert = np.maximum(0.0, pert)
            
            # Fine-tune with high beta
            result = _local_search(pert, 300.0, maxiter=200, ftol=1e-15, gtol=1e-13)
            track(result)

    # Final ultra-fine polish on the best
    for _ in range(5):
        if time.time() > deadline:
            break
        pert = best_h + RNG.normal(0.0, 0.01, N)
        pert = np.maximum(0.0, pert)
        result = _local_search(pert, 500.0, maxiter=300, ftol=1e-16, gtol=1e-14)
        track(result)

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
