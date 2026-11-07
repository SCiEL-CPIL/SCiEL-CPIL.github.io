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
    deadline = start_time + (TIME_LIMIT - 20.0) # 20s safety margin

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
    # We use L-BFGS-B with bounds [0, inf)
    bounds = [(0, None)] * N
    
    # We try multiple betas (smoothness parameters) and starts
    betas = [20.0, 50.0, 100.0]
    
    for s in starts:
        if time.time() > deadline:
            break
        
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            try:
                res = opt.minimize(
                    lambda x: _smooth_value(x, beta),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < 0: # -value, so lower is better
                    track(res.x)
            except Exception:
                pass

    # Polish phase: Perturb best and re-optimize with high beta
    for _ in range(30):
        if time.time() > deadline:
            break
        pert = best_h + RNG.normal(0.0, 0.1, N)
        pert = np.maximum(0.0, pert)
        
        # Use high beta for sharpness
        try:
            res = opt.minimize(
                lambda x: _smooth_value(x, 200.0),
                pert,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
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
