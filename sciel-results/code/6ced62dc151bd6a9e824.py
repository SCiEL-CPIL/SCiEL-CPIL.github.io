"""
Structural change: Replace local gradient-based optimization with a global
population-based differential evolution (DE/rand/1/bin) search.
This avoids getting trapped in local optima and explores the 50-dimensional
landscape more thoroughly. We use a smooth log-sum-exp surrogate for
norm_inf to ensure gradient-free optimizers can navigate the landscape
effectively, but the key is the global search strategy.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.optimize import differential_evolution

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

def _smooth_value(h, beta=100.0):
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
    if g_max <= 0:
        return 0.0
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
    deadline = start_time + (TIME_LIMIT - 20.0)  # 20s safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # Initial evaluation of diverse starting points
    starts = _generate_starts()
    for s in starts:
        track(s)

    # Phase 1: Global search with differential evolution
    # Use a smooth surrogate for DE to avoid non-differentiability issues
    # DE works well with bounded variables, so we constrain heights to [0, 5]
    # The optimal heights are unlikely to exceed 5 in magnitude
    
    bounds = [(0, 5)] * N
    
    # Use a moderate beta for the smooth surrogate during global search
    beta_global = 50.0
    
    # Run differential evolution with a population size and max iterations
    # that fit within our time budget
    try:
        # Use a callback to track the best solution
        def callback(xk, convergence):
            nonlocal best_val, best_h
            h = np.maximum(0.0, xk)
            v = compute_value(h)
            if v > best_val:
                best_val = v
                best_h = h.copy()
            # Check time
            if time.time() > deadline:
                raise KeyboardInterrupt()
        
        res = differential_evolution(
            lambda x: _smooth_value(x, beta_global),
            bounds,
            maxiter=300,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=RNG,
            callback=callback,
            polish=True  # Polish with L-BFGS-B at the end
        )
        if res.fun < 0:  # -value, so lower is better
            track(res.x)
    except KeyboardInterrupt:
        pass
    except Exception:
        pass

    # Phase 2: Local refinement around the best solution found
    if time.time() < deadline:
        # Use L-BFGS-B with high beta for sharpness
        bounds_local = [(0, None)] * N
        for _ in range(10):
            if time.time() > deadline:
                break
            pert = best_h + RNG.normal(0.0, 0.05, N)
            pert = np.maximum(0.0, pert)
            
            try:
                res = opt.minimize(
                    lambda x: _smooth_value(x, 200.0),
                    pert,
                    method='L-BFGS-B',
                    bounds=bounds_local,
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
