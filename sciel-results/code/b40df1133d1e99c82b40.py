"""Bootstrap portfolio: model-and-optimize for Erdos minimum overlap."""
import numpy as np
import time

N = 100
DEADLINE = 140.0  # safety margin below 150s kill
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    # h: (N,) in [0,1], sum ~ N/2
    # value = max(correlate(h, 1-h, full)) / N * 2
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def objective_smooth(h, beta=100.0):
    """Smooth surrogate using log-sum-exp approximation of max."""
    c = np.correlate(h, 1.0 - h, mode="full")
    # c is the correlation vector. We want to minimize max(c).
    # max(c) ~ (1/beta) * log(sum(exp(beta*c)))
    # To avoid overflow, shift by max(c)
    c_max = np.max(c)
    if beta * (c_max - c_max) > 700: # just in case, though 0 is safe
         pass
    # log-sum-exp trick: log(sum(exp(beta*c))) = beta*c_max + log(sum(exp(beta*(c-c_max))))
    # max(c) approx = c_max + (1/beta)*log(sum(exp(beta*(c-c_max))))
    # Note: sum(exp(0)) = len(c). So log(len(c))/beta is the offset.
    # Actually, we just want to minimize max(c).
    # Let's use a simpler soft-max:
    # f(x) = (1/beta) * log(sum(exp(beta*x)))
    # This is close to max(x).
    # We want to minimize this.
    c_shifted = c - c_max
    # exp(beta * c_shifted) is <= 1.
    s = np.sum(np.exp(beta * c_shifted))
    # soft_max = c_max + (1/beta) * log(s)
    return (c_max + (1.0/beta) * np.log(s)) / N * 2.0

def project(h):
    """Project to [0,1]^N and fix sum to N/2 (within tolerance)."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    target = N / 2.0
    if s > 0:
        # scale, then re-clip; do a few iterations
        for _ in range(100):
            h = np.clip(h, 0.0, 1.0)
            s = h.sum()
            if abs(s - target) < 1e-12:
                break
            h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h

def fourier_h(theta):
    """theta: [a0, (f1,p1,a1), (f2,p2,a2), ...] -> h vector."""
    h = np.full(N, 0.5)
    # a0 already in 0.5 base; interpret theta[0] as small dc offset
    h = h + theta[0]
    idx = 1
    i = np.arange(N)
    while idx + 3 <= len(theta):
        f, p, a = theta[idx], theta[idx+1], theta[idx+2]
        h = h + a * np.sin(2*np.pi*f*i/N + p)
        idx += 3
    return h

def optimize_fourier(theta0, max_iter=2000):
    from scipy.optimize import minimize
    def obj(x):
        h = project(fourier_h(x))
        return objective_smooth(h, beta=50.0)
    try:
        res = minimize(obj, theta0, method="L-BFGS-B",
                       options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50})
        return res.x
    except Exception:
        return theta0

def main_search():
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0

    rng = np.random.default_rng(0)

    # Try a few parameterized families with random init
    # Increase harmonics to 10 for more flexibility
    n_harm = 10
    dim = 1 + 3*n_harm
    for trial in range(80):
        if remaining() < 10.0:
            break
        theta0 = np.zeros(dim)
        theta0[0] = rng.uniform(-0.05, 0.05)
        for k in range(n_harm):
            # choose integer-ish frequencies to create structure
            # Higher frequencies might be useful for finer structure
            theta0[1+3*k]   = rng.integers(1, N//2+1)          # f
            theta0[2+3*k]   = rng.uniform(0, 2*np.pi)          # p
            theta0[3+3*k]   = rng.uniform(-0.4, 0.4)           # a
        theta_opt = optimize_fourier(theta0, max_iter=3000)
        h = project(fourier_h(theta_opt))
        v = objective(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
        if remaining() < 20.0:
            break

    # Local polish on best_h with small random perturbations (direct)
    # Increase iterations and use adaptive step size
    step_size = 0.05
    for i in range(2000):
        if remaining() < 5.0:
            break
        # Decay step size
        step_size = 0.05 * (1.0 - i/4000.0)
        if step_size < 0.001:
            step_size = 0.001
        h = best_h + rng.normal(0, step_size, N)
        h = project(h)
        v = objective(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
        else:
            # If we haven't improved in a while, we might be stuck, but L-BFGS-B already did heavy lifting.
            # Just continue with smaller steps.
            pass

    return best_h, best_val

# EVOLVE-BLOCK-START
def construct():
    global _CACHED
    try:
        if '_CACHED' in globals():
            return _CACHED
        h, _ = main_search()
        h = project(h)
        # enforce sum within tolerance
        s = h.sum()
        h = h + (N/2.0 - s)/N
        h = np.clip(h, 0.0, 1.0)
        assert np.all(np.isfinite(h))
        _CACHED = h
        return h
    except Exception:
        return np.full(N, 0.5)
# EVOLVE-BLOCK-END
