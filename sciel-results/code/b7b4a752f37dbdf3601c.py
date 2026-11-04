"""Rewritten: ADMM spectral-peak minimization with multi-sine parameterization.

Key structural change: instead of optimizing all 400 heights directly, we
parameterize x as a sum of a small number of sinusoids (10-16 params),
reducing the effective dimensionality. We then minimize the spectral peak
(max |autoconv| / S^2) using ADMM with a spectral-domain dual variable.

The ADMM formulation:
  min_x  (1/2)||F(x)||_inf  s.t.  sum(x) = 1
where F(x) maps x to its autoconvolution (excluding zero-lag).

We use a soft-max surrogate for the max and solve via alternating:
  x-update: minimize (1/2)||F(x)||^2_smooth + (rho/2)||x - z + u||^2
  z-update: prox of the max-norm ball
  u-update: dual ascent

Combined with multi-sine parameterization, this enables global optimization
of the low-dimensional parameter vector via differential evolution or
L-BFGS-B on the smooth surrogate.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400
FFT_SIZE = 1024  # power of 2 >= 2*N-1


def _autoconv_lin(x):
    """Full linear autoconvolution of x (length 2n-1)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c)))


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _multi_sine_x(params, N):
    """Reconstruct x from multi-sine parameters.
    
    params = [a0, f1, p1, a1, f2, p2, a2, ..., fK, pK, aK]
    where a0 is DC offset, (fj, pj, aj) are frequency, phase, amplitude.
    """
    x = np.zeros(N)
    idx = np.arange(N)
    if len(params) < 1:
        return x
    x += params[0]
    # Triplets: (freq, phase, amplitude)
    for j in range(1, len(params), 3):
        if j + 2 >= len(params):
            break
        f = params[j]
        p = params[j + 1]
        a = params[j + 2]
        x += a * np.sin(2 * np.pi * f * idx / N + p)
    return x


def _objective_from_params(params, N, beta=20.0):
    """Compute smooth metric from multi-sine params."""
    x = _multi_sine_x(params, N)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e6
    c = _autoconv_lin(x)
    abs_c = np.abs(c[1:])  # off-diagonal
    m = np.max(abs_c)
    if m < 1e-15:
        return 1e-15 / (s * s)
    # Soft-max surrogate
    sm = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return sm / (s * s)


def _try_multi_sine_optimization(deadline, rng, n_sines=6):
    """Optimize multi-sine parameters via L-BFGS-B and DE."""
    best_val = float('inf')
    best_params = None
    
    # Parameter bounds: DC in [-1,1], freq in [0.5, 50], phase in [0, 2pi], amp in [0, 2]
    # Total params: 1 + 3*n_sines
    n_params = 1 + 3 * n_sines
    
    bounds = [(-1.0, 1.0)]  # DC
    for _ in range(n_sines):
        bounds.append((0.5, 50.0))   # frequency
        bounds.append((0.0, 2.0 * np.pi))  # phase
        bounds.append((0.0, 2.0))    # amplitude
    
    # Try multiple random initializations with L-BFGS-B
    for trial in range(15):
        if time.time() > deadline:
            break
        p0 = np.array([rng.uniform(-0.5, 0.5)] + 
                      [rng.uniform(0.5, 50.0), rng.uniform(0, 2*np.pi), rng.uniform(0.1, 1.5)] * n_sines)
        
        try:
            res = minimize(_objective_from_params, p0, args=(N, 20.0),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Try differential evolution for global search
    if time.time() < deadline - 10 and best_val > 0.01:
        try:
            de_res = differential_evolution(_objective_from_params, bounds,
                                           args=(N, 20.0),
                                           maxiter=50, popsize=15,
                                           tol=1e-8, seed=int(rng.integers(0, 1000000)),
                                           polish=True)
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    if best_params is not None:
        x = _multi_sine_x(best_params, N)
        # Normalize to sum=1
        s = np.sum(x)
        if abs(s) > 1e-12:
            x = x / s
        return x, _metric(x)
    
    return None, float('inf')


def _admm_refine(x0, deadline, rng, iters=5000, rho=10.0, beta=15.0):
    """ADMM refinement of continuous heights.
    
    min_x  (1/beta) log(sum exp(beta|c_k|))  s.t. sum(x)=1
    
    ADMM split: min f(x) + g(z) s.t. x=z
    f(x) = smooth max of |autoconv(x)|
    g(z) = indicator{sum(z)=1}
    """
    n = N
    x = np.array(x0, dtype=float).copy()
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s
    z = x.copy()
    u = np.zeros(n)
    
    best = _metric(x)
    bestx = x.copy()
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        # x-update: minimize f(x) + (rho/2)||x - z + u||^2
        # Use a few gradient steps on the smooth objective + quadratic term
        for inner in range(3):
            c = _autoconv_lin(x)
            abs_c = np.abs(c[1:])
            m = np.max(abs_c)
            if m < 1e-15:
                break
            weights = np.exp(beta * (abs_c - m))
            weights /= weights.sum()
            
            # Gradient of soft-max w.r.t. x
            g = np.zeros(n)
            for k in range(1, n):
                wk = weights[k - 1]
                if wk < 1e-8:
                    continue
                sign_ck = np.sign(c[k])
                g[:n - k] += wk * sign_ck * x[k:]
                g[k:] += wk * sign_ck * x[:n - k]
            
            # Add quadratic term from ADMM
            g += rho * (x - z + u)
            
            # Gradient step
            x = x - 0.01 * g / (1.0 + rho * 0.01)
        
        # z-update: project onto sum(z)=1
        z = x + u
        z = z - (np.sum(z) - 1.0) / n
        
        # u-update
        u = u + x - z
        
        # Track best
        v = _metric(x)
        if v < best:
            best = v
            bestx = x.copy()
    
    return bestx, best


def _try_gradient_descent_smooth(x0, deadline, rng, iters=20000, beta=25.0):
    """Direct gradient descent on smooth surrogate with sum=1 constraint."""
    n = N
    x = np.array(x0, dtype=float).copy()
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s
    
    best = _metric(x)
    bestx = x.copy()
    lr = 1e-4
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        m = np.max(abs_c)
        if m < 1e-15:
            break
        weights = np.exp(beta * (abs_c - m))
        weights /= weights.sum()
        
        g = np.zeros(n)
        for k in range(1, n):
            wk = weights[k - 1]
            if wk < 1e-8:
                continue
            sign_ck = np.sign(c[k])
            g[:n - k] += wk * sign_ck * x[k:]
            g[k:] += wk * sign_ck * x[:n - k]
        
        # Project gradient to sum=0 subspace
        g = g - np.mean(g)
        
        x = x - lr * g
        x = x - (np.sum(x) - 1.0) / n
        x = np.clip(x, -2.0, 2.0)
        x = x - (np.sum(x) - 1.0) / n
        
        v = _metric(x)
        if v < best:
            best = v
            bestx = x.copy()
            lr = min(lr * 1.02, 1e-3)
        else:
            lr *= 0.999
    
    return bestx, best


def _try_local_search(x, deadline, rng, max_rounds=30000):
    """Local search: flip a few signs and accept improvements."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        k = int(rng.integers(1, min(4, N // 4) + 1))
        idx = rng.choice(N, size=k, replace=False)
        xf = best_x.copy()
        xf[idx] *= -1.0
        v = _metric(xf)
        if v < best_val:
            best_val = v
            best_x = xf
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 3000:
                break
    return best_x, best_val


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = np.random.default_rng(2024)

    best_x = np.ones(N)
    best_val = _metric(best_x)

    def consider(x, v):
        nonlocal best_x, best_val
        if v < best_val:
            best_val = v
            best_x = x.copy()

    # Phase 1: Multi-sine parameterization optimization (different dimensionality)
    t_phase1 = deadline - 60
    for n_sines in [4, 6, 8]:
        if time.time() > t_phase1:
            break
        x, v = _try_multi_sine_optimization(t_phase1, rng, n_sines)
        if x is not None:
            consider(x, v)

    # Phase 2: ADMM refinement from best multi-sine solution
    if time.time() < deadline - 40:
        x, v = _admm_refine(best_x, deadline - 40, rng, iters=3000, rho=5.0, beta=15.0)
        consider(x, v)

    # Phase 3: Smooth gradient descent refinement
    if time.time() < deadline - 25:
        x, v = _try_gradient_descent_smooth(best_x, deadline - 25, rng, iters=30000, beta=25.0)
        consider(x, v)

    # Phase 4: Another ADMM pass with different parameters
    if time.time() < deadline - 15:
        x, v = _admm_refine(best_x, deadline - 15, rng, iters=2000, rho=20.0, beta=10.0)
        consider(x, v)

    # Phase 5: Local search on signs of best continuous solution
    if time.time() < deadline - 5:
        x_sign = np.where(best_x >= 0, 1.0, -1.0)
        x, v = _try_local_search(x_sign, deadline - 5, rng, 20000)
        consider(x, v)

    # Phase 6: Final polish with very small steps
    if time.time() < deadline - 2:
        x, v = _try_gradient_descent_smooth(best_x, deadline - 2, rng, iters=10000, beta=40.0)
        consider(x, v)

    return best_x
