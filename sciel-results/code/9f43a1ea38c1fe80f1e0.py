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

def _corr_grad(h):
    """Gradient of max(c) w.r.t. h where c = correlate(h, 1-h, full).
    Only the argmax component matters. c[k] = sum_i h[i]*(1-h[i+k-N+1]) (shifted).
    d c[k]/d h[j] = (1-h[j+k-N+1]) - h[j-k+N-1]  (with out-of-range terms = 0).
    We return the gradient of max(c) = c[argmax] w.r.t. h.
    """
    c = np.correlate(h, 1.0 - h, mode="full")
    kmax = int(np.argmax(c))
    # c[k] = sum_{i} h[i] * (1-h[i - (k - (N-1))])  -- need careful indexing
    # numpy correlate(a, v, mode='full')[k] = sum_n a[n] * v[n - (len(v)-1-k)]
    # Here a=h, v=1-h. len(v)=N. So c[k] = sum_n h[n] * (1-h[n - (N-1-k)])
    # Let d = N-1-k. Then c[k] = sum_n h[n] * (1-h[n-d]) where h indexed 0..N-1, out-of-range=0.
    d = N - 1 - kmax
    # dc/dh[j] = (1 - h[j-d]) - h[j+d]  (with h out-of-range = 0)
    grad = np.zeros(N)
    for j in range(N):
        jd = j - d
        jp = j + d
        term1 = 0.0
        if 0 <= jd < N:
            term1 = 1.0 - h[jd]
        term2 = 0.0
        if 0 <= jp < N:
            term2 = h[jp]
        grad[j] = term1 - term2
    return grad

def _lbfgs_smooth(h0, beta=80.0, max_iter=5000, ftol=1e-14):
    """Direct L-BFGS-B on h with smooth objective and projected gradient."""
    from scipy.optimize import minimize
    h0 = project(h0).copy()
    def obj_and_grad(x):
        h = project(x)
        c = np.correlate(h, 1.0 - h, mode="full")
        c_max = np.max(c)
        c_shifted = c - c_max
        s = np.sum(np.exp(beta * c_shifted))
        val = (c_max + (1.0/beta) * np.log(s)) / N * 2.0
        # Gradient via subgradient of max + smooth part
        # d/dh [soft_max(c)] ≈ (1/beta) * sum_k exp(beta*(c[k]-cmax)) * dc[k]/dh
        # For simplicity, use gradient of the hard max at argmax (subgradient)
        # weighted by softmax weights.
        w = np.exp(beta * c_shifted)
        w /= w.sum()
        # gradient = sum_k w[k] * dc[k]/dh
        # dc[k]/dh[j] = (1-h[j-(N-1-k)]) - h[j+(N-1-k)] with out-of-range=0
        # Vectorized: for each k, d = N-1-k
        # We'll compute via loops over N (100) and 2N-1 (199) -- 19900 ops, fine.
        grad = np.zeros(N)
        for k in range(2*N - 1):
            if w[k] < 1e-15:
                continue
            d = N - 1 - k
            for j in range(N):
                jd = j - d
                jp = j + d
                t1 = (1.0 - h[jd]) if 0 <= jd < N else 0.0
                t2 = h[jp] if 0 <= jp < N else 0.0
                grad[j] += w[k] * (t1 - t2)
        grad /= N * 2.0  # chain rule from /N*2
        return val, grad
    try:
        res = minimize(obj_and_grad, h0, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 1.0)]*N,
                       options={"maxiter": max_iter, "ftol": ftol, "gtol": 1e-10, "maxls": 50})
        return project(res.x)
    except Exception:
        return h0

def _make_block_pattern(block_len, rng):
    """Create a pattern with alternating high/low blocks to reduce overlap."""
    h = np.zeros(N)
    # Fill with blocks of block_len: high (0.8) then low (0.2) etc., adjusted for sum=N/2
    n_blocks = N // block_len
    for b in range(n_blocks):
        start = b * block_len
        if b % 2 == 0:
            h[start:start+block_len] = 0.7
        else:
            h[start:start+block_len] = 0.3
    # Fill remainder
    if N % block_len != 0:
        rem = N % block_len
        h[N-rem:] = 0.5
    # Add small random perturbation
    h += rng.normal(0, 0.02, N)
    return project(h)

def main_search():
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0

    rng = np.random.default_rng(0)

    # Phase 1: Fourier-parametrized search (global structure)
    n_harm = 8
    dim = 1 + 3*n_harm
    for trial in range(40):
        if remaining() < 15.0:
            break
        theta0 = np.zeros(dim)
        theta0[0] = rng.uniform(-0.05, 0.05)
        for k in range(n_harm):
            theta0[1+3*k]   = rng.integers(1, N//2+1)
            theta0[2+3*k]   = rng.uniform(0, 2*np.pi)
            theta0[3+3*k]   = rng.uniform(-0.4, 0.4)
        theta_opt = optimize_fourier(theta0, max_iter=2000)
        h = project(fourier_h(theta_opt))
        v = objective(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
        if remaining() < 25.0:
            break

    # Phase 2: Block-pattern initializations + direct L-BFGS polish
    block_lengths = [2, 3, 4, 5, 7, 10, 20, 25, 50]
    for bl in block_lengths:
        if remaining() < 15.0:
            break
        for attempt in range(3):
            if remaining() < 10.0:
                break
            h_init = _make_block_pattern(bl, rng)
            h_opt = _lbfgs_smooth(h_init, beta=80.0, max_iter=3000)
            v = objective(h_opt)
            if v < best_val:
                best_val = v
                best_h = h_opt.copy()

    # Phase 3: Random restarts + direct L-BFGS
    for trial in range(30):
        if remaining() < 10.0:
            break
        h_init = np.clip(best_h + rng.normal(0, 0.15, N), 0, 1)
        h_init = project(h_init)
        h_opt = _lbfgs_smooth(h_init, beta=60.0, max_iter=2000)
        v = objective(h_opt)
        if v < best_val:
            best_val = v
            best_h = h_opt.copy()

    # Phase 4: Final fine polish on best
    if remaining() > 5.0:
        h_opt = _lbfgs_smooth(best_h, beta=200.0, max_iter=5000, ftol=1e-15)
        v = objective(h_opt)
        if v < best_val:
            best_val = v
            best_h = h_opt.copy()

    # Phase 5: Tiny random walk polish
    step_size = 0.01
    for i in range(500):
        if remaining() < 3.0:
            break
        h = best_h + rng.normal(0, step_size, N)
        h = project(h)
        v = objective(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
            step_size = max(0.001, step_size * 1.1)
        else:
            step_size = max(0.001, step_size * 0.95)

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
