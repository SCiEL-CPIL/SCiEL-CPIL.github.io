"""
Improved L-BFGS-B optimization with DCT-basis parameterization.
Key ideas:
1. Represent h as nonnegative combination of DCT basis functions (low-rank smoothness)
2. Use smooth log-sum-exp surrogate for norm_inf
3. Correct analytic gradient via the chain rule through convolution structure
4. Adaptive beta schedule with focused optimization phases
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

# DCT basis matrix (type-II DCT, orthonormal)
def _make_dct_basis(n_basis):
    """Create orthonormal DCT-II basis matrix of shape (N, n_basis)."""
    k = np.arange(N)
    basis = np.zeros((N, n_basis))
    for j in range(n_basis):
        if j == 0:
            basis[:, j] = 1.0 / np.sqrt(N)
        else:
            basis[:, j] = np.sqrt(2.0 / N) * np.cos(np.pi * j * (k + 0.5) / N)
    return basis

DCT_BASIS = _make_dct_basis(8)  # 8 DCT modes for smoothness

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

def _smooth_value_and_grad(h, beta=50.0):
    """
    Smooth surrogate with correct analytic gradient.
    Returns (negative_value, gradient_of_negative_value).
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    
    if norm_1 <= 0:
        return 0.0, np.zeros(N)
    
    # Smooth max via log-sum-exp
    g_max = np.max(g)
    exp_terms = np.exp(beta * (g - g_max))
    sum_exp = np.sum(exp_terms)
    if sum_exp <= 0:
        return 0.0, np.zeros(N)
    lse = g_max + (1.0 / beta) * np.log(sum_exp)
    
    if lse <= 0:
        return 0.0, np.zeros(N)
    
    val = l2 / (norm_1 * lse)
    
    # Compute gradients using the structure:
    # l2 = (W/3) * [2*sum_j g[j]^2 + sum_j g[j]*g[j+1]]
    # dl2/dg[j] = (W/3) * [4*g[j] + g[j-1] + g[j+1]] (with boundary handling)
    
    dl2_dg = np.zeros(m)
    dl2_dg += 4.0 * g
    if m > 1:
        dl2_dg[1:] += g[:-1]
        dl2_dg[:-1] += g[1:]
    dl2_dg *= (W / 3.0)
    
    dnorm1_dg = np.ones(m) / (m + 1)
    dlse_dg = beta * exp_terms / sum_exp
    
    # Gradient of a linear combination of g w.r.t. h:
    # If f = sum_j c[j] * g[j], then df/dh[i] = sum_j c[j] * dg[j]/dh[i]
    # g[j] = sum_k h[k] * h[k+j] for valid indices
    # dg[j]/dh[i] = h[i+j] (if i+j < N) + h[i-j] (if i-j >= 0)
    
    def _grad_of_g_linear(c):
        """Compute gradient of sum_j c[j]*g[j] w.r.t. h."""
        grad = np.zeros(N)
        for i in range(N):
            s = 0.0
            for j in range(m):
                if i + j < N:
                    s += c[j] * h[i + j]
                if i - j >= 0:
                    s += c[j] * h[i - j]
            grad[i] = s
        return grad
    
    # Vectorized version using convolution
    def _grad_of_g_linear_fast(c):
        """Faster gradient computation using convolution."""
        # Pad c to handle the convolution properly
        # We want: grad[i] = sum_j c[j] * (h[i+j] + h[i-j]) (with bounds)
        
        # Term 1: sum_j c[j] * h[i+j] for i+j < N
        # This is a correlation-like operation
        # Let's compute it as: for each i, sum over j where j < N-i
        # Equivalent to: (c * h_flipped)[something]
        
        # Actually, let's use the fact that:
        # sum_j c[j] * h[i+j] = sum_k c[k] * h[i+k] where k=j
        # This is a cross-correlation of c and h
        
        # More efficiently:
        # grad1[i] = sum_{j=0}^{N-1-i} c[j] * h[i+j]
        # grad2[i] = sum_{j=0}^{i} c[j] * h[i-j]
        
        # grad1 can be computed as: for each offset j, add c[j]*h[j:] to grad1[:N-j]
        # grad2 can be computed as: for each offset j, add c[j]*h[:N-j] to grad2[j:]
        
        grad = np.zeros(N)
        
        # grad1: sum_j c[j] * h[i+j] for i in [0, N-1-j]
        for j in range(min(m, N)):
            if c[j] != 0:
                grad[:N-j] += c[j] * h[j:]
        
        # grad2: sum_j c[j] * h[i-j] for i in [j, N-1]
        for j in range(min(m, N)):
            if c[j] != 0:
                grad[j:] += c[j] * h[:N-j]
        
        return grad
    
    dl2_dh = _grad_of_g_linear_fast(dl2_dg)
    dnorm1_dh = _grad_of_g_linear_fast(dnorm1_dg)
    dlse_dh = _grad_of_g_linear_fast(dlse_dg)
    
    # dval/dh[i] = (1/(norm_1*lse)) * (dl2_dh[i] - val * (dlse_dh[i]/lse + dnorm1_dh[i]/norm_1))
    dval_dh = (1.0 / (norm_1 * lse)) * (dl2_dh - val * (dlse_dh / lse + dnorm1_dh / norm_1))
    
    # We're minimizing -val, so gradient of objective = -dval_dh
    grad_obj = -dval_dh
    
    return -val, grad_obj

def _project_to_dct(coeffs):
    """Convert DCT coefficients to heights (ensure nonnegativity)."""
    h = DCT_BASIS @ coeffs
    return np.maximum(0.0, h)

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
    # Raised cosine
    starts.append(0.5 * (1 + np.cos(np.pi * (2 * x / (N - 1) - 1))))
    # Plateau
    plateau = np.ones(N)
    plateau[:5] = 0.5
    plateau[-5:] = 0.5
    starts.append(plateau)
    # Random smooth profiles
    for _ in range(20):
        n_freqs = RNG.randint(2, 8)
        freqs = RNG.randint(1, 8, size=n_freqs)
        phases = RNG.uniform(0, 2 * np.pi, size=n_freqs)
        amps = RNG.uniform(0.3, 1.0, size=n_freqs)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        prof = np.maximum(0.05, prof)
        starts.append(prof)
    
    # Edge-concentrated
    edge = np.exp(-np.abs(x - N/2) / (N/6))
    starts.append(edge)
    
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

    starts = _generate_starts()
    
    # Initial evaluation
    for s in starts:
        track(s)

    bounds = [(0, None)] * N
    
    # Phase 1: Broad exploration with low beta from diverse starts
    beta_explore = 15.0
    n_starts = min(12, len(starts))
    for idx in range(n_starts):
        if time.time() > deadline:
            break
        
        x0 = np.maximum(0.0, np.asarray(starts[idx], dtype=float))
        
        try:
            res = opt.minimize(
                lambda x: _smooth_value_and_grad(x, beta_explore)[0],
                x0,
                jac=lambda x: _smooth_value_and_grad(x, beta_explore)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 40}
            )
            track(res.x)
        except Exception:
            pass
    
    # Phase 2: Refinement with medium beta from best few
    beta_medium = 60.0
    for _ in range(5):
        if time.time() > deadline:
            break
        
        # Perturb best and refine
        x0 = best_h + RNG.normal(0.0, 0.02, N)
        x0 = np.maximum(0.0, x0)
        
        try:
            res = opt.minimize(
                lambda x: _smooth_value_and_grad(x, beta_medium)[0],
                x0,
                jac=lambda x: _smooth_value_and_grad(x, beta_medium)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            track(res.x)
        except Exception:
            pass
    
    # Phase 3: High-beta polish from best
    if time.time() < deadline - 15:
        x0 = best_h.copy()
        try:
            res = opt.minimize(
                lambda x: _smooth_value_and_grad(x, 200.0)[0],
                x0,
                jac=lambda x: _smooth_value_and_grad(x, 200.0)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 80}
            )
            track(res.x)
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish on exact value
    if time.time() < deadline - 5:
        x0 = best_h.copy()
        try:
            res = opt.minimize(
                lambda x: -compute_value(x),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'fatol': 1e-12, 'xatol': 1e-10, 'adaptive': True}
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
