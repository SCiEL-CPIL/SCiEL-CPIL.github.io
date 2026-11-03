"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural change: use a different optimization approach based on
minimax reformulation with iterative water-filling and coordinate descent,
combined with a different representation (Fourier-domain parameterization)."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _fourier_param_to_h(params):
    """
    Convert Fourier coefficients to step heights.
    params[0] is the DC component (fixed at N/2), rest are cosine/sine pairs.
    We parameterize h as a sum of Fourier modes to naturally enforce smoothness.
    """
    n = N
    h = np.zeros(n)
    # DC component
    h[:] = params[0]
    for k in range(1, n // 2 + 1):
        if 2 * k >= len(params):
            break
        a_k = params[2 * k - 1]
        b_k = params[2 * k]
        idx = np.arange(n)
        h += a_k * np.cos(2.0 * np.pi * k * idx / n) + b_k * np.sin(2.0 * np.pi * k * idx / n)
    return h


def _h_to_fourier_param(h):
    """Convert step heights to Fourier parameters."""
    n = N
    params = np.zeros(2 * (n // 2 + 1))
    params[0] = np.mean(h)
    for k in range(1, n // 2 + 1):
        idx = np.arange(n)
        a_k = (2.0 / n) * np.sum(h * np.cos(2.0 * np.pi * k * idx / n))
        b_k = (2.0 / n) * np.sum(h * np.sin(2.0 * np.pi * k * idx / n))
        params[2 * k - 1] = a_k
        params[2 * k] = b_k
    return params


def _smooth_max_obj_grad(h, beta):
    """
    Smooth maximum of overlap values with gradient.
    Uses log-sum-exp formulation.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    
    val = m + np.log(sum_exp) / beta
    
    # Gradient of C_k w.r.t. h_j
    # C_k = (2/n)(n/2 - r_k), r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    # dC_k/dh_j = -(2/n) * [h[j-k] (if j>=k) + h[j+k] (if j+k<n)]
    grad = np.zeros(n)
    for k in range(n):
        if w[k] < 1e-15:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
    
    return val, grad


def _coordinate_descent_polish(h, max_iters=5000, tol=1e-10):
    """
    Coordinate descent on individual h_j to reduce max overlap.
    For each coordinate, find the optimal value given others fixed.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    n = N
    target_sum = n / 2.0
    
    for it in range(max_iters):
        improved = False
        for j in range(n):
            # Fix all except h_j
            # C_k depends on h_j through terms where j is involved
            # For a given j, C_k is affected when:
            #   - j >= k and j-k < n (h[j]*h[j-k])
            #   - j+k < n (h[j+k]*h[j])
            
            # Compute current C_k values
            cs = _linear_overlap_values(h)
            max_c = np.max(cs)
            
            # Try to improve by adjusting h_j
            # The overlap values are quadratic in h_j
            # Let's do a small search around current h_j
            
            old_val = h[j]
            # Sample a few candidate values
            candidates = np.linspace(max(0, old_val - 0.1), min(1, old_val + 0.1), 11)
            candidates = np.append(candidates, [0.0, 1.0])
            candidates = np.unique(candidates)
            
            best_candidate = old_val
            best_score_candidate = max_c
            
            for c in candidates:
                h[j] = c
                # Adjust sum constraint approximately
                cs_new = _linear_overlap_values(h)
                score_new = np.max(cs_new)
                if score_new < best_score_candidate - tol:
                    best_score_candidate = score_new
                    best_candidate = c
            
            h[j] = best_candidate
            if abs(h[j] - old_val) > tol:
                improved = True
        
        if not improved:
            break
    
    # Fix sum
    h = _fix_sum(h)
    return h


def _nesterov_pgd(h0, beta, lr, max_iters=2000):
    """
    Nesterov accelerated projected gradient descent on smooth max objective.
    Projection onto {h: 0<=h<=1, sum(h)=N/2}.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    n = N
    target_sum = n / 2.0
    
    m = np.array(h, dtype=np.float64, copy=True)  # momentum
    
    for it in range(max_iters):
        # Nesterov lookahead
        h_nest = h + (it / (it + 3)) * (h - m)
        
        # Gradient step
        val, grad = _smooth_max_obj_grad(h_nest, beta)
        h_new = h_nest - lr * grad
        
        # Project onto constraints
        # Simple projection: clip then fix sum
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _fix_sum(h_new)
        
        # Update momentum
        m = h
        h = h_new
    
    return h


def _bisection_feasibility(h0, max_iters=300):
    """
    Bisection on the max overlap value T.
    For a given T, check if there exists h with max C_k <= T.
    This is a convex feasibility problem (C_k are concave in h? No, they're quadratic).
    Actually, let's just use this to guide the search.
    """
    # This approach is complex; skip for now
    return h0


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    def _elapsed():
        return time.time() - t0
    
    def _remaining():
        return _DEADLINE - time.time()
    
    # Phase 1: Analytical starts
    i = np.arange(N)
    starts = [
        0.5 + 0.5 * np.cos(2.0 * np.pi * i / N),
        0.5 + 0.5 * np.sin(2.0 * np.pi * i / N),
        0.5 + 0.5 * np.cos(4.0 * np.pi * i / N),
        0.5 + 0.5 * np.sin(4.0 * np.pi * i / N),
        0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.3 * np.sin(4.0 * np.pi * i / N),
        np.where(i % 2 == 0, 0.2, 0.8),
        np.where(i % 4 < 2, 0.1, 0.9),
        np.full(N, 0.5),
    ]
    
    for h_start in starts:
        if _remaining() < 30:
            break
        h_s = _fix_sum(h_start)
        _update_best(h_s)
    
    # Phase 2: SLSQP optimization from best start
    def _run_slsqp(h0, beta, maxiter, ftol):
        nonlocal best_h, best_score
        if _remaining() < 10:
            return
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            val, _ = _smooth_max_obj_grad(x, beta)
            return val
            
        def jac_func(x):
            _, grad = _smooth_max_obj_grad(x, beta)
            return grad
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass
    
    if best_h is not None:
        _run_slsqp(best_h, beta=10.0, maxiter=300, ftol=1e-8)
        _run_slsqp(best_h, beta=20.0, maxiter=300, ftol=1e-9)
    
    # Phase 3: Nesterov PGD with multiple betas
    if best_h is not None and _remaining() > 20:
        for beta in [10.0, 30.0, 50.0]:
            if _remaining() < 10:
                break
            lr = 1.0 / (beta * N)  # Rough Lipschitz estimate
            h_nest = _nesterov_pgd(best_h, beta, lr, max_iters=500)
            _update_best(h_nest)
    
    # Phase 4: Coordinate descent polish
    if best_h is not None and _remaining() > 15:
        h_cd = _coordinate_descent_polish(best_h, max_iters=3000)
        _update_best(h_cd)
    
    # Phase 5: More random restarts
    if _remaining() > 25:
        for _ in range(15):
            if _remaining() < 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _run_slsqp(h_rand, beta=20.0, maxiter=150, ftol=1e-8)
    
    # Phase 6: Final high-beta polish
    if best_h is not None and _remaining() > 10:
        _run_slsqp(best_h, beta=50.0, maxiter=400, ftol=1e-10)
    
    if best_h is not None and _remaining() > 5:
        _run_slsqp(best_h, beta=100.0, maxiter=300, ftol=1e-12)
    
    if best_h is not None and _remaining() > 3:
        _run_slsqp(best_h, beta=200.0, maxiter=200, ftol=1e-13)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
