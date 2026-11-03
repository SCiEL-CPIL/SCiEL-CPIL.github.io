"""
Second autocorrelation inequality: maximize ratio of squared L2 norm to L1 norm times sup norm of autoconvolution.

Strategy:
1. Normalize L1=1 (ratio is scale-invariant)
2. Formulate as: minimize M subject to sum(h)=1, h>=0, M >= (h*h)_k for all k
3. Use SLSQP with the auxiliary variable M (smooth formulation)
4. Add spectral refinement using eigendecomposition of autoconvolution Toeplitz matrix
5. Multi-start from diverse initializations (sparse peaks, uniform, structured)
6. Soft-max parameterization to smooth the max operator and allow gradient-based exploration
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Interval width for [-1/4, 1/4] divided into N steps

def _autoconv(h):
    """Linear autoconvolution of h with itself."""
    return np.convolve(h, h, mode='full')

def _score_raw(h):
    """Raw score: L2^2 / (delta * L1 * max(autoconv))."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    c = _autoconv(h)
    max_c = np.max(c)
    if max_c <= 0:
        return 0.0
    return l2_sq / (DELTA * l1 * max_c)

def _solve_slsqp(h_init, maxiter=500, ftol=1e-10):
    """
    Solve: minimize M subject to:
      - sum(h) = 1
      - M - (h*h)_k >= 0 for all k
      - h >= 0, M >= 0
    Returns optimized h (normalized to L1=1) or None on failure.
    """
    h_init = np.asarray(h_init, dtype=np.float64)
    s = np.sum(h_init)
    if s <= 0:
        return None
    h_init = h_init / s
    
    c_init = _autoconv(h_init)
    M_init = np.max(c_init) + 1e-6
    
    x0 = np.concatenate([h_init, [M_init]])
    
    def obj(x):
        return x[N]  # Minimize M
    
    def obj_grad(x):
        g = np.zeros(N + 1)
        g[N] = 1.0
        return g
    
    def cons_eq(x):
        return np.sum(x[:N]) - 1.0
    
    def cons_eq_grad(x):
        g = np.zeros(N + 1)
        g[:N] = 1.0
        return g
    
    def cons_ineq(x):
        h = x[:N]
        M = x[N]
        c = _autoconv(h)
        return M - c  # M - c_k >= 0
    
    def cons_ineq_grad(x):
        h = x[:N]
        grad = np.zeros((2*N - 1, N + 1))
        for k in range(2*N - 1):
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    grad[k, i] = -2.0 * h[j]
            grad[k, N] = 1.0
        return grad
    
    bounds = [(0.0, None)] * N + [(0.0, None)]
    
    constraints = [
        {'type': 'eq', 'fun': cons_eq, 'jac': cons_eq_grad},
        {'type': 'ineq', 'fun': cons_ineq, 'jac': cons_ineq_grad}
    ]
    
    try:
        res = minimize(
            obj, x0, method='SLSQP', jac=obj_grad,
            bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        h_res = np.maximum(0, res.x[:N])
        s_res = np.sum(h_res)
        if s_res > 1e-12:
            h_res = h_res / s_res
            return h_res
        return None
    except Exception:
        return None

def _soft_max_refine(h, iterations=100, beta_start=10.0, beta_end=100.0):
    """
    Refine h using a soft-max parameterization to smooth the max operator.
    We parameterize h_i = exp(z_i / beta) / sum(exp(z_j / beta)) and optimize z.
    The objective is to maximize L2^2 / max(autoconv) with L1=1.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    s = np.sum(h)
    if s <= 0:
        return h
    h = h / s
    
    # Initialize z from h: h_i = exp(z_i/beta) / sum(exp(z_j/beta))
    # => z_i = beta * log(h_i) (approximately, ignoring normalization)
    # We'll use a log-sum-exp stable formulation
    
    def soft_max_obj(z, beta):
        """Minimize -log(ratio) = -log(L2^2) + log(max(autoconv)) with soft max."""
        # h = softmax(z / beta)
        z_scaled = z / beta
        # Log-sum-exp for stability
        z_max = np.max(z_scaled)
        exp_z = np.exp(z_scaled - z_max)
        sum_exp = np.sum(exp_z)
        h = exp_z / sum_exp  # L1 = 1
        
        l2_sq = np.sum(h**2)
        if l2_sq <= 0:
            return 1e10
        
        c = _autoconv(h)
        max_c = np.max(c)
        if max_c <= 0:
            return 1e10
        
        # We want to minimize log(max_c) - log(l2_sq)
        # = log(max_c / l2_sq)
        return np.log(max_c) - np.log(l2_sq)
    
    def soft_max_grad(z, beta):
        """Gradient of soft_max_obj wrt z."""
        z_scaled = z / beta
        z_max = np.max(z_scaled)
        exp_z = np.exp(z_scaled - z_max)
        sum_exp = np.sum(exp_z)
        h = exp_z / sum_exp  # L1 = 1
        
        l2_sq = np.sum(h**2)
        c = _autoconv(h)
        max_c = np.max(c)
        
        # Gradient of log(max_c) wrt h:
        # d(max_c)/dh_i = sum over active k of d(c_k)/dh_i / max_c
        # For simplicity, use the subgradient at the max index
        active_k = np.argmax(c)
        # d(c_k)/dh_i = 2 * h[k-i] if valid
        grad_max_c = np.zeros(N)
        for i in range(N):
            j = active_k - i
            if 0 <= j < N:
                grad_max_c[i] = 2.0 * h[j]
        
        # Gradient of log(max_c) = grad_max_c / max_c
        grad_log_max = grad_max_c / max_c
        
        # Gradient of log(l2_sq) = 2h / l2_sq
        grad_log_l2 = 2.0 * h / l2_sq
        
        # Gradient of obj = grad_log_max - grad_log_l2
        grad_h = grad_log_max - grad_log_l2
        
        # Chain rule: d/dz = d/dh * dh/dz
        # h_i = exp(z_i/beta) / sum(exp(z_j/beta))
        # dh_i/dz_j = (1/beta) * h_i * (delta_ij - h_j)
        # So grad_z_j = (1/beta) * h_j * (grad_h_j - sum_i h_i * grad_h_i)
        
        mean_grad = np.sum(h * grad_h)
        grad_z = (1.0 / beta) * h * (grad_h - mean_grad)
        
        return grad_z
    
    # Start with z from h
    h_safe = np.maximum(h, 1e-10)
    z0 = beta_start * np.log(h_safe)
    z0 -= np.mean(z0)  # Center for numerical stability
    
    current_z = z0.copy()
    current_beta = beta_start
    
    for it in range(iterations):
        # Animate beta from beta_start to beta_end
        progress = it / max(1, iterations - 1)
        current_beta = beta_start + (beta_end - beta_start) * progress
        
        try:
            res = minimize(
                soft_max_obj, current_z, method='BFGS',
                jac=lambda z: soft_max_grad(z, current_beta),
                options={'maxiter': 50, 'gtol': 1e-6, 'disp': False}
            )
            current_z = res.x
        except Exception:
            pass
        
        # Extract h
        z_scaled = current_z / current_beta
        z_max = np.max(z_scaled)
        exp_z = np.exp(z_scaled - z_max)
        sum_exp = np.sum(exp_z)
        h_new = exp_z / sum_exp
        
        # Check improvement
        sc_old = _score_raw(h)
        sc_new = _score_raw(h_new)
        if sc_new > sc_old * (1 - 1e-8):
            h = h_new
    
    return h

def _spectral_refine(h, iterations=200):
    """
    Refine h by finding the direction in the null space of the 
    active autoconvolution constraints that increases L2^2.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    s = np.sum(h)
    if s <= 0:
        return h
    h = h / s
    
    for _ in range(iterations):
        c = _autoconv(h)
        max_c = np.max(c)
        
        active = np.where(c >= max_c - 1e-8)[0]
        
        A = np.zeros((len(active), N))
        for idx, k in enumerate(active):
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    A[idx, i] = 2.0 * h[j]
        
        grad = 2.0 * h
        
        try:
            U, S_vals, Vt = np.linalg.svd(A, full_matrices=True)
            rank = np.sum(S_vals > 1e-10)
            if rank < N:
                null_space = Vt[rank:].T
                ones = np.ones(N)
                null_space_centered = null_space - null_space.mean(axis=0)
                
                proj = null_space_centered.T @ grad
                norm_proj = np.linalg.norm(proj)
                if norm_proj > 1e-12:
                    c_vec = proj / norm_proj
                    d = null_space_centered @ c_vec
                    d_norm = np.linalg.norm(d)
                    if d_norm > 1e-12:
                        d = d / d_norm
                        best_alpha = 0
                        best_l2 = np.sum(h**2)
                        for alpha in np.linspace(0, 0.5, 20):
                            h_try = h + alpha * d
                            h_try = np.maximum(0, h_try)
                            s_try = np.sum(h_try)
                            if s_try > 0:
                                h_try = h_try / s_try
                                l2_try = np.sum(h_try**2)
                                c_try = _autoconv(h_try)
                                max_c_try = np.max(c_try)
                                if max_c_try <= max_c * 1.01:
                                    if l2_try > best_l2:
                                        best_l2 = l2_try
                                        best_alpha = alpha
                        if best_alpha > 0:
                            h = h + best_alpha * d
                            h = np.maximum(0, h)
                            s = np.sum(h)
                            if s > 0:
                                h = h / s
                            continue
        except Exception:
            pass
        
        improved = False
        for i in range(N):
            orig = h[i]
            for factor in [0.8, 0.9, 1.1, 1.2]:
                h_try = h.copy()
                h_try[i] = orig * factor
                h_try = np.maximum(0, h_try)
                s_try = np.sum(h_try)
                if s_try > 0:
                    h_try = h_try / s_try
                    c_try = _autoconv(h_try)
                    max_c_try = np.max(c_try)
                    if max_c_try <= max_c * 1.005:
                        l2_try = np.sum(h_try**2)
                        l2_orig = np.sum(h**2)
                        if l2_try > l2_orig:
                            h = h_try
                            improved = True
                            break
        if not improved:
            break
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    def update_best(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(0, h)
        s = np.sum(h)
        if s <= 0:
            return
        h = h / s
        sc = _score_raw(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N) / N)
    
    # Sparse peaks with various gaps
    for gap in [2, 3, 4, 5]:
        for start in range(gap):
            h = np.zeros(N)
            count = 0
            for i in range(start, N, gap):
                h[i] = 1.0
                count += 1
            if count > 0:
                initializations.append(h / np.sum(h))
    
    # Random sparse configurations
    np.random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 60:
            break
        num_peaks = np.random.randint(3, 20)
        indices = np.random.choice(N, num_peaks, replace=False)
        h = np.zeros(N)
        h[indices] = np.random.uniform(0.5, 2.0, size=num_peaks)
        initializations.append(h / np.sum(h))
    
    # Structured: alternating high/low
    for period in [2, 3, 4]:
        for offset in range(period):
            h = np.zeros(N)
            for i in range(N):
                if i % period == offset:
                    h[i] = 1.0
            if np.sum(h) > 0:
                initializations.append(h / np.sum(h))
    
    # Phase 2: SLSQP optimization from each initialization
    for h_init in initializations:
        if time.time() > deadline - 20:
            break
        
        h_opt = _solve_slsqp(h_init, maxiter=300, ftol=1e-10)
        if h_opt is not None:
            update_best(h_opt)
    
    # Phase 3: Soft-max refinement on best solutions
    if best_h is not None and time.time() < deadline - 10:
        h_refined = _soft_max_refine(best_h, iterations=50, beta_start=5.0, beta_end=50.0)
        update_best(h_refined)
    
    # Phase 4: More SLSQP restarts with perturbation of best
    if time.time() < deadline - 10:
        for restart in range(30):
            if time.time() > deadline - 5:
                break
            
            h_pert = best_h.copy()
            mask = np.random.random(N) < 0.3
            h_pert[mask] *= np.random.uniform(0.5, 2.0, size=np.sum(mask))
            h_pert = np.maximum(0, h_pert)
            s = np.sum(h_pert)
            if s > 0:
                h_pert = h_pert / s
            
            h_opt = _solve_slsqp(h_pert, maxiter=200, ftol=1e-10)
            if h_opt is not None:
                update_best(h_opt)
    
    # Phase 5: Final spectral refinement
    if best_h is not None and time.time() < deadline - 2:
        h_refined = _spectral_refine(best_h, iterations=50)
        update_best(h_refined)
    
    # Phase 6: Final soft-max polish
    if best_h is not None and time.time() < deadline - 1:
        h_refined = _soft_max_refine(best_h, iterations=30, beta_start=20.0, beta_end=200.0)
        update_best(h_refined)
    
    # Ensure valid output
    best_h = np.maximum(0, best_h)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return list(best_h)
