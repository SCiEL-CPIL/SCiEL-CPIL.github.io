"""
Optimized program for third autocorrelation inequality.
Combines coordinate descent, gradient-based soft-max smoothing, and ADMM-style
peak-constrained energy minimization.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 400
DEADLINE = 145.0  # safety margin before 150s kill


def _autoconv_fft(h):
    """Compute full autoconvolution using FFT."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * n - 1]


def _objective(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _soft_max_and_weights(conv, beta):
    """Compute soft-max and its weights for the autoconvolution."""
    m = np.max(conv)
    if m <= 0:
        return m, np.zeros_like(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    soft_max = m + np.log(np.sum(exp_shifted)) / beta
    return soft_max, weights


def _grad_autoconv(w, h):
    """
    Gradient of sum_j w_j * conv_j w.r.t. h.
    conv = full autoconvolution of h.
    d(conv_j)/d(h_l) = 2 * h[j-l] if 0 <= j-l < N, else 0.
    So grad_h[l] = 2 * sum_j w_j * h[j-l] (for valid j-l).
    Using FFT: cross_corr = ifft(fft(w) * conj(fft(h)))
    """
    L = 2 * N
    w_pad = np.zeros(L)
    w_pad[:len(w)] = w
    h_pad = np.zeros(L)
    h_pad[:N] = h
    
    W = np.fft.fft(w_pad, n=L)
    H = np.fft.fft(h_pad, n=L)
    
    cross = np.fft.ifft(W * np.conj(H)).real
    grad_h = 2.0 * cross[:N]
    return grad_h


def _objective_soft(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    soft_max, _ = _soft_max_and_weights(conv, beta)
    return 2.0 * N * soft_max / (s * s)


def _objective_soft_grad(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _autoconv_fft(h)
    soft_max, weights = _soft_max_and_weights(conv, beta)
    
    if np.max(weights) < 1e-15:
        return np.zeros_like(h)
    
    grad_sm = _grad_autoconv(weights, h)
    
    # f = 2*N*soft_max/s^2
    # df/dh_i = 2*N * (grad_sm_i * s^2 - soft_max * 2*s) / s^4
    #         = 2*N * (grad_sm_i * s - 2*soft_max) / s^3
    return 2.0 * N * (grad_sm * s - 2.0 * soft_max) / (s * s * s)


def _spectral_penalty(h, lam=0.1):
    """Penalty for spectral concentration (L4 norm of DCT coefficients)."""
    dct_coeffs = dct(h, type=2, norm='ortho')
    return lam * np.sum(np.abs(dct_coeffs)**4) / N


def _spectral_penalty_grad(h, lam=0.1):
    """Gradient of spectral penalty."""
    dct_coeffs = dct(h, type=2, norm='ortho')
    grad = 4.0 * lam / N * idct(dct_coeffs**3, type=2, norm='ortho')
    return grad


def _combined_obj(h, beta=50.0, lam=0.05):
    return _objective_soft(h, beta) + _spectral_penalty(h, lam)


def _combined_grad(h, beta=50.0, lam=0.05):
    return _objective_soft_grad(h, beta) + _spectral_penalty_grad(h, lam)


def _admm_peak_minimize(h_init, max_iters=200, lambda_step=0.5, deadline=None, start_time=None):
    """
    ADMM-style optimization to minimize peak of autoconvolution.
    
    Formulation:
      min_h  max_j conv_j(h)
      s.t.   sum(h) = 1 (normalized)
    
    ADMM splits:
      min  max_j conv_j + (rho/2)||h - z + u||^2
      s.t. sum(z) = 1
    
    We use a dual ascent approach with soft-max smoothing of the max.
    """
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = DEADLINE
    
    h = h_init.copy()
    # Normalize to sum=1
    s = np.sum(h)
    if abs(s) < 1e-12:
        h[0] += 1.0
        s = np.sum(h)
    h = h / s
    
    best_val = _objective(h)
    best_h = h.copy()
    
    rho = 0.1  # ADMM penalty parameter
    beta = 20.0  # soft-max temperature
    
    for it in range(max_iters):
        if time.time() - start_time > deadline - 3:
            break
        
        # Compute current autoconvolution and soft-max weights
        conv = _autoconv_fft(h)
        sm, weights = _soft_max_and_weights(conv, beta)
        
        # Gradient step on h to reduce weighted peak
        if np.max(weights) > 1e-15:
            grad = _grad_autoconv(weights, h)
            # Normalize gradient by sum to maintain constraint
            s = np.sum(h)
            grad = grad - np.mean(grad)  # zero-mean to preserve sum
            h = h - lambda_step * grad / (np.max(np.abs(grad)) + 1e-10)
        
        # Project onto sum=1
        s = np.sum(h)
        if abs(s) < 1e-12:
            h[0] += 1.0
            s = np.sum(h)
        h = h / s
        
        # Increase beta gradually (sharpen the max approximation)
        if it % 20 == 0:
            beta = min(beta * 1.3, 200.0)
        
        # Track best
        val = _objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val


def _coordinate_descent(h, max_iters=3000, step_scale=0.1, deadline=None, start_time=None):
    """
    Coordinate descent / hill climbing on h.
    Tries small perturbations in each coordinate direction.
    """
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = DEADLINE
    
    best_val = _objective(h)
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-12:
        h[0] += 1.0
        s = np.sum(h)
    
    # Normalize for stability
    h_norm = h / s
    best_val = _objective(h_norm)
    
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > deadline - 5:
            break
        
        if no_improve_count > 300:
            break
        
        # Try random coordinate perturbations
        n_try = 15
        improved = False
        for _ in range(n_try):
            i = np.random.randint(N)
            # Try both positive and negative steps
            for sign in [1.0, -1.0]:
                step = sign * step_scale * np.random.rand()
                h_new = h_norm.copy()
                h_new[i] += step
                s_new = np.sum(h_new)
                if abs(s_new) < 1e-12:
                    continue
                val_new = _objective(h_new)
                if val_new < best_val:
                    h_norm = h_new
                    best_val = val_new
                    improved = True
                    break
        if not improved:
            no_improve_count += 1
            # Occasionally try larger steps
            if no_improve_count % 50 == 0:
                step_scale *= 0.7
        else:
            no_improve_count = 0
    
    return h_norm, best_val


def _gradient_descent(h_init, max_iters=500, lr=0.01, deadline=None, start_time=None):
    """
    Gradient descent with line search on the soft-max objective.
    """
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = DEADLINE
    
    h = h_init.copy()
    s = np.sum(h)
    if abs(s) < 1e-12:
        h[0] += 1.0
        s = np.sum(h)
    h = h / s
    
    best_val = _objective(h)
    best_h = h.copy()
    
    beta = 30.0
    
    for it in range(max_iters):
        if time.time() - start_time > deadline - 5:
            break
        
        # Compute gradient
        grad = _objective_soft_grad(h, beta)
        # Make gradient zero-mean to preserve sum constraint
        grad = grad - np.mean(grad)
        
        # Line search
        alpha = lr
        found = False
        for _ in range(10):
            h_new = h - alpha * grad
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-12:
                alpha *= 0.5
                continue
            h_new = h_new / s_new
            val_new = _objective(h_new)
            if val_new < best_val:
                h = h_new
                best_val = val_new
                best_h = h.copy()
                found = True
                break
            alpha *= 0.5
        
        if not found:
            # Try smaller beta
            beta = max(beta * 0.8, 5.0)
            if beta <= 5.0 and it > 50:
                break
        
        # Increase beta gradually
        if it % 30 == 0:
            beta = min(beta * 1.2, 150.0)
    
    return best_h, best_val


def _refine(h, deadline):
    """Refine a candidate h using a sequence of optimization phases."""
    h_curr = h.copy()
    
    # Phase 1: Gradient descent with moderate beta
    if time.time() < deadline - 15.0:
        h_curr, _ = _gradient_descent(h_curr, max_iters=200, lr=0.02, deadline=deadline, start_time=time.time())
    
    # Phase 2: ADMM peak minimization
    if time.time() < deadline - 10.0:
        h_curr, _ = _admm_peak_minimize(h_curr, max_iters=100, lambda_step=0.3, deadline=deadline, start_time=time.time())
    
    # Phase 3: Coordinate descent polishing
    if time.time() < deadline - 5.0:
        h_curr, _ = _coordinate_descent(h_curr, max_iters=2000, step_scale=0.05, deadline=deadline, start_time=time.time())
    
    # Phase 4: Final gradient polish
    if time.time() < deadline - 2.0:
        h_curr, _ = _gradient_descent(h_curr, max_iters=100, lr=0.005, deadline=deadline, start_time=time.time())
    
    return h_curr


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective(best_h)  # should be 2.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    i = np.arange(N)
    
    # 1. Seed
    initializations.append(np.ones(N))
    
    # 2. Low-frequency sinusoids with DC offset
    for k in [1, 2, 3, 4]:
        for a in [0.5, 1.0, 2.0]:
            h0 = a + np.sin(2 * np.pi * k * i / N)
            initializations.append(h0)
    
    # 3. Comb patterns
    for period in [2, 3, 4, 5, 8, 10]:
        h0 = np.where((i % period) < period // 2, 1.0, -1.0)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) > 1.0:
            initializations.append(h0)
    
    # 4. Random smooth functions
    for _ in range(5):
        h0 = np.sin(2 * np.pi * rng.uniform(0.1, 2.0) * i / N + rng.uniform(0, 2 * np.pi))
        h0 += 1.0
        initializations.append(h0)
    
    # 5. Sparse random
    for n_sparse in [50, 100, 150]:
        h0 = np.zeros(N)
        idx = rng.choice(N, size=n_sparse, replace=False)
        h0[idx] = rng.randn(n_sparse)
        if abs(np.sum(h0)) < 1e-10:
            h0[0] += 1.0
        initializations.append(h0)
    
    # Optimize each initialization
    for h0 in initializations:
        if time.time() > deadline - 30.0:
            break
        
        h_refined = _refine(h0, deadline)
        val = _objective(h_refined)
        if val < best_val:
            best_val = val
            best_h = h_refined.copy()
    
    # Final refinement on the best found
    if time.time() < deadline - 10.0:
        h_final = _refine(best_h, deadline)
        val_final = _objective(h_final)
        if val_final < best_val:
            best_h = h_final
            best_val = val_final
    
    return best_h


# Quick test
if __name__ == '__main__':
    h = construct()
    val = _objective(h)
    print(f"Value: {val}")
    print(f"Score: {2.0 / val}")
