import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 138  # Safety margin before 150s kill


def _linear_overlap_values(h):
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
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0

    if np.all(h >= 0.0) and np.all(h <= 1.0) and abs(np.sum(h) - target) < 1e-14:
        return h

    alpha_low = -max(h) - 1.0
    alpha_high = 1.0 - min(h) + 1.0

    for _ in range(60):
        alpha_mid = (alpha_low + alpha_high) / 2.0
        h_new = np.clip(h + alpha_mid, 0.0, 1.0)
        s = np.sum(h_new)
        if s < target:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    h_final = np.clip(h + (alpha_low + alpha_high) / 2.0, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        mask = (h_final > 1e-10) & (h_final < 1.0 - 1e-10)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_final)
        else:
            idx = np.argmax(h_final)
        h_final[idx] += diff
        h_final = np.clip(h_final, 0.0, 1.0)

    return h_final


def _solve_inner_quadratic(p, h0, max_iter=100):
    """
    Solve min_h sum_k p_k * C_k(h) subject to 0<=h<=1, sum(h)=N/2.
    C_k(h) = (2/N)(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}
    
    This is a convex quadratic in h. We use projected gradient with the
    Hessian-vector product computed via FFT.
    """
    n = N
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    
    # Precompute: C_k = (2/N)(N/2 - r_k)
    # r_k = sum_{i=k}^{N-1} h_i * h_{i-k} (for k=0, this is sum h_i^2)
    # grad of C_k w.r.t. h: 
    #   d(r_k)/d(h_j) = h_{j-k} (if j>=k) + h_{j+k} (if j+k<N), for k>0
    #   for k=0: d(r_0)/d(h_j) = 2*h_j
    #   So d(C_k)/d(h_j) = -(2/N) * [h_{j-k} + h_{j+k}] (with boundary handling)
    
    def _grad_weighted(p_vec, h_vec):
        """Gradient of sum_k p_k * C_k(h)"""
        n = len(h_vec)
        grad = np.zeros(n)
        for k in range(n):
            pk = p_vec[k]
            if pk < 1e-15:
                continue
            factor = -(2.0 / n) * pk
            if k == 0:
                grad += factor * 2.0 * h_vec
            else:
                # d(r_k)/d(h_j) = h_{j-k} (if j>=k) + h_{j+k} (if j+k<n)
                # For j in [0, n-k-1]: contributes h_{j-k} from first term (when j-k >= 0)
                # Actually: r_k = sum_{i=k}^{n-1} h_i h_{i-k}
                # d(r_k)/d(h_j): 
                #   if j >= k: term h_j * h_{j-k} exists (i=j), so derivative has h_{j-k}
                #   if j+k < n: term h_{j+k} * h_j exists (i=j+k), so derivative has h_{j+k}
                for j in range(n):
                    contrib = 0.0
                    if j >= k:
                        contrib += h_vec[j - k]
                    if j + k < n:
                        contrib += h_vec[j + k]
                    grad[j] += factor * contrib
        return grad
    
    # Faster gradient using convolution
    def _grad_weighted_fast(p_vec, h_vec):
        n = len(h_vec)
        n_fft = 2 * n
        h_pad = np.zeros(n_fft)
        h_pad[:n] = h_vec
        
        # For each k, the gradient contribution is -(2/n)*p_k * (h shifted by k + h shifted by -k)
        # We can compute all at once using FFT
        grad = np.zeros(n)
        
        # Method: for each k, grad_k = -(2/n)*p_k * (h_{j-k} + h_{j+k})
        # This is like a weighted sum of shifted h
        # Use FFT: create a filter where the k-th shift has weight p_k
        # Actually, let's just do it with the loop but vectorized per k
        
        for k in range(n):
            pk = p_vec[k]
            if pk < 1e-15:
                continue
            factor = -(2.0 / n) * pk
            if k == 0:
                grad += factor * 2.0 * h_vec
            else:
                # h_{j-k} for j>=k, h_{j+k} for j+k<n
                if n - k > 0:
                    grad[:n-k] += factor * h_vec[k:]  # h_{j+k} for j in [0, n-k-1]
                    grad[k:] += factor * h_vec[:n-k]  # h_{j-k} for j in [k, n-1]
        return grad
    
    # Use the fast gradient
    h_prev = h.copy()
    t_k = 1.0
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        grad = _grad_weighted_fast(p, h)
        h_new = h - lr * grad
        h_new = _fix_sum(h_new)
        
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)
        
        h_prev = h
        h = h_next
        t_k = t_new
        
        if it % 50 == 49:
            lr = max(0.001, min(0.05, lr * 0.95))
    
    return h


def _minimax_alternating(h0, num_outer=50, inner_iter=50):
    """
    Alternating minimax: 
    Outer: update p (weights on overlap values) via multiplicative weights
    Inner: solve min_h sum_k p_k C_k(h) for fixed p
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Initialize p uniformly
    p = np.ones(N) / N
    
    for outer in range(num_outer):
        if time.time() > _DEADLINE - 10:
            break
        
        # Inner: solve for h given p
        h = _solve_inner_quadratic(p, h, max_iter=inner_iter)
        
        # Evaluate
        cs = _linear_overlap_values(h)
        current_score = np.max(cs)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Outer: update p via multiplicative weights (exponential weights on active constraints)
        # p_k <- p_k * exp(beta * C_k) / sum_j p_j * exp(beta * C_j)
        beta = 20.0 + outer * 2  # Increase sharpness over time
        exp_cs = np.exp(beta * cs - np.max(cs))
        p_new = p * exp_cs
        p_new = p_new / np.sum(p_new)
        p = p_new
    
    return best_h, best_score


def _smooth_max_val_grad(h, beta):
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    # FFT-based gradient
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)

    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]

    h_flip = np.zeros(n)
    h_flip[0] = h[0]
    h_flip[1:] = h[1:][::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv_whf = np.fft.irfft(W * HF, n=n_fft)[:n]

    grad = -(2.0 / n) * (conv_wh + conv_whf)
    return val, grad


def _accelerated_smooth_max(h0, beta, max_iter=1000, lr_init=0.05):
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    best_val = _score(h)
    best_h = h.copy()

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _smooth_max_val_grad(h, beta)
        h_new = h - lr * grad
        h_new = _fix_sum(h_new)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        if it % 100 == 99:
            lr = max(0.001, min(0.1, lr * 0.95))

        if it % 50 == 49:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()

    return best_h, best_val


def construct():
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

    i = np.arange(N)

    # Phase 1: Generate diverse structured starts
    starts = []

    for _ in range(6):
        h_rand = np.random.rand(N)
        starts.append(_fix_sum(h_rand))

    for f in [1, 2, 3, 4, 5, 6, 7, 8]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * f * i / N + phase)
            starts.append(_fix_sum(h_sine))

    for f in [1, 2, 3]:
        for amp in [0.2, 0.3, 0.4]:
            h_cos = 0.5 + amp * np.cos(2 * np.pi * f * i / N)
            starts.append(_fix_sum(h_cos))

    h_2f = 0.5 + 0.3 * np.cos(2 * np.pi * i / N) + 0.2 * np.cos(4 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))
    h_2f2 = 0.5 + 0.25 * np.sin(2 * np.pi * i / N) + 0.25 * np.sin(3 * np.pi * i / N)
    starts.append(_fix_sum(h_2f2))

    h_saw = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_fix_sum(h_saw))

    for h0 in starts:
        _update_best(h0)

    # Phase 2: Minimax alternating from best starts
    if best_h is not None:
        for num_outer in [30, 50]:
            if time.time() > _DEADLINE - 40:
                break
            h_candidate, s_candidate = _minimax_alternating(best_h, num_outer=num_outer, inner_iter=40)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 3: Smooth max accelerated gradient (refinement)
    if best_h is not None:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > _DEADLINE - 20:
                break
            h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta, max_iter=500, lr_init=0.05)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 4: More minimax with higher beta
    if best_h is not None and time.time() < _DEADLINE - 15:
        h_candidate, s_candidate = _minimax_alternating(best_h, num_outer=40, inner_iter=60)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 5: Final high-beta polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=400.0, max_iter=300, lr_init=0.02)
        if s_candidate < best_score:
            _update_best(h_candidate)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
