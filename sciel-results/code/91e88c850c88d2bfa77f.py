import time
import numpy as np

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


def _smooth_max_val_grad(h, beta):
    """
    Compute the value and gradient of the smooth max surrogate:
    f(h) = (1/beta) * log(sum_k exp(beta * C_k(h)))
    where C_k(h) = (2/N) * (N/2 - r_k), r_k = autocorrelation at lag k.
    
    The gradient is computed via FFT for efficiency.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    # Gradient of C_k w.r.t. h_j:
    # C_k = (2/N)(N/2 - r_k), r_k = sum_{i=k}^{n-1} h_i h_{i-k}
    # d(r_k)/d(h_j) = h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) for k>0
    #                = 2*h_j for k=0
    # So d(C_k)/d(h_j) = -(2/N) * [h_{j-k} + h_{j+k}] (boundary handled)
    
    # We need: grad_h = sum_k w_k * d(C_k)/d(h)
    # = -(2/N) * sum_k w_k * [h_{j-k} + h_{j+k}]
    
    # This is a weighted sum of shifts of h. We can compute it via FFT.
    # Let W be the FFT of w (padded), H be the FFT of h (padded).
    # The sum_k w_k * h_{j+k} is a circular convolution of w and h.
    # The sum_k w_k * h_{j-k} is a circular correlation.
    
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    W = np.fft.rfft(w_pad, n=n_fft)
    
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    
    # sum_k w_k * h_{j+k} = circular convolution (w * h)_j
    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # sum_k w_k * h_{j-k} = circular correlation = conv of w with flipped h
    h_flip = np.zeros(n_fft)
    h_flip[0] = h[0]
    h_flip[1:n] = h[1:][::-1]
    HF = np.fft.rfft(h_flip, n=n_fft)
    conv_whf = np.fft.irfft(W * HF, n=n_fft)[:n]
    
    grad = -(2.0 / n) * (conv_wh + conv_whf)
    
    return val, grad


def _accelerated_smooth_max(h0, beta, max_iter=1000, lr_init=0.05):
    """Nesterov-accelerated gradient descent on the smooth max surrogate."""
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
        
        # Gradient step
        h_new = h - lr * grad
        h_new = _fix_sum(h_new)
        
        # Nesterov momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)
        
        h_prev = h
        h = h_next
        t_k = t_new
        
        # Decay learning rate
        if it % 100 == 99:
            lr = max(0.001, min(0.1, lr * 0.95))
        
        # Track best
        if it % 50 == 49:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()

    return best_h, best_val


def _minimax_alternating(h0, num_outer=50, inner_iter=50):
    """
    Alternating minimax:
    Outer: update p (weights on overlap values) via multiplicative weights
    Inner: solve min_h sum_k p_k C_k(h) for fixed p using accelerated gradient
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
        # We use a weighted smooth max as a proxy for the linear objective
        # Actually, let's directly minimize sum_k p_k C_k(h)
        # The gradient of sum_k p_k C_k(h) is:
        # -(2/N) * sum_k p_k * [h_{j-k} + h_{j+k}]
        # which is the same form as _smooth_max_val_grad but with w=p
        
        h_inner = h.copy()
        h_prev_inner = h_inner.copy()
        t_k_inner = 1.0
        lr_inner = 0.02
        
        for it in range(inner_iter):
            if time.time() > _DEADLINE - 5:
                break
            
            # Compute gradient of sum_k p_k C_k(h)
            n = len(h_inner)
            n_fft = 2 * n
            p_pad = np.zeros(n_fft)
            p_pad[:n] = p
            P = np.fft.rfft(p_pad, n=n_fft)
            
            h_pad = np.zeros(n_fft)
            h_pad[:n] = h_inner
            H = np.fft.rfft(h_pad, n=n_fft)
            
            conv_ph = np.fft.irfft(P * H, n=n_fft)[:n]
            
            h_flip = np.zeros(n_fft)
            h_flip[0] = h_inner[0]
            h_flip[1:n] = h_inner[1:][::-1]
            HF = np.fft.rfft(h_flip, n=n_fft)
            conv_phf = np.fft.irfft(P * HF, n=n_fft)[:n]
            
            grad = -(2.0 / n) * (conv_ph + conv_phf)
            
            h_new = h_inner - lr_inner * grad
            h_new = _fix_sum(h_new)
            
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k_inner * t_k_inner)) / 2.0
            h_next = h_new + ((t_k_inner - 1.0) / t_new) * (h_new - h_prev_inner)
            h_next = _fix_sum(h_next)
            
            h_prev_inner = h_inner
            h_inner = h_next
            t_k_inner = t_new
        
        h = h_inner
        
        # Evaluate
        cs = _linear_overlap_values(h)
        current_score = np.max(cs)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Outer: update p via multiplicative weights
        beta = 20.0 + outer * 2
        exp_cs = np.exp(beta * cs - np.max(cs))
        p_new = p * exp_cs
        p_new = p_new / np.sum(p_new)
        p = p_new
    
    return best_h, best_score


def _coordinate_descent_polish(h0, max_iter=2000):
    """
    Coordinate descent polish: iteratively adjust individual h_j to reduce the max overlap.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    best_h = h.copy()
    best_score = _score(h)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        improved = False
        # Try each coordinate in random order
        indices = np.random.permutation(N)
        for j in indices:
            if time.time() > _DEADLINE - 5:
                break
            
            old_val = h[j]
            # Try small perturbations
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h[j] = np.clip(old_val + delta, 0.0, 1.0)
                # Fix sum by adjusting another coordinate
                diff = (N / 2.0) - np.sum(h)
                if abs(diff) > 1e-14:
                    # Find a coordinate to adjust
                    mask = (np.arange(N) != j)
                    mask &= (h > 1e-10) & (h < 1.0 - 1e-10)
                    if np.any(mask):
                        h[mask] += diff / np.sum(mask)
                        h = np.clip(h, 0.0, 1.0)
                    else:
                        h[j] = old_val
                        continue
                
                s = _score(h)
                if s < best_score - 1e-10:
                    best_score = s
                    best_h = h.copy()
                    improved = True
                    break
            h[j] = old_val
        
        if not improved:
            # Try larger steps
            for j in indices[:10]:
                if time.time() > _DEADLINE - 5:
                    break
                old_val = h[j]
                for delta in [0.05, -0.05, 0.1, -0.1]:
                    h[j] = np.clip(old_val + delta, 0.0, 1.0)
                    diff = (N / 2.0) - np.sum(h)
                    if abs(diff) > 1e-14:
                        mask = (np.arange(N) != j)
                        mask &= (h > 1e-10) & (h < 1.0 - 1e-10)
                        if np.any(mask):
                            h[mask] += diff / np.sum(mask)
                            h = np.clip(h, 0.0, 1.0)
                        else:
                            h[j] = old_val
                            continue
                    s = _score(h)
                    if s < best_score - 1e-10:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                        break
                h[j] = old_val
    
    return best_h, best_score


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

    for _ in range(8):
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

    # Add more structured starts
    for f in [1, 2, 3, 4]:
        for g in [2, 3, 5, 7]:
            h_prod = 0.5 + 0.2 * np.sin(2 * np.pi * f * i / N) * np.cos(2 * np.pi * g * i / N)
            starts.append(_fix_sum(h_prod))

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

    # Phase 5: Coordinate descent polish
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_candidate, s_candidate = _coordinate_descent_polish(best_h, max_iter=1500)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 6: Final high-beta polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=400.0, max_iter=300, lr_init=0.02)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 7: Another round of coordinate descent
    if best_h is not None and time.time() < _DEADLINE - 3:
        h_candidate, s_candidate = _coordinate_descent_polish(best_h, max_iter=500)
        if s_candidate < best_score:
            _update_best(h_candidate)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
