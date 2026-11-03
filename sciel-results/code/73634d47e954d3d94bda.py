"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-based smooth-max/epigraph optimization with Nesterov acceleration,
adaptive beta schedule, and high-precision coordinate-descent polish.
"""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 142.0


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}. Uses FFT."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the actual maximum overlap score."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """Project h onto {x : 0 <= x_i <= 1, sum(x) = target_sum} via bisection on offset."""
    h = np.asarray(h, dtype=np.float64)
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0
    if np.sum(np.clip(h + low, 0.0, 1.0)) > target_sum:
        low = -10.0
    if np.sum(np.clip(h + high, 0.0, 1.0)) < target_sum:
        high = 10.0
    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = np.sum(np.clip(h + mid, 0.0, 1.0))
        if s_mid > target_sum:
            high = mid
        else:
            low = mid
    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
    return h_proj


def _smooth_max_obj_grad(h, beta):
    """
    Compute value and gradient of smooth max of C_k using log-sum-exp.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    """
    cs = _linear_overlap_values(h)
    n = len(h)
    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp
    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)
    
    # Gradient via FFT: grad_F_j = -(2/N) * (A_j + B_j)
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)
    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)
    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]
    grad = -(2.0 / N) * (A + B)
    return val, grad


def _nesterov_optimize(h0, beta, n_iter=300, lr=0.05):
    """Nesterov accelerated gradient descent on smooth max with adaptive LR."""
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    h_prev = h.copy()
    t = 1.0
    best_h = h.copy()
    best_score = _score(h)
    lr_current = lr
    max_backtracks = 8
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        y = h + momentum * (h - h_prev)
        val, grad = _smooth_max_obj_grad(y, beta)
        h_new = y - lr_current * grad
        h_new = _project(h_new, N / 2.0)
        val_new = _score(h_new)
        backtrack_count = 0
        while val_new > val + 1e-10 * np.abs(val) and backtrack_count < max_backtracks:
            lr_current *= 0.5
            h_new = y - lr_current * grad
            h_new = _project(h_new, N / 2.0)
            val_new = _score(h_new)
            backtrack_count += 1
        if backtrack_count == max_backtracks:
            lr_current = max(lr * 0.1, 1e-6)
        h_prev = h
        h = h_new
        t = t_new
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
    return best_h, best_score


def _epigraph_optimize(h0, beta, n_iter=300, lr_t=0.02, lr_h=0.05):
    """
    Epigraph formulation: minimize t + (1/beta)*log(sum(exp(beta*(C_k - t))))
    over (h, t) with h in [0,1]^N, sum(h)=N/2, and t free.
    This can be more numerically stable than pure smooth-max.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    t = _score(h)
    
    h_prev = h.copy()
    t_prev = t
    t_k = 1.0
    
    best_h = h.copy()
    best_t = t
    best_score = t
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        t_new_k = (1 + np.sqrt(1 + 4 * t_k * t_k)) / 2
        momentum = (t_k - 1) / t_new_k
        
        # Nesterov extrapolation
        y_h = h + momentum * (h - h_prev)
        y_t = t + momentum * (t - t_prev)
        
        # Compute C_k at y_h
        cs = _linear_overlap_values(y_h)
        diff = cs - y_t
        cs_max = np.max(diff)
        exp_diff = np.exp(beta * (diff - cs_max))
        sum_exp = np.sum(exp_diff)
        w = exp_diff / sum_exp
        
        # Objective: t + (1/beta)*(log(sum_exp) + beta*cs_max) = t + cs_max + log(sum_exp)/beta
        obj_val = y_t + cs_max + np.log(sum_exp) / beta
        
        # Gradient w.r.t. t: 1 - sum(w) = 0 (since sum(w)=1)
        # Actually: d/dt [t + (1/beta)*log(sum(exp(beta*(C_k-t))))]
        # = 1 + (1/beta) * sum(exp(beta*(C_k-t)) * (-beta) * (1/sum(exp(...))))
        # = 1 - 1 = 0
        # Wait, that's not right. Let me recompute.
        # F(t) = t + (1/beta) * log(sum_k exp(beta*(C_k - t)))
        # dF/dt = 1 + (1/beta) * (sum_k exp(beta*(C_k-t)) * (-beta)) / sum_k exp(beta*(C_k-t))
        # = 1 - 1 = 0
        # Hmm, that means t doesn't appear in the gradient? That can't be right.
        # Let me re-examine. The issue is that C_k depends on h, not t.
        # So dF/dt = 1 - (sum_k w_k) = 1 - 1 = 0.
        # This means the optimal t* = max_k C_k exactly, and the smooth approximation
        # just adds a small offset. So we should just set t = max(C_k) + small correction.
        
        # Actually, for the epigraph, at optimum: t* = max_k C_k + (1/beta)*log(N)
        # approximately. Let's just use the smooth max approach but with t as a variable.
        
        # Better approach: just use the smooth max value as a proxy for t.
        # The gradient w.r.t. h is the same as smooth max gradient.
        # Let's just do Nesterov on h with the smooth max objective, and track t = max(C_k).
        
        # Gradient w.r.t. h: same as smooth max
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        w_pad = np.zeros(n_fft)
        w_pad[:N] = w
        y_h_pad = np.zeros(n_fft)
        y_h_pad[:N] = y_h
        W = np.fft.rfft(w_pad)
        H = np.fft.rfft(y_h_pad)
        conv = np.fft.irfft(W * H, n=n_fft)
        A = conv[:N]
        w_rev = w[::-1]
        w_rev_pad = np.zeros(n_fft)
        w_rev_pad[:N] = w_rev
        W_rev = np.fft.rfft(w_rev_pad)
        conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
        B = conv_rev[N-1 : N-1+N]
        grad_h = -(2.0 / N) * (A + B)
        
        # Update h
        h_new = y_h - lr_h * grad_h
        h_new = _project(h_new, N / 2.0)
        
        # Update t: t_new = max(C_k at h_new) + (1/beta)*log(N) as approximation
        cs_new = _linear_overlap_values(h_new)
        t_new = np.max(cs_new) + np.log(N) / beta
        
        # Check improvement
        actual_score = _score(h_new)
        if actual_score < best_score:
            best_score = actual_score
            best_h = h_new.copy()
            best_t = t_new
        
        h_prev = h
        t_prev = t
        h = h_new
        t = t_new
        t_k = t_new_k
    
    return best_h, best_score


def _coordinate_descent_polish(h, n_sweeps=300, tol=1e-10):
    """
    Coordinate descent polish: for each j, find h_j that minimizes max_k C_k
    subject to 0 <= h_j <= 1, then project to maintain sum constraint.
    """
    h = np.array(h, dtype=np.float64)
    h = _project(h, N / 2.0)
    n = len(h)
    best_score = _score(h)
    
    for sweep in range(n_sweeps):
        if time.time() > _DEADLINE - 3:
            break
        improved = False
        for j in range(n):
            if time.time() > _DEADLINE - 3:
                break
            h_save = h[j]
            cs = _linear_overlap_values(h)
            
            # Compute coeff_k = h_{j-k}[j>=k] + h_{j+k}[j+k<N]
            coeff = np.zeros(n)
            for k in range(n):
                c = 0.0
                if j >= k:
                    c += h[j - k]
                if j + k < n:
                    c += h[j + k]
                coeff[k] = c
            
            a = (2.0 / n) * coeff
            b = cs + a * h_save
            
            # Minimize max_k (b_k - a_k * t) over t in [0,1] via ternary search
            lo, hi = 0.0, 1.0
            for _ in range(40):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                f1 = np.max(b - a * m1)
                f2 = np.max(b - a * m2)
                if f1 < f2:
                    hi = m2
                else:
                    lo = m1
            t_opt = (lo + hi) / 2.0
            f_opt = np.max(b - a * t_opt)
            f_current = np.max(cs)
            
            if f_opt < f_current - 1e-12:
                h[j] = t_opt
                h = _project(h, n / 2.0)
                new_score = _score(h)
                if new_score < best_score - 1e-12:
                    best_score = new_score
                    improved = True
            else:
                h[j] = h_save
        if not improved:
            break
    return h, best_score


def _high_precision_polish(h, n_iter=100, lr=0.003, beta=5000.0):
    """
    Final high-precision polish: very high beta, tiny learning rate,
    combined with coordinate descent on active set.
    """
    h = np.array(h, dtype=np.float64)
    h = _project(h, N / 2.0)
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_obj_grad(h, beta)
        h_new = h - lr * grad
        h_new = _project(h_new, N / 2.0)
        current_score = _score(h_new)
        if current_score < best_score - 1e-15:
            best_score = current_score
            best_h = h_new.copy()
            h = h_new
        else:
            # Try half step
            h_new = h - lr * 0.5 * grad
            h_new = _project(h_new, N / 2.0)
            current_score = _score(h_new)
            if current_score < best_score - 1e-15:
                best_score = current_score
                best_h = h_new.copy()
                h = h_new
            else:
                break
    
    # Coordinate descent on the result
    if time.time() < _DEADLINE - 3:
        h_cd, score_cd = _coordinate_descent_polish(best_h, n_sweeps=500)
        if score_cd < best_score:
            best_score = score_cd
            best_h = h_cd.copy()
    
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Diverse initializations
    i = np.arange(N)
    initializations = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.sin(2 * np.pi * i / N),
        0.5 + 0.3 * np.cos(2 * np.pi * i / N),
        0.5 + 0.2 * np.random.randn(N),
        np.random.uniform(0.0, 1.0, N),
        0.5 + 0.3 * np.sin(4 * np.pi * i / N),
        0.5 + 0.2 * np.cos(2 * np.pi * 3 * i / N),
        0.5 + 0.15 * np.sin(2 * np.pi * 5 * i / N) + 0.1 * np.cos(2 * np.pi * 7 * i / N),
        np.where(i % 2 == 0, 0.7, 0.3),
    ]
    
    inits = []
    for init in initializations:
        h_init = np.clip(init, 0.0, 1.0)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
    
    # Beta schedule: start low, increase aggressively
    beta_schedule = [
        (5.0, 400, 0.05),
        (20.0, 350, 0.04),
        (100.0, 300, 0.02),
        (500.0, 250, 0.01),
        (2000.0, 200, 0.005),
        (10000.0, 150, 0.002),
    ]
    
    for beta, n_iter, lr in beta_schedule:
        if time.time() > _DEADLINE - 20:
            break
        
        for h_init in inits:
            if time.time() > _DEADLINE - 10:
                break
            
            # Use Nesterov smooth max
            h_opt, s_opt = _nesterov_optimize(h_init, beta, n_iter=n_iter, lr=lr)
            
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
    
    # High-precision final polish
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_polished, s_polished = _high_precision_polish(best_h, n_iter=100, lr=0.003, beta=5000.0)
        if s_polished < best_score:
            best_score = s_polished
            best_h = h_polished.copy()
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure exact sum and bounds
    best_h = _project(best_h, N / 2.0)
    
    return best_h
