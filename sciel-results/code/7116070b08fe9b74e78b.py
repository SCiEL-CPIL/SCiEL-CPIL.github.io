"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.

Strategy:
1. Parameterise the sequence as a low-dimensional manifold:
   h_i = 0.5 + a*cos(2*pi*f*i/N + phi) + b*sin(2*pi*f*i/N + phi) + noise
   This captures the smooth, low-autocorrelation structure.
2. Use Nesterov accelerated gradient descent on a smooth-max surrogate
   for global optimization in the full N-dimensional space.
3. Use coordinate descent to equalize the top active constraints,
   targeting the multi-constraint tie structure at the optimum.
4. Adaptive beta schedule to sharpen the max approximation.
"""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
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
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0
    
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
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
    Compute the value and gradient of the smooth max of C_k.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    grad F = sum(w_k * grad C_k)
    """
    cs = _linear_overlap_values(h)
    
    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp
    
    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)
    
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
        
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    
    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)
    
    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:n]
    
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:n] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)
    
    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[n-1 : n-1+n]
    
    grad = -(2.0 / n) * (A + B)
    
    return val, grad


def _nesterov_accelerated_gradient(h0, beta, n_iter=300, lr=0.05):
    """
    Nesterov accelerated gradient descent on the smooth max objective
    with backtracking line search.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    h_prev = h.copy()
    t = 1.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        
        y = h + momentum * (h - h_prev)
        
        val_y, grad = _smooth_max_obj_grad(y, beta)
        
        step_lr = lr
        h_new = None
        for bt in range(20):
            h_candidate = y - step_lr * grad
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-6:
                h_new = h_candidate
                break
            step_lr *= 0.5
        
        if h_new is None:
            h_new = y - lr * grad
            h_new = _project(h_new, N / 2.0)
        
        h_prev = h
        h = h_new
        t = t_new
        
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h, best_score


def _coordinate_descent_refine(h0, max_time=30.0, n_top=5):
    """
    Coordinate descent to equalize the top active C_k constraints.
    For each pair of active constraints, adjust h[i], h[j] to reduce the max.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    best_h = h.copy()
    best_score = _score(h)
    
    start_time = time.time()
    
    while time.time() < start_time + max_time and time.time() < _DEADLINE - 3:
        cs = _linear_overlap_values(h)
        cs_sorted = np.argsort(cs)[::-1]
        top_indices = cs_sorted[:n_top]
        
        improved = False
        for k1 in top_indices:
            for k2 in top_indices:
                if k1 >= k2:
                    continue
                # Try adjusting pairs of h values
                for i in range(N):
                    for j in range(i + 1, N):
                        # Compute current C_k1 and C_k2 contributions
                        # If we increase h[i] by delta, C_k changes by -(2/N)*delta*h[i-k]
                        # We want to find delta that equalizes C_k1 and C_k2
                        h_i = h[i]
                        h_j = h[j]
                        
                        # Compute gradient of C_k1 and C_k2 w.r.t h[i] and h[j]
                        dC1_di = -(2.0/N) * (h[(i-k1)%N] if i >= k1 else h[(i-k1+N)%N])
                        dC1_dj = -(2.0/N) * (h[(j-k1)%N] if j >= k1 else h[(j-k1+N)%N])
                        dC2_di = -(2.0/N) * (h[(i-k2)%N] if i >= k2 else h[(i-k2+N)%N])
                        dC2_dj = -(2.0/N) * (h[(j-k2)%N] if j >= k2 else h[(j-k2+N)%N])
                        
                        # Solve for delta_i, delta_j to equalize C_k1 and C_k2
                        # C_k1 + dC1_di*delta_i + dC1_dj*delta_j = C_k2 + dC2_di*delta_i + dC2_dj*delta_j
                        # (dC1_di - dC2_di)*delta_i + (dC1_dj - dC2_dj)*delta_j = C_k2 - C_k1
                        
                        A11 = dC1_di - dC2_di
                        A12 = dC1_dj - dC2_dj
                        b = cs[k2] - cs[k1]
                        
                        if abs(A11) < 1e-12 and abs(A12) < 1e-12:
                            continue
                            
                        # Use least squares if singular
                        if abs(A11) > 1e-12:
                            # delta_i = (b - A12*delta_j) / A11
                            # Minimize (C_k1 + dC1_di*delta_i + dC1_dj*delta_j)^2 + (C_k2 + dC2_di*delta_i + dC2_dj*delta_j)^2
                            # This is a 2x2 system
                            A = np.array([[A11, A12],
                                          [dC1_di + dC2_di, dC1_dj + dC2_dj]])
                            b_vec = np.array([b, -(cs[k1] + cs[k2])])
                            
                            try:
                                delta = np.linalg.solve(A, b_vec)
                            except np.linalg.LinAlgError:
                                delta = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                        else:
                            delta_j = b / A12
                            delta_i = 0.0
                            delta = np.array([delta_i, delta_j])
                        
                        delta_i = delta[0]
                        delta_j = delta[1]
                        
                        # Limit step size
                        max_step = 0.05
                        delta_i = np.clip(delta_i, -max_step, max_step)
                        delta_j = np.clip(delta_j, -max_step, max_step)
                        
                        # Ensure bounds
                        new_hi = np.clip(h_i + delta_i, 0.0, 1.0)
                        new_hj = np.clip(h_j + delta_j, 0.0, 1.0)
                        
                        if abs(new_hi - h_i) < 1e-10 and abs(new_hj - h_j) < 1e-10:
                            continue
                            
                        h_test = h.copy()
                        h_test[i] = new_hi
                        h_test[j] = new_hj
                        h_test = _project(h_test, N / 2.0)
                        
                        new_score = _score(h_test)
                        if new_score < best_score - 1e-12:
                            h = h_test
                            best_score = new_score
                            best_h = h.copy()
                            improved = True
                            
        if not improved:
            break
            
    return best_h, best_score


def _gradient_polish(h0, beta=500.0, max_time=15.0):
    """
    Refine h using Nesterov accelerated gradient descent with a high beta.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    best_h = h.copy()
    best_score = _score(h)
    
    start_time = time.time()
    lr = 0.02
    t = 1.0
    h_prev = h.copy()
    
    while time.time() < start_time + max_time:
        if time.time() > _DEADLINE - 2:
            break
            
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        y = h + momentum * (h - h_prev)
        
        val_y, grad = _smooth_max_obj_grad(y, beta)
        
        h_new = y - lr * grad
        h_new = _project(h_new, N / 2.0)
        
        s_new = _score(h_new)
        if s_new < best_score - 1e-12:
            best_score = s_new
            best_h = h_new.copy()
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.95
            if lr < 1e-5:
                break
                
        h_prev = h
        h = h_new
        t = t_new
        
    return best_h, best_score


def _parametric_init(freq, amp_cos, amp_sin, noise_scale, seed):
    """
    Generate a parameterized initial guess:
    h_i = 0.5 + amp_cos*cos(2*pi*freq*i/N + phase) + amp_sin*sin(2*pi*freq*i/N + phase) + noise
    """
    np.random.seed(seed)
    i = np.arange(N)
    phase = np.random.uniform(0, 2*np.pi)
    h = 0.5 + amp_cos * np.cos(2*np.pi*freq*i/N + phase) + amp_sin * np.sin(2*np.pi*freq*i/N + phase)
    h += noise_scale * np.random.randn(N)
    h = np.clip(h, 0.0, 1.0)
    h = _project(h, N / 2.0)
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Phase 1: Generate diverse initializations from the parameterized manifold
    inits = []
    
    # Structured inits from the manifold
    for freq in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for amp in [0.2, 0.3, 0.4, 0.5]:
            for noise in [0.0, 0.05, 0.1]:
                seed = hash((freq, amp, noise)) % (2**31)
                h_init = _parametric_init(freq, amp, amp*0.5, noise, seed)
                inits.append(h_init)
    
    # Add random inits
    for _ in range(5):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
    
    # Add a few standard inits
    inits.append(np.full(N, 0.5))
    inits.append(_project(0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N), N/2.0))
    
    # Phase 2: Nesterov optimization with adaptive beta
    beta = 10.0
    n_restarts = 8
    
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 25:
            break
            
        for h_init in inits:
            if time.time() > _DEADLINE - 10:
                break
                
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=200, lr=0.05)
            
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
        beta *= 1.4
        
    # Phase 3: Coordinate descent refinement
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_ref, s_ref = _coordinate_descent_refine(best_h, max_time=min(20.0, _DEADLINE - time.time() - 8), n_top=5)
        if s_ref < best_score:
            best_score = s_ref
            best_h = h_ref.copy()
            
    # Phase 4: High-precision gradient polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_ref, s_ref = _gradient_polish(best_h, beta=500.0, max_time=min(10.0, _DEADLINE - time.time() - 3))
        if s_ref < best_score:
            best_score = s_ref
            best_h = h_ref.copy()
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h, N / 2.0)
    return best_h
