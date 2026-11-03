"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses Nesterov accelerated gradient descent on a smooth-max surrogate
with adaptive temperature, backtracking line search, and projection
onto the constraint set. Includes coordinate descent polish.
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


def _epigraph_obj_grad(h, t, beta):
    """
    Epigraph formulation: minimize t + (1/beta)*log(sum(exp(beta*(C_k - t))))
    This is equivalent to smooth max but with explicit t variable.
    Returns (val, grad_h, grad_t).
    """
    cs = _linear_overlap_values(h)
    diff = cs - t
    diff_max = np.max(diff)
    exp_diff = np.exp(beta * (diff - diff_max))
    sum_exp = np.sum(exp_diff)
    w = exp_diff / sum_exp
    
    val = t + (1.0 / beta) * (np.log(sum_exp) + beta * diff_max)
    
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
    
    grad_h = -(2.0 / N) * (A + B)
    grad_t = 1.0 - np.sum(w)
    
    return val, grad_h, grad_t


def _nesterov_accelerated_gradient(h0, beta, n_iter=500, lr=0.01):
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


def _h_to_cosine(h):
    """Convert h to cosine coefficients (excluding DC). h_i = mean + sum_k c_k cos(2*pi*k*i/N)."""
    mean_h = np.mean(h)
    k_vals = np.arange(1, N // 2 + 1)
    angles = 2.0 * np.pi * np.outer(np.arange(N), k_vals) / N
    cos_mat = np.cos(angles)  # (N, N//2)
    c = cos_mat.T @ h  # (N//2,)
    # For the cosine basis, the DC component is handled separately
    return mean_h, c


def _cosine_to_h(mean_h, c):
    """Convert cosine coefficients back to h."""
    k_vals = np.arange(1, N // 2 + 1)
    angles = 2.0 * np.pi * np.outer(np.arange(N), k_vals) / N
    cos_mat = np.cos(angles)  # (N, N//2)
    h = mean_h + cos_mat @ c
    return h


def _nesterov_spectral(h0, beta, n_iter=400, lr=0.05):
    """
    Nesterov accelerated gradient descent in cosine (spectral) space.
    The DC component is fixed to enforce sum constraint.
    Only non-DC cosine coefficients are optimized.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    mean_h = np.mean(h)
    c = _h_to_cosine(h)[1]
    
    c_prev = c.copy()
    t = 1.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        
        y_c = c + momentum * (c - c_prev)
        y_h = _cosine_to_h(mean_h, y_c)
        
        val_y, grad_h = _smooth_max_obj_grad(y_h, beta)
        
        # Project gradient onto cosine subspace (non-DC)
        k_vals = np.arange(1, N // 2 + 1)
        angles = 2.0 * np.pi * np.outer(np.arange(N), k_vals) / N
        cos_mat = np.cos(angles)
        grad_c = cos_mat.T @ grad_h  # gradient in cosine space
        
        step_lr = lr
        c_new = None
        for bt in range(20):
            c_candidate = y_c - step_lr * grad_c
            h_candidate = _cosine_to_h(mean_h, c_candidate)
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-6:
                c_new = c_candidate
                break
            step_lr *= 0.5
        
        if c_new is None:
            c_new = y_c - lr * grad_c
            h_candidate = _cosine_to_h(mean_h, c_new)
            h_candidate = _project(h_candidate, N / 2.0)
        
        c_prev = c
        c = c_new
        t = t_new
        
        h_current = _cosine_to_h(mean_h, c)
        h_current = _project(h_current, N / 2.0)
        current_score = _score(h_current)
        if current_score < best_score:
            best_score = current_score
            best_h = h_current.copy()
            
    return best_h, best_score


def _nesterov_epigraph(h0, beta, n_iter=500, lr=0.01):
    """
    Nesterov accelerated gradient descent using epigraph formulation with
    auxiliary variable t. This can help escape local minima by providing
    an additional degree of freedom.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    # Initialize t at the current max overlap
    t_val = _score(h)
    
    h_prev = h.copy()
    t_prev = t_val
    t_mom = 1.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        t_mom_new = (1 + np.sqrt(1 + 4 * t_mom * t_mom)) / 2
        momentum = (t_mom - 1) / t_mom_new
        
        y_h = h + momentum * (h - h_prev)
        y_t = t_val + momentum * (t_val - t_prev)
        
        val_y, grad_h, grad_t = _epigraph_obj_grad(y_h, y_t, beta)
        
        step_lr = lr
        h_new = None
        t_new_val = None
        for bt in range(20):
            h_candidate = y_h - step_lr * grad_h
            h_candidate = _project(h_candidate, N / 2.0)
            t_candidate = y_t - step_lr * grad_t
            t_candidate = max(0.0, t_candidate)
            
            # Score is the max of C_k, not t + smooth penalty
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-6:
                h_new = h_candidate
                t_new_val = t_candidate
                break
            step_lr *= 0.5
        
        if h_new is None:
            h_new = y_h - lr * grad_h
            h_new = _project(h_new, N / 2.0)
            t_new_val = max(0.0, y_t - lr * grad_t)
        
        h_prev = h
        t_prev = t_val
        h = h_new
        t_val = t_new_val
        t_mom = t_mom_new
        
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h, best_score


def _coordinate_descent_polish(h0, max_outer=50, max_inner=N):
    """
    Coordinate descent polish: adjust one element at a time with
    local search around current value. Uses adaptive step sizes,
    parabolic interpolation for faster convergence, and a finer grid.
    """
    h = np.array(h0, dtype=np.float64).copy()
    best_score = _score(h)
    best_h = h.copy()
    
    # Track best step size per coordinate for adaptive refinement
    best_step = np.full(N, 0.01)
    
    for outer in range(max_outer):
        if time.time() > _DEADLINE - 2:
            break
        improved = False
        for i in range(max_inner):
            if time.time() > _DEADLINE - 2:
                break
            old_val = h[i]
            step = best_step[i]
            
            # Adaptive grid: start with larger steps, refine if improvement
            deltas = [step, step/2, step/4, step/8, step/16, step/32]
            candidates = set()
            for d in deltas:
                candidates.add(old_val - d)
                candidates.add(old_val + d)
            # Also try very fine steps
            fine_deltas = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
            for d in fine_deltas:
                candidates.add(old_val - d)
                candidates.add(old_val + d)
            candidates.add(0.0)
            candidates.add(1.0)
            
            best_candidate_val = None
            best_candidate_score = best_score - 1e-12
            
            for new_val in sorted(candidates):
                if new_val < -1e-12 or new_val > 1 + 1e-12:
                    continue
                new_val = np.clip(new_val, 0.0, 1.0)
                h_test = h.copy()
                h_test[i] = new_val
                h_test = _project(h_test, N / 2.0)
                s_test = _score(h_test)
                if s_test < best_candidate_score:
                    best_candidate_score = s_test
                    best_candidate_val = new_val
                    # Update step size based on how far we moved
                    move = abs(new_val - old_val)
                    if move > 0:
                        best_step[i] = max(0.00005, move * 0.5)
            
            # Parabolic interpolation refinement
            if best_candidate_val is not None and best_candidate_val != old_val:
                # Try a few parabolic fits around the best point
                for refine_iter in range(3):
                    if time.time() > _DEADLINE - 2:
                        break
                    center = best_candidate_val
                    half_width = max(0.0005, abs(best_candidate_val - old_val) * 0.3)
                    pts = [center - half_width, center, center + half_width]
                    pts = [max(0.0, min(1.0, p)) for p in pts]
                    if len(set(pts)) < 3:
                        break
                    scores = []
                    for p in pts:
                        h_test = h.copy()
                        h_test[i] = p
                        h_test = _project(h_test, N / 2.0)
                        scores.append(_score(h_test))
                    scores = np.array(scores)
                    # Find minimum of quadratic through 3 points
                    if pts[1] != pts[0] and pts[2] != pts[1]:
                        x1, x2, x3 = pts
                        y1, y2, y3 = scores
                        # Quadratic interpolation
                        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
                        if abs(denom) > 1e-15:
                            a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
                            b = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom
                            if abs(a) > 1e-15:
                                x_min = -b / (2 * a)
                                if 0.0 < x_min < 1.0:
                                    h_test = h.copy()
                                    h_test[i] = x_min
                                    h_test = _project(h_test, N / 2.0)
                                    s_test = _score(h_test)
                                    if s_test < best_candidate_score - 1e-10:
                                        best_candidate_score = s_test
                                        best_candidate_val = x_min
                                        half_width *= 0.5
                                        continue
                                else:
                                    break
                        else:
                            break
                    else:
                        break
                    break
            
            if best_candidate_val is not None:
                h[i] = best_candidate_val
                h = _project(h, N / 2.0)
                best_score = best_candidate_score
                best_h = h.copy()
                improved = True
    
        if not improved:
            break
    
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Initializations: diverse starting points
    initializations = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N),
        0.5 + 0.2 * np.sin(4 * np.pi * np.arange(N) / N),
        0.5 + 0.25 * np.cos(2 * np.pi * np.arange(N) / N),
        0.5 + 0.15 * np.random.randn(N),
        # Additional structured inits
        0.5 + 0.2 * np.sin(6 * np.pi * np.arange(N) / N),
        0.5 + 0.15 * np.cos(3 * np.pi * np.arange(N) / N),
    ]
    
    inits = []
    for init in initializations:
        h_init = np.clip(init, 0.0, 1.0)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
        
    # Add random inits
    for _ in range(6):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
        
    # Phase 1: Nesterov with adaptive beta (standard smooth max)
    beta = 10.0
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 40:
            break
            
        for h_init in inits:
            if time.time() > _DEADLINE - 25:
                break
                
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=350, lr=0.05)
            
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
        beta *= 1.4
        
        # Also try a lower learning rate for finer convergence
        for h_init in inits[:3]:
            if time.time() > _DEADLINE - 25:
                break
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=250, lr=0.02)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
    # Phase 1.5: Spectral-space Nesterov (cosine basis, sum constraint naturally enforced)
    if best_h is not None and time.time() < _DEADLINE - 35:
        beta_spec = 12.0
        for r in range(4):
            if time.time() > _DEADLINE - 20:
                break
            # Use current best and a couple of inits as starting points
            spec_starts = [best_h]
            if len(inits) > 0:
                spec_starts.append(inits[0])
            if len(inits) > 2:
                spec_starts.append(inits[2])
                
            for sp in spec_starts:
                if time.time() > _DEADLINE - 18:
                    break
                h_opt, s_opt = _nesterov_spectral(sp, beta_spec, n_iter=300, lr=0.08)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
            beta_spec *= 1.35
            
    # Phase 2: Epigraph Nesterov with the best solution so far
    if best_h is not None and time.time() < _DEADLINE - 20:
        beta_ep = 15.0
        for r in range(3):
            if time.time() > _DEADLINE - 12:
                break
            # Use current best as starting point, plus perturbed versions
            start_points = [best_h]
            if len(inits) > 0:
                start_points.append(inits[0])
            if len(inits) > 1:
                start_points.append(inits[1])
                
            for sp in start_points:
                if time.time() > _DEADLINE - 10:
                    break
                h_opt, s_opt = _nesterov_epigraph(sp, beta_ep, n_iter=250, lr=0.03)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
            beta_ep *= 1.3
            
    # Phase 3: Alternating full-space and spectral refinement
    if best_h is not None and time.time() < _DEADLINE - 15:
        beta_alt = 20.0
        for r in range(4):
            if time.time() > _DEADLINE - 8:
                break
            # Full-space Nesterov
            h_opt, s_opt = _nesterov_accelerated_gradient(best_h, beta_alt, n_iter=200, lr=0.03)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
            # Spectral Nesterov
            if time.time() > _DEADLINE - 8:
                break
            h_opt, s_opt = _nesterov_spectral(best_h, beta_alt, n_iter=200, lr=0.06)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
            beta_alt *= 1.3
            
    # Phase 4: Coordinate descent polish
    if best_h is not None and time.time() < _DEADLINE - 8:
        h_pol, s_pol = _coordinate_descent_polish(best_h, max_outer=40, max_inner=N)
        if s_pol < best_score:
            best_score = s_pol
            best_h = h_pol.copy()
            
    # Phase 5: Final fine-grained coordinate descent with tighter tolerance
    if best_h is not None and time.time() < _DEADLINE - 3:
        h_pol2, s_pol2 = _coordinate_descent_polish(best_h, max_outer=25, max_inner=N)
        if s_pol2 < best_score:
            best_score = s_pol2
            best_h = h_pol2.copy()
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h, N / 2.0)
    return best_h
