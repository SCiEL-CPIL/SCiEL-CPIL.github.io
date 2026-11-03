"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines the current spectral + Nesterov design with the epigraph formulation
and FFT-accelerated gradient from the inspiration, plus a bisection-based
projection for tighter constraint handling and a final high-beta polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 138  # Safety margin before 150s kill


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
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto {x : 0 <= x_i <= 1, sum(x) = target_sum} via bisection
    on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0

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


def _fix_sum(h):
    """Alias for _project for backward compatibility."""
    return _project(h, N / 2.0)


def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k using FFT.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
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


def _nesterov_smooth_max(h0, beta, max_iter=500, lr_init=0.05):
    """
    Nesterov accelerated projected gradient on smooth max with adaptive
    backtracking line search.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)

    h_prev = h.copy()
    t_mom = 1.0
    lr_current = lr_init
    best_h = h.copy()
    best_score = _score(h)
    max_backtracks = 12

    for it in range(max_iter):
        if time.time() > _DEADLINE - 8:
            break

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom)) / 2.0
        momentum = (t_mom - 1.0) / t_new
        y = h + momentum * (h - h_prev)

        val_y, grad = _smooth_max_obj_grad(y, beta)

        step_lr = lr_current
        h_new = None
        for bt in range(max_backtracks):
            h_candidate = y - step_lr * grad
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                h_new = h_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            h_new = y - lr_current * 0.5 * grad
            h_new = _project(h_new, N / 2.0)
            lr_current *= 0.5

        h_prev = h
        h = h_new
        t_mom = t_new

        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    return best_h, best_score


def _nesterov_epigraph(h0, beta, max_iter=500, lr_init=0.03):
    """
    Nesterov accelerated gradient descent using epigraph formulation.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)
    t_val = _score(h)

    h_prev = h.copy()
    t_prev = t_val
    t_mom = 1.0
    lr_current = lr_init
    best_h = h.copy()
    best_score = _score(h)
    max_backtracks = 12

    for it in range(max_iter):
        if time.time() > _DEADLINE - 8:
            break

        t_mom_new = (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom)) / 2.0
        momentum = (t_mom - 1.0) / t_mom_new

        y_h = h + momentum * (h - h_prev)
        y_t = t_val + momentum * (t_val - t_prev)

        val_y, grad_h, grad_t = _epigraph_obj_grad(y_h, y_t, beta)

        step_lr = lr_current
        h_new = None
        t_new_val = None
        for bt in range(max_backtracks):
            h_candidate = y_h - step_lr * grad_h
            h_candidate = _project(h_candidate, N / 2.0)
            t_candidate = y_t - step_lr * grad_t
            t_candidate = max(0.0, t_candidate)

            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                h_new = h_candidate
                t_new_val = t_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            h_new = y_h - lr_current * 0.5 * grad_h
            h_new = _project(h_new, N / 2.0)
            t_new_val = max(0.0, y_t - lr_current * 0.5 * grad_t)
            lr_current *= 0.5

        h_prev = h
        t_prev = t_val
        h = h_new
        t_val = t_new_val
        t_mom = t_mom_new

        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    return best_h, best_score


def _spectral_h_from_coeffs(coeffs):
    """
    Reconstruct h from real DFT coefficients (modes 1..K).
    coeffs = [a1, b1, a2, b2, ..., aK, bK]
    h_i = 0.5 + sum_{k=1}^K (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    n_modes = len(coeffs) // 2
    h = np.full(N, 0.5)
    i = np.arange(N)
    for k in range(1, n_modes + 1):
        a = coeffs[2 * (k - 1)]
        b = coeffs[2 * (k - 1) + 1]
        h += a * np.cos(2.0 * np.pi * k * i / N)
        h += b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral_coeffs(h, K):
    """
    Extract the first K real DFT coefficients from h.
    Returns [a1, b1, ..., aK, bK]
    """
    H = np.fft.rfft(h)
    out = []
    for k in range(1, K + 1):
        a = 2.0 / N * H[k].real
        b = -2.0 / N * H[k].imag
        out.append(a)
        out.append(b)
    return out


def _optimize_spectral(h_init, K, beta, maxiter=200, lr_init=0.01):
    """
    Optimize in spectral coefficient space using L-BFGS-B with penalty for bounds.
    """
    coeffs0 = _h_to_spectral_coeffs(h_init, K)

    def obj(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = np.sum(np.maximum(0, -h) ** 2) + np.sum(np.maximum(0, h - 1) ** 2)
        val, _ = _smooth_max_obj_grad(h, beta)
        return val + 1.0 * penalty

    def grad(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        _, grad_h = _smooth_max_obj_grad(h, beta)

        penalty_grad = np.zeros(N)
        penalty_grad[h < 0] += -2.0 * h[h < 0]
        penalty_grad[h > 1] += 2.0 * (h[h > 1] - 1.0)

        total_grad_h = grad_h + penalty_grad

        n_modes = len(coeffs) // 2
        i = np.arange(N)
        grad_coeffs = np.zeros(len(coeffs))
        for k in range(1, n_modes + 1):
            cos_k = np.cos(2.0 * np.pi * k * i / N)
            sin_k = np.sin(2.0 * np.pi * k * i / N)
            grad_coeffs[2 * (k - 1)] = np.sum(total_grad_h * cos_k)
            grad_coeffs[2 * (k - 1) + 1] = np.sum(total_grad_h * sin_k)

        return grad_coeffs

    try:
        res = minimize(obj, coeffs0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < obj(coeffs0):
            h_opt = _spectral_h_from_coeffs(res.x)
            return h_opt
    except Exception:
        pass

    # Fallback: Nesterov in spectral space
    coeffs = np.array(coeffs0, dtype=np.float64, copy=True)
    coeffs_prev = coeffs.copy()
    t_k = 1.0
    lr = lr_init
    best_coeffs = coeffs.copy()
    best_val = obj(coeffs)

    for it in range(max(100, maxiter)):
        if time.time() > _DEADLINE - 8:
            break
        val = obj(coeffs)
        g = grad(coeffs)

        coeffs_new = coeffs - lr * g
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        coeffs_next = coeffs_new + ((t_k - 1.0) / t_new) * (coeffs_new - coeffs_prev)

        coeffs_prev = coeffs
        coeffs = coeffs_next
        t_k = t_new

        if it % 50 == 49:
            lr = max(0.001, lr * 0.95)

        v = obj(coeffs)
        if v < best_val:
            best_val = v
            best_coeffs = coeffs.copy()

    h_opt = _spectral_h_from_coeffs(best_coeffs)
    return h_opt


def _run_slsqp(h0, beta, maxiter=300, ftol=1e-12, gtol=1e-10):
    """
    Run SLSQP optimization on the smooth max objective.
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
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
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    except Exception:
        return h0


def _high_beta_polish(h0, n_iter=200, beta=300.0, lr_init=0.01):
    """
    Final high-precision polish pass with very high beta and small learning rate.
    Uses projected gradient descent with backtracking line search.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)

    best_h = h.copy()
    best_score = _score(h)
    lr = lr_init

    for i in range(n_iter):
        if time.time() > _DEADLINE - 3:
            break

        val, grad = _smooth_max_obj_grad(h, beta)
        
        # Backtracking line search
        step = lr
        h_new = None
        for bt in range(10):
            h_candidate = h - step * grad
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val + 1e-8:
                h_new = h_candidate
                break
            step *= 0.5
        
        if h_new is None:
            h_new = h - step * grad
            h_new = _project(h_new, N / 2.0)
            
        h = h_new
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
        # Decay learning rate slowly
        lr *= 0.99

    return best_h, best_score


def _coordinate_descent_polish(h0, beta=200.0, max_iter=300):
    """
    Coordinate-descent polish: iteratively optimize each coordinate while holding
    others fixed, using the smooth-max surrogate. This can escape from plateaus
    that gradient methods miss.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)
    best_h = h.copy()
    best_score = _score(h)

    for it in range(max_iter):
        if time.time() > _DEADLINE - 3:
            break

        improved = False
        for j in range(N):
            if time.time() > _DEADLINE - 3:
                break
            old_val = h[j]
            # Try a few candidate values around current
            candidates = [max(0.0, min(1.0, old_val + d)) for d in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]]
            best_cand = old_val
            best_cand_score = _score(h)

            for c in candidates:
                if abs(c - old_val) < 1e-10:
                    continue
                h[j] = c
                s = _score(h)
                if s < best_cand_score - 1e-10:
                    best_cand_score = s
                    best_cand = c
            h[j] = best_cand

            # Re-project to maintain sum constraint
            h = _project(h, N / 2.0)
            s = _score(h)
            if s < best_score - 1e-10:
                best_score = s
                best_h = h.copy()
                improved = True

        if not improved and it > 20:
            break

    return best_h, best_score


def _cellular_automaton_starts(n_starts=12, n_steps=20):
    """
    Generate diverse initial points using 1D cellular automata.
    A binary seed evolves under a random local rule; the result is smoothed
    via a temperature parameter to produce heights in [0,1] with sum N/2.
    """
    starts = []
    for _ in range(n_starts):
        # Random binary seed
        seed = np.random.randint(0, 2, N)

        # Apply CA steps with a random rule (rule = random mapping from 3-bit neighborhood to 0/1)
        ca_state = seed.copy()
        for _ in range(n_steps):
            left = np.roll(ca_state, 1)
            right = np.roll(ca_state, -1)
            # Random rule: use a fixed random threshold on weighted neighborhood
            w_l = np.random.uniform(0, 1)
            w_c = np.random.uniform(0, 1)
            w_r = np.random.uniform(0, 1)
            total_w = w_l + w_c + w_r + 1e-10
            weighted = (w_l * left + w_c * ca_state + w_r * right) / total_w
            threshold = np.random.uniform(0.3, 0.7)
            ca_state = (weighted > threshold).astype(np.float64)

        # Smooth the binary output with a temperature-like mixing factor
        # Mix the CA output with its moving average to get values in [0,1]
        kernel_size = np.random.choice([3, 5, 7, 9])
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(ca_state, kernel, mode='same')

        # Apply a sigmoid-like transformation with random temperature
        temp = np.random.uniform(0.3, 1.5)
        centered = smoothed - 0.5
        h_ca = 0.5 + 0.5 * np.tanh(centered / temp)

        h_ca = _project(h_ca, N / 2.0)
        starts.append(h_ca)

    return starts


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)

    # --- Phase 1: Diverse structured starts ---
    starts = []

    # Cosine fundamental
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_project(h_cos, N / 2.0))

    # Sine fundamental
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_project(h_sin, N / 2.0))

    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(_project(h_2f, N / 2.0))

    # Three-frequency
    h_3f = 0.5 + 0.25 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N) + 0.15 * np.cos(6.0 * np.pi * i / N)
    starts.append(_project(h_3f, N / 2.0))

    # Random sines
    for _ in range(15):
        f = np.random.uniform(1, 6)
        phase = np.random.uniform(0, 2 * np.pi)
        h_s = 0.5 + 0.5 * np.sin(2 * np.pi * f * i / N + phase)
        starts.append(_project(h_s, N / 2.0))

    # Random
    for _ in range(15):
        h_r = np.random.rand(N)
        starts.append(_project(h_r, N / 2.0))

    for h0 in starts:
        _update_best(h0)

    # --- Phase 2: Spectral optimization from best start ---
    for K in [2, 4, 8, 16, 32, 48]:
        if time.time() - t0 > _DEADLINE - 35:
            break
        h_spec = _optimize_spectral(best_h, K, beta=50.0, maxiter=200)
        _update_best(h_spec)

    # --- Phase 3: Nesterov smooth max refinement ---
    if best_h is not None and time.time() < _DEADLINE - 25:
        h_cand, s_cand = _nesterov_smooth_max(best_h, beta=50.0, max_iter=400, lr_init=0.05)
        _update_best(h_cand)

    if best_h is not None and time.time() < _DEADLINE - 20:
        h_cand2, s_cand2 = _nesterov_epigraph(best_h, beta=80.0, max_iter=300, lr_init=0.03)
        _update_best(h_cand2)

    # --- Phase 4: Random restarts with spectral polish ---
    if time.time() < _DEADLINE - 30:
        for _ in range(8):
            if time.time() - t0 > _DEADLINE - 20:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _project(h_rand, N / 2.0)
            h_spec_rand = _optimize_spectral(h_rand, K=8, beta=50.0, maxiter=100)
            _update_best(h_spec_rand)

    # --- Phase 4b: Cellular automaton starts with spectral polish ---
    if time.time() < _DEADLINE - 25:
        ca_starts = _cellular_automaton_starts(n_starts=10, n_steps=15)
        for h_ca in ca_starts:
            if time.time() - t0 > _DEADLINE - 15:
                break
            _update_best(h_ca)
            # Also try spectral optimization from each CA start
            if time.time() - t0 > _DEADLINE - 12:
                break
            h_spec_ca = _optimize_spectral(h_ca, K=6, beta=50.0, maxiter=80)
            _update_best(h_spec_ca)

    # --- Phase 5: SLSQP polish ---
    if best_h is not None and time.time() < _DEADLINE - 12:
        h_opt = _run_slsqp(best_h, beta=100.0, maxiter=300, ftol=1e-12, gtol=1e-10)
        _update_best(h_opt)

    # --- Phase 6: High-beta polish ---
    if best_h is not None and time.time() < _DEADLINE - 8:
        h_final, _ = _high_beta_polish(best_h, n_iter=200, beta=500.0, lr_init=0.01)
        _update_best(h_final)

    # --- Phase 6b: Coordinate descent polish ---
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_cd, _ = _coordinate_descent_polish(best_h, beta=200.0, max_iter=150)
        _update_best(h_cd)

    # --- Phase 7: Final SLSQP with very high beta ---
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_final2 = _run_slsqp(best_h, beta=500.0, maxiter=500, ftol=1e-14, gtol=1e-12)
        _update_best(h_final2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _project(best_h, N / 2.0)
    return best_h
