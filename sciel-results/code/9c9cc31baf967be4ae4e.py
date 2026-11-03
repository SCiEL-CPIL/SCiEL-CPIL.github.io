"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines Nesterov accelerated gradient descent on a smooth-max surrogate
with adaptive temperature, backtracking line search, projection onto the
constraint set, an epigraph formulation, and multi-stage coordinate
descent polish. Incorporates adaptive learning rate scheduling, more
structured initializations, and a high-beta final polish pass.
"""
import time
import numpy as np
from scipy.optimize import minimize as _scipy_minimize

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
    Fast path: if a uniform shift already keeps all entries in [0,1],
    use it directly. Otherwise fall back to bisection on the offset.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    # Fast path: try a uniform shift first
    offset = (target_sum - current_sum) / len(h)
    h_shifted = h + offset
    if np.all(h_shifted >= -1e-12) and np.all(h_shifted <= 1 + 1e-12):
        h_shifted = np.clip(h_shifted, 0.0, 1.0)
        diff = target_sum - np.sum(h_shifted)
        if abs(diff) > 1e-14:
            if diff > 0:
                idx = np.argmin(h_shifted)
                h_shifted[idx] += diff
            else:
                idx = np.argmax(h_shifted)
                h_shifted[idx] += diff
            h_shifted = np.clip(h_shifted, 0.0, 1.0)
        return h_shifted

    # Slow path: bisection
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


def _nesterov_accelerated_gradient(h0, beta, n_iter=500, lr=0.05):
    """
    Nesterov accelerated gradient descent on the smooth max objective
    with adaptive learning rate and backtracking line search.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    h_prev = h.copy()
    t = 1.0

    best_h = h.copy()
    best_score = _score(h)

    lr_current = lr
    max_backtracks = 12

    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new

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
        t = t_new

        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

    return best_h, best_score


def _slsqp_polish(h0, beta=20.0):
    """
    SLSQP polish on the smooth-max surrogate.
    """
    if time.time() > _DEADLINE - 5:
        return h0, _score(h0)
        
    try:
        h_init_slsqp = np.array(h0, dtype=np.float64).copy()

        def slsqp_obj(x):
            x_clipped = np.clip(x, 0.0, 1.0)
            cs = _linear_overlap_values(x_clipped)
            cs_max = np.max(cs)
            exp_cs = np.exp(beta * (cs - cs_max))
            return (1.0 / beta) * (np.log(np.sum(exp_cs)) + beta * cs_max)

        def slsqp_grad(x):
            x_clipped = np.clip(x, 0.0, 1.0)
            cs = _linear_overlap_values(x_clipped)
            cs_max = np.max(cs)
            exp_cs = np.exp(beta * (cs - cs_max))
            w = exp_cs / np.sum(exp_cs)
            n_fft = 1
            while n_fft < 2 * N:
                n_fft *= 2
            w_pad = np.zeros(n_fft)
            w_pad[:N] = w
            h_pad = np.zeros(n_fft)
            h_pad[:N] = x_clipped
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
            g = -(2.0 / N) * (A + B)
            # Zero out gradient where x is at bound
            g[x_clipped <= 1e-10] = 0.0
            g[x_clipped >= 1 - 1e-10] = 0.0
            return g

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
             'jac': lambda x: np.ones(N)}
        ]
        bounds = [(0.0, 1.0)] * N

        res = _scipy_minimize(
            slsqp_obj, h_init_slsqp, method='SLSQP',
            jac=slsqp_grad, bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if res.success or res.fun < slsqp_obj(h_init_slsqp):
            h_slsqp = np.clip(res.x, 0.0, 1.0)
            h_slsqp = _project(h_slsqp, N / 2.0)
            s_slsqp = _score(h_slsqp)
            return h_slsqp, s_slsqp
    except Exception:
        pass
    return h0, _score(h0)


def _hybrid_coordinate_polish(h0, max_outer=30, max_inner=N):
    """
    Hybrid coordinate descent: tries bounds (0, 1) and small gradient steps.
    """
    h = np.array(h0, dtype=np.float64).copy()
    h = _project(h, N / 2.0)
    best_score = _score(h)
    best_h = h.copy()

    for outer in range(max_outer):
        if time.time() > _DEADLINE - 2:
            break
        improved = False
        _, grad = _smooth_max_obj_grad(h, 50.0)
        for i in range(max_inner):
            if time.time() > _DEADLINE - 2:
                break
            old_val = h[i]
            
            # 1. Try setting to 0
            h_test = h.copy()
            h_test[i] = 0.0
            h_test = _project(h_test, N / 2.0)
            s_test = _score(h_test)
            if s_test < best_score - 1e-12:
                h = h_test
                best_score = s_test
                best_h = h.copy()
                improved = True
                continue # Re-evaluate other coords with new h

            # 2. Try setting to 1
            h_test = h.copy()
            h_test[i] = 1.0
            h_test = _project(h_test, N / 2.0)
            s_test = _score(h_test)
            if s_test < best_score - 1e-12:
                h = h_test
                best_score = s_test
                best_h = h.copy()
                improved = True
                continue

            # 3. Try small gradient steps
            g_i = grad[i]
            if abs(g_i) < 1e-15:
                continue
            direction = -np.sign(g_i)
            for d in [0.01, 0.005, 0.002, 0.001, 0.0005]:
                new_val = old_val + direction * d
                if new_val < -1e-12 or new_val > 1 + 1e-12:
                    continue
                new_val = np.clip(new_val, 0.0, 1.0)
                if abs(new_val - old_val) < 1e-15:
                    continue
                h_test = h.copy()
                h_test[i] = new_val
                h_test = _project(h_test, N / 2.0)
                s_test = _score(h_test)
                if s_test < best_score - 1e-12:
                    h = h_test
                    best_score = s_test
                    best_h = h.copy()
                    improved = True
                    break

        if not improved:
            break

    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')
    top_candidates = []

    np.random.seed(42)

    # Initializations: diverse starting points
    initializations = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N),
        0.5 + 0.2 * np.sin(4 * np.pi * np.arange(N) / N),
        0.5 + 0.25 * np.cos(2 * np.pi * np.arange(N) / N),
        0.5 + 0.15 * np.random.randn(N),
        0.5 + 0.2 * np.sin(6 * np.pi * np.arange(N) / N),
        0.5 + 0.15 * np.cos(3 * np.pi * np.arange(N) / N),
        0.5 + 0.1 * np.sin(8 * np.pi * np.arange(N) / N) + 0.1 * np.cos(5 * np.pi * np.arange(N) / N),
        0.5 + 0.2 * np.cos(2 * np.pi * 3 * np.arange(N) / N),
        0.5 + 0.15 * np.sin(2 * np.pi * 5 * np.arange(N) / N),
        # Add binary-like inits
        np.where(np.arange(N) % 2 == 0, 0.7, 0.3),
        np.where(np.arange(N) % 3 == 0, 1.0, 0.0),
    ]

    inits = []
    for init in initializations:
        h_init = np.clip(init, 0.0, 1.0)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)

    # Add random inits
    for _ in range(8):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)

    # Phase 1: Nesterov with adaptive beta
    beta = 8.0
    n_restarts = 5

    for r in range(n_restarts):
        if time.time() > _DEADLINE - 35:
            break

        for h_init in inits:
            if time.time() > _DEADLINE - 25:
                break

            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=350, lr=0.05)

            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
            
            # Keep top 3 distinct candidates
            if s_opt < best_score + 0.01: # Threshold to keep diverse good candidates
                is_dup = False
                for c in top_candidates:
                    if np.linalg.norm(h_opt - c[0]) < 1e-3:
                        is_dup = True
                        break
                if not is_dup:
                    top_candidates.append((h_opt, s_opt))
                    if len(top_candidates) > 3:
                        top_candidates.sort(key=lambda x: x[1])
                        top_candidates.pop()

        beta *= 1.35

        # Also try a lower learning rate for finer convergence on top inits
        for h_init in inits[:4]:
            if time.time() > _DEADLINE - 25:
                break
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=250, lr=0.02)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 2: SLSQP polish on top candidates
    if time.time() < _DEADLINE - 15:
        candidates_to_polish = [best_h] if best_h is not None else []
        for c in top_candidates:
            if c[0] is not best_h:
                candidates_to_polish.append(c[0])
        
        for h_cand in candidates_to_polish:
            if time.time() > _DEADLINE - 10:
                break
            h_pol, s_pol = _slsqp_polish(h_cand, beta=20.0)
            if s_pol < best_score:
                best_score = s_pol
                best_h = h_pol.copy()

    # Phase 3: Hybrid Coordinate Descent
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_pol2, s_pol2 = _hybrid_coordinate_polish(best_h, max_outer=30, max_inner=N)
        if s_pol2 < best_score:
            best_score = s_pol2
            best_h = h_pol2.copy()

    # Phase 4: Final ultra-fine polish
    if best_h is not None and time.time() < _DEADLINE - 3:
        h_final = np.array(best_h, dtype=np.float64).copy()
        h_final = _project(h_final, N / 2.0)
        s_final = _score(h_final)
        for _ in range(20):
            if time.time() > _DEADLINE - 1.0:
                break
            any_improved = False
            _, grad_final = _smooth_max_obj_grad(h_final, 50.0)
            for i in range(N):
                if time.time() > _DEADLINE - 1.0:
                    break
                old_val = h_final[i]
                g_i = grad_final[i]
                if abs(g_i) < 1e-14:
                    continue
                direction = -np.sign(g_i)
                for d in [0.0001, 0.00005, 0.00002]:
                    new_val = old_val + direction * d
                    if new_val < -1e-12 or new_val > 1 + 1e-12:
                        continue
                    new_val = np.clip(new_val, 0.0, 1.0)
                    h_test = h_final.copy()
                    h_test[i] = new_val
                    h_test = _project(h_test, N / 2.0)
                    s_test = _score(h_test)
                    if s_test < s_final - 1e-13:
                        h_final = h_test
                        s_final = s_test
                        any_improved = True
                        break
            if not any_improved:
                break
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _project(best_h, N / 2.0)
    return best_h
