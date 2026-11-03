import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def _time_left(deadline):
    return deadline - time.time()

def _ratio(h):
    h = np.asarray(h, dtype=float)
    l1 = float(np.sum(h))
    if l1 <= 0:
        return 0.0
    l2_sq = float(np.sum(h * h))
    c = np.convolve(h, h)
    peak = float(np.max(c))
    if peak <= 0:
        return 0.0
    return l2_sq / (l1 * peak)

def _softmax_c(h, tau):
    """Smooth surrogate for max of autoconvolution via log-sum-exp."""
    c = np.convolve(h, h)
    m = np.max(c)
    if m < 0:
        m = 0.0
    z = c / tau
    # numerical stability
    zmax = np.max(z)
    return m + tau * (zmax + np.log(np.sum(np.exp(z - zmax))))

def _softmax_c_grad(h, tau):
    """Gradient of soft-max of autoconvolution w.r.t. h."""
    c = np.convolve(h, h)
    m = np.max(c)
    if m < 0:
        m = 0.0
    z = c / tau
    zmax = np.max(z)
    e = np.exp(z - zmax)
    w = e / np.sum(e)  # softmax weights, sums to 1
    # d(soft)/dh_i = (1/tau) * sum_k w_k * d c_k / d h_i = (1/tau) * sum_k w_k * 2 h_{k-i}
    grad = np.zeros(N)
    for k in range(2 * N - 1):
        wk = w[k]
        if wk == 0.0:
            continue
        for i in range(N):
            j = k - i
            if 0 <= j < N:
                grad[i] += wk * h[j]
    grad *= 2.0 / tau
    return grad

def _find_sidon_support(K, attempts=300, seed=None):
    if seed is not None:
        random.seed(seed)
    for _ in range(attempts):
        candidates = list(range(N))
        random.shuffle(candidates)
        S = []
        sums = set()
        for cand in candidates:
            if len(S) == K:
                break
            conflict = False
            new_sums = set()
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums or sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        if len(S) == K:
            return S
    return None

def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin for 150s kill

    best_h = None
    best_ratio = 0.0

    def consider(h):
        nonlocal best_h, best_ratio
        if h is None:
            return
        h = np.asarray(h, dtype=float)
        h = np.clip(h, 0.0, None)
        s = float(np.sum(h))
        if s <= 0:
            return
        h = h / s
        r = _ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()

    # Build initial guesses
    guesses = []

    # 1. Uniform
    h_uni = np.ones(N) / N
    guesses.append(h_uni.copy())

    # 2. Sidon supports (sparser, higher ratio)
    for K in range(14, 4, -1):
        if _time_left(deadline) < 20.0:
            break
        S = _find_sidon_support(K, attempts=200)
        if S is not None:
            h = np.zeros(N)
            h[S] = 1.0 / K
            guesses.append(h.copy())
            consider(h)

    # 3. Random sparse guesses
    rng = np.random.RandomState(42)
    for _ in range(30):
        if _time_left(deadline) < 10.0:
            break
        K_rand = int(rng.randint(5, 16))
        support = rng.choice(N, K_rand, replace=False)
        h = np.zeros(N)
        h[support] = 1.0 / K_rand
        guesses.append(h.copy())
        consider(h)

    # Objective: minimize soft-max of autoconvolution, with ||h||_1 = 1
    # Use log-variables to enforce h >= 0: h_i = exp(x_i)
    # Constraint: sum exp(x_i) = 1  ->  log(sum exp(x_i)) = 0
    # Objective: soft-max of autoconvolution of h = exp(x)

    def obj_log(x):
        h = np.exp(x)
        return _softmax_c(h, 0.05)

    def obj_log_grad(x):
        h = np.exp(x)
        g = _softmax_c_grad(h, 0.05)
        return g * h

    def cons_log(x):
        h = np.exp(x)
        return np.log(np.sum(h))

    def cons_log_grad(x):
        h = np.exp(x)
        s = np.sum(h)
        return h / s

    # Optimize each guess with L-BFGS-B
    for h0 in guesses:
        if _time_left(deadline) < 5.0:
            break
        # log-variables
        h0c = np.clip(h0, 1e-12, None)
        x0 = np.log(h0c)
        # normalize
        x0 = x0 - np.log(np.sum(np.exp(x0)))

        try:
            res = minimize(
                obj_log, x0, method='L-BFGS-B', jac=obj_log_grad,
                constraints=[{'type': 'eq', 'fun': cons_log, 'jac': cons_log_grad}],
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if res.x is not None:
                h_res = np.exp(res.x)
                consider(h_res)
        except Exception:
            pass

        # Also try a direct (non-log) L-BFGS-B with bounds, starting from h0
        if _time_left(deadline) < 5.0:
            break
        def obj_direct(h):
            return _softmax_c(h, 0.05)
        def obj_direct_grad(h):
            return _softmax_c_grad(h, 0.05)
        try:
            res = minimize(
                obj_direct, h0, method='L-BFGS-B', jac=obj_direct_grad,
                bounds=[(1e-12, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if res.x is not None:
                consider(res.x)
        except Exception:
            pass

    # Polish: high-precision refinement on the best solution
    if best_h is not None and _time_left(deadline) > 5.0:
        h0 = best_h.copy()
        h0c = np.clip(h0, 1e-12, None)
        x0 = np.log(h0c)
        x0 = x0 - np.log(np.sum(np.exp(x0)))
        for tau in [0.02, 0.01, 0.005]:
            if _time_left(deadline) < 3.0:
                break
            def obj_p(x):
                h = np.exp(x)
                return _softmax_c(h, tau)
            def obj_p_grad(x):
                h = np.exp(x)
                g = _softmax_c_grad(h, tau)
                return g * h
            try:
                res = minimize(
                    obj_p, x0, method='L-BFGS-B', jac=obj_p_grad,
                    constraints=[{'type': 'eq', 'fun': cons_log, 'jac': cons_log_grad}],
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
                )
                if res.x is not None:
                    h_res = np.exp(res.x)
                    consider(h_res)
                    x0 = np.log(np.clip(h_res, 1e-12, None))
                    x0 = x0 - np.log(np.sum(np.exp(x0)))
            except Exception:
                pass

    # Final fallback
    if best_h is None:
        best_h = np.ones(N) / N

    best_h = np.clip(best_h, 0.0, None)
    s = float(np.sum(best_h))
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
