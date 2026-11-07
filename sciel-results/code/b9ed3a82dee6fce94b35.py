"""Refined program: vectorized gradient + smooth soft-max surrogate + L-BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
M = 2 * N - 1          # length of g = convolve(h,h)
W = 1.0 / (M + 1)      # spacing w = (0.5-(-0.5))/(m+2-1) = 1/(m+1)
_START = time.time()
_DEADLINE = _START + 110  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)              # length M
    y = np.zeros(M + 2)
    y[1:-1] = g
    y2 = y * y
    l2 = (W / 3.0) * np.sum(y2[:-1] + y[:-1] * y[1:] + y2[1:])
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    norm_inf = np.max(np.abs(g))
    denom = norm_1 * norm_inf
    if denom <= 0:
        return 0.0
    return l2 / denom

def _value_soft(h, tau):
    """Smooth value using log-sum-exp surrogate for norm_inf (tau>0)."""
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    y = np.zeros(M + 2)
    y[1:-1] = g
    y2 = y * y
    l2 = (W / 3.0) * np.sum(y2[:-1] + y[:-1] * y[1:] + y2[1:])
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    if norm_1 <= 0:
        return 0.0
    # soft-max of |g|: tau * logsumexp(|g|/tau)
    ag = np.abs(g)
    amax = np.max(ag)
    if amax <= 0:
        return 0.0
    norm_inf_soft = tau * (np.log(np.sum(np.exp((ag - amax) / tau))) + amax / tau)
    denom = norm_1 * norm_inf_soft
    if denom <= 0:
        return 0.0
    return l2 / denom

def _value_grad(h, tau=0.0):
    """Return (value, grad) of value wrt h.
    If tau>0, use smooth soft-max surrogate; else exact (sub)gradient."""
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)              # length M
    y = np.zeros(M + 2)
    y[1:-1] = g
    y2 = y * y
    l2 = (W / 3.0) * np.sum(y2[:-1] + y[:-1] * y[1:] + y2[1:])
    norm_1 = np.sum(np.abs(g)) / (M + 1)

    if tau > 0:
        ag = np.abs(g)
        amax = np.max(ag)
        if amax <= 0:
            return 0.0, np.zeros_like(h)
        norm_inf_soft = tau * (np.log(np.sum(np.exp((ag - amax) / tau))) + amax / tau)
        # d norm_inf_soft / d ag = softmax weights
        wsoft = np.exp((ag - amax) / tau)
        wsoft /= np.sum(wsoft)
        dnh_dg = wsoft * np.sign(g)
        denom = norm_1 * norm_inf_soft
        val = l2 / denom
    else:
        norm_inf = np.max(np.abs(g))
        if norm_inf <= 0:
            return 0.0, np.zeros_like(h)
        imax = int(np.argmax(np.abs(g)))
        dnh_dg = np.zeros(M)
        dnh_dg[imax] = np.sign(g[imax])
        denom = norm_1 * norm_inf
        val = l2 / denom

    # d l2 / d y  (vectorized)
    dly = np.zeros(M + 2)
    a = W / 3.0
    # for i in 0..M: term i contributes a*(2y[i]+y[i+1]) to dly[i] and a*(y[i]+2y[i+1]) to dly[i+1]
    dly[:-1] += a * (2 * y[:-1] + y[1:])
    dly[1:]  += a * (y[:-1] + 2 * y[1:])
    dlg = dly[1:-1]                     # d l2 / d g[k]

    # d l2 / d h[j] = sum_k dlg[k] * (h[j-k] 1[0<=j-k<N] + h[j+k] 1[0<=j+k<N])
    # This equals (dlg * h)[j] + (dlg_rev * h)[j] where dlg_rev[k]=dlg[-1-k]
    conv1 = np.convolve(dlg, h, mode='full')      # length 2N-1 = M
    conv2 = np.convolve(dlg[::-1], h, mode='full')
    dgh = conv1[:N] + conv2[N - 1:N - 1 + N]
    # check lengths: conv1 length M=2N-1, we need indices 0..N-1
    # conv2 length M, we need indices (N-1)..(2N-2)
    dgh = conv1[:N] + conv2[N-1:]

    # d norm_1 / d g[k] = sign(g[k])/(M+1)
    dn1_dg = np.sign(g) / (M + 1)
    dnh1 = np.convolve(dn1_dg, h, mode='full')[:N] + np.convolve(dn1_dg[::-1], h, mode='full')[N-1:]
    dnhinf = np.convolve(dnh_dg, h, mode='full')[:N] + np.convolve(dnh_dg[::-1], h, mode='full')[N-1:]

    if tau > 0:
        norm_inf = norm_inf_soft
    else:
        norm_inf = np.max(np.abs(g))

    ddenom = dnh1 * norm_inf + norm_1 * dnhinf
    grad_val = dgh / denom - l2 * ddenom / (denom * denom)
    return val, grad_val

def _project(h):
    h = np.clip(h, 0.0, None)
    s = np.sum(h)
    if s <= 0:
        return np.ones(N) / N
    return h / s

def _grad_ascent(h0, max_iter=200):
    """Projected gradient ascent with backtracking on the smooth (tau-annealed) objective."""
    h = _project(h0.copy())
    best_h = h.copy()
    best_v = _value(h)
    tau = 1.0
    lr = 0.1
    for it in range(max_iter):
        if _time_left() < 5:
            break
        # anneal tau down
        tau = max(1e-3, 1.0 / (1.0 + it))
        val, gv = _value_soft(h, tau)
        # backtracking line search on the smooth objective
        step = lr
        improved = False
        for _ in range(12):
            h_new = _project(h + step * gv)
            vn = _value_soft(h_new, tau)
            if vn > val:
                h = h_new
                val = vn
                improved = True
                break
            step *= 0.5
        if improved:
            v_exact = _value(h)
            if v_exact > best_v:
                best_v = v_exact
                best_h = h.copy()
            lr = min(step * 1.1, 1.0)
        else:
            lr *= 0.7
            if lr < 1e-5:
                break
    # final exact polish
    v_exact = _value(best_h)
    if v_exact > best_v:
        best_v = v_exact
    return best_h, best_v

def _lbfgs_polish(h0, max_iter=300):
    """L-BFGS-B polish on the smooth objective with small tau, then exact."""
    h = _project(h0.copy())
    tau = 1e-3
    def neg_obj(x):
        x = np.clip(x, 0, None)
        s = np.sum(x)
        if s <= 0:
            return 1e6
        x = x / s
        return -_value_soft(x, tau)
    def neg_grad(x):
        x = np.clip(x, 0, None)
        s = np.sum(x)
        if s <= 0:
            return np.zeros(N)
        x = x / s
        _, gv = _value_soft(x, tau)
        return -gv
    try:
        res = minimize(neg_obj, h, method='L-BFGS-B', jac=neg_grad,
                       options={'maxiter': max_iter, 'gtol': 1e-8, 'maxls': 20})
        h2 = _project(res.x)
        v2 = _value(h2)
        if v2 > _value(h):
            return h2, v2
    except Exception:
        pass
    return h, _value(h)

def _optimize():
    rng = np.random.default_rng(0)
    # diverse starts
    inits = [
        np.ones(N),
        np.linspace(0, 1, N),
        rng.uniform(0, 1, N),
        np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 5.0 ** 2)),
        np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 8.0 ** 2)),
        np.sin(np.pi * np.arange(N) / N) + 0.1,
        np.concatenate([np.ones(N // 2), np.zeros(N - N // 2)]),
        np.concatenate([np.zeros(N // 3), np.ones(N - 2 * (N // 3)), np.zeros(N // 3)]),
    ]
    best_h = None
    best_v = -1.0
    for h0 in inits:
        if _time_left() < 10:
            break
        h0 = np.clip(np.asarray(h0, float), 0, None)
        if not np.all(np.isfinite(h0)):
            continue
        s = np.sum(h0)
        if s <= 0:
            continue
        h0 = h0 / s
        # gradient ascent with annealing
        h1, v1 = _grad_ascent(h0, max_iter=150)
        if v1 > best_v:
            best_v = v1
            best_h = h1.copy()
        # L-BFGS polish on the best so far from this start
        h2, v2 = _lbfgs_polish(h1, max_iter=200)
        if v2 > best_v:
            best_v = v2
            best_h = h2.copy()
    # final polish on global best
    if best_h is not None and _time_left() > 5:
        h3, v3 = _lbfgs_polish(best_h, max_iter=400)
        if v3 > best_v:
            best_v = v3
            best_h = h3.copy()
    if best_h is None:
        best_h = np.ones(N)
        best_v = _value(best_h)
    return best_h, best_v

def construct():
    try:
        h, v = _optimize()
        h = np.asarray(h, float)
        h = np.clip(h, 0, None)
        if not np.all(np.isfinite(h)):
            h = np.ones(N)
        s = np.sum(h)
        if s <= 0:
            h = np.ones(N)
        return list(h)
    except Exception:
        return [1.0] * N
