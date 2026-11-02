"""Second autocorrelation inequality: maximize (||h||_2^2)/(||h||_1 * ||h*h||_inf)
over nonnegative step heights h in R^N, N=50. Scale-invariant, so fix sum(h)=1.
Ratio >= 1 always (Cauchy-Schwarz on each lag); improve by flattening the
autoconvolution so max_k (h*h)_k is small relative to sum h_i^2."""

import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 150.0          # wall-clock kill by evaluator
SAFETY = 5.0              # return this many seconds before the limit
START = time.time()

def _remaining():
    return DEADLINE - SAFETY - (time.time() - START)

def _autoconv(h):
    # full linear convolution of h with itself (length 2N-1)
    return np.convolve(h, h, mode='full')

def _ratio(h):
    S = h.sum()
    if S <= 0:
        return 0.0
    c = _autoconv(h)
    peak = c.max()
    if peak <= 0:
        return 0.0
    return (h @ h) / (S * peak)

def _smooth_neg_ratio(h, beta):
    # minimize this = maximize ratio, using soft-max surrogate for max_k c_k
    S = h.sum()
    if S <= 1e-12:
        return 1e6
    c = _autoconv(h)
    # soft-max of c
    m = c.max()
    logsumexp = m + np.log(np.sum(np.exp(c - m)))
    # objective: log(softmax c) - log(sum h_i^2)  (S fixed = 1 by constraint handled outside)
    sq = h @ h
    if sq <= 1e-12:
        return 1e6
    return logsumexp - np.log(sq)

def _sinkhorn_like(h0, iters, beta):
    """Multiplicative min-max updates: attacker = soft-max over lags,
    defender = push mass in h away from lags producing high peaks."""
    h = h0.copy()
    h = np.clip(h, 1e-9, None)
    h /= h.sum()
    for it in range(iters):
        if _remaining() < 2.0:
            break
        c = _autoconv(h)                 # length 2N-1
        m = c.max()
        w = np.exp(beta * (c - m))       # attacker weights over lags
        w /= w.sum()
        # For each height index i, the lags k where h[i] participates in c[k]
        # are k = i (self, c_k includes h[i]*h[i]) and k = i+j, j = i-j for j in range.
        # Approximate per-index pressure: sum over lags k of w_k * (d c_k / d h_i)
        # d c_k / d h_i = h_{i+k} + h_{i-k} (with bounds), i.e. the convolution partner.
        # Build pressure vector p_i = sum_k w_k * (h_{i+k} + h_{i-k})_clipped
        p = np.zeros(N)
        for k in range(2 * N - 1):
            wk = w[k]
            if wk < 1e-12:
                continue
            # indices i such that i+k in [0,N-1] and i-k in [0,N-1]
            # c_k = sum_{i} h_i h_{i+k} where i in [0, N-1-k]
            # d c_k/d h_i = h_{i+k} (i in [0,N-1-k]) + h_{i-k} (i in [k, N-1])
            if k == 0:
                p += 2.0 * wk * h
            else:
                lo = np.arange(0, N - k)
                hi = np.arange(k, N)
                p[lo] += wk * h[lo + k]
                p[hi] += wk * h[hi - k]
        # multiplicative update: reduce h where pressure is high
        # h <- h * exp(-eta * p)  (eta small)
        eta = 0.5
        h = h * np.exp(-eta * p)
        h = np.clip(h, 1e-9, None)
        h /= h.sum()
    return h

def _polish(h0, beta, maxiter):
    """High-precision polish: scipy minimize of smooth objective with nonnegativity."""
    h0 = np.clip(h0, 1e-9, None)
    h0 = h0 / h0.sum()
    cons = [{'type': 'eq', 'fun': lambda x: x.sum() - 1.0,
             'jac': lambda x: np.ones_like(x)}]
    best_x, best_f = h0.copy(), _smooth_neg_ratio(h0, beta)
    for b in ([beta, beta * 2.0] if beta * 2.0 < 1e4 else [beta]):
        if _remaining() < 3.0:
            break
        try:
            res = minimize(_smooth_neg_ratio, best_x.copy(), args=(b,),
                           method='SLSQP', bounds=[(1e-9, None)] * N,
                           constraints=cons,
                           options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.success or (res.fun < best_f - 1e-12):
                cand = np.clip(res.x, 1e-9, None)
                cand /= cand.sum()
                if _ratio(cand) > _ratio(best_x):
                    best_x = cand
                    best_f = res.fun
        except Exception:
            pass
    return best_x

def construct():
    t0 = time.time()
    deadline = DEADLINE - SAFETY
    rng = np.random.default_rng(12345)

    best = np.full(N, 1.0 / N)
    best_r = _ratio(best)

    def consider(h):
        nonlocal best, best_r
        h = np.clip(h, 0.0, None)
        if h.sum() <= 0:
            return
        h = h / h.sum()
        r = _ratio(h)
        if r > best_r:
            best, best_r = h.copy(), r

    # Phase 1: Sinkhorn-style multiplicative min-max from several inits
    inits = [
        np.full(N, 1.0 / N),
        np.ones(N) / N,
        np.abs(rng.standard_normal(N)) + 0.1,
        # two separated blocks
        np.concatenate([np.ones(N // 2), np.zeros(N - N // 2)]),
        # geometric
        np.power(0.85, np.arange(N)),
    ]
    for h0 in inits:
        if time.time() > deadline - 15.0:
            break
        h0 = np.clip(h0, 1e-9, None); h0 /= h0.sum()
        for beta in [4.0, 8.0, 16.0]:
            if time.time() > deadline - 10.0:
                break
            h = _sinkhorn_like(h0, iters=200, beta=beta)
            consider(h)

    # Phase 2: smooth polish on the best so far
    if time.time() < deadline - 5.0:
        h = _polish(best, beta=16.0, maxiter=400)
        consider(h)
        if time.time() < deadline - 3.0:
            h = _polish(best, beta=32.0, maxiter=300)
            consider(h)

    # Phase 3: tiny random-restart local polish if time remains
    while time.time() < deadline - 3.0:
        h0 = best + 0.02 * rng.standard_normal(N)
        h0 = np.clip(h0, 1e-9, None); h0 /= h0.sum()
        h = _polish(h0, beta=16.0, maxiter=150)
        consider(h)

    return [float(x) for x in best]
