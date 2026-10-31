"""Optimize equal-width step heights to minimize max overlap of h with shifted 1-h."""
import numpy as np
import time

N = 100
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _max_overlap(h):
    """Exact max overlap: max_k sum_i h[i]*(1-h[i-k]) for all shifts k."""
    one_minus = 1.0 - h
    best = 0.0
    n = len(h)
    for k in range(n):
        a = h[k:]
        b = one_minus[:n - k]
        ov = np.dot(a, b)
        if ov > best:
            best = ov
    return best


def _logsumexp(vals, beta):
    m = np.max(vals)
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta


def _objective(h, beta):
    """Smooth surrogate for max overlap using log-sum-exp."""
    one_minus = 1.0 - h
    n = len(h)
    vals = np.empty(n)
    for k in range(n):
        a = h[k:]
        b = one_minus[:n - k]
        vals[k] = np.dot(a, b)
    return _logsumexp(vals, beta)


def _project(h, target_sum):
    """Project onto {0<=x<=1, sum x = target_sum} via bisection on Lagrange multiplier."""
    lo, hi = -100.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    mid = (lo + hi) / 2.0
    return np.clip(h - mid, 0.0, 1.0)


def _grad_objective(h, beta):
    """Gradient of log-sum-exp surrogate."""
    one_minus = 1.0 - h
    n = len(h)
    vals = np.empty(n)
    for k in range(n):
        a = h[k:]
        b = one_minus[:n - k]
        vals[k] = np.dot(a, b)
    m = np.max(vals)
    exps = np.exp(beta * (vals - m))
    weights = exps / np.sum(exps)

    # d/dh_i of vals[k] = sum_j h[j]*(1-h[j-k])
    # = (1-h[i-k]) [if i>=k, from h[i] term] + (-h[i]) [if i-k>=0, i.e. i>=k, from (1-h[i-k]) term]
    # Actually: vals[k] = sum_{j=k}^{n-1} h[j]*(1-h[j-k])
    # d/dh[i]: term h[i]*(1-h[i-k]) contributes (1-h[i-k]) if i>=k
    #          term h[i+k]*(1-h[i]) contributes -h[i+k] if i+k < n
    grad = np.zeros(n)
    for k in range(n):
        w = weights[k]
        # contribution from h[i] for i in [k, n-1]: (1-h[i-k])
        grad[k:] += w * (1.0 - h[:n-k])
        # contribution from (1-h[i]) for i in [0, n-k-1]: -h[i+k]
        grad[:n-k] += w * (-h[k:])
    return grad


def _gradient_descent(h0, beta, steps=2000, lr_init=0.01):
    h = h0.copy()
    target = N / 2.0
    best_h = h.copy()
    best_obj = _objective(h, beta)
    lr = lr_init
    for it in range(steps):
        if time.time() > DEADLINE - 5.0:
            break
        g = _grad_objective(h, beta)
        h_new = h - lr * g
        h_new = _project(h_new, target)
        obj = _objective(h_new, beta)
        if obj < best_obj:
            best_obj = obj
            best_h = h_new.copy()
        h = h_new
        if it % 50 == 0:
            lr *= 0.95
    return best_h, best_obj


def construct():
    target = N / 2.0
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)

    rng = np.random.default_rng(42)
    n_restarts = 20
    betas = [5.0, 10.0, 20.0, 50.0]

    for r in range(n_restarts):
        if time.time() > DEADLINE - 10.0:
            break
        for beta in betas:
            if time.time() > DEADLINE - 10.0:
                break
            # Initialize: mix of random and structured
            if r % 4 == 0:
                h0 = np.full(N, 0.5)
            elif r % 4 == 1:
                h0 = np.clip(0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N * (r + 1)), 0, 1)
                h0 = _project(h0, target)
            elif r % 4 == 2:
                h0 = rng.uniform(0, 1, N)
                h0 = _project(h0, target)
            else:
                h0 = np.clip(0.5 + 0.4 * rng.standard_normal(N), 0, 1)
                h0 = _project(h0, target)

            h, obj = _gradient_descent(h0, beta, steps=1500, lr_init=0.01)
            score = _max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()

    # Final polish: more steps, higher beta
    if time.time() < DEADLINE - 15.0:
        h, obj = _gradient_descent(best_h, 100.0, steps=3000, lr_init=0.005)
        score = _max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Ensure exact sum
    best_h = _project(best_h, target)
    return best_h
