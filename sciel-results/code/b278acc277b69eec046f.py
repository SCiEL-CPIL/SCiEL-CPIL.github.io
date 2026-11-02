"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
STEP = 1.0 / N  # interval width on [-1/4, 1/4]

# EVOLVE-BLOCK-START
def _autoconv(h):
    # discrete autoconvolution: c[k] = sum_i h[i]*h[k-i] over valid indices
    # h has length N, indices 0..N-1. c index from -(N-1) to (N-1).
    h = np.asarray(h, dtype=float)
    c = np.convolve(h, h, mode='full')
    return c

def _score_from_h(h):
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 <= 0:
        return -1.0
    l2sq = np.sum(h * h)
    c = _autoconv(h)
    sup = np.max(c)
    if sup <= 0:
        return -1.0
    return l2sq / (l1 * sup)

def _objective(v, tau):
    # v: log-heights. h = exp(v)
    h = np.exp(v)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2sq = np.sum(h * h)
    c = _autoconv(h)
    # soft-max surrogate for sup(c) = max c
    m = np.max(c)
    # log-sum-exp scaled: softmax = m + tau*log(sum(exp((c-m)/tau)))
    if tau <= 0:
        sup = m
    else:
        z = (c - m) / tau
        # guard overflow
        zmax = np.max(z)
        sup = m + tau * (zmax + np.log(np.sum(np.exp(z - zmax))))
    if sup <= 0:
        return 0.0
    return l2sq / (l1 * sup)

def _neg_obj(v, tau):
    return -_objective(v, tau)

def _optimize_from(v0, tau, deadline, maxiter=2000):
    from scipy.optimize import minimize
    t0 = time.time()
    try:
        res = minimize(_neg_obj, v0, args=(tau,), method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxfun': maxiter * 2,
                                'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
    except Exception:
        return v0
    if time.time() > deadline - 1.0:
        return res.x if hasattr(res, 'x') else v0
    return res.x

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = np.random.default_rng(0)

    # Candidate initial log-height profiles
    def tri_profile(a):
        # triangular: peak 1 at center, linear to 0 at ends, scaled by a
        i = np.arange(N)
        c = (N - 1) / 2.0
        prof = np.maximum(0.0, 1.0 - np.abs(i - c) / (c + 1e-9))
        return a * prof

    def gauss_profile(a, s):
        i = np.arange(N)
        c = (N - 1) / 2.0
        prof = np.exp(-0.5 * ((i - c) / s) ** 2)
        return a * prof

    inits = []
    # flat
    inits.append(np.zeros(N))
    # triangular variants
    inits.append(np.log(tri_profile(1.0, 0) + 1e-9))
    inits.append(np.log(tri_profile(0.7, 0) + 1e-9))
    inits.append(np.log(tri_profile(1.4, 0) + 1e-9))
    # gaussian variants
    for s in [3.0, 5.0, 7.0, 9.0]:
        inits.append(np.log(gauss_profile(1.0, s) + 1e-9))
        inits.append(np.log(gauss_profile(0.8, s) + 1e-9))
    # random smooth profiles
    for _ in range(6):
        base = np.random.randn(N)
        # smooth via cumsum / moving average
        kern = np.ones(5) / 5.0
        sm = np.convolve(base, kern, mode='same')
        # make unimodal-ish: add a bump
        c = (N - 1) / 2.0
        sm = sm + 1.5 * np.exp(-0.5 * ((np.arange(N) - c) / 6.0) ** 2)
        inits.append(sm)

    best_h = None
    best_val = -1.0

    def consider(v):
        nonlocal best_h, best_val
        h = np.exp(v)
        val = _score_from_h(h)
        if val > best_val:
            best_val = val
            best_h = h

    # Stage 1: coarse soft-max optimization from each init
    tau_coarse = 0.5
    for v0 in inits:
        if time.time() > deadline - 30.0:
            break
        v = _optimize_from(v0, tau_coarse, deadline, maxiter=1000)
        consider(v)

    # Stage 2: refine with tighter soft-max (near-exact sup) from best few
    # Keep top candidates by current score
    cand = []
    for v0 in inits:
        h = np.exp(v0)
        cand.append((_score_from_h(h), v0))
    # Also add stage-1 results implicitly via best_h
    # Re-optimize best_h's log with fine tau
    if best_h is not None:
        vbest = np.log(best_h + 1e-12)
        for tau_fine in [0.1, 0.02, 0.005]:
            if time.time() > deadline - 15.0:
                break
            vbest = _optimize_from(vbest, tau_fine, deadline, maxiter=3000)
            consider(vbest)

    # Final high-precision polish: tighten tolerance, more iterations on the best
    if best_h is not None:
        vbest = np.log(best_h + 1e-12)
        vbest = _optimize_from(vbest, 0.001, deadline, maxiter=5000)
        consider(vbest)

    if best_h is None:
        return [1.0] * N

    # Ensure nonnegative and finite
    best_h = np.clip(best_h, 0.0, None)
    if not np.all(np.isfinite(best_h)):
        return [1.0] * N
    # Normalize scale-invariance is fine (ratio is scale-invariant), keep as is
    return list(best_h)
# EVOLVE-BLOCK-END
