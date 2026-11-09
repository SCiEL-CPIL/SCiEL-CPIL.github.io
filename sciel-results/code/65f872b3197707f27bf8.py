import time
import numpy as np

N = 50
_DEADLINE = 130.0  # safety margin below the 150s kill

def _autoconv_supremum(h):
    """Supremum of the autoconvolution f*f for step heights h on N equal intervals.
    The k-th lag (k in 0..N-1) contributes sum_i h[i]*h[i+k]; the max over k is the sup."""
    n = len(h)
    best = 0.0
    for k in range(n):
        s = np.dot(h[:n-k], h[k:])
        if s > best:
            best = s
    return best

def _score(h):
    h = np.asarray(h, dtype=float)
    if h.size != N or np.any(h < -1e-12):
        return 0.0
    h = np.clip(h, 0.0, None)
    l2sq = float(np.dot(h, h))
    if l2sq <= 0.0:
        return 0.0
    l1 = float(np.sum(h))
    sup = _autoconv_supremum(h)
    if l1 <= 0.0 or sup <= 0.0:
        return 0.0
    return l2sq / (l1 * sup)

def _initial_candidates():
    cands = []
    cands.append(np.ones(N))
    cands.append(np.full(N, 0.5))
    # a few structured guesses
    cands.append(np.sin(np.linspace(0, np.pi, N)) ** 2 + 1e-6)
    cands.append(np.cos(np.linspace(-np.pi/2, np.pi/2, N)) ** 2 + 1e-6)
    cands.append(np.concatenate([np.ones(N//2), np.zeros(N - N//2)]))
    cands.append(np.concatenate([np.zeros(N//2), np.ones(N - N//2)]))
    # random
    rng = np.random.default_rng(0)
    for _ in range(8):
        cands.append(rng.random(N) + 0.05)
    return cands

def construct():
    t0 = time.time()
    best_h = None
    best_s = -1.0

    for c in _initial_candidates():
        c = np.clip(np.asarray(c, dtype=float), 0.0, None)
        s = _score(c)
        if s > best_s:
            best_s = s
            best_h = c.copy()

    # Try a global search with differential evolution if we have time.
    try:
        from scipy.optimize import differential_evolution
        def _neg(x):
            return -_score(x)
        bounds = [(0.0, 1.0)] * N
        popsize = 20
        # limit maxiter so we don't blow the budget; we'll check time in the callback
        stop = [False]
        def _callback(xk, convergence):
            if time.time() > _DEADLINE:
                stop[0] = True
            return stop[0]
        # start from a few seeds to help DE
        init = []
        for c in _initial_candidates()[:5]:
            init.append(np.clip(c, 0.0, 1.0))
        res = differential_evolution(
            _neg, bounds,
            maxiter=400, popsize=popsize,
            init=init,
            tol=1e-8,
            polish=True,
            callback=_callback,
            seed=12345,
            updating='deferred',
            x0=None,
        )
        if res.x is not None:
            cand = np.clip(res.x, 0.0, None)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best_h = cand.copy()
    except Exception:
        pass

    # Local refinement with L-BFGS-B on the negative score, if time remains.
    if best_h is not None and time.time() < _DEADLINE - 5:
        try:
            from scipy.optimize import minimize
            def _neg2(x):
                return -_score(x)
            best_local = best_h.copy()
            best_local_s = best_s
            # a couple of restarts from scaled versions
            for scale in (1.0, 0.7, 1.3):
                if time.time() > _DEADLINE - 3:
                    break
                x0 = np.clip(best_h * scale, 0.0, None)
                res = minimize(
                    _neg2, x0,
                    method='L-BFGS-B',
                    bounds=[(0.0, None)] * N,
                    options={'maxiter': 200, 'ftol': 1e-12},
                )
                if res.x is not None:
                    cand = np.clip(res.x, 0.0, None)
                    s = _score(cand)
                    if s > best_local_s:
                        best_local_s = s
                        best_local = cand.copy()
            if best_local_s > best_s:
                best_s = best_local_s
                best_h = best_local.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)
    # Return as a plain sequence of floats.
    return [float(v) for v in best_h]
