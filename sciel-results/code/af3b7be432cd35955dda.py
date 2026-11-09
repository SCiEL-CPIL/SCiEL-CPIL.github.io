# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize
from scipy.special import comb
import concurrent.futures

_START_TIME = time.time()

class _EvalStruct:
    """Precompute pair-sum structure for a fixed support; reuse across weight changes."""
    __slots__ = ('n', 'support', 'pair_sums', 'order', 'cum_p', 'sorted_probs', 'cdf_lookup', 'P')

    def __init__(self, support):
        support = np.ascontiguousarray(support, dtype=np.float64)
        self.support = support
        self.n = n = len(support)
        ps = (support[:, None] + support[None, :]).ravel()
        self.pair_sums = ps
        self.order = np.argsort(ps, kind='stable')
        self.sorted_probs = np.empty(n * n, dtype=np.float64)
        self.cum_p = np.empty(n * n, dtype=np.float64)
        self.cdf_lookup = np.empty(n * n, dtype=np.float64)
        self.P = 0.0

    def update(self, p):
        p = np.ascontiguousarray(p, dtype=np.float64)
        self.sorted_probs = (p[:, None] * p[None, :]).ravel()[self.order]
        self.cum_p = np.cumsum(self.sorted_probs)
        self.cdf_lookup = np.empty_like(self.cum_p)
        self.cdf_lookup[0] = 0.0
        if len(self.cum_p) > 1:
            self.cdf_lookup[1:] = self.cum_p[:-1]
        P = np.sum((p[:, None] * p[None, :]) * self.cdf_lookup.reshape(self.n, self.n))
        self.P = float(P)
        return self.P

def _eval_support(support):
    """Optimize weights for fixed support, return (P, p)."""
    support = np.sort(np.maximum(np.asarray(support, dtype=np.float64), 0.0))
    n = len(support)
    struct = _EvalStruct(support)
    p = np.full(n, 1.0 / n, dtype=np.float64)
    struct.update(p)
    best_P = struct.P
    for _ in range(5000):
        if time.time() - _START_TIME > 598:
            break
        improved = False
        for idx in range(n):
            base = p.copy()
            base_sum = base.sum()
            if base_sum <= 1e-15:
                base = np.full(n, 1.0 / n)
                base_sum = 1.0
            others = base.copy()
            others[idx] = 0.0
            os = others.sum()
            if os < 1e-15:
                g = np.zeros(n)
            else:
                g = others / os
            best_pv = -1.0
            best_pval = p[idx]
            for t in np.linspace(0.0, 1.0, 16):
                p_t = (1.0 - t) * g
                p_t[idx] = t
                p_t /= p_t.sum()
                pv = struct.update(p_t)
                if pv > best_pv:
                    best_pv = pv
                    best_pval = t
            if best_pv > best_P + 1e-15:
                p = (1.0 - best_pval) * g
                p[idx] = best_pval
                p /= p.sum()
                struct.update(p)
                best_P = struct.P
                improved = True
        if not improved:
            break
    return best_P, p

def _pair_cdf(n, s):
    """C(k,l)=P[X1+X2 < 2*s_l - s_k] under uniform weights on s."""
    s2 = s[:, None] + s[None, :]
    thr = 2.0 * s[None, :] - s[:, None]
    C = (s2[None, :, None, :] < thr[:, None, :, None]).sum(axis=(2, 3)) / (n * n)
    return C

def _search_n(n, seed, t_deadline):
    try:
        struct = _EvalStruct(np.zeros(n))
        def f(params):
            if time.time() > t_deadline:
                return 1e6
            s = np.concatenate(([0.0], params[: n - 1]))
            s = np.sort(np.maximum(s, 0.0))
            w = np.maximum(params[n - 1:], 0.0)
            ws = w.sum()
            if ws <= 1e-9:
                return 1e6
            struct.support = s
            struct.n = n
            ps = (s[:, None] + s[None, :]).ravel()
            struct.pair_sums = ps
            struct.order = np.argsort(ps, kind='stable')
            struct.update(w / ws)
            return -struct.P
        # analytic CDF-based heuristic initialization
        s_init = np.sort(np.random.RandomState(seed).uniform(0.1, 1.0, n - 1))
        C = _pair_cdf(n, np.concatenate(([0.0], s_init)))
        w_init = np.random.RandomState(seed + 1).dirichlet(np.ones(n) * 1.5)
        x0 = np.concatenate([s_init, w_init])
        bounds = [(0.0, 2.0)] * (n - 1) + [(0.0, 1.0)] * n
        res = differential_evolution(
            f, bounds, x0=x0, seed=seed, maxiter=120, popsize=20,
            tol=1e-9, polish=False, mutation=(0.5, 1.0), recombination=0.8,
            init='sobol')
        x = res.x
        s = np.sort(np.maximum(np.concatenate(([0.0], x[: n - 1])), 0.0))
        w = np.maximum(x[n - 1:], 0.0)
        ws = w.sum()
        if ws <= 1e-9:
            return None
        P, p = _eval_support(s)
        return (P, s, p)
    except Exception:
        return None

def beat_the_average_measure():
    t0 = time.time()
    deadline = t0 + 600
    # analytic 2-point seed
    p2 = (1.0 + np.sqrt(33.0)) / 16.0
    s2 = np.array([0.0, 1.0])
    w2 = np.array([1.0 - p2, p2])
    best_P, best_s, best_w = _eval_support(s2)

    def try_candidate(s, w):
        nonlocal best_P, best_s, best_w
        P, p = _eval_support(np.maximum(np.asarray(s, dtype=np.float64), 0.0))
        if P > best_P:
            best_P, best_s, best_w = P, np.sort(p is not None and s) if False else np.sort(np.maximum(np.asarray(s, dtype=np.float64), 0.0)), p

    # parallel DE restarts across n
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        jobs = {}
        for n in (3, 4, 5):
            for r in range(4 if n <= 4 else 2):
                jobs[ex.submit(_search_n, n, seed=1000 * n + r, t_deadline=deadline)] = (n, r)
        for fut in concurrent.futures.as_completed(jobs):
            res = fut.result()
            if res is not None:
                P, s, p = res
                if P > best_P:
                    best_P, best_s, best_w = P, s, p

    # polish best
    if time.time() < deadline - 30 and best_s is not None:
        n = len(best_s)
        def f(x):
            s = np.sort(np.maximum(x[:n], 0.0))
            logw = x[n:] - x[n:].max()
            w = np.exp(logw); w /= w.sum()
            P, _ = _eval_support(s)  # not used; recompute cheaply via struct
            return -P
        x0 = np.concatenate([best_s, np.log(np.maximum(best_w, 1e-15))])
        try:
            r = minimize(f, x0, method='Nelder-Mead',
                         options={'maxiter': 4000, 'adaptive': True, 'fatol': 1e-15})
            s = np.sort(np.maximum(r.x[:n], 0.0))
            logw = r.x[n:] - r.x[n:].max(); w = np.exp(logw); w /= w.sum()
            P, p = _eval_support(s)
            if P > best_P:
                best_P, best_s, best_w = P, s, p
        except Exception:
            pass

    best_s = np.nan_to_num(np.maximum(best_s, 0.0))
    best_w = np.nan_to_num(np.maximum(best_w, 0.0))
    ws = best_w.sum()
    if ws <= 1e-9:
        best_s, best_w = s2, w2
    return best_s, best_w
# EVOLVE-BLOCK-END
