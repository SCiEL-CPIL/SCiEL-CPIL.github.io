"""Rewritten: signed step heights minimizing max|autoconv| / (sum s)^2.

Design family (kept): smooth L-BFGS-B on a differentiable log-sum-exp surrogate
+ discrete sign-flip local search + simulated annealing + final high-precision
polish. Improvements: efficient top-K log-sum-exp, batched vectorized flip
evaluation (many more local-search passes), more gradient restarts, tighter
polish, explicit deadline guard.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve


def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int


def _soft_max_ac(s, temperature=1.0, K=128):
    """Differentiable surrogate for max|autoconv| via top-K log-sum-exp."""
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    # Only the largest K values matter for the log-sum-exp; the rest are
    # exponentially suppressed (ac_sq <= max_val => exp(ac_sq - max_val) <= 1).
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    lse = max_val + np.log(np.sum(np.exp(top - max_val)))
    return np.sqrt(lse)


def _soft_objective(x, temperature=1.0, K=128):
    """Soft objective: log-sum-exp surrogate for max|ac| divided by (sum s)^2."""
    s = np.tanh(x)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    return _soft_max_ac(s, temperature, K) / sq_int


def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # 7 s safety margin before the 150 s kill

    def _now():
        return time.time()

    best_score = np.inf
    best_s = None
    np.random.seed(42)

    # ---------------- Phase 1: gradient-based optimization ----------------
    # Multiple diverse initializations; temperature anneals down so the
    # surrogate sharpens toward the true max|ac|.
    n_restarts = 20
    for it in range(n_restarts):
        if _now() > deadline - 30:
            break

        if it < 6:
            x0 = np.random.randn(N) * 0.1
        elif it < 12:
            t = np.arange(N)
            x0 = 0.6 * np.sin(2 * np.pi * t * np.random.uniform(0.05, 0.6)) \
                + np.random.randn(N) * 0.05
        elif it < 16:
            x0 = np.random.randn(N) * 0.3
        else:
            # Perturb the best-so-far continuous landscape (if available).
            x0 = np.random.randn(N) * 0.15

        temp = 3.0 * (0.6 ** (it / 10.0))
        try:
            result = minimize(
                _soft_objective,
                x0,
                method='L-BFGS-B',
                args=(temp, 128),
                options={'maxiter': 120, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            s_cont = np.tanh(result.x)
            s_bin = np.sign(s_cont)
            s_bin[s_bin == 0] = 1.0

            sc_b = _score_sequence(s_bin)
            if sc_b < best_score:
                best_score = sc_b
                best_s = s_bin.copy()
            sc_c = _score_sequence(s_cont)
            if sc_c < best_score:
                best_score = sc_c
                best_s = s_cont.copy()
        except Exception:
            continue

    # ---------------- Phase 2: discrete local search + SA ----------------
    if best_s is not None:
        cur = best_s.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        cur_score = best_score

        def _try_flip(idx):
            """Return (new_ac, new_sum, new_score) after flipping cur[idx]."""
            delta = -2.0 * cur[idx]
            new_ac = cur_ac + delta * np.concatenate(
                (np.zeros(idx), cur, np.zeros(idx))
            )
            new_sum = cur_sum + delta
            nsq = new_sum ** 2
            if nsq < 1e-10:
                return new_ac, new_sum, np.inf
            return new_ac, new_sum, float(np.max(np.abs(new_ac)) / nsq)

        def _batched_greedy_pass():
            """Vectorized: evaluate every single flip at once, keep the best."""
            deltas = -2.0 * cur
            # ac[k] + 2*delta[j]*cur[k-j]  =>  2 * (cur * deltas)
            contrib = 2.0 * fftconvolve(cur, deltas, mode='full')
            cand_ac = cur_ac[None, :] + contrib          # (N, 2N-1)
            cand_sum = cur_sum + deltas                  # (N,)
            cand_sum_sq = cand_sum ** 2
            bad = cand_sum_sq < 1e-10
            cand_max = np.max(np.abs(cand_ac), axis=1)   # (N,)
            cand_score = np.where(bad, np.inf, cand_max / np.where(bad, 1.0, cand_sum_sq))
            j = int(np.argmin(cand_score))
            return j, float(cand_score[j])

        # Greedy local search: repeatedly take the best single flip.
        max_passes = 600
        for _ in range(max_passes):
            if _now() > deadline - 20:
                break
            j, js = _batched_greedy_pass()
            if js >= cur_score - 1e-12:
                break
            cur[j] = -cur[j]
            cur_ac, cur_sum, cur_score = _try_flip(j)
            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

        # Simulated annealing to escape local minima.
        T_start = 0.04
        sa_steps = 2000
        for it in range(sa_steps):
            if _now() > deadline - 20:
                break
            idx = int(np.random.randint(N))
            new_ac, new_sum, new_score = _try_flip(idx)
            if new_score == np.inf:
                continue
            d = new_score - cur_score
            T = max(T_start * (0.99 ** it), 1e-6)
            if d < 0 or np.random.rand() < np.exp(-d / T):
                cur[idx] = -cur[idx]
                cur_ac, cur_sum, cur_score = new_ac, new_sum, new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_s = cur.copy()

    # ---------------- Phase 3: final high-precision polish ----------------
    if best_s is not None and _now() < deadline - 8:
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        for temp, iters in ((0.4, 200), (0.15, 200)):
            if _now() > deadline - 8:
                break
            try:
                result = minimize(
                    _soft_objective,
                    x_start,
                    method='L-BFGS-B',
                    args=(temp, 256),
                    options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-9},
                )
                s_final = np.tanh(result.x)
                x_start = result.x
                sf = _score_sequence(s_final)
                if sf < best_score:
                    best_score = sf
                    best_s = s_final.copy()
                sf_b = _score_sequence(np.sign(s_final) * 1.0)
                if sf_b < best_score:
                    best_score = sf_b
                    best_s = np.sign(s_final)
                    best_s[best_s == 0] = 1.0
            except Exception:
                pass

    if best_s is None:
        return np.ones(N)
    return best_s
# EVOLVE-BLOCK-END
