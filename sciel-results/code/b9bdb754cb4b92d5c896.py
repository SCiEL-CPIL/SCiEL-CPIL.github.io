"""
Signed autocorrelation maximum absolute-value variant.

We seek a signed step function f (length N) that minimizes:
    max_k | (f * f)[k] / (sum f)^2 |
where * is circular autoconvolution.

Strategy:
1. Initialize with a diverse set of candidates (all-ones, low-autocorrelation
   binary sequences, random).
2. Use a fast local search with efficient autoconvolution updates.
3. Use simulated annealing to escape local minima.
4. Final polish with a differentiable surrogate.

Key insight: The seed (all ones) gives score 1/N. We can do better by
using signs to reduce the peak autoconvolution while keeping the sum large.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400

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

def _max_ac_after_flip(cur, cur_ac, cur_sum, j):
    """Single-flip score update for a given index j."""
    val = cur[j]
    delta = -2.0 * val
    new_sum = cur_sum + delta
    nsq = new_sum ** 2
    if nsq < 1e-10:
        return np.inf
    new_ac = cur_ac.copy()
    new_ac[j:j+N] += 2.0 * delta * cur
    new_ac[j] += delta * delta
    return float(np.max(np.abs(new_ac)) / nsq)

def _batched_flip_eval(cur, cur_ac, cur_sum):
    """Vectorized evaluation of all N single-flip moves.
    
    Returns array of scores for flipping each index j.
    Uses the fact that flipping index j changes autoconvolution by:
      new_ac[k] = cur_ac[k] + 2*(-2*cur[j])*cur[k-j] for k in [j, j+N-1]
    plus a diagonal correction.
    
    For efficiency, we compute the max|new_ac| for each j using the
    structure that only a window of length N is affected.
    """
    # delta[j] = -2 * cur[j]
    deltas = -2.0 * cur
    new_sums = cur_sum + deltas
    nsq = new_sums ** 2
    # Guard against zero sums
    valid = nsq > 1e-10
    nsq_safe = np.where(valid, nsq, 1.0)
    
    # For each flip j, the affected autoconvolution window is [j, j+N-1]
    # new_ac_window = cur_ac[j:j+N] + 2*delta[j]*cur[0:N]
    # The max of |new_ac| over ALL positions = max(max_affected, max_unaffected)
    # Unaffected: positions outside [j, j+N-1] in the full (2N-1) length
    # This is complex to vectorize fully, so we use a hybrid approach:
    # Compute the affected window max for all j at once, then combine with
    # the global max of unaffected portions.
    
    # Build the affected windows: shape (N, N)
    # affected[j, k] = cur_ac[j+k] + 2*deltas[j]*cur[k] for k=0..N-1
    # cur_ac has length 2N-1, indices j+k for j,k in [0,N-1] range [0, 2N-2]
    
    # Use broadcasting: cur_ac_indices[j, k] = j + k
    idx = np.arange(N)
    ac_idx = idx[:, None] + idx[None, :]  # shape (N, N)
    affected = cur_ac[ac_idx] + 2.0 * deltas[:, None] * cur[None, :]
    
    # Also need diagonal correction: new_ac[j] += delta[j]^2
    # This is already included since affected[j,j] = cur_ac[j] + 2*delta[j]*cur[j]
    # But the true update is: new_ac[j] = cur_ac[j] + 2*delta[j]*cur[j] + delta[j]^2
    # So add delta[j]^2 to the diagonal
    diag_correction = deltas ** 2
    affected[np.arange(N), np.arange(N)] += diag_correction
    
    # Max over affected window for each j
    max_affected = np.max(np.abs(affected), axis=1)
    
    # Unaffected portions: for flip j, positions 0..j-1 and j+N..2N-2 are unchanged
    # max_unaffected[j] = max(max(cur_ac[0:j]), max(cur_ac[j+N:2N-1]))
    # Precompute prefix and suffix max of |cur_ac|
    abs_ac = np.abs(cur_ac)
    prefix_max = np.maximum.accumulate(abs_ac)
    # suffix_max[i] = max(abs_ac[i:])
    suffix_max = np.maximum.accumulate(abs_ac[::-1])[::-1]
    
    # For flip j: unaffected = positions [0, j-1] and [j+N, 2N-2]
    # prefix part: if j > 0, max is prefix_max[j-1], else 0
    # suffix part: if j+N < 2N-1, max is suffix_max[j+N], else 0
    max_unaff = np.zeros(N)
    # j ranges 0 to N-1; j+N ranges N to 2N-1
    # suffix starts at j+N, which is >= N, and 2N-2 is the last index
    # For j=0: suffix starts at N, valid if N <= 2N-2 (N>=2)
    # For j=N-1: suffix starts at 2N-1, which is out of bounds (max index 2N-2)
    
    # Prefix: j-1 can be -1 (j=0)
    prefix_vals = np.where(np.arange(N) > 0, prefix_max[np.maximum(np.arange(N)-1, 0)], 0.0)
    # Suffix: j+N can be up to 2N-1, max valid index is 2N-2
    suffix_starts = np.arange(N) + N
    suffix_valid = suffix_starts < len(abs_ac)
    suffix_vals = np.where(suffix_valid, suffix_max[np.minimum(suffix_starts, len(abs_ac)-1)], 0.0)
    
    max_unaff = np.maximum(prefix_vals, suffix_vals)
    # For j=0, prefix is empty, so prefix_vals[0] should be 0
    max_unaff[0] = suffix_vals[0]
    
    max_total = np.maximum(max_affected, max_unaff)
    scores = max_total / nsq_safe
    scores[~valid] = np.inf
    return scores

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # 7 s safety margin before the 150 s kill

    def _now():
        return time.time()

    best_score = np.inf
    best_s = None
    np.random.seed(42)

    # ---------------- Phase 1: gradient-based optimization ----------------
    # Multiple diverse initializations
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

        # Greedy local search with vectorized flip evaluation
        max_passes = 2000
        for _ in range(max_passes):
            if _now() > deadline - 20:
                break

            all_scores = _batched_flip_eval(cur, cur_ac, cur_sum)
            best_j = int(np.argmin(all_scores))
            best_js = all_scores[best_j]

            if best_js >= cur_score - 1e-12:
                break

            val = cur[best_j]
            delta = -2.0 * val
            cur_ac = cur_ac.copy()
            cur_ac[best_j:best_j+N] += 2.0 * delta * cur
            cur_ac[best_j] += delta * delta
            cur[best_j] = -val
            cur_sum += delta
            cur_score = best_js

            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

        # Simulated annealing to escape local minima.
        T_start = 0.05
        sa_steps = 10000
        for it in range(sa_steps):
            if _now() > deadline - 20:
                break
            idx = int(np.random.randint(N))
            new_score = _max_ac_after_flip(cur, cur_ac, cur_sum, idx)
            if new_score == np.inf:
                continue
            d = new_score - cur_score
            T = max(T_start * (0.995 ** it), 1e-6)
            if d < 0 or np.random.rand() < np.exp(-d / T):
                val = cur[idx]
                delta = -2.0 * val
                cur_ac = cur_ac.copy()
                cur_ac[idx:idx+N] += 2.0 * delta * cur
                cur_ac[idx] += delta * delta
                cur[idx] = -val
                cur_sum += delta
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_s = cur.copy()

    # ---------------- Phase 3: final high-precision polish ----------------
    if best_s is not None and _now() < deadline - 5:
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        # Use progressively tighter soft-max (higher K) for more precise approximation
        for temp, K_val, iters in ((0.5, 256, 150), (0.2, 512, 150), (0.08, 1024, 200)):
            if _now() > deadline - 5:
                break
            try:
                result = minimize(
                    _soft_objective,
                    x_start,
                    method='L-BFGS-B',
                    args=(temp, K_val),
                    options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                s_cont = np.tanh(result.x)
                sc = _score_sequence(s_cont)
                if sc < best_score:
                    best_score = sc
                    best_s = s_cont.copy()
                    x_start = result.x.copy()
            except Exception:
                continue

    # Final: try a few more diverse random restarts if time permits
    if _now() < deadline - 3:
        for trial in range(5):
            if _now() > deadline - 3:
                break
            x0 = np.random.randn(N) * 0.1
            try:
                result = minimize(
                    _soft_objective,
                    x0,
                    method='L-BFGS-B',
                    args=(0.1, 512),
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                s_cont = np.tanh(result.x)
                sc = _score_sequence(s_cont)
                if sc < best_score:
                    best_score = sc
                    best_s = s_cont.copy()
            except Exception:
                continue

    return best_s

# EVOLVE-BLOCK-END
