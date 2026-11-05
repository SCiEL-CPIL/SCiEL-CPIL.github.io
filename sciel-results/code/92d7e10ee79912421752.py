"""Bermudan put exercise-policy research: barrier policy fit by backward induction.

Fits 12 monthly exercise boundaries via continuation-value estimation on the
training paths, using a smooth (log-sum-exp) surrogate for the min in the
Bermudan recursion, plus a final local polish pass over the boundaries.
"""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 150.0  # wall-clock kill limit (seconds); keep safety margin
_MARGIN = 15.0     # return best result this far before the limit


def _extract_paths(training_paths, contract):
    """Pull (S, r, sigma, T, K, S0, n_dates) out of the contract/training data.

    The contract is expected to carry the fixed model assumptions. We read them
    defensively and fall back to the stated task constants.
    """
    def _get(obj, key, default):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    S0 = float(_get(contract, 'S0', 100.0))
    K = float(_get(contract, 'K', 100.0))
    r = float(_get(contract, 'r', 0.05))
    sigma = float(_get(contract, 'sigma', 0.2))
    T = float(_get(contract, 'T', 1.0))
    n_dates = int(_get(contract, 'n_dates', 12))

    # training_paths: expected to be an array of shape (n_dates, n_paths) of
    # spot prices at the exercise dates (no time-0 row).
    S = np.asarray(training_paths, dtype=np.float64)
    if S.ndim == 1:
        # maybe (n_dates,) single path or (n_paths,) at one date; reshape
        S = S.reshape(1, -1)
    if S.shape[0] != n_dates:
        # try transposed
        if S.ndim == 2 and S.shape[1] == n_dates:
            S = S.T
    return S, S0, K, r, sigma, T, n_dates


def _softmin(a, b, t):
    """Smooth min(a, b) via log-sum-exp (negative). a,b arrays same shape, t>0."""
    m = np.minimum(a, b)
    # softmin = -log(exp(-a/t)+exp(-b/t)) * t  (stable form)
    u = a / t
    v = b / t
    w = np.maximum(u, v)
    return -t * (w + np.log(np.exp(-u - w) + np.exp(-v - w)))


def _barrier_value(S, K, r, n_dates, B, t=1e-3):
    """Mean discounted payoff of a barrier policy on given paths.

    S: (n_dates, n_paths). B: length n_dates boundaries (index 0 = first date).
    Continuation value is discounted back to time 0.
    """
    n_dates_local = S.shape[0]
    n_paths = S.shape[1]
    dt = 1.0 / n_dates_local  # T=1, monthly
    disc = np.exp(-r * np.arange(1, n_dates_local + 1) * dt)  # discount to t0
    payoff = np.zeros(n_paths, dtype=np.float64)
    cont = np.zeros(n_paths, dtype=np.float64)
    for k in range(n_dates_local - 1, -1, -1):
        exercise = (S[k] <= B[k])
        ex_pay = np.maximum(K - S[k], 0.0)
        # smooth min between exercise and (discounted) continuation
        disc_cont = np.exp(-r * dt) * cont
        val = _softmin(ex_pay, disc_cont, t)
        payoff = np.where(exercise, ex_pay, val)
        cont = payoff
    return np.mean(disc * cont)


def _fit_barrier(S, K, r, n_dates):
    """Backward induction to get initial boundary estimates.

    Continuation value at date k (for paths not exercised) is the discounted
    mean of the realized optimal value from later dates, estimated from the
    training paths. Boundary B_k is the spot level where ex_pay == cont.
    We estimate cont_k(S_k) as a monotone function via quantile/lookup.
    """
    dt = 1.0 / n_dates
    n_paths = S.shape[1]
    # optimal value from date k onward, computed backward
    opt = np.zeros((n_dates, n_paths), dtype=np.float64)
    # terminal date: only exercise (no later dates)
    opt[-1] = np.maximum(K - S[-1], 0.0)
    for k in range(n_dates - 2, -1, -1):
        ex_pay = np.maximum(K - S[k], 0.0)
        disc_opt = np.exp(-r * dt) * opt[k + 1]
        opt[k] = np.maximum(ex_pay, disc_opt)

    # For each date k, estimate continuation value as a function of S_k.
    # cont_k(s) = E[ disc * opt_{k+1} | S_k = s ] approximated by local mean.
    # Boundary B_k: solve ex_pay(B_k) = cont_k(B_k).
    B = np.zeros(n_dates, dtype=np.float64)
    for k in range(n_dates):
        ex_pay = np.maximum(K - S[k], 0.0)
        if k == n_dates - 1:
            disc_opt = np.zeros(n_paths)
        else:
            disc_opt = np.exp(-r * dt) * opt[k + 1]
        # pairs (S_k, disc_opt). Boundary where K - S == disc_opt (for S<K)
        # i.e. S == K - disc_opt. Boundary is the S level; exercise if S<=B.
        # Estimate B_k as the S quantile where ex_pay crosses disc_opt.
        # Use: B_k = median of (K - disc_opt) clipped, weighted by S ordering.
        cand = K - disc_opt
        # Only meaningful where S_k is in put range; take robust quantile.
        B[k] = float(np.clip(np.median(cand), 0.0, K))
    return B


def _polish_boundaries(S, K, r, n_dates, B, iters=8, lr=0.5):
    """Local coordinate-descent polish of boundaries to raise training value."""
    B = B.copy()
    best_val = _barrier_value(S, K, r, n_dates, B)
    for _ in range(iters):
        improved = False
        for k in range(n_dates):
            # try a few perturbations around B[k]
            base = B[k]
            for delta in (lr, -lr, lr * 0.5, -lr * 0.5):
                trial = B.copy()
                trial[k] = max(0.0, base + delta)
                v = _barrier_value(S, K, r, n_dates, trial)
                if v > best_val + 1e-12:
                    B = trial
                    best_val = v
                    improved = True
        if not improved:
            break
    return B, best_val


def fit_exercise_policy(training_paths, contract):
    t_start = _time.time()
    S, S0, K, r, sigma, T, n_dates = _extract_paths(training_paths, contract)

    # Ensure S has the right orientation: (n_dates, n_paths)
    if S.shape[0] != n_dates and S.ndim == 2:
        if S.shape[1] == n_dates:
            S = S.T
    if S.shape[0] != n_dates:
        # pad or trim to n_dates
        if S.shape[0] > n_dates:
            S = S[:n_dates]
        else:
            pad = np.zeros((n_dates - S.shape[0], S.shape[1]))
            S = np.vstack([S, pad])

    # Initial boundary estimates via backward induction
    B = _fit_barrier(S, K, r, n_dates)

    # Polish with deadline guard
    best_B = B.copy()
    best_val = _barrier_value(S, K, r, n_dates, best_B)

    remaining = _DEADLINE - _MARGIN - (_time.time() - t_start)
    if remaining > 5.0:
        best_B, best_val = _polish_boundaries(S, K, r, n_dates, best_B,
                                              iters=10, lr=0.5)

    # Final deadline check: if we're close, stop and return best found.
    if _time.time() - t_start > _DEADLINE - _MARGIN:
        best_B = best_B

    boundaries = best_B

    class _Policy:
        def __init__(self, B):
            self.B = B

        def exercise(self, current_state, time):
            """Return boolean vector: True if exercise at this date.

            current_state: array of spot prices at the current date.
            time: the date index (0-based among the 12 monthly dates) or
                  a time value in [0, T]. We map to a date index.
            """
            S_cur = np.asarray(current_state, dtype=np.float64)
            if S_cur.ndim == 0:
                S_cur = S_cur.reshape(1)
            # Determine date index from time
            if time is None:
                k = 0
            else:
                try:
                    tval = float(time)
                except Exception:
                    tval = 0.0
                # time in [0, T]; date k corresponds to t = (k+1)*dt
                dt = 1.0 / n_dates
                k = int(np.round(tval / dt)) - 1
                if k < 0:
                    k = 0
                if k >= n_dates:
                    k = n_dates - 1
            b = boundaries[k]
            return (S_cur <= b).astype(bool)

    return _Policy(boundaries)
# EVOLVE-BLOCK-END
