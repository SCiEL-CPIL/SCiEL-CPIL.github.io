# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _extract_contract_params(contract):
    """Robustly extract K, r, sigma, T, n_exercise from contract object."""
    def get(name, default):
        # try attribute
        try:
            v = getattr(contract, name)
            if v is not None:
                return v
        except Exception:
            pass
        # try dict
        try:
            if isinstance(contract, dict) and name in contract:
                return contract[name]
        except Exception:
            pass
        # try common aliases
        aliases = {
            'K': ['K', 'strike', 'strike_price', 'K0', 'spot_strike'],
            'r': ['r', 'rate', 'risk_free', 'rf', 'R'],
            'sigma': ['sigma', 'vol', 'volatility', 'Sigma', 's'],
            'T': ['T', 'maturity', 'expiry', 'horizon', 'tau'],
            'n_exercise': ['n_exercise', 'n_exercise_dates', 'n_steps', 'n_dates', 'num_exercise', 'n'],
        }
        for a in aliases.get(name, [name]):
            try:
                v = getattr(contract, a)
                if v is not None:
                    return v
            except Exception:
                pass
            try:
                if isinstance(contract, dict) and a in contract:
                    return contract[a]
            except Exception:
                pass
        return default
    K = float(get('K', 100.0))
    r = float(get('r', 0.05))
    sigma = float(get('sigma', 0.2))
    T = float(get('T', 1.0))
    n_ex = int(get('n_exercise', 12))
    return K, r, sigma, T, n_ex


def _get_state_at_dates(training_paths, n_ex, n_cols):
    """Return array of shape (n_paths, n_ex) of states at the n_ex exercise dates."""
    tp = np.asarray(training_paths, dtype=float)
    if tp.ndim == 1:
        tp = tp.reshape(1, -1)
    n_paths, n_cols = tp.shape
    if n_cols >= n_ex + 1:
        # assume column 0 is time 0, exercise dates are columns 1..n_ex
        return tp[:, 1:n_ex + 1]
    elif n_cols == n_ex:
        return tp[:, :n_ex]
    else:
        # fallback: resample linearly to n_ex points
        # assume tp columns are equally spaced over [0, T]
        idx = np.linspace(0, n_cols - 1, n_ex)
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, n_cols - 1)
        frac = idx - i0
        out = tp[:, i0] * (1 - frac) + tp[:, i1] * frac
        return out


def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    DEADLINE = 150.0
    MARGIN = 10.0  # safety margin

    K, r, sigma, T, n_ex = _extract_contract_params(contract)

    # time grid
    dt = T / n_ex
    times = np.linspace(0.0, T, n_ex + 1)  # times[0]=0, times[k]=k*dt

    # states at exercise dates: shape (n_paths, n_ex), index j -> date j (0-based), date j is at time (j+1)*dt
    states = _get_state_at_dates(training_paths, n_ex, None)
    n_paths, _ = states.shape

    # We'll do backward induction. V_next(s) is the value function at the next date.
    # Start at maturity: V at the last date (date n_ex-1, time T) = max(K - s, 0)
    # Actually the last exercise date is at time T (maturity). At maturity, continuation = 0.
    # Let's define:
    #   date index j = 0..n_ex-1, at time (j+1)*dt
    #   V_j(s) = optimal value at date j given spot s
    #   At j = n_ex-1 (maturity): V = max(K-s, 0), exercise if s < K
    #   For j < n_ex-1: C_j(s) = e^{-r dt} E[V_{j+1}(S_{j+1}) | S_j = s]
    #                    V_j(s) = max(K - s, C_j(s)), exercise if K - s >= C_j(s) i.e. s <= threshold_j

    # We represent V_j as a function. For regression, at step j we need V_{j+1} evaluated at S_{j+1}.
    # So we maintain V_{j+1} as a callable (polynomial coefficients).

    # Thresholds: threshold[j] for date j (0-based). Exercise if s <= threshold[j].
    thresholds = np.zeros(n_ex)

    # Maturity date (j = n_ex - 1): V = max(K - s, 0), threshold = K
    thresholds[n_ex - 1] = K

    # V_next at maturity as a function: max(K - s, 0). For the regression at j = n_ex-2,
    # we need V_{n_ex-1}(S_{n_ex-1}) = max(K - S, 0).
    # We'll represent V_next as a polynomial fit of max(K-s,0) over the relevant range, OR just evaluate directly.
    # Simpler: keep V_next as a callable that we can evaluate on arrays.

    def v_maturity(s):
        return np.maximum(K - s, 0.0)

    V_next = v_maturity

    # Scale for regression: use s/100
    scale = 100.0
    deg = 5

    # Precompute Vandermonde for each date's S_j (scaled)
    # For date j, S_j = states[:, j]
    def fit_poly(y, x):
        """Fit polynomial of degree deg to y = f(x). x is 1D. Returns coefficients (highest first) for np.polyval,
        or a callable. We'll return a callable and also coefficients in scaled space."""
        x = np.asarray(x, dtype=float) / scale
        # Build Vandermonde: [x^deg, x^(deg-1), ..., x, 1]
        Vm = np.vander(x, N=deg + 1, increasing=False)
        # Solve least squares
        try:
            coeffs, *_ = np.linalg.lstsq(Vm, y, rcond=None)
        except Exception:
            coeffs = np.zeros(deg + 1)
            coeffs[-1] = np.mean(y)
        # callable
        def f(s):
            xs = np.asarray(s, dtype=float) / scale
            return np.polyval(coeffs, xs)
        return coeffs, f

    # Backward induction
    for j in range(n_ex - 2, -1, -1):
        # Check time budget
        if time.time() - t_start > DEADLINE - MARGIN:
            # fill remaining thresholds with a heuristic and break
            for jj in range(j, -1, -1):
                thresholds[jj] = K  # default: exercise if ITM
            break

        S_j = states[:, j]
        S_next = states[:, j + 1]
        # continuation values at next date for each path
        V_next_vals = V_next(S_next)  # shape (n_paths,)
        y = np.exp(-r * dt) * V_next_vals

        # Regress y on S_j
        coeffs, V_j_cont = fit_poly(y, S_j)

        # Now V_j(s) = max(K - s, C_j(s)) where C_j(s) = V_j_cont(s)
        # threshold_j: solve K - s = C_j(s) for s, i.e., C_j(s) + s - K = 0
        # C_j is a polynomial in s (via scaled). Let's find the root in [0, K] (or [0, 2K]).
        # Define g(s) = C_j(s) + s - K. We want g(s) = 0.
        # For a put, as s -> 0, C_j(0) ~ e^{-r dt} E[V_next(S_next)|S_j=0]. S_next given S_j=0 is 0 (GBM), so V_next(0)=K, C_j(0) ~ e^{-r dt} K. g(0) = e^{-r dt} K + 0 - K = K(e^{-r dt} - 1) < 0.
        # As s -> K, C_j(K) is small (OTM continuation), g(K) = C_j(K) + K - K = C_j(K) > 0 (small positive).
        # So there's a root in (0, K) where g crosses from negative to positive.
        # Actually we want the threshold where exercise switches: for s < threshold, exercise (intrinsic >= continuation); for s > threshold, continue.
        # Intrinsic = K - s (decreasing in s). Continuation C_j(s) (decreasing in s).
        # At low s: intrinsic is large, continuation is large too. Which is bigger?
        # At s=0: intrinsic = K, continuation = e^{-r dt} K < K. So intrinsic > continuation -> exercise.
        # At high s: intrinsic -> 0, continuation -> 0 but continuation decays slower? 
        # The threshold is where intrinsic = continuation.
        # g(s) = C_j(s) - (K - s) = C_j(s) + s - K.
        # At s=0: g(0) = C_j(0) - K = e^{-r dt}K - K < 0 -> continuation < intrinsic -> exercise.
        # At s=K: g(K) = C_j(K) - 0 = C_j(K) > 0 -> continuation > intrinsic -> continue.
        # So g crosses 0 from negative to positive at the threshold. Exercise if g(s) < 0, i.e., s < threshold.
        # Find root of g(s) = 0 in [0, K].

        def g(s):
            return V_j_cont(s) + s - K

        # Use bisection / brentq
        lo, hi = 0.0, K
        # Ensure sign change
        glo = g(lo)
        ghi = g(hi)
        if glo * ghi > 0:
            # no clean sign change; fallback
            # if g(K) < 0, always exercise (threshold = K)
            # if g(0) > 0, never exercise until... threshold = 0
            if ghi < 0:
                thr = K
            else:
                thr = 0.0
        else:
            # bisection
            a, b = lo, hi
            fa, fb = glo, ghi
            for _ in range(80):
                m = 0.5 * (a + b)
                fm = g(m)
                if fa * fm <= 0:
                    b = m
                    fb = fm
                else:
                    a = m
                    fa = fm
                if abs(b - a) < 1e-9:
                    break
            thr = 0.5 * (a + b)
        thresholds[j] = thr

        # V_j for the next iteration: V_j(s) = max(K - s, C_j(s))
        # Represent as callable
        def V_j(s, c=V_j_cont):
            s = np.asarray(s, dtype=float)
            intr = np.maximum(K - s, 0.0)
            cont = c(s)
            return np.maximum(intr, cont)
        V_next = V_j

    # Build the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, times, n_ex, K):
            self.thresholds = thresholds
            self.times = times
            self.n_ex = n_ex
            self.K = K

        def exercise(self, current_state, time):
            s = np.asarray(current_state, dtype=float)
            # determine exercise date index from time
            # times[j] = (j+1)*dt for j in 0..n_ex-1
            # find j such that time is closest to times[j]
            t = float(time)
            # map t to index
            # t in [0, T]. exercise dates at (j+1)*T/n_ex
            # index = round(t / dt) - 1, clamp to [0, n_ex-1]
            dt_local = T / self.n_ex
            idx = int(round(t / dt_local)) - 1
            idx = max(0, min(self.n_ex - 1, idx))
            thr = self.thresholds[idx]
            return s <= thr

    policy = ExercisePolicy(thresholds, times, n_ex, K)
    return policy
# EVOLVE-BLOCK-END
