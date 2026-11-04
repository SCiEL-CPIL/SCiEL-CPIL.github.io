"""Second autocorrelation inequality: maximize L2^2 / (L1 * sup|autoconv| for 50 nonneg heights."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
# interval [-1/4, 1/4], 50 equal sub-intervals
_h = np.linspace(-0.25, 0.25, N + 1)
_dx = _h[1] - _h[0]
_x = 0.5 * (_h[:-1] + _h[1:])          # midpoints of the 50 intervals


def _score(h):
    """Exact objective: L2^2 / (L1 * sup|autoconv|). Returns (value, valid)."""
    h = np.asarray(h, dtype=float)
    L1 = float(np.sum(h))
    if L1 <= 0.0:
        return 0.0, False
    ac = np.convolve(h, h, mode="full")
    sup = float(np.max(ac))
    if sup <= 0.0:
        return 0.0, False
    val = float(np.dot(h, h)) / (L1 * sup)
    return val, np.all(h >= -1e-12)


def _get_sidon_set(n, max_val):
    """Generate a Sidon set of size n within [0, max_val]."""
    S = [0]
    sums = set([0])
    for i in range(1, max_val + 1):
        if len(S) >= n:
            break
        # Check if i can be added
        # New sums will be i+s for s in S, and 2*i
        new_sums = set()
        valid = True
        for s in S:
            new_sum = i + s
            if new_sum in sums or new_sum in new_sums:
                valid = False
                break
            new_sums.add(new_sum)
        two_i = 2 * i
        if two_i in sums or two_i in new_sums:
            valid = False
        
        if valid:
            S.append(i)
            sums.update(new_sums)
            sums.add(two_i)
    return S


def construct():
    import time
    t0 = time.time()
    DEADLINE = t0 + 105.0          # safety margin under 120s wall-clock
    rnd = np.random.default_rng(20240517)

    best = np.ones(N)
    best_v, _ = _score(best)

    def _try(h):
        nonlocal best, best_v
        h = np.asarray(h, dtype=float)
        v, ok = _score(h)
        if ok and v > best_v:
            best_v = v
            best = np.array(h, dtype=float)

    # ---- 1) Sidon set initialization (from inspiration 1) ----
    # Sidon sets have the property that all pairwise sums are distinct.
    # This leads to a "flat" autoconvolution profile (many small peaks instead of one large one),
    # which minimizes the sup norm relative to L1 and L2.
    # We generate several Sidon sets of varying sizes to use as starting points for local search.
    sidon_starts = []
    for size in range(4, 15):
        s_set = _get_sidon_set(size, N - 1)
        if len(s_set) >= 4:
            h_s = np.zeros(N)
            for idx in s_set:
                h_s[idx] = 1.0
            # Normalize to unit L2 for better conditioning in local search
            norm = np.linalg.norm(h_s)
            if norm > 0:
                h_s = h_s / norm
                sidon_starts.append(h_s)
    
    # Also try random sparse starts
    for _ in range(10):
        mask = rnd.random(N) < 0.2
        if np.sum(mask) < 2:
            continue
        h_r = np.zeros(N)
        h_r[mask] = 1.0
        norm = np.linalg.norm(h_r)
        if norm > 0:
            h_r = h_r / norm
            sidon_starts.append(h_r)

    # ---- 2) Local optimization on smooth surrogate ----
    # We use a smooth surrogate that approximates the non-smooth max in the denominator.
    # We optimize log(ratio) = log(L2^2) - log(L1) - log(sup|ac|).
    # To make it smooth, we approximate log(sup|ac|) with a soft-max:
    # log(sum(exp(ac_i / beta)) * beta) ~ log(sup) as beta -> 0.
    
    def _soft_max_log(ac, beta):
        """Approximate log(max(ac)) using log-sum-exp."""
        # Clip to avoid overflow
        ac_clipped = np.clip(ac, -1e30, 1e30)
        shifted = ac_clipped / beta
        m = np.max(shifted)
        if m > 700:
            shifted = shifted - m
        s = np.sum(np.exp(shifted))
        if s < 1e-30:
            return np.log(1e-30)
        return m + np.log(s)

    def _smooth_log_ratio_neg(v, beta):
        """
        Negative of the smooth log ratio.
        v is the log-amplitude vector: h = exp(v).
        This parameterization ensures h > 0, avoiding boundary issues with L-BFGS-B.
        """
        h = np.exp(v)
        L2sq = float(np.dot(h, h))
        L1 = float(np.sum(h))
        if L2sq <= 0.0 or L1 <= 0.0:
            return 1e10, np.zeros_like(v)
        
        ac = np.convolve(h, h, mode="full")
        # Use soft-max for the peak
        smax = _soft_max_log(ac, beta)
        
        # log_ratio = log(L2sq) - log(L1) - smax
        # We want to minimize -log_ratio
        val = -(np.log(L2sq) - np.log(L1) - smax)
        
        # Gradient computation (analytical)
        # d/dv log_ratio = d/dv (log(L2sq) - log(L1) - smax)
        # Note: h_i = exp(v_i), dh_i/dv_i = h_i
        
        # d log(L2sq)/dv_i = (2 h_i) / L2sq * h_i = 2 h_i^2 / L2sq
        # d log(L1)/dv_i = h_i / L1
        # d smax/dv_i = sum_j (d smax/dac_j) * (dac_j/dv_i)
        #   ac_j = sum_k h_k h_{j-k}
        #   dac_j/dv_i = h_i * (sum_k' h_{j-k} + h_{j-i} ... ) -- this is getting complex.
        #   Let's use numerical gradient for the soft-max part to be safe and simple, 
        #   or just use L-BFGS-B with numerical Jacobian if analytical is too slow/complex.
        #   Given N=50, numerical gradient is fast enough (50 evals per step).
        
        return val, None

    # We will use L-BFGS-B with numerical gradient for the log-parameterized problem.
    # This is robust.
    
    beta = 0.05  # Temperature for soft-max
    
    # Start from the best Sidon set found so far (or uniform)
    if sidon_starts:
        # Convert best Sidon start to log-space
        # Ensure no zeros in log space
        h_start = sidon_starts[0].copy()
        h_start[h_start == 0] = 1e-6
        v_start = np.log(h_start)
    else:
        v_start = np.zeros(N)

    # Run local search from multiple starts
    starts_to_try = []
    if sidon_starts:
        for hs in sidon_starts[:5]: # Limit number of starts to save time
            hs_safe = hs.copy()
            hs_safe[hs_safe == 0] = 1e-6
            starts_to_try.append(np.log(hs_safe))
    
    # Add a uniform start
    starts_to_try.append(np.zeros(N))
    
    # Add a few random log-space starts
    for _ in range(5):
        v_rand = rnd.normal(0, 1, N)
        starts_to_try.append(v_rand)

    for v0 in starts_to_try:
        if time.time() > DEADLINE - 10:
            break
            
        res = minimize(
            _smooth_log_ratio_neg, v0,
            args=(beta,),
            method="L-BFGS-B",
            options={"maxiter": 200, "maxls": 50}
        )
        
        if res.x is not None:
            h_opt = np.exp(res.x)
            _try(h_opt)
            
            # Also try a second pass with smaller beta for sharper peak approximation
            if time.time() < DEADLINE - 5:
                beta2 = 0.01
                res2 = minimize(
                    _smooth_log_ratio_neg, res.x,
                    args=(beta2,),
                    method="L-BFGS-B",
                    options={"maxiter": 100, "maxls": 30}
                )
                if res2.x is not None:
                    h_opt2 = np.exp(res2.x)
                    _try(h_opt2)

    # ---- 3) Final local polish of the global best ----
    if time.time() < DEADLINE - 2.0:
        # Convert best to log space for final polish
        best_safe = best.copy()
        best_safe[best_safe == 0] = 1e-8
        v_best = np.log(best_safe)
        
        res = minimize(
            _smooth_log_ratio_neg, v_best,
            args=(0.005,),
            method="L-BFGS-B",
            options={"maxiter": 200, "maxls": 50}
        )
        if res.x is not None:
            h_final = np.exp(res.x)
            _try(h_final)

    return [max(0.0, float(x)) for x in best]
