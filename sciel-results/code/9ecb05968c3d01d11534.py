import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 132.0  # safety margin before 150s kill

def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr) / N * 2.0

def _corr_vector(h):
    """Return the full correlation vector (length 2N-1)."""
    return np.correlate(h, 1.0 - h, mode='full')

def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(1000):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-15:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        n_free = int(np.sum(mask))
        if n_free == 0:
            break
        h[mask] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_objective(x, beta):
    """
    x = [h (N), t (1)]
    Objective: t + (1/beta) * log(sum_lag exp(beta * (corr_lag - t)))
    This is a smooth surrogate for max(t, max_lag corr_lag).
    """
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    # corr_lag - t
    shifted = beta * (corr - t)
    m = np.max(shifted)
    if m > 700:
        shifted = shifted - m
        logsum = m + np.log(np.sum(np.exp(shifted)))
    else:
        logsum = np.log(np.sum(np.exp(shifted)))
    return t + logsum / beta

def _smooth_objective_grad(x, beta):
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    shifted = beta * (corr - t)
    m = np.max(shifted)
    if m > 700:
        shifted = shifted - m
        logsum = m + np.log(np.sum(np.exp(shifted)))
    else:
        logsum = np.log(np.sum(np.exp(shifted)))
    # weights: p_lag = exp(beta*(corr_lag - t)) / sum_lag exp(...)
    p = np.exp(shifted)
    p_sum = np.sum(p)
    p = p / p_sum  # softmax weights over lags

    # Gradient w.r.t t: 1 - (1/beta)*beta*sum_lag p_lag = 1 - 1 = 0? 
    # d/dt [t + (1/beta) logsumexp(beta*(corr - t))]
    # = 1 + (1/beta) * sum_lag p_lag * (-beta) = 1 - sum_lag p_lag = 1 - 1 = 0
    # So gradient w.r.t t is 0 (good, t is free).
    grad_t = 0.0

    # Gradient w.r.t h: (1/beta) * sum_lag p_lag * beta * d(corr_lag)/dh
    # = sum_lag p_lag * d(corr_lag)/dh
    # d(corr_lag)/dh[i]: corr at lag k (k = idx - (N-1)) is sum_i h[i]*(1-h[i+k])
    # d/dh[i] of corr_k = (1-h[i+k]) if 0<=i+k<N  [from term i]
    #                   + (-h[i-k]) if 0<=i-k<N   [from term i-k, where h[i-k]*(1-h[i])]
    # So total: sum_k p_k * [ (1-h[i+k]) * 1_{0<=i+k<N} - h[i-k] * 1_{0<=i-k<N} ]
    # We can vectorize: for each lag k, the contribution to grad[i] is p_k * (1-h[i+k]) - p_k * h[i-k]
    # Let's build it via convolution-like ops.
    # For a fixed k, term A_k[i] = (1-h[i+k]) if 0<=i+k<N else 0
    # term B_k[i] = h[i-k] if 0<=i-k<N else 0
    # grad_h[i] = sum_k p_k * (A_k[i] - B_k[i])
    # = sum_k p_k * A_k[i] - sum_k p_k * B_k[i]
    # sum_k p_k * A_k[i] = sum_k p_k * (1-h[i+k])  = sum_k p_k - sum_k p_k*h[i+k]
    #   = 1 - sum_k p_k * h[i+k]
    # sum_k p_k * B_k[i] = sum_k p_k * h[i-k]
    # Note: sum_k p_k * h[i+k] and sum_k p_k * h[i-k] are related by reversing p.
    # Let's just compute directly using the correlation structure.
    # Actually, corr_k = sum_i h[i]*(1-h[i+k]). We can write the gradient more directly.
    # d(corr_k)/dh[i] = (1-h[i+k]) - h[i-k]  (with bounds)
    # Let's compute grad_h = sum_k p_k * d(corr_k)/dh
    # Build d(corr_k)/dh as a function.
    # For efficiency, note that d(corr_k)/dh[i] depends on k and i.
    # Let's use: for each i, grad_h[i] = sum_k p_k * (1-h[i+k]) * 1_{0<=i+k<N} - sum_k p_k * h[i-k] * 1_{0<=i-k<N}
    # First sum: sum_k p_k * (1-h[i+k]) over valid k = sum_k p_k - sum_k p_k*h[i+k]
    #   = 1 - (p * h shifted)[i]  where (p*h)[i] = sum_k p[k]*h[i+k]
    # Second sum: sum_k p_k * h[i-k] = (p_rev * h)[i] where p_rev[k] = p[-k]
    # Let's compute these via np.convolve or direct.
    # Since N=100, a direct loop over k is fine (199 lags * 100 = 20k ops).
    grad_h = np.zeros(N)
    for k_idx in range(len(p)):
        pk = p[k_idx]
        if pk < 1e-15:
            continue
        k = k_idx - (N - 1)  # actual lag
        for i in range(N):
            # term A: (1-h[i+k]) if 0<=i+k<N
            j = i + k
            if 0 <= j < N:
                grad_h[i] += pk * (1.0 - h[j])
            # term B: -h[i-k] if 0<=i-k<N
            j2 = i - k
            if 0 <= j2 < N:
                grad_h[i] -= pk * h[j2]
    return np.concatenate([grad_h, [grad_t]])

def _epigraph_objective(x, beta):
    """Epigraph: minimize t s.t. corr_lag <= t for all lag. Use soft penalty."""
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    # Penalty for corr_lag > t: (1/beta)*logsumexp(beta*(corr - t))
    shifted = beta * (corr - t)
    m = np.max(shifted)
    if m > 700:
        shifted = shifted - m
        logsum = m + np.log(np.sum(np.exp(shifted)))
    else:
        logsum = np.log(np.sum(np.exp(shifted)))
    return t + 0.1 * logsum / beta  # small penalty to keep t >= max corr

def _epigraph_grad(x, beta):
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    shifted = beta * (corr - t)
    m = np.max(shifted)
    if m > 700:
        shifted = shifted - m
        logsum = m + np.log(np.sum(np.exp(shifted)))
    else:
        logsum = np.log(np.sum(np.exp(shifted)))
    p = np.exp(shifted)
    p = p / np.sum(p)
    # grad_t = 1 + 0.1 * (1/beta) * sum p_k * (-beta) = 1 - 0.1
    grad_t = 1.0 - 0.1
    # grad_h = 0.1 * sum_k p_k * d(corr_k)/dh
    grad_h = np.zeros(N)
    for k_idx in range(len(p)):
        pk = p[k_idx]
        if pk < 1e-15:
            continue
        k = k_idx - (N - 1)
        for i in range(N):
            j = i + k
            if 0 <= j < N:
                grad_h[i] += pk * (1.0 - h[j])
            j2 = i - k
            if 0 <= j2 < N:
                grad_h[i] -= pk * h[j2]
    grad_h *= 0.1
    return np.concatenate([grad_h, [grad_t]])

def _coordinate_polish(h, max_iter=200):
    """
    Coordinate descent: for each i, optimize h[i] given others,
    projecting to keep sum = TARGET_SUM and 0<=h<=1.
    Since the objective is piecewise linear in h[i] (for fixed others,
    each corr_lag is linear in h[i]), the max is piecewise linear,
    so the minimum over h[i] is at a kink or boundary.
    We evaluate h[i] at all candidate points (where any corr_lag's slope changes)
    and boundaries, then pick the best.
    """
    h = _project_feasible(h).copy()
    best_val = _correlation_max(h)
    for it in range(max_iter):
        improved = False
        for i in range(N):
            h_old = h[i]
            # Candidate values for h[i]:
            # The objective as a function of x = h[i] (others fixed) is max_lag (a_lag * x + b_lag)
            # where a_lag = d(corr_lag)/dh[i], b_lag = corr_lag - a_lag * h_old
            # Kinks occur where two corr_lag are equal: a_l1*x+b_l1 = a_l2*x+b_l2
            # Also boundaries x=0, x=1, and x that keeps sum=TARGET_SUM.
            # For simplicity, sample x at: 0, 1, h_old, and midpoints.
            # Better: compute all corr_lag as linear functions of x, find kinks.
            # corr_lag(x) = corr_lag(h) + a_lag * (x - h_old)
            # where a_lag = d(corr_lag)/dh[i]
            # Compute a_lag for all lags.
            a_lag = np.zeros(2 * N - 1)
            for k_idx in range(2 * N - 1):
                k = k_idx - (N - 1)
                # d(corr_k)/dh[i] = (1-h[i+k]) if 0<=i+k<N, minus h[i-k] if 0<=i-k<N
                val = 0.0
                j = i + k
                if 0 <= j < N:
                    val += (1.0 - h[j])
                j2 = i - k
                if 0 <= j2 < N:
                    val -= h[j2]
                a_lag[k_idx] = val
            b_lag = _corr_vector(h) - a_lag * h_old
            # Kinks: for each pair (l1, l2) with a_l1 != a_l2, x = (b_l2 - b_l1)/(a_l1 - a_l2)
            # That's O(L^2) = 199^2 = ~40k pairs, times 100 i's = 4M, too slow.
            # Instead, sample x on a fine grid in [0,1] and also at 0, 1.
            # Since N=100 and we do this for each i, a grid of 50 points is 50*100=5k evals per i,
            # each eval is O(N) for correlate... too slow.
            # Alternative: just try a few smart points: 0, 1, h_old, and the point that
            # minimizes the current max lag.
            # For now, use a coarse approach: try x in {0, 1, h_old} and a few around.
            candidates = [0.0, 1.0, h_old]
            # Add a few around h_old
            for dx in [-0.1, -0.05, 0.05, 0.1]:
                candidates.append(h_old + dx)
            # Also, the optimal x for a single active lag l is where the slope changes sign,
            # but since it's max of linear, the min is at a kink.
            # Let's just evaluate at candidates and pick best.
            best_x = h_old
            best_v = best_val
            for x in candidates:
                x = max(0.0, min(1.0, x))
                # Adjust to keep sum: if we change h[i] by (x - h_old), we need to adjust others.
                # For a true coordinate step, we'd redistribute. But that complicates.
                # Instead, project after setting h[i]=x.
                h_test = h.copy()
                h_test[i] = x
                h_test = _project_feasible(h_test)
                v = _correlation_max(h_test)
                if v < best_v - 1e-12:
                    best_v = v
                    best_x = h_test.copy()
            if best_v < best_val - 1e-12:
                h = best_x
                best_val = best_v
                improved = True
        if not improved:
            break
    return h, best_val

def construct():
    start_time = time.time()

    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)

    # Phase 1: Structured initializations
    starts = [np.full(N, 0.5)]

    # Alternating patterns
    for base in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
        starts.append(_project_feasible(alt))

    # Sine waves
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]:
        for amp in [0.2, 0.3, 0.4, 0.45, 0.49]:
            t = np.arange(N)
            h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t / N)
            starts.append(_project_feasible(h_sin))
            h_cos = 0.5 + amp * np.cos(2 * np.pi * freq * t / N)
            starts.append(_project_feasible(h_cos))

    # Two-block patterns
    for split in [20, 30, 40, 50, 60, 70, 80]:
        h_block = np.zeros(N)
        h_block[:split] = 1.0
        h_block = _project_feasible(h_block)
        starts.append(h_block)
        h_block2 = np.zeros(N)
        h_block2[split:] = 1.0
        h_block2 = _project_feasible(h_block2)
        starts.append(h_block2)

    # Linear ramps
    starts.append(_project_feasible(np.linspace(0.0, 1.0, N)))
    starts.append(_project_feasible(np.linspace(1.0, 0.0, N)))

    # Triangular
    t = np.arange(N)
    tri = 1.0 - np.abs(t - N / 2.0) / (N / 2.0)
    starts.append(_project_feasible(tri))
    tri2 = np.abs(t - N / 2.0) / (N / 2.0)
    starts.append(_project_feasible(tri2))

    # Random smooth seeds
    rng = np.random.RandomState(42)
    for _ in range(15):
        noise = rng.randn(N)
        kernel = np.ones(9) / 9.0
        smooth = np.convolve(noise, kernel, mode='same')
        h_pert = 0.5 + 0.4 * smooth
        starts.append(_project_feasible(h_pert))

    # Random blocky seeds
    for _ in range(10):
        h_rand = rng.rand(N)
        # Make it blocky
        block_size = rng.randint(3, 15)
        for b in range(0, N, block_size):
            h_rand[b:b+block_size] = h_rand[b]
        starts.append(_project_feasible(h_rand))

    # Phase 2: Refine with smooth epigraph SLSQP
    bounds = [(0.0, 1.0) for _ in range(N)] + [(-1.0, 2.0)]
    constraint = {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - TARGET_SUM,
                  'jac': lambda x: np.concatenate([np.ones(N), [0.0]])}

    # Use a moderate beta for smoothness
    beta = 20.0

    for h0 in starts:
        if time.time() - start_time > DEADLINE - 25:
            break
        try:
            x0 = np.concatenate([h0, [_correlation_max(h0) * N / 2.0 + 0.1]])
            res = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                jac=lambda x: _smooth_objective_grad(x, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            h_res = _project_feasible(res.x[:N])
            val = _correlation_max(h_res)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass

    # Phase 3: Coordinate polish on best candidate
    if time.time() - start_time < DEADLINE - 10:
        try:
            h_polished, val_polished = _coordinate_polish(best_h, max_iter=100)
            if val_polished < best_val - 1e-12:
                best_val = val_polished
                best_h = h_polished
        except Exception:
            pass

    # Phase 4: Final high-precision SLSQP polish
    if time.time() - start_time < DEADLINE - 5:
        try:
            # Use the smooth objective with higher beta for sharper max
            beta2 = 50.0
            x0 = np.concatenate([best_h, [_correlation_max(best_h) * N / 2.0 + 0.05]])
            res = minimize(
                _smooth_objective,
                x0,
                args=(beta2,),
                jac=lambda x: _smooth_objective_grad(x, beta2),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
            )
            h_final = _project_feasible(res.x[:N])
            val_final = _correlation_max(h_final)
            if val_final < best_val - 1e-13:
                best_val = val_final
                best_h = h_final
        except Exception:
            pass

    # Ensure final validity
    best_h = _project_feasible(best_h)

    return best_h
