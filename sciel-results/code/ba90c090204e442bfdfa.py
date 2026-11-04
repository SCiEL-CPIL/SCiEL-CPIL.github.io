import numpy as np
import time

N = 400

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return float(np.max(np.abs(acf[1:N])))

def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf

def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _gradient_step(h, acf):
    """
    Find the single best sign flip to reduce PSL.
    For flipping h[i], the ACF change at lag k is:
      delta[k-1] = -2 * h[i] * (h[i-k] + h[i+k])  (where indices valid)
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    acf_lags = acf[1:n]
    best_i = -1
    best_new_psl = float('inf')

    for i in range(n):
        neighbor_sum = np.zeros(n - 1)
        max_k1 = min(i, n - 1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
        max_k2 = min(n - 1 - i, n - 1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf_lags + delta
        new_psl = float(np.max(np.abs(new_acf_lags)))
        if new_psl < best_new_psl:
            best_new_psl = new_psl
            best_i = i

    return best_i, float(best_new_psl), current_psl

def _fast_psl_optimize(h, max_passes=300, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    if start_time is None:
        start_time = time.time()

    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)

    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break

        i, new_psl, cur_psl = _gradient_step(h, acf)
        if i == -1 or new_psl >= cur_psl:
            break

        h[i] = -h[i]
        passes += 1
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)

        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()

        if best_psl < 1e-10:
            break

    return best_h, best_psl

def _chirp_sequence(n, alpha, beta, gamma, delta):
    """
    Generate a signed sequence from the sign of a sum of chirp components.
    alpha: linear chirp frequency (quadratic phase)
    beta: quadratic chirp frequency (cubic phase)
    gamma: additional linear phase (rotation)
    delta: DC offset
    """
    t = np.arange(n, dtype=np.float64)
    # Linear chirp: exp(i * alpha * t^2)
    # Quadratic chirp: exp(i * beta * t^3)
    # We take the real part and add a DC offset, then threshold
    val = (np.cos(alpha * t * t)
           + np.cos(beta * t * t * t)
           + gamma * np.cos(2.0 * np.pi * t / n)
           + delta)
    return np.sign(val)

def _chirp_family_candidates(rng):
    """
    Generate diverse candidates from the chirp family.
    Each candidate is sign of a sum of two orthogonal chirp signals
    with optimized frequency scaling factors.
    """
    candidates = []
    t = np.arange(N, dtype=np.float64)

    # Zadoff-Chu-like: pure quadratic chirp with various roots
    # For N=400, try various quadratic phase coefficients
    for root in [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        # Quadratic chirp: cos(2*pi*root*t^2/N)
        val = np.cos(2.0 * np.pi * root * t * t / N)
        candidates.append(np.sign(val))

    # Mixed linear + quadratic chirp
    for trial in range(40):
        alpha = rng.uniform(0.01, 0.5)
        beta = rng.uniform(0.001, 0.1)
        gamma = rng.uniform(0.0, 2.0 * np.pi)
        delta = rng.uniform(-0.5, 0.5)
        val = (np.cos(alpha * t * t)
               + beta * np.cos(2.0 * np.pi * t / N + gamma)
               + delta)
        candidates.append(np.sign(val))

    # Pure quadratic with phase offset
    for trial in range(20):
        alpha = rng.uniform(0.05, 0.3)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        val = np.cos(alpha * t * t + phase)
        candidates.append(np.sign(val))

    # Cubic chirp
    for trial in range(15):
        beta = rng.uniform(0.005, 0.05)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        val = np.cos(beta * t * t * t + phase)
        candidates.append(np.sign(val))

    # Two orthogonal chirps summed
    for trial in range(25):
        a1 = rng.uniform(0.05, 0.3)
        a2 = rng.uniform(0.05, 0.3)
        p1 = rng.uniform(0.0, 2.0 * np.pi)
        p2 = rng.uniform(0.0, 2.0 * np.pi)
        val = np.cos(a1 * t * t + p1) + np.cos(a2 * t * t + p2)
        candidates.append(np.sign(val))

    # Linear + quadratic (orthogonal chirp pair)
    for trial in range(25):
        lin_freq = rng.uniform(1.0, 10.0)
        quad_freq = rng.uniform(0.05, 0.3)
        p1 = rng.uniform(0.0, 2.0 * np.pi)
        p2 = rng.uniform(0.0, 2.0 * np.pi)
        val = (np.cos(2.0 * np.pi * lin_freq * t / N + p1)
               + np.cos(quad_freq * t * t + p2))
        candidates.append(np.sign(val))

    # Ensure no all-zeros
    for i, c in enumerate(candidates):
        if np.all(c == 0):
            candidates[i] = np.ones(N)

    return candidates

def _simulated_annealing_psl(h, time_budget=15.0, start_time=None, rng=None):
    """
    Simulated annealing on PSL using fast gradient evaluation.
    """
    if rng is None:
        rng = np.random.default_rng()

    h = h.copy().astype(np.float64)
    n = len(h)
    if start_time is None:
        start_time = time.time()

    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    best_h = h.copy()
    best_psl = current_psl

    T = 5.0
    T_min = 0.05
    cooling = 0.93

    while T > T_min:
        if time.time() - start_time > time_budget:
            break

        i = int(rng.integers(n))
        neighbor_sum = np.zeros(n - 1)
        max_k1 = min(i, n - 1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
        max_k2 = min(n - 1 - i, n - 1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]

        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf[1:n] + delta
        new_psl = float(np.max(np.abs(new_acf_lags)))
        delta_psl = new_psl - current_psl

        if delta_psl < 0 or np.random.random() < np.exp(-delta_psl / T):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()

        T *= cooling

    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    best_h = None
    best_score = float('inf')
    rng = np.random.default_rng(42)

    # Generate chirp-family candidates
    candidates = _chirp_family_candidates(rng)

    # Also add a few random and periodic candidates for diversity
    for i in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    for period in [3, 5, 7, 11]:
        pattern = np.sign(np.cos(2.0 * np.pi * np.arange(N) / period))
        candidates.append(pattern)

    # Score all candidates
    scored = []
    for h in candidates:
        s = _score(h)
        scored.append((s, h))
    scored.sort(key=lambda x: x[0])

    # Refine top candidates with fast PSL optimization
    top_k = 8
    for i, (s, h) in enumerate(scored[:top_k]):
        if time.time() > deadline - 15:
            break
        time_limit = deadline - 10
        h_refined, s_refined = _fast_psl_optimize(
            h, max_passes=400, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()

    # Random restarts with fast PSL optimization
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 5, time.time() + 4)
        h_refined, s_refined = _fast_psl_optimize(
            h_init, max_passes=80, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()

    # SA on best found so far
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(10.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 2.0:
            h_sa, s_sa = _simulated_annealing_psl(
                best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()

    if best_h is None:
        best_h = np.ones(N)

    return best_h
