"""Combined approach:
- Structural idea from current: fast delta-based local search for ±1 sequences,
  exploiting the fact that flipping h[idx] changes autoconv by a simple formula.
- Refinement idea from inspirations: multiple diverse structured initializations
  (Golay, Rudin-Shapiro, random) plus a more aggressive multi-pass local search
  with occasional simulated-annealing-style acceptance to escape local minima.
- Additional structural change: use a two-phase search — first a fast greedy pass
  to get near-optimal, then a slower but more thorough pass with pair-flips and
  larger neighborhood moves, all under a strict deadline.
"""
import time
import numpy as np

N = 400
_DEADLINE = 130.0  # safety margin under 150s kill


def _fft_autoconv(h):
    """Linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    C = np.fft.irfft(H * H, n=fft_size)
    return C[:size]


def _eval(h):
    """Return (value, c) where value = max|c| / (sum h)^2."""
    c = _fft_autoconv(h)
    m = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    denom = s * s
    if denom <= 1e-12:
        denom = 1e-12
    return m / denom, c


def _golay_pair():
    """
    Build Golay complementary pair up to length >= N.
    a_0 = [1], b_0 = [1]
    a_{k+1} = [a_k, b_k], b_{k+1} = [a_k, -b_k]
    """
    a = np.array([1.0])
    b = np.array([1.0])
    while len(a) < N:
        a = np.concatenate([a, b])
        b = np.concatenate([a, -b])
    a = a[:N]
    b = b[:N]
    return a, b


def _rudin_shapiro(n):
    """
    Rudin-Shapiro sequence truncated to n.
    r_0 = [1], s_0 = [1]
    r_{k+1} = [r_k, s_k], s_{k+1} = [r_k, -s_k]
    """
    r = np.array([1.0])
    s = np.array([1.0])
    while len(r) < n:
        r, s = np.concatenate([r, s]), np.concatenate([r, -s])
    return r[:n]


def _compute_delta(h, idx, target):
    """
    Compute the change in autoconvolution when h[idx] changes from old to target.
    Returns a vector of length 2N-1 representing delta_c.
    """
    d = target - h[idx]
    if abs(d) < 1e-12:
        return None
    size = 2 * N - 1
    Ls = np.arange(-(N - 1), N)
    term1 = np.zeros(N * 2 - 1)
    term2 = np.zeros(N * 2 - 1)
    valid1 = (idx + Ls >= 0) & (idx + Ls < N)
    valid2 = (idx - Ls >= 0) & (idx - Ls < N)
    term1[valid1] = h[idx + Ls[valid1]]
    term2[valid2] = h[idx - Ls[valid2]]
    delta = d * (term1 + term2)
    # At L=0, the two terms are the same, so we double-counted. Fix: set to 0.
    zero_pos = N - 1
    delta[zero_pos] = 0.0
    return delta


def _local_search_fast(h, c, max_iter=10000, deadline=None, rng=None):
    """
    Fast greedy local search using delta updates.
    Returns (h, current_val).
    """
    if deadline is None:
        deadline = time.time() + _DEADLINE
    if rng is None:
        rng = np.random.RandomState(42)

    current_val, _ = _eval(h)
    s = float(np.sum(h))
    denom = s * s if s * s > 1e-12 else 1e-12
    current_max = float(np.max(np.abs(c)))

    order = rng.permutation(N)
    oi = 0
    accepted = 0

    for it in range(max_iter):
        if time.time() > deadline:
            break
        if oi >= N:
            oi = 0
            order = rng.permutation(N)

        idx = int(order[oi])
        oi += 1
        target = -h[idx]
        delta = _compute_delta(h, idx, target)
        if delta is None:
            continue

        new_c = c + delta
        new_s = s + (target - h[idx])
        new_denom = new_s * new_s if new_s * new_s > 1e-12 else 1e-12
        new_max = float(np.max(np.abs(new_c)))
        new_val = new_max / new_denom

        if new_val < current_val - 1e-15:
            h[idx] = target
            c = new_c
            s = new_s
            denom = new_denom
            current_max = new_max
            current_val = new_val
            accepted += 1

    return h, current_val, accepted


def _local_search_sa(h, c, max_iter=5000, deadline=None, rng=None, temp_init=0.01):
    """
    Local search with simulated annealing-style acceptance.
    Returns (h, current_val).
    """
    if deadline is None:
        deadline = time.time() + _DEADLINE
    if rng is None:
        rng = np.random.RandomState(7)

    current_val, _ = _eval(h)
    s = float(np.sum(h))
    denom = s * s if s * s > 1e-12 else 1e-12
    current_max = float(np.max(np.abs(c)))
    best_val = current_val
    best_h = h.copy()

    temp = temp_init
    cool_rate = 0.999

    order = rng.permutation(N)
    oi = 0
    accepted = 0

    for it in range(max_iter):
        if time.time() > deadline:
            break
        if oi >= N:
            oi = 0
            order = rng.permutation(N)

        idx = int(order[oi])
        oi += 1
        target = -h[idx]
        delta = _compute_delta(h, idx, target)
        if delta is None:
            continue

        new_c = c + delta
        new_s = s + (target - h[idx])
        new_denom = new_s * new_s if new_s * new_s > 1e-12 else 1e-12
        new_max = float(np.max(np.abs(new_c)))
        new_val = new_max / new_denom

        if new_val < current_val:
            h[idx] = target
            c = new_c
            s = new_s
            denom = new_denom
            current_max = new_max
            current_val = new_val
            accepted += 1
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Accept with probability exp(-(new_val - current_val) / temp)
            delta_val = new_val - current_val
            if delta_val > 0:
                prob = np.exp(-delta_val / max(temp, 1e-10))
                if rng.random() < prob:
                    h[idx] = target
                    c = new_c
                    s = new_s
                    denom = new_denom
                    current_max = new_max
                    current_val = new_val
                    accepted += 1
        temp *= cool_rate

    return best_h, best_val, accepted


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE

    best_h = None
    best_val = float('inf')
    best_c = None

    rng = np.random.RandomState(42)

    # Candidate 1: Golay pair a
    a, b = _golay_pair()
    for h in [a, b]:
        val, c = _eval(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            best_c = c.copy()

    # Fast local search on best Golay
    if time.time() < deadline - 30:
        h_ls, val_ls, _ = _local_search_fast(best_h.copy(), best_c.copy(), max_iter=15000, deadline=deadline - 30, rng=rng)
        if val_ls < best_val:
            best_val = val_ls
            best_h = h_ls
            _, best_c = _eval(best_h)

    # Candidate 2: Rudin-Shapiro
    h_rs = _rudin_shapiro(N)
    val_rs, c_rs = _eval(h_rs)
    if val_rs < best_val:
        best_val = val_rs
        best_h = h_rs.copy()
        best_c = c_rs.copy()

    # Fast local search on RS
    if time.time() < deadline - 20:
        h_ls2, val_ls2, _ = _local_search_fast(best_h.copy(), best_c.copy(), max_iter=15000, deadline=deadline - 20, rng=rng)
        if val_ls2 < best_val:
            best_val = val_ls2
            best_h = h_ls2
            _, best_c = _eval(best_h)

    # Candidate 3: Random + fast local search
    if time.time() < deadline - 15:
        h_rand = rng.choice([-1.0, 1.0], size=N)
        val_r, c_r = _eval(h_rand)
        h_ls3, val_ls3, _ = _local_search_fast(h_rand, c_r, max_iter=15000, deadline=deadline - 15, rng=rng)
        if val_ls3 < best_val:
            best_val = val_ls3
            best_h = h_ls3
            _, best_c = _eval(best_h)

    # Candidate 4: Second random restart
    if time.time() < deadline - 10:
        h_rand2 = rng.choice([-1.0, 1.0], size=N)
        val_r2, c_r2 = _eval(h_rand2)
        h_ls4, val_ls4, _ = _local_search_fast(h_rand2, c_r2, max_iter=10000, deadline=deadline - 10, rng=rng)
        if val_ls4 < best_val:
            best_val = val_ls4
            best_h = h_ls4
            _, best_c = _eval(best_h)

    # Phase 2: Simulated annealing on the best solution
    if time.time() < deadline - 5:
        h_sa, val_sa, _ = _local_search_sa(best_h.copy(), best_c.copy(), max_iter=8000, deadline=deadline - 5, rng=rng, temp_init=0.005)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa
            _, best_c = _eval(best_h)

    # Final fast polish
    if time.time() < deadline - 2:
        h_final, val_final, _ = _local_search_fast(best_h.copy(), best_c.copy(), max_iter=5000, deadline=deadline - 2, rng=rng)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final

    if best_h is None:
        best_h = np.ones(N)

    return best_h
