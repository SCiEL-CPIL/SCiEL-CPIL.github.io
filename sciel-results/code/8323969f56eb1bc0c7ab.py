import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2

# ---------------------------------------------------------------------------
# Fast exact single-flip evaluation.
#
# A[k] = sum_i h[i]*h[i+k]  (zero-padded, k = 0..N-1)
# Flipping bit i changes A[k] by  Delta[k] = 2*h[i]*(h[i+k]-h[i-k]),
# with out-of-range terms treated as 0.  Delta is computed in O(N) with
# vectorised slices, so every single-flip trial is O(N) (no FFT).
# ---------------------------------------------------------------------------
def _delta_for_flip(h, i):
    hi = h[i]
    return 2.0 * hi * (h[i:] + h[:i] - h[i::-1] - h[:i - 1:-1])


def _compute_acf(h):
    """Aperiodic autocorrelation ac[0..N-1] via FFT (used to seed a pass)."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe(h):
    """max_{k>=1} |sum_i h[i] h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _anneal_single_flip(seed, time_budget):
    """Simulated annealing with exact O(N) single-flip moves."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best_h = h.copy()
    best = cur
    T = 1.0
    T_min = 1e-8
    start = time.time()
    while time.time() - start < time_budget:
        i = int(rng.integers(0, N))
        d = _delta_for_flip(h, i)
        ac_new = ac + d
        new_val = float(np.max(np.abs(ac_new[1:])))
        delta = new_val - cur
        if delta <= 0.0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            ac = ac_new
            cur = new_val
            h[i] *= -1.0
            if cur < best:
                best = cur
                best_h = h.copy()
        T = max(T_min, T * 0.9999)
    return best_h, best


def _anneal_block(seed, time_budget):
    """Simulated annealing with block-flip moves (FFT evaluation)."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-8
    start = time.time()
    while time.time() - start < time_budget:
        r = rng.random()
        if r < 0.5:
            L = int(rng.integers(1, 5))
        elif r < 0.8:
            L = int(rng.integers(4, 13))
        elif r < 0.95:
            L = int(rng.integers(8, 25))
        else:
            L = int(rng.integers(16, 33))
        L = min(L, N)
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        ac_new = _compute_acf(h)
        new_val = float(np.max(np.abs(ac_new[1:])))
        delta = new_val - cur
        if delta <= 0.0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            ac = ac_new
            cur = new_val
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99995)
    return best_h, best


def _anneal_smooth(seed, time_budget):
    """Simulated annealing on a smooth log-sum-exp surrogate of the
    nonsmooth max|.| objective, using O(N) single-flip increments.
    Returns the best *exact* peak encountered along the way."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _compute_acf(h)
    a = np.abs(ac[1:])
    cur = float(np.max(a))
    best_h = h.copy()
    best = cur
    beta = 4.0
    beta_max = 120.0
    T = 2.0
    T_min = 1e-8
    start = time.time()
    while time.time() - start < time_budget:
        i = int(rng.integers(0, N))
        d = _delta_for_flip(h, i)
        a_new = np.abs(ac[1:] + d[1:])
        m = float(np.max(a_new))
        s = float(np.sum(np.exp(beta * a_new - beta * m)))
        new_val = m + np.log(s) / beta
        delta = new_val - cur
        if delta <= 0.0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            ac = ac + d
            cur = new_val
            h[i] *= -1.0
            exact = float(np.max(a_new))
            if exact < best:
                best = exact
                best_h = h.copy()
        T = max(T_min, T * 0.9999)
        beta = min(beta_max, beta * 1.001)
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """Repeated full single-flip passes (O(N) each) + block passes until a
    local optimum is reached or the deadline is hit."""
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        # ---- single-flip pass ----
        improved = False
        for i in range(N):
            d = _delta_for_flip(h, i)
            ac_try = ac + d
            c = float(np.max(np.abs(ac_try[1:])))
            if c < cur - 1e-15:
                ac = ac_try
                cur = c
                h[i] *= -1.0
                improved = True
                if cur < best:
                    best = cur
                    best_h = h.copy()
        if not improved:
            break
        # ---- small block-flip pass (O(N^2), cheap) ----
        improved = False
        for L in (1, 2, 3):
            for i in range(0, N - L + 1):
                h[i:i + L] *= -1.0
                ac_try = _compute_acf(h)
                c = float(np.max(np.abs(ac_try[1:])))
                if c < cur - 1e-15:
                    ac = ac_try
                    cur = c
                    improved = True
                    if cur < best:
                        best = cur
                        best_h = h.copy()
                else:
                    h[i:i + L] *= -1.0
        if not improved:
            break
    return best_h, best


def construct():
    try:
        hard_deadline = 140.0
        start = time.time()
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        # Long single-flip anneal from a random +-1 start.
        budget = min(45.0, hard_deadline - (time.time() - start) - 30.0)
        if budget > 1.0:
            h, c = _anneal_single_flip(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h

        # A few diverse block-anneal restarts.
        restart_id = 1
        while time.time() - start < hard_deadline - 45.0:
            sub = min(10.0, hard_deadline - (time.time() - start) - 45.0)
            if sub < 1.0:
                break
            h, c = _anneal_block(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1

        # Smooth-surrogate (log-sum-exp) anneal to escape local minima.
        budget = min(25.0, hard_deadline - (time.time() - start) - 15.0)
        if budget > 1.0:
            h, c = _anneal_smooth(987654, budget)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Final high-precision greedy polish with whatever time remains.
        remaining = hard_deadline - (time.time() - start)
        if remaining > 2.0:
            h, c = _greedy_polish(global_best_h, remaining - 3.0, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return global_best_h
    except Exception:
        return np.ones(N)
