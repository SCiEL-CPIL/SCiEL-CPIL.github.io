"""Low-peak-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound
below the all-ones baseline.

Design family (kept): simulated annealing + greedy polish, now with
  * an O(N) incremental autocorrelation update for single-flips (no FFT per trial),
  * a smooth epigraph / soft-peak (log-sum-exp) surrogate for exploration,
  * a high-precision greedy single-flip polish,
  * a multi-start strategy with diverse initializers (Random, Hadamard-like, Turing-like),
  * an explicit 140 s deadline guard (10 s safety under the 150 s kill).
"""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def _ac_full(h):
    """Full aperiodic autocorrelation ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k], k=0..N-1."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe(h):
    ac = _ac_full(h)
    return float(np.max(np.abs(ac[1:])))


def _soft_peak(ac, tau):
    """Smooth epigraph surrogate: log-sum-exp of ac[k]^2/tau over k>=1."""
    v = ac[1:] ** 2 / tau
    m = float(np.max(v))
    return m + float(np.log(np.sum(np.exp(v - m))))


def _flip_ac_fast(ac, h, i):
    """Vectorised O(N) update of ac after flipping h[i]."""
    new_ac = ac.copy()
    hi = h[i]
    k_hi = N - 1 - i
    if k_hi >= 0:
        new_ac[0:k_hi + 1] += -2.0 * hi * h[i:i + k_hi + 1]
    k_lo = i
    if k_lo >= 1:
        new_ac[1:k_lo + 1] += -2.0 * hi * h[i - k_lo:i]
    return new_ac


# ---------------------------------------------------------------------------
# Initializers
# ---------------------------------------------------------------------------

def _init_random(seed):
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=N)

def _init_hadamard(seed):
    # Generate a pseudo-random Walsh-Hadamard-like sequence
    # Use a random projection of a Hadamard matrix row, or just a structured random
    # For simplicity, use a random sequence but with a specific structure:
    # h[i] = sign(sin(2*pi*i/N * k)) for random k? 
    # Let's use a random linear combination of low-frequency sines.
    rng = np.random.default_rng(seed)
    h = np.zeros(N)
    for k in range(1, 5):
        phase = rng.random() * 2 * np.pi
        h += np.sin(2 * np.pi * k * np.arange(N) / N + phase)
    return np.sign(h)

def _init_turing(seed):
    # Simple 1D reaction-diffusion proxy: 
    # Start with noise, apply a simple smoothing + thresholding iteration
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    # Iterate a simple cellular automaton rule
    for _ in range(10):
        # Local average
        h_smooth = np.convolve(h, np.ones(5)/5.0, mode='same')
        # Threshold with hysteresis or just sign
        h = np.sign(h_smooth)
        # Add a bit of noise to break symmetry
        mask = rng.random(N) < 0.05
        h[mask] *= -1.0
    return h


# ---------------------------------------------------------------------------
# Annealing
# ---------------------------------------------------------------------------

def _anneal(h_init, seed, time_budget):
    rng = np.random.default_rng(seed)
    h = h_init.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 4.0
    T_min = 1e-7
    tau = 8.0
    start = time.time()
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        # Prefer single flips for precision, occasionally blocks
        if rng.random() < 0.8:
            i = int(rng.integers(0, N))
            h[i] *= -1.0
            ac_new = _flip_ac_fast(ac, h, i)
            new = float(np.max(np.abs(ac_new[1:])))
            d = _soft_peak(ac_new, tau) - _soft_peak(ac, tau)
            if new <= cur or rng.random() < np.exp(-d / max(T, 1e-9)):
                cur = new
                ac = ac_new
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                h[i] *= -1.0
        else:
            L = int(rng.integers(2, 10))
            i = int(rng.integers(0, N - L + 1))
            h[i:i + L] *= -1.0
            ac_new = _ac_full(h)
            new = float(np.max(np.abs(ac_new[1:])))
            d = _soft_peak(ac_new, tau) - _soft_peak(ac, tau)
            if new <= cur or rng.random() < np.exp(-d / max(T, 1e-9)):
                cur = new
                ac = ac_new
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                h[i:i + L] *= -1.0
        T = max(T_min, T * 0.9999)
        tau = max(1.0, tau * 0.999)
    return best_h, best


# ---------------------------------------------------------------------------
# Greedy Polish
# ---------------------------------------------------------------------------

def _greedy_single_only(h, time_budget, start):
    """Tight single-flip polish with the O(N) update."""
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c:
                best_c = c
                best_i = i
        if best_i < 0:
            break
        h[best_i] *= -1.0
        ac = _flip_ac_fast(ac, h, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        def _remaining():
            return hard_deadline - (time.time() - start)

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        # Multi-start strategy with diverse initializers
        initializers = [
            ("random", _init_random(1)),
            ("random2", _init_random(2)),
            ("hadamard", _init_hadamard(3)),
            ("turing", _init_turing(4)),
            ("random3", _init_random(5)),
            ("turing2", _init_turing(6)),
        ]

        # Phase 1: Quick anneal + polish on each initializer
        for name, h_init in initializers:
            if _remaining() < 20.0:
                break
            budget = min(10.0, _remaining() - 20.0)
            if budget < 2.0:
                continue
            
            h, c = _anneal(h_init, hash(name) % 10000, budget)
            if c < global_best:
                global_best = c
                global_best_h = h.copy()
            
            # Polish
            if _remaining() > 15.0:
                sub_budget = min(5.0, _remaining() - 15.0)
                h2, c2 = _greedy_single_only(global_best_h, sub_budget, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: Long polish on the best found so far
        if _remaining() > 10.0:
            h, c = _greedy_single_only(global_best_h, _remaining() - 5.0, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 3: Final safety check
        if _remaining() > 2.0:
             h, c = _greedy_single_only(global_best_h, _remaining() - 1.0, time.time())
             if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
