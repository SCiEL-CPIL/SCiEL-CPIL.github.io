"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (as +1/-1 array)."""
    # Thue-Morse: t[0]=0, t[2k]=t[k], t[2k+1]=1-t[k]
    # Build iteratively
    seq = np.array([1], dtype=np.int8)  # start with +1
    while len(seq) < n:
        seq = np.concatenate([seq, -seq])
    return seq[:n].astype(np.float64)


def _autoconv_peak(x):
    """Compute max |autocorrelation| at non-zero lags for a ±1 sequence."""
    n = len(x)
    # Full autocorrelation via FFT
    f = np.fft.fft(x, n=2 * n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    # ac[0] = sum x^2 = n; we want max |ac[k]| for k=1..n-1
    if n <= 1:
        return 0.0
    return np.max(np.abs(ac[1:]))


def _autoconv_peak_direct(x):
    """Direct O(n^2) autocorrelation peak for verification/small n."""
    n = len(x)
    if n <= 1:
        return 0.0
    best = 0.0
    for k in range(1, n):
        s = np.dot(x[:n - k], x[k:])
        if abs(s) > best:
            best = abs(s)
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Start with Thue-Morse sequence
    x = _thue_morse(N)
    current_peak = _autoconv_peak(x)
    best_x = x.copy()
    best_peak = current_peak

    # Local search: try flipping individual bits
    rng = np.random.default_rng(42)
    flip_order = rng.permutation(N)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in flip_order:
            if time.time() > deadline:
                break
            # Try flipping bit i
            x[i] *= -1
            new_peak = _autoconv_peak(x)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                best_x = x.copy()
                improved = True
            else:
                x[i] *= -1  # revert
        if improved:
            # Restart flip order
            flip_order = rng.permutation(N)

    # Second phase: random restarts with local search if time permits
    while time.time() < deadline - 5.0:
        # Random starting point
        x = np.where(rng.random(N) < 0.5, 1.0, -1.0)
        current_peak = _autoconv_peak(x)
        if current_peak >= best_peak:
            continue

        # Local search from this random start
        flip_order = rng.permutation(N)
        local_improved = True
        while local_improved and time.time() < deadline - 2.0:
            local_improved = False
            for i in flip_order:
                if time.time() > deadline - 2.0:
                    break
                x[i] *= -1
                new_peak = _autoconv_peak(x)
                if new_peak < best_peak - 1e-12:
                    best_peak = new_peak
                    best_x = x.copy()
                    local_improved = True
                else:
                    x[i] *= -1
            if local_improved:
                flip_order = rng.permutation(N)

    return best_x
# EVOLVE-BLOCK-END
