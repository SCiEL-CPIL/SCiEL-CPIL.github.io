import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    # Precompute phase factors for incremental FFT updates
    k = np.arange(fft_size // 2 + 1)
    exp_factors = np.exp(-2j * np.pi * np.outer(np.arange(N), k) / fft_size)

    def _score_fast(x, X=None):
        """Fast score computation using precomputed FFT."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        if X is None:
            X = np.fft.rfft(x, fft_size)
        ac = np.fft.irfft(X * np.conj(X), fft_size)[:2*N-1]
        max_ac = np.max(np.abs(ac))
        return max_ac / (s * s)

    def _flip_score(x, i, X):
        """Compute score after flipping index i, without modifying x or X."""
        old_val = x[i]
        X_new = X - 2.0 * old_val * exp_factors[i]
        s = np.sum(x) - 2.0 * old_val
        if abs(s) < 1e-12:
            return float('inf'), X_new
        ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
        max_ac = np.max(np.abs(ac))
        return max_ac / (s * s), X_new

    # Phase 1: Initial candidates
    rng = np.random.default_rng(12345)
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)

    # Try structured candidates
    candidates = [
        np.ones(N),
        np.array([1.0 if i < N//2 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 3 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if (i // 8) % 2 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if (i // 16) % 2 == 0 else -1.0 for i in range(N)]),
        # Sparse candidates
        np.array([1.0 if i % 5 == 0 else 0.0 for i in range(N)]),
        np.array([1.0 if i % 7 == 0 else -1.0 if i % 7 == 3 else 0.0 for i in range(N)]),
    ]
    for c in candidates:
        c = np.asarray(c, dtype=np.float64)
        sc = _score(c)
        if sc < best_score:
            best_score = sc
            best_x = c.copy()

    # Try random ±1 candidates
    for seed in range(200):
        if time.time() > deadline - 90:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # Phase 2: Sparse Bernoulli candidates
    # The key insight: for a sparse random sign vector, the autoconvolution
    # has smaller peak relative to the sum squared
    for density in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
        for trial in range(50):
            if time.time() > deadline - 80:
                break
            mask = rng.random(N) < density
            signs = rng.choice([-1.0, 1.0], size=N)
            x = np.where(mask, signs, 0.0)
            s = np.sum(x)
            if abs(s) < 1e-6:
                continue
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()

    # Phase 3: Local search with single flips
    x = best_x.copy()
    X = np.fft.rfft(x, fft_size)
    cur_score = _score_fast(x, X)

    rng2 = np.random.default_rng(67890)
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline - 20 and no_improve < max_no_improve:
        i = rng2.integers(0, N)
        old_val = x[i]
        new_score, X_new = _flip_score(x, i, X)

        if new_score < cur_score:
            x[i] = -old_val
            X = X_new
            cur_score = new_score
            no_improve = 0
            if cur_score < best_score:
                best_score = cur_score
                best_x = x.copy()
        else:
            no_improve += 1

    # Phase 4: Systematic sweep
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(N):
            if time.time() > deadline - 10:
                break
            old_val = x[i]
            new_score, X_new = _flip_score(x, i, X)
            if new_score < cur_score:
                x[i] = -old_val
                X = X_new
                cur_score = new_score
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()

    # Phase 5: Multi-flip search
    if time.time() < deadline - 5:
        x = best_x.copy()
        X = np.fft.rfft(x, fft_size)
        cur_score = _score_fast(x, X)
        rng3 = np.random.default_rng(11111)
        no_improve = 0
        max_no_improve = 200

        while time.time() < deadline - 5 and no_improve < max_no_improve:
            i1 = rng3.integers(0, N)
            i2 = rng3.integers(0, N)
            if i1 == i2:
                continue
            v1, v2 = x[i1], x[i2]
            X_new = X - 2.0 * v1 * exp_factors[i1] - 2.0 * v2 * exp_factors[i2]
            s = np.sum(x) - 2.0 * v1 - 2.0 * v2
            if abs(s) < 1e-12:
                new_score = float('inf')
            else:
                ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
                max_ac = np.max(np.abs(ac))
                new_score = max_ac / (s * s)

            if new_score < cur_score:
                x[i1] = -v1
                x[i2] = -v2
                X = X_new
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()
            else:
                no_improve += 1

    # Final verification
    final_score = _score(best_x)
    if final_score < best_score:
        best_score = final_score

    return best_x
