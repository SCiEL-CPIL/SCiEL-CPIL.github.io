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
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    # Phase 1: Gradient descent on continuous values to find good starting region
    x_best = np.ones(N, dtype=np.float64)
    best_score = _score(x_best)

    x = np.ones(N, dtype=np.float64)
    x /= np.sum(x)  # normalize sum to 1

    rng = np.random.default_rng(42)

    # Smooth max surrogate for gradient computation
    T = 30.0

    def _smooth_max_abs_ac(ac):
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        shifted = T * (abs_ac - m)
        shifted = np.clip(shifted, -100, 100)
        exp_shifted = np.exp(shifted)
        log_sum = np.log(np.sum(exp_shifted))
        value = m + log_sum / T
        weights = exp_shifted / np.sum(exp_shifted)
        grad_ac = weights * np.sign(ac)
        return value, grad_ac

    def _grad_smooth_max(x):
        ac = _autoconv(x)
        value, grad_ac = _smooth_max_abs_ac(ac)

        grad_ac_padded = np.zeros(fft_size, dtype=np.float64)
        grad_ac_padded[:len(grad_ac)] = grad_ac

        X = np.fft.rfft(x, fft_size)
        G = np.fft.rfft(grad_ac_padded, fft_size)
        grad_x = np.fft.irfft(2 * X * G, fft_size)[:N]

        return value, grad_x

    step_size = 0.003
    no_improve = 0
    max_no_improve = 80

    for iter in range(3000):
        if time.time() > deadline - 60:
            break

        value, grad_x = _grad_smooth_max(x)

        s = np.sum(x)
        ac = _autoconv(x)
        max_ac = np.max(np.abs(ac))
        cur_score = max_ac / (s * s)

        if cur_score < best_score:
            best_score = cur_score
            x_best = x.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break

        grad_norm = np.linalg.norm(grad_x)
        if grad_norm < 1e-15:
            break
        grad_normed = grad_x / grad_norm

        x_new = x - step_size * grad_normed
        x_new = x_new - np.mean(x_new) + 1.0 / N
        x = x_new

        if no_improve > 25:
            step_size *= 0.85

    # Phase 2: Convert to binary and do fast local search with incremental FFT
    # Threshold the continuous solution to get binary starting point
    x_bin = np.sign(x_best)
    x_bin[x_bin == 0] = 1.0  # handle zeros

    # Also try a few random binary initializations
    for trial in range(10):
        if time.time() > deadline - 55:
            break
        x_rand = rng.choice([-1.0, 1.0], size=N)
        sc = _score(x_rand)
        if sc < _score(x_bin):
            x_bin = x_rand.copy()

    # Incremental FFT local search on binary sequence
    x = x_bin.copy()
    X = np.fft.rfft(x, fft_size)

    # Precompute phase factors for incremental updates
    k = np.arange(fft_size // 2 + 1)
    exp_factors = np.exp(-2j * np.pi * np.outer(np.arange(N), k) / fft_size)

    def _flip_score(x, i, X):
        """Compute score after flipping x[i] using incremental FFT update."""
        old_val = x[i]
        delta = -2.0 * old_val
        X_new = X + delta * exp_factors[i]
        ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
        max_ac = np.max(np.abs(ac))
        s = np.sum(x) + delta  # sum changes by delta
        if abs(s) < 1e-12:
            return float('inf'), X_new
        return max_ac / (s * s), X_new

    cur_score = _score(x)
    if cur_score < best_score:
        best_score = cur_score
        x_best = x.copy()

    # Local search: random flips with acceptance
    rng2 = np.random.default_rng(123)
    T_sa = 0.5
    T_min = 1e-4
    cooling = 0.9998
    no_improve = 0
    max_no_improve = 200

    while time.time() < deadline - 25 and no_improve < max_no_improve:
        if T_sa < T_min:
            T_sa = T_min

        i = rng2.integers(0, N)
        old_val = x[i]
        x[i] = -old_val
        X_new = X - 2.0 * old_val * exp_factors[i]

        ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
        max_ac = np.max(np.abs(ac))
        s = np.sum(x)
        if abs(s) < 1e-12:
            new_score = float('inf')
        else:
            new_score = max_ac / (s * s)

        # Accept if better or with probability exp(-delta/T)
        if new_score < cur_score or (np.random.random() < np.exp(-(new_score - cur_score) / max(T_sa, 1e-10))):
            X = X_new
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                x_best = x.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            x[i] = old_val  # revert
            no_improve += 1

        T_sa *= cooling

    # Phase 3: Exhaustive single-flip polish
    x = x_best.copy()
    X = np.fft.rfft(x, fft_size)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old_val = x[i]
            x[i] = -old_val
            X_new = X - 2.0 * old_val * exp_factors[i]
            ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
            max_ac = np.max(np.abs(ac))
            s = np.sum(x)
            if abs(s) < 1e-12:
                sc = float('inf')
            else:
                sc = max_ac / (s * s)
            if sc < best_score:
                best_score = sc
                x_best = x.copy()
                X = X_new
                improved = True
            else:
                x[i] = old_val  # revert

    # Final verification
    final_score = _score(x_best)
    if final_score < best_score:
        best_score = final_score

    return x_best
