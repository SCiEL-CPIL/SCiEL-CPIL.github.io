import numpy as np
import time

N = 400

# Precompute FFT size
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

# Precompute phase factors for incremental FFT updates
# exp_factors[i, k] = exp(-2j*pi*i*k/FFT_SIZE)
_k = np.arange(FFT_SIZE // 2 + 1)
EXP_FACTORS = np.exp(-2j * np.pi * np.outer(np.arange(N), _k) / FFT_SIZE)


def _autoconv(x):
    """Compute full autoconvolution of x via FFT."""
    n = len(x)
    X = np.fft.rfft(x, FFT_SIZE)
    ac = np.fft.irfft(X * np.conj(X), FFT_SIZE)[:2 * n - 1]
    return ac


def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)


def _ac_from_fft(X):
    return np.fft.irfft(X * np.conj(X), FFT_SIZE)[:2 * N - 1]


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # ---------- Phase 0: Structured + random ±1 candidates ----------
    rng = np.random.default_rng(42)
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)

    candidates = [
        np.ones(N),
        np.array([1.0] * (N // 2) + [-1.0] * (N - N // 2)),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 4 < 2 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 8 < 4 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 16 < 8 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 32 < 16 else -1.0 for i in range(N)]),
    ]
    for c in candidates:
        c = np.asarray(c, dtype=np.float64)
        sc = _score(c)
        if sc < best_score:
            best_score = sc
            best_x = c.copy()

    # Quick random ±1 scan
    for _ in range(200):
        if time.time() > deadline - 90:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # ---------- Phase 1: Fast binary local search (incremental FFT) ----------
    x = best_x.copy()
    cur_score = best_score
    X = np.fft.rfft(x, FFT_SIZE)

    # Precompute sum for fast scoring
    s = np.sum(x)

    def _fast_flip_score(x, i, X, s):
        """Flip x[i], return (new_score, new_X, new_s)."""
        old_val = x[i]
        delta = -2.0 * old_val
        new_s = s + delta
        if abs(new_s) < 1e-12:
            return float('inf'), X, new_s
        X_new = X + delta * EXP_FACTORS[i]
        ac = _ac_from_fft(X_new)
        max_ac = np.max(np.abs(ac))
        score = max_ac / (new_s * new_s)
        return score, X_new, new_s

    # Simulated annealing on ±1 flips
    T = 0.05
    T_min = 1e-7
    cooling = 0.999
    rng2 = np.random.default_rng(123)
    no_improve = 0
    max_no_improve = 2000

    while time.time() < deadline - 40 and no_improve < max_no_improve:
        i = rng2.integers(0, N)
        old_val = x[i]
        x[i] = -old_val
        new_score, X_new, new_s = _fast_flip_score(x, i, X, s)

        if new_score < cur_score or rng2.random() < np.exp(-(new_score - cur_score) / max(T, 1e-12)):
            # Accept
            X = X_new
            s = new_s
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_x = x.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Reject
            x[i] = old_val
            no_improve += 1

        T *= cooling

        # Occasional restart from best
        if no_improve > 500:
            x = best_x.copy()
            X = np.fft.rfft(x, FFT_SIZE)
            s = np.sum(x)
            cur_score = best_score
            no_improve = 0
            T = max(T, 0.01)

    # ---------- Phase 2: Continuous local search (subgradient on max|ac|) ----------
    # Start from best binary solution, normalize to sum=1
    x = best_x.astype(np.float64).copy()
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s  # normalize to sum=1

    best_cont_score = _score(x)
    best_cont_x = x.copy()

    step_size = 0.02
    no_improve_c = 0
    max_no_improve_c = 300

    for it in range(20000):
        if time.time() > deadline - 5:
            break

        ac = _autoconv(x)
        abs_ac = np.abs(ac)
        k_max = np.argmax(abs_ac)
        val_max = ac[k_max]

        s = np.sum(x)
        cur_sc = abs(val_max) / (s * s) if abs(s) > 1e-12 else float('inf')

        if cur_sc < best_cont_score:
            best_cont_score = cur_sc
            best_cont_x = x.copy()
            no_improve_c = 0
        else:
            no_improve_c += 1
            if no_improve_c > max_no_improve_c:
                # Decay step size and reset counter
                step_size *= 0.7
                if step_size < 1e-6:
                    break
                no_improve_c = 0

        # Gradient of ac[k_max] w.r.t. x[j]
        # ac[k] = sum_i x[i]*x[k-i], d(ac[k])/dx[j] = x[k-j] + x[j-k]
        k = k_max
        grad = np.zeros(N)
        # Vectorized: for j in 0..N-1, term1 = x[k-j] if 0<=k-j<N, term2 = x[j-k] if 0<=j-k<N
        # term1: indices j where 0 <= k-j < N  =>  k-N < j <= k, i.e. max(0,k-N+1) <= j <= min(N-1,k)
        j_lo1 = max(0, k - N + 1)
        j_hi1 = min(N - 1, k)
        if j_lo1 <= j_hi1:
            js = np.arange(j_lo1, j_hi1 + 1)
            grad[js] += x[k - js]
        # term2: indices j where 0 <= j-k < N  =>  k <= j < k+N, i.e. max(0,k) <= j <= min(N-1, k+N-1)
        j_lo2 = max(0, k)
        j_hi2 = min(N - 1, k + N - 1)
        if j_lo2 <= j_hi2:
            js = np.arange(j_lo2, j_hi2 + 1)
            grad[js] += x[js - k]

        direction = -np.sign(val_max)
        x_new = x + step_size * direction * grad

        # Enforce sum(x_new) = 1
        x_new = x_new - np.mean(x_new) + 1.0 / N

        x = x_new

    # ---------- Final: compare and return best ----------
    final_cont_score = _score(best_cont_x)
    if final_cont_score < best_score:
        best_score = final_cont_score
        best_x = best_cont_x

    return best_x
