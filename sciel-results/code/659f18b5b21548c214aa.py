"""Third autocorrelation inequality — chaotic-map seeding + simulated annealing."""
import numpy as np
import time

N = 400
DEADLINE = 115.0  # safety margin below 150s evaluator limit
FFT_SIZE = 4096   # next power of 2 > 3*N = 1200, ensures linear conv


def _compute_c3(h):
    """
    Compute C3[k] = sum_i h[i] * h[i+k] * h[i+2k] via FFT.
    This is the linear convolution of h with (h*h).
    Returns array of length N (lags 0..N-1).
    """
    h_pad = np.zeros(FFT_SIZE, dtype=np.float64)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    C3_full = np.fft.irfft(H * H * H, n=FFT_SIZE)
    # Linear convolution of length 3N-1; lags 0..N-1 are at indices 0..N-1
    return C3_full[:N]


def _score(h):
    """Upper bound on C3: max |C3[k]| for k=0..N-1. Lower is better."""
    c3 = _compute_c3(h)
    return np.max(np.abs(c3))


def _logistic_sign_seq(seed, r=4.0, n=N):
    """Generate a ±1 sequence from logistic map sampled at n points."""
    x = np.random.RandomState(seed).uniform(0.01, 0.99)
    signs = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = r * x * (1.0 - x)
        signs[i] = 1.0 if x >= 0.5 else -1.0
    return signs


def _golden_angle_sign_seq(n=N):
    """Generate a ±1 sequence via golden-angle rotation (low-discrepancy)."""
    golden = (1 + np.sqrt(5)) / 2.0
    signs = np.empty(n, dtype=np.float64)
    for i in range(n):
        theta = (i * golden) % 1.0
        signs[i] = 1.0 if theta < 0.5 else -1.0
    return signs


def _random_block_flip(h, rng, block_sizes):
    """Flip a random contiguous block of random size from block_sizes."""
    h_new = h.copy()
    blen = rng.choice(block_sizes)
    start = rng.randint(0, N - blen + 1)
    h_new[start:start+blen] *= -1.0
    return h_new


def construct():
    """
    Construct a signed step function with 400 equal intervals.
    Returns a sequence of signed step heights (±1).
    Score is max|C3|, lower is better.
    """
    start_time = time.time()

    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    # --- Phase 1: Generate diverse initial candidates ---
    candidates = []

    # Chaotic logistic map seeds
    for seed in range(20):
        candidates.append(_logistic_sign_seq(seed))

    # Golden angle sequence
    candidates.append(_golden_angle_sign_seq())

    # Sine-based patterns
    x = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64]:
        for phase in [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]:
            candidates.append(np.sign(np.sin(2 * np.pi * freq * x + phase)))

    # Block patterns
    for blen in [2, 3, 4, 5, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 80, 100]:
        for half in range(1, blen // 2 + 1):
            pattern = np.ones(blen)
            pattern[half:] = -1
            candidates.append(np.tile(pattern, N // blen + 1)[:N])

    # Random
    rng0 = np.random.RandomState(42)
    for _ in range(10):
        candidates.append(rng0.choice([-1.0, 1.0], size=N))

    # Score all candidates, keep best
    best_h = None
    best_score = np.inf
    for h in candidates:
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    if not elapsed_ok():
        return np.sign(best_h)

    # --- Phase 2: Simulated annealing with block flips ---
    h_cur = best_h.copy()
    s_cur = best_score
    h_best = best_h.copy()
    s_best = best_score

    rng = np.random.RandomState(123)
    block_sizes = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 80])

    # Annealing schedule
    T_start = 2.0
    T_end = 0.001
    # We'll do as many iterations as fit in remaining time
    n_iterations_target = 200000

    # Pre-compute time per iteration estimate
    t_iter_start = time.time()
    # Warm up: time a few iterations
    for _ in range(5):
        h_try = _random_block_flip(h_cur, rng, block_sizes)
        s_try = _score(h_try)
        h_cur = h_try
        s_cur = s_try
    t_warm = (time.time() - t_iter_start) / 5.0

    remaining_time = DEADLINE - (time.time() - start_time)
    n_iters = min(n_iterations_target, int(remaining_time / max(t_iter_start, 1e-6)))
    n_iters = max(n_iters, 100)

    for it in range(n_iters):
        if not elapsed_ok():
            break

        # Temperature: geometric cooling
        frac = it / max(n_iters - 1, 1)
        T = T_start * (T_end / T_start) ** frac

        # Propose: random block flip
        h_try = _random_block_flip(h_cur, rng, block_sizes)
        s_try = _score(h_try)

        delta = s_try - s_cur
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            h_cur = h_try
            s_cur = s_try
            if s_cur < s_best:
                s_best = s_cur
                h_best = h_cur.copy()

    # --- Phase 3: Local greedy refinement (single-bit flips) ---
    if elapsed_ok():
        h_work = h_best.copy()
        s_work = s_best
        improved = True
        max_rounds = 5
        for rnd in range(max_rounds):
            if not elapsed_ok():
                break
            improved = False
            order = rng.permutation(N)
            for idx in order:
                if not elapsed_ok():
                    break
                h_try = h_work.copy()
                h_try[idx] *= -1.0
                s_try = _score(h_try)
                if s_try < s_work - 1e-12:
                    h_work = h_try
                    s_work = s_try
                    improved = True
            if s_work < s_best:
                s_best = s_work
                h_best = h_work.copy()
            if not improved:
                break

    # --- Phase 4: A few more SA restarts if time permits ---
    if elapsed_ok():
        for restart in range(3):
            if not elapsed_ok():
                break
            # Perturb the best solution
            h_restart = h_best.copy()
            for _ in range(20):
                h_restart = _random_block_flip(h_restart, rng, block_sizes)
            s_restart = _score(h_restart)

            h_cur = h_restart
            s_cur = s_restart
            if s_cur < s_best:
                s_best = s_cur
                h_best = h_cur.copy()

            # Short SA from this restart
            T_start_r = 0.5
            T_end_r = 0.01
            n_iters_r = 20000
            for it in range(n_iters_r):
                if not elapsed_ok():
                    break
                frac = it / max(n_iters_r - 1, 1)
                T = T_start_r * (T_end_r / T_start_r) ** frac
                h_try = _random_block_flip(h_cur, rng, block_sizes)
                s_try = _score(h_try)
                delta = s_try - s_cur
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
                    h_cur = h_try
                    s_cur = s_try
                    if s_cur < s_best:
                        s_best = s_cur
                        h_best = h_cur.copy()

    # Final: ensure ±1
    best_h = np.sign(h_best).astype(np.float64)
    if np.all(best_h == 0):
        best_h = np.ones(N)
    return best_h


if __name__ == "__main__":
    h = construct()
    print("Constructed first 20:", h[:20])
    print("Score:", _score(h))
