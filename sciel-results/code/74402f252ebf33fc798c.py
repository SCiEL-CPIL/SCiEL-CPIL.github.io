import numpy as np
import time

N = 400

def _autoconv_max_abs(f):
    """Compute max absolute value of autoconvolution of f using FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size=size)
    ac = np.fft.irfft(F * np.conj(F), size=size)[:n]
    return np.max(np.abs(ac))

def _squared_sum(f):
    return (np.sum(f)) ** 2

def _score(f):
    s = _squared_sum(f)
    if s == 0:
        return np.inf
    return _autoconv_max_abs(f) / s

def _chirp_init(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    phase = 2 * np.pi * (t * t) / (2 * N)
    f = np.sign(np.cos(phase))
    flip_mask = rng.random(N) < 0.1
    f[flip_mask] *= -1
    return f

def _block_init(seed):
    rng = np.random.default_rng(seed)
    f = np.ones(N)
    pos = 0
    while pos < N:
        block_len = int(rng.integers(3, 20))
        block_len = min(block_len, N - pos)
        sign = int(rng.choice([-1, 1]))
        f[pos:pos + block_len] = sign
        pos += block_len
    return f

def _legendre_init(seed):
    rng = np.random.default_rng(seed)
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    p = N
    while not is_prime(p):
        p -= 1
    f = np.ones(p)
    for i in range(1, p):
        ls = pow(i, (p - 1) // 2, p)
        f[i] = 1 if ls == 1 else -1
    f_ext = np.tile(f, N // p + 1)[:N]
    flip_mask = rng.random(N) < 0.05
    f_ext[flip_mask] *= -1
    return f_ext

def _local_search(f, max_iters=50000, time_limit=10):
    f = f.copy()
    best_score = _score(f)
    start_time = time.time()

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        i = np.random.randint(N)
        f[i] *= -1
        new_score = _score(f)
        if new_score < best_score:
            best_score = new_score
        else:
            f[i] *= -1

    return f, best_score

def _simulated_annealing(f, time_limit=15):
    f = f.copy()
    best_f = f.copy()
    best_score = _score(f)
    start_time = time.time()
    T = 0.1
    T_min = 0.001
    cooling = 0.995

    while time.time() - start_time < time_limit:
        if T < T_min:
            break
        i = np.random.randint(N)
        f[i] *= -1
        new_score = _score(f)
        delta = new_score - best_score

        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        else:
            f[i] *= -1

        T *= cooling

    return best_f, _score(best_f)

def construct():
    start_time = time.time()
    deadline = start_time + 110

    best_f = None
    best_score = np.inf

    inits = []
    inits.append(_chirp_init(42))
    inits.append(_chirp_init(123))
    inits.append(_block_init(7))
    inits.append(_block_init(99))
    inits.append(_legendre_init(5))
    inits.append(_legendre_init(88))
    inits.append(np.ones(N))

    for f in inits:
        s = _score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()

    scored_inits = sorted(inits, key=lambda x: _score(x))

    for f_init in scored_inits[:3]:
        if time.time() > deadline:
            break
        f_ls, score_ls = _local_search(f_init, max_iters=30000, time_limit=15)
        if score_ls < best_score:
            best_score = score_ls
            best_f = f_ls.copy()

    if time.time() < deadline:
        f_sa, score_sa = _simulated_annealing(best_f, time_limit=20)
        if score_sa < best_score:
            best_score = score_sa
            best_f = f_sa.copy()

    if time.time() < deadline:
        f_final, score_final = _local_search(best_f, max_iters=20000, time_limit=10)
        if score_final < best_score:
            best_score = score_final
            best_f = f_final.copy()

    return best_f
