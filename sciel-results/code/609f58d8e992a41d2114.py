import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 130.0

def _elapsed():
    return time.time() - _START

def _score_set(U):
    U = np.asarray(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0:
        return 0.0
    if U[0] != 0:
        U = np.concatenate(([0], U))
        n = len(U)
    M = int(U[-1])
    if M < 1:
        return 0.0
    
    # Use FFT for sumset and diffset sizes
    vec = np.zeros(M + 1, dtype=np.float64)
    vec[U] = 1.0
    
    # Sumset U+U
    f = np.fft.rfft(vec, n=2 * (M + 1))
    s = np.fft.irfft(f * f, n=2 * (M + 1))
    sumset = int(np.sum(s >= 0.5))
    
    # Diffset U-U
    # U-U is the support of the autocorrelation?
    # g[k] = sum u[i] u[i+k]. This is U - U shifted.
    # Actually, np.fft.irfft(f * np.conj(f)) gives autocorrelation.
    # The support of the autocorrelation corresponds to U-U.
    g = np.fft.irfft(f * np.conj(f), n=2 * (M + 1))
    diffset = int(np.sum(g >= 0.5))
    
    ratio = diffset / sumset if sumset > 0 else 0.0
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def construct():
    best_U = [0]
    best_score = 0.0
    
    # Candidate 1: The "0, 1, 4, 6" pattern extension
    # Steps: 1, 3, 2, 3, 2, 3...
    def generate_pattern(k):
        U = [0]
        steps = [1, 3, 2]
        for i in range(k - 1):
            if i < len(steps):
                U.append(U[-1] + steps[i])
            else:
                # Continue pattern 3, 2
                U.append(U[-1] + (3 if (i - len(steps)) % 2 == 0 else 2))
        return U

    for k in range(2, 20):
        U = generate_pattern(k)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U

    # Candidate 2: Powers of 2 (baseline)
    for k in range(1, 15):
        U = [0] + [2**i for i in range(k)]
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U

    # Candidate 3: Random sparse sets
    rng = np.random.RandomState(42)
    for max_val in [10, 20, 40, 80, 160, 320, 640]:
        for size in range(3, min(20, max_val//4 + 1)):
            if _elapsed() > 30:
                break
            for _ in range(10):
                candidates = rng.choice(range(1, max_val+1), size=size, replace=False)
                U = sorted([0] + list(candidates))
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U
        if _elapsed() > 30:
            break

    # Local search
    U = list(best_U)
    cur_score = best_score
    improved = True
    while improved and _elapsed() < 110:
        improved = False
        max_val = max(U) if U else 1
        
        # Try adding elements
        for _ in range(50):
            if _elapsed() > 115:
                break
            c = rng.randint(1, max_val * 2 + 1)
            if c not in U:
                trial = sorted(U + [c])
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
        
        # Try removing elements
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > 115:
                    break
                trial = U[:idx] + U[idx+1:]
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True

    return best_U
