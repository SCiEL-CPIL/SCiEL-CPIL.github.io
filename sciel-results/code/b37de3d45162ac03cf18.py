"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(x):
    """Compute max absolute value of linear autoconvolution and squared sum."""
    # Linear autoconvolution: c[k] = sum_i x[i] * x[i+k] for k=0..N-1
    # Use FFT for efficiency
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    conv = np.fft.irfft(f * np.conj(f), n=2*n)[:n]
    max_abs = np.max(np.abs(conv))
    s = np.sum(x)
    sq_sum = s * s
    if sq_sum == 0:
        return max_abs, 0.0
    return max_abs, sq_sum

def _score(x):
    max_abs, sq_sum = _autoconv_max_abs(x)
    if sq_sum == 0:
        return float('inf')
    return max_abs / sq_sum

def _try_flip(x, i, current_max, current_sq):
    """Try flipping x[i], return new score or None if worse."""
    x[i] = -x[i]
    max_abs, sq_sum = _autoconv_max_abs(x)
    x[i] = -x[i]  # undo
    if sq_sum == 0:
        return None
    return max_abs / sq_sum

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Start with a random ±1 sequence
    np.random.seed(42)
    x = np.random.choice([-1, 1], size=N)
    
    current_score = _score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Local search: try flipping each element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] = -x[i]
            max_abs, sq_sum = _autoconv_max_abs(x)
            if sq_sum != 0:
                new_score = max_abs / sq_sum
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                else:
                    x[i] = -x[i]  # undo
            else:
                x[i] = -x[i]  # undo
    
    # Simulated annealing for further improvement
    if time.time() < deadline - 5:
        T = 1.0
        x = best_x.copy()
        current_score = best_score
        for step in range(100000):
            if time.time() > deadline:
                break
            i = np.random.randint(N)
            x[i] = -x[i]
            max_abs, sq_sum = _autoconv_max_abs(x)
            if sq_sum == 0:
                x[i] = -x[i]
                continue
            new_score = max_abs / sq_sum
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                x[i] = -x[i]
            T *= 0.999
    
    return best_x
# EVOLVE-BLOCK-END
