"""Rewritten program: signed step function search for autoconvolution upper bound."""
import time
import numpy as np

N = 400
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _autoconv_max_abs(x):
    """
    Compute the maximum absolute value of the autoconvolution of x.
    x is a signed step function with N equal intervals on [-1/4, 1/4].
    The autoconvolution (circular or linear?) - for a step function on [-1/4,1/4],
    the convolution is over the support. Let's use linear convolution.
    Actually, for autoconvolution upper bound C3, we need max |x * x(t)| where * is convolution.
    With N equal intervals, x is represented by N heights. The convolution at each point
    is a sum of products of heights weighted by overlap lengths.
    
    For N equal intervals on [-1/4, 1/4], interval width h = (1/2)/N = 1/(2N).
    The autoconvolution at point t is integral over s of x(s)*x(t-s) ds.
    Since x is piecewise constant, this can be computed via discrete convolution of the heights
    scaled by h.
    
    Let's use: conv = np.convolve(x, x, mode='full') * h
    where h = 1/(2N).
    Then return max(abs(conv)).
    """
    h = 1.0 / (2 * N)
    conv = np.convolve(x, x, mode='full') * h
    return np.max(np.abs(conv))


def _eval(x):
    """Score function: lower is better."""
    return _autoconv_max_abs(x)


def _block_flip(x, start, length, rng):
    """Flip signs of a contiguous block."""
    x_new = x.copy()
    x_new[start:start + length] *= -1
    return x_new


def _random_init(rng):
    """Random signed step function."""
    return rng.choice([-1.0, 1.0], size=N)


def _simulated_annealing(x0, time_budget, rng):
    """
    Simulated annealing with block-flip moves.
    """
    x = x0.copy()
    best_x = x.copy()
    best_score = _eval(x)
    
    # Temperature schedule
    T_start = 0.1
    T_end = 1e-6
    n_steps = 50000
    
    start_time = time.time()
    
    for i in range(n_steps):
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        # Choose block size and position
        block_size = rng.integers(4, 33)  # 4 to 32
        if block_size > N:
            block_size = N
        start = rng.integers(0, N - block_size + 1)
        
        # Propose flip
        x_new = _block_flip(x, start, block_size, rng)
        s_new = _eval(x_new)
        
        # Accept or reject
        if s_new < best_score:
            best_score = s_new
            best_x = x_new.copy()
        
        delta = s_new - _eval(x)
        if delta < 0:
            x = x_new
        else:
            # Temperature annealing
            T = T_start * (T_end / T_start) ** (i / max(n_steps - 1, 1))
            if rng.random() < np.exp(-delta / max(T, 1e-12)):
                x = x_new
    
    return best_x, best_score


def _greedy_polish(x, time_budget, rng):
    """
    Greedy local search: try single-bit flips and small block flips.
    """
    x = x.copy()
    best_score = _eval(x)
    best_x = x.copy()
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        # Try single-bit flips
        for i in range(N):
            if (time.time() - start_time) > time_budget:
                break
            x_new = x.copy()
            x_new[i] *= -1
            s_new = _eval(x_new)
            if s_new < best_score - 1e-12:
                x = x_new
                best_score = s_new
                best_x = x.copy()
                improved = True
        
        # Try small block flips (size 2-4)
        for block_size in [2, 3, 4]:
            if (time.time() - start_time) > time_budget:
                break
            for start in range(N - block_size + 1):
                if (time.time() - start_time) > time_budget:
                    break
                x_new = x.copy()
                x_new[start:start + block_size] *= -1
                s_new = _eval(x_new)
                if s_new < best_score - 1e-12:
                    x = x_new
                    best_score = s_new
                    best_x = x.copy()
                    improved = True
    
    return best_x, best_score


def construct():
    """
    Main entry point. Returns a sequence of N signed step heights.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Initial random start
    x = _random_init(rng)
    best_x = x.copy()
    best_score = _eval(x)
    
    # Phase 1: Simulated annealing with multiple restarts
    remaining = TIME_LIMIT - (time.time() - start_time) - 20.0  # reserve 20s for polish
    n_restarts = 5
    time_per_restart = max(remaining / n_restarts, 1.0)
    
    for r in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 20.0:
            break
        
        # New random start or perturb best
        if r == 0:
            x_curr = x
        else:
            # Start from a perturbation of the best found so far
            x_curr = best_x.copy()
            # Flip a random block to diversify
            bs = rng.integers(8, 41)
            bs = min(bs, N)
            st = rng.integers(0, N - bs + 1)
            x_curr[st:st + bs] *= -1
        
        x_res, s_res = _simulated_annealing(x_curr, time_per_restart, rng)
        if s_res < best_score:
            best_score = s_res
            best_x = x_res.copy()
    
    # Phase 2: Greedy polish
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 2.0:
        x_polished, s_polished = _greedy_polish(best_x, remaining - 1.0, rng)
        if s_polished < best_score:
            best_score = s_polished
            best_x = x_polished.copy()
    
    # Ensure we return within time limit
    return best_x


# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s")
    print(f"Score: {_eval(result)}")
    print(f"Shape: {result.shape}")
