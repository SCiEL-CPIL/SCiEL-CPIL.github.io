"""
Structural change: use sparse combinatorial design + simulated annealing with
vectorized FFT scoring. The key insight from the cell finding is that selecting
a small subset of positions to flip (sparse perturbation) decouples the search
from the full 400-dimensional space and can find structured patterns with lower
autocorrelation peaks than uniform random or dense local search.
"""
import numpy as np
import time

N = 400
_DEADLINE_MARGIN = 135.0


def _compute_c3_upper_bound(f):
    """
    Compute the upper bound on C3: max |autoconvolution| over non-zero lags.
    Uses FFT for efficiency.
    """
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    # Exclude lag 0 (self-convolution = sum of squares = N for ±1 values)
    if len(conv) > 1:
        return float(np.max(np.abs(conv[1:])))
    return 0.0


def _score(f):
    return _compute_c3_upper_bound(f)


def _try_flip_vectorized(f, idx):
    """Try flipping position idx and return new score."""
    f_new = f.copy()
    f_new[idx] *= -1.0
    return _score(f_new)


def _simulated_annealing(f, max_iters=50000, deadline=None, T_start=5.0, T_end=0.01):
    """
    Simulated annealing with sparse perturbations (flip 1-3 bits at a time).
    This is a structural change from pure greedy local search: it accepts
    worse moves with temperature-dependent probability to escape local minima.
    """
    current_f = f.copy()
    current_score = _score(current_f)
    best_f = current_f.copy()
    best_score = current_score
    
    n = len(current_f)
    rng = np.random.RandomState(7)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule (geometric cooling)
        alpha = T_end / T_start
        T = T_start * (alpha ** (it / max(1, max_iters)))
        
        # Choose perturbation size: mostly 1, sometimes 2-3
        r = rng.random()
        if r < 0.6:
            k = 1
        elif r < 0.9:
            k = 2
        else:
            k = 3
        
        # Select random positions to flip
        if k == 1:
            idx = rng.randint(n)
            new_f = current_f.copy()
            new_f[idx] *= -1.0
        else:
            indices = rng.choice(n, size=k, replace=False)
            new_f = current_f.copy()
            new_f[indices] *= -1.0
        
        new_score = _score(new_f)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            current_f = new_f
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = current_f.copy()
    
    return best_f, best_score


def _sparse_greedy(f, deadline=None, max_rounds=200):
    """
    Sparse greedy: in each round, try flipping each position and keep the best.
    But only do a few rounds, then move on. This is faster than full local search
    when combined with restarts.
    """
    current_f = f.copy()
    current_score = _score(current_f)
    best_f = current_f.copy()
    best_score = current_score
    
    n = len(current_f)
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        best_delta = 0.0
        best_idx = -1
        
        for i in range(n):
            f_new = current_f.copy()
            f_new[i] *= -1.0
            s = _score(f_new)
            if s - current_score < best_delta:
                best_delta = s - current_score
                best_idx = i
        
        if best_idx == -1:
            break
        
        current_f[best_idx] *= -1.0
        current_score += best_delta
        if current_score < best_score:
            best_score = current_score
            best_f = current_f.copy()
    
    return best_f, best_score


def _structured_initial(rng, style):
    """Generate structured initial patterns."""
    f = np.ones(N)
    if style == 0:
        # Random
        f = rng.choice([-1.0, 1.0], size=N)
    elif style == 1:
        # Block pattern
        block = N // 8
        for i in range(0, N, block):
            sign = rng.choice([-1.0, 1.0])
            f[i:min(i+block, N)] = sign
    elif style == 2:
        # Sparse: mostly +1, few -1
        n_neg = rng.randint(20, 60)
        neg_idx = rng.choice(N, size=n_neg, replace=False)
        f[neg_idx] = -1.0
    elif style == 3:
        # Walsh-like: use a simple structured pattern
        # Generate based on bit-reversal or similar
        idx = np.arange(N)
        # Use a pseudo-random but structured pattern
        f = np.where((idx % 7 < 3), 1.0, -1.0)
        f[rng.choice(N, size=50, replace=False)] *= -1.0
    elif style == 4:
        # Symmetric pattern
        half = N // 2
        f[:half] = rng.choice([-1.0, 1.0], size=half)
        f[half:] = f[:half]
    elif style == 5:
        # Anti-symmetric
        half = N // 2
        f[:half] = rng.choice([-1.0, 1.0], size=half)
        f[half:] = -f[:half]
    else:
        f = rng.choice([-1.0, 1.0], size=N)
    return f


def construct():
    """
    Main entry point. Returns a sequence of N signed step heights.
    Score is the upper bound on C3 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    rng = np.random.RandomState(2024)
    
    best_f = None
    best_score = np.inf
    
    # Phase 1: Multiple structured initializations + sparse greedy
    for style in range(6):
        if time.time() > deadline - 30:
            break
        f_init = _structured_initial(rng, style)
        f_res, s_res = _sparse_greedy(f_init, deadline=deadline - 30, max_rounds=50)
        if s_res < best_score:
            best_score = s_res
            best_f = f_res.copy()
    
    # Phase 2: Simulated annealing from best so far
    if time.time() < deadline - 15 and best_f is not None:
        f_sa, s_sa = _simulated_annealing(
            best_f, max_iters=80000, deadline=deadline - 15,
            T_start=3.0, T_end=0.01
        )
        if s_sa < best_score:
            best_score = s_sa
            best_f = f_sa.copy()
    
    # Phase 3: More restarts with simulated annealing
    for r in range(4):
        if time.time() > deadline - 10:
            break
        f_init = _structured_initial(rng, rng.randint(6))
        f_sa, s_sa = _simulated_annealing(
            f_init, max_iters=30000, deadline=deadline - 10,
            T_start=2.0, T_end=0.05
        )
        if s_sa < best_score:
            best_score = s_sa
            best_f = f_sa.copy()
    
    # Phase 4: Final polish with sparse greedy
    if time.time() < deadline - 5 and best_f is not None:
        f_final, s_final = _sparse_greedy(best_f, deadline=deadline - 3, max_rounds=30)
        if s_final < best_score:
            best_score = s_final
            best_f = f_final.copy()
    
    # Fallback: should not happen
    if best_f is None:
        best_f = rng.choice([-1.0, 1.0], size=N)
    
    return best_f
