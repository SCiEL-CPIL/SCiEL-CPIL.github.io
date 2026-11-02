import numpy as np
import time

N = 400
DEADLINE = 145.0

def _compute_c3_bound(signs):
    """
    Compute an upper bound on C3 for a signed step function.
    Uses L1 norm of autoconvolution as the metric.
    """
    dx = 0.5 / N
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return np.sum(np.abs(conv)) * dx

def _block_flip_move(signs, block_size, rng):
    """
    Flip a random contiguous block of signs.
    Returns new signs array.
    """
    new_signs = signs.copy()
    start = rng.randint(0, N - block_size + 1)
    new_signs[start:start + block_size] = -new_signs[start:start + block_size]
    return new_signs

def _simulated_annealing(signs, start_temp=1.0, end_temp=0.01, n_iter=5000, block_sizes=None):
    """
    Simulated annealing with block-flip moves.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32]
    
    rng = np.random.RandomState(12345)
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(n_iter):
        # Linear cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (i / max(n_iter - 1, 1))
        
        # Choose a random block size with bias toward smaller blocks
        bs = block_sizes[rng.choice(len(block_sizes), p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1])]
        if bs > N:
            bs = N
        
        new_signs = _block_flip_move(current, bs, rng)
        new_score = _compute_c3_bound(new_signs)
        
        delta = new_score - current_score
        if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / max(temp, 1e-10))):
            current = new_signs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = current.copy()
    
    return best, best_score

def _pairwise_swap_optimize(signs, max_rounds=20):
    """
    Try swapping pairs of signs to find improvements.
    More effective than single flips for certain structures.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    rng = np.random.RandomState(999)
    
    for _ in range(max_rounds):
        improved = False
        # Try random pairs
        for _ in range(500):
            i, j = rng.choice(N, size=2, replace=False)
            if current[i] == current[j]:
                continue
            # Swap signs at i and j
            current[i], current[j] = current[j], current[i]
            new_score = _compute_c3_bound(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i], current[j] = current[j], current[i]
        if not improved:
            break
    
    return current, current_score

def _compute_gradient(f):
    """
    Compute the subgradient of the L1 autocorrelation objective.
    The objective is sum_k |R[k]| where R[k] is the autocorrelation at lag k.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    R = np.fft.irfft(F * np.conj(F), n=n_fft)
    
    # R[0..N-1] are the linear autocorrelation at lags 0..N-1
    # The objective is sum_{k=-(N-1)}^{N-1} |R[k]| = |R[0]| + 2*sum_{k=1}^{N-1} |R[k]|
    # Since R[0] = N > 0, |R[0]| = N, which is constant.
    # So we minimize 2*sum_{k=1}^{N-1} |R[k]|
    
    # The subgradient of |R[k]| w.r.t. f[i] is:
    # sign(R[k]) * (f[i+k] if i+k < N else 0) + sign(R[k]) * (f[i-k] if i-k >= 0 else 0)
    # for k >= 1.
    
    grad = np.zeros(N)
    s = np.sign(R[:N])
    
    # For each lag k, add sign(R[k]) * (f[i+k] + f[i-k]) appropriately
    for k in range(1, N):
        sk = s[k]
        if sk == 0:
            continue
        # Contribution from f[i+k] for i = 0..N-1-k
        grad[:N-k] += sk * f[k:]
        # Contribution from f[i-k] for i = k..N-1
        grad[k:] += sk * f[:N-k]
    
    # Multiply by 2 for the symmetric lags
    grad *= 2.0
    
    return grad

def _projected_gradient_descent(f_init, max_iters=1000, lr=0.05, time_limit=30.0):
    """
    Run projected gradient descent on the continuous relaxation.
    Allows f[i] in [-1, 1] and projects back to signs periodically.
    """
    start_time = time.time()
    f = f_init.astype(float).copy()
    
    best_f = np.sign(f).astype(float)
    best_f[best_f == 0] = 1.0
    best_val = _compute_c3_bound(best_f)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        grad = _compute_gradient(f)
        f = f - lr * grad
        # Project to [-1, 1]
        f = np.clip(f, -1.0, 1.0)
        
        # Sign projection to get a discrete candidate
        f_sign = np.sign(f)
        f_sign[f_sign == 0] = 1.0
        val_sign = _compute_c3_bound(f_sign)
        
        if val_sign < best_val:
            best_val = val_sign
            best_f = f_sign.copy()
        
        # Adaptive learning rate: decrease if no improvement in last 50 iters
        if it % 100 == 99:
            lr *= 0.95
    
    return best_f, best_val

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Structural Change: Focus on Projected Gradient Descent (PGD) on continuous relaxation
    # This is a differentiable optimization approach, distinct from the previous combinatorial search.
    
    best_pattern = np.ones(N)
    best_score = _compute_c3_bound(best_pattern)
    
    # We will run PGD from several different random starts, using the available time budget.
    # Each run gets a fraction of the time.
    num_restarts = 5
    time_per_restart = (DEADLINE - 10) / num_restarts  # 10s safety margin
    
    for r in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        # Generate a random initial sign pattern
        init_signs = rng.choice([1, -1], size=N)
        
        # Run PGD with a time limit specific to this restart
        # We pass a smaller max_iters but rely on time_limit
        optimized, opt_score = _projected_gradient_descent(
            init_signs, 
            max_iters=2000, 
            lr=0.05, 
            time_limit=time_per_restart
        )
        
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Final local refinement using pairwise swaps on the best found pattern
    # This is a fast combinatorial step to polish the continuous result.
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Limit rounds to ensure we don't exceed time
        # 500 swaps per round, 5 rounds max
        optimized2, opt_score2 = _pairwise_swap_optimize(best_pattern, max_rounds=5)
        if opt_score2 < best_score:
            best_score = opt_score2
            best_pattern = optimized2
    
    result = np.sign(best_pattern).astype(int)
    
    # Constraint Check: heights must not sum to zero
    # If sum is 0, flip the first element to break the symmetry
    # This changes the score slightly but satisfies the constraint.
    if np.sum(result) == 0:
        result[0] = -result[0]
        
    return result
