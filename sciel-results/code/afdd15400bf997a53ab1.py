import numpy as np
import time
import math

N = 400

def _score(h):
    """Compute the upper bound on C3: max absolute autocorrelation at non-zero lags."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf

def _build_interaction_matrix(h):
    """
    Build matrix S where S[i, k] = h[i-k] + h[i+k] (with valid indices only).
    S shape: (N, N-1)
    k index in S corresponds to lag k+1.
    """
    n = len(h)
    S = np.zeros((n, n - 1), dtype=np.float64)
    for k in range(1, n):
        idx = k - 1
        # h[i-k] for i >= k
        S[k:, idx] += h[:n-k]
        # h[i+k] for i < n-k
        S[:n-k, idx] += h[k:n]
    return S

def _soft_descent(h_init, time_limit, alpha=10.0, lr=0.5, momentum=0.9):
    """
    Optimize the sequence using soft-descent on the log-sum-exp peak objective.
    """
    n = len(h_init)
    # Initialize continuous potentials x such that sign(x) ~ h_init
    # We add small noise to break symmetry if needed, but start close to h_init
    x = h_init.copy().astype(np.float64)
    # Add small random perturbation to avoid staying exactly at the discrete point
    x += 0.1 * np.random.randn(n)
    
    # Clamp to [-1, 1] initially? No, let it flow.
    # But we evaluate h = sign(x).
    
    v = np.zeros(n) # Momentum accumulator
    
    best_h = h_init.copy()
    best_score = _score(best_h)
    current_h = np.sign(x)
    current_score = _score(current_h)
    
    last_improvement_time = time.time()
    
    while time.time() < time_limit:
        # Compute ACF for current discrete h
        acf = _compute_acf(current_h)[1:n]
        
        # Compute soft peak weights
        # rho_k = acf[k-1]
        rho_sq = acf ** 2
        # To avoid overflow in exp, subtract max
        max_rho_sq = np.max(rho_sq)
        if max_rho_sq > 1000:
            # If values are huge, alpha effectively becomes 0 or we need to scale
            # For our problem, ACF values are bounded by N, so rho_sq ~ N^2 = 160000.
            # exp(10 * 160000) overflows.
            # We must scale alpha or use a different soft max.
            # Let's use alpha * (rho_sq - max_rho_sq) + max_rho_sq * alpha
            # Actually, log(sum exp(alpha * rho_sq)) = alpha * max + log(sum exp(alpha * (rho_sq - max)))
            # Gradient depends on weights w_k = exp(alpha * rho_sq) / sum(...)
            # w_k = exp(alpha * (rho_sq - max)) / sum(exp(alpha * (rho_sq - max)))
            # This is numerically stable.
            shifted = rho_sq - max_rho_sq
            exp_shifted = np.exp(alpha * shifted)
            sum_exp = np.sum(exp_shifted)
            w = exp_shifted / sum_exp
        else:
            exp_rho = np.exp(alpha * rho_sq)
            sum_exp = np.sum(exp_rho)
            w = exp_rho / sum_exp
            
        # Build interaction matrix S
        S = _build_interaction_matrix(current_h)
        
        # Compute gradient contribution g_i = -2 * h_i * sum_k w_k * S[i, k]
        # sum_k w_k * S[i, k] is a matrix-vector product: S @ w
        # w shape (N-1), S shape (N, N-1). Result shape (N,).
        grad_contrib = S @ w
        grad = -2.0 * current_h * grad_contrib
        
        # Update x using momentum
        v = momentum * v + lr * grad
        x_new = x + v
        
        # Determine new discrete h
        h_new = np.sign(x_new)
        # Handle zeros: if 0, keep previous or random? Keep previous to be stable.
        h_new[h_new == 0] = current_h[h_new == 0]
        
        # Check for changes
        if not np.array_equal(h_new, current_h):
            # Evaluate new score
            new_score = _score(h_new)
            
            # Accept if better, or with some probability (simulated annealing style)
            # But for simplicity and speed, let's use deterministic hill climbing on the soft landscape
            # but only accept discrete improvements.
            if new_score < current_score:
                current_h = h_new
                current_score = new_score
                x = x_new
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                    last_improvement_time = time.time()
            else:
                # If no improvement, we might still move x to explore, 
                # but we risk getting stuck in a "soft" local min that doesn't map to discrete improvement.
                # Let's update x anyway to continue the flow, but if we stagnate too long, restart.
                x = x_new
                # If we haven't improved in 5 seconds, try a random kick
                if time.time() - last_improvement_time > 5.0:
                    # Random restart of the soft descent from the current best
                    x = best_h.copy().astype(np.float64) + 0.1 * np.random.randn(n)
                    v = np.zeros(n)
                    last_improvement_time = time.time()
        else:
            # No discrete change, update x to keep moving
            x = x_new
            
    return best_h, best_score

def _local_polish(h, time_limit):
    """
    Fast local search using the greedy best-flip strategy.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    while time.time() < time_limit:
        acf = _compute_acf(current_h)[1:n]
        S = _build_interaction_matrix(current_h)
        
        # Compute new peaks for all flips
        deltas = -2.0 * current_h[:, np.newaxis] * S
        new_acfs = acf[np.newaxis, :] + deltas
        new_peaks = np.max(np.abs(new_acfs), axis=1)
        
        best_i = np.argmin(new_peaks)
        best_new_score = new_peaks[best_i]
        
        if best_new_score < current_score - 1e-9:
            current_h[best_i] = -current_h[best_i]
            current_score = best_new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            # Stagnation. Try a few random flips or break.
            # If we break, we exit the loop.
            # Let's try a random perturbation to escape.
            i = np.random.randint(n)
            current_h[i] = -current_h[i]
            new_s = _score(current_h)
            if new_s < current_score:
                current_score = new_s
                if new_s < best_score:
                    best_score = new_s
                    best_h = current_h.copy()
            else:
                # Revert and break to save time for other methods
                current_h[i] = -current_h[i]
                break
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_h = None
    best_score = float('inf')
    
    # 1. Generate diverse initial candidates
    candidates = []
    rng = np.random.default_rng(42)
    
    # Random
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # Structured
    t = np.arange(N)
    candidates.append(np.sign(np.cos(2 * np.pi * t / N)))
    candidates.append(np.sign(np.cos(2 * np.pi * t * 3 / N)))
    candidates.append(np.sign(np.cos(2 * np.pi * t * 7 / N)))
    candidates.append(np.array([1.0 if i%2==0 else -1.0 for i in range(N)]))
    
    # Evaluate initial candidates
    for h in candidates:
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 2. Run Soft Descent from the best candidate
    # Allocate most of the time here
    soft_time_limit = start_time + 100.0
    if soft_time_limit < deadline - 20:
        h_soft, s_soft = _soft_descent(best_h, soft_time_limit, alpha=5.0, lr=0.1, momentum=0.5)
        if s_soft < best_score:
            best_score = s_soft
            best_h = h_soft

    # 3. Run Local Polish on the result
    polish_time_limit = deadline - 5.0
    if time.time() < polish_time_limit:
        h_polish, s_polish = _local_polish(best_h, polish_time_limit)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish

    # 4. One last quick check/restart if time permits
    if time.time() < deadline - 2.0:
        # Try a random restart with local polish
        h_rand = np.random.choice([-1.0, 1.0], size=N)
        h_final, s_final = _local_polish(h_rand, deadline - 1.0)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final

    return best_h
