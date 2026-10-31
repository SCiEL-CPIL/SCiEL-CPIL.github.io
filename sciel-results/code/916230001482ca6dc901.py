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

def _gradient_step(x, lr, target_sum, ac, fft_size):
    """
    Perform one projected gradient descent step on the continuous relaxation.
    Objective: max|ac| subject to sum(x) = target_sum (and optionally |x_i| <= 1)
    We use a soft-max penalty: f(x) = log(sum(exp(beta * |ac|))) as proxy for max|ac|
    Gradient of f w.r.t. x_i involves the autoconvolution sensitivity.
    
    Actually, for autoconvolution ac[t] = sum_j x[j]*x[j+t], 
    d(ac[t])/dx[i] = x[i+t] (if i+t < N) + x[i-t] (if i-t >= 0)
    
    So gradient of max|ac| w.r.t. x[i] is sum_t sign(ac[t]) * (x[i+t] + x[i-t])
    
    For efficiency, we compute this via FFT or direct correlation.
    """
    n = len(x)
    # Compute gradient of the soft-max of |ac|
    # g[i] = sum_t sign(ac[t]) * (x[(i+t) % N] + x[(i-t) % N])  -- but ac is linear, not circular
    # Actually ac[t] for t in [-(N-1), N-1]
    # d ac[t] / dx[i] = x[i+t] if 0 <= i+t < N else 0, plus x[i-t] if 0 <= i-t < N else 0
    # (because ac[t] = sum_j x[j] x[j+t], derivative w.r.t. x[i] is x[i+t] + x[i-t] with bounds)
    
    # Let's compute gradient via the relation:
    # ac = x * x (linear convolution)
    # d/dx[i] max|ac| ≈ sum_t sign(ac[t]) * (d ac[t]/dx[i])
    # d ac[t]/dx[i] = x[i+t] (if 0<=i+t<N) + x[i-t] (if 0<=i-t<N)
    # This is the same as (x * sign_ac)(i) where sign_ac is the sign of ac, but with linear convolution
    # Actually it's the cross-correlation of x with sign(ac)
    
    # sign_ac has length 2N-1, indices 0..2N-2 corresponding to lags -(N-1)..(N-1)
    # Let's reindex: ac_full[0] corresponds to lag -(N-1), ac_full[N-1] to lag 0, etc.
    # Actually in our _autoconv, ac[0] is the convolution at index 0, which is sum_j x[j]x[j] = sum x[j]^2 (lag 0)
    # Wait, let me check: np.fft.irfft(X*conj(X)) gives the circular convolution.
    # For linear autoconvolution of length N, the result has length 2N-1.
    # ac[0] = x[0]*x[0]
    # ac[1] = x[0]*x[1] + x[1]*x[0] = 2*x[0]*x[1]
    # ...
    # ac[N-1] = sum_{j=0}^{N-1} x[j]*x[j] = sum x[j]^2  (this is the peak, lag 0 in autocorrelation sense)
    # ac[N] = x[0]*x[N-1] + x[1]*x[N-1-1] ... wait no.
    
    # Let's just compute the gradient numerically or via a helper.
    # For a ±1 sequence, x[i]^2 = 1, so ac[N-1] = N always.
    # For continuous x, we need the gradient.
    
    # Let's use the fact that for linear convolution y = x * x:
    # dy/dx[i] is a vector of length 2N-1 where:
    # (dy/dx[i])[t] = x[t-i] if 0<=t-i<N else 0  (from the x[j] term)
    #                + x[i-t] if 0<=i-t<N else 0  (from the x[j+t] term, j=i-t)
    # So (dy/dx[i])[t] = x[t-i] (if valid) + x[i-t] (if valid)
    
    # This is equivalent to: gradient[i] = sum_t sign(ac[t]) * (x[t-i] (if 0<=t-i<N) + x[i-t] (if 0<=i-t<N))
    
    # Let's compute this efficiently.
    # Let s = sign(ac), length 2N-1.
    # gradient[i] = sum_t s[t] * x[t-i] (for t-i in [0,N-1]) + sum_t s[t] * x[i-t] (for i-t in [0,N-1])
    # First term: let u = t-i, so t = u+i, u in [0, N-1-i] (since t < 2N-1 and u >= 0)
    #   = sum_{u=0}^{N-1-i} s[u+i] * x[u]
    # Second term: let v = i-t, so t = i-v, v in [0, i] (since t >= 0)
    #   = sum_{v=0}^{i} s[i-v] * x[v]
    
    # This is a correlation. We can compute it via FFT or directly.
    # For N=400, direct computation is O(N^2) = 160k, which is fine per step.
    
    s = np.sign(ac)
    grad = np.zeros(n)
    for i in range(n):
        # First term: sum_{u=0}^{n-1-i} s[u+i] * x[u]
        if i < n:
            grad[i] += np.dot(s[i:i+n], x[:n-i])
        # Second term: sum_{v=0}^{i} s[i-v] * x[v]
        if i >= 0:
            grad[i] += np.dot(s[:i+1][::-1], x[:i+1])
    
    # Project gradient onto the constraint sum(x) = target_sum
    # We want to move in the direction that reduces the objective while maintaining the sum.
    # Use Lagrange multiplier: x_new = x - lr * (grad - lambda * 1)
    # where lambda is chosen so that sum(x_new) = target_sum
    # sum(x_new) = sum(x) - lr * (sum(grad) - lambda * n) = target_sum
    # lambda = (sum(x) - target_sum + lr * sum(grad)) / (lr * n)
    
    sum_x = np.sum(x)
    lambda_val = (sum_x - target_sum + lr * np.sum(grad)) / (lr * n)
    x_new = x - lr * (grad - lambda_val)
    
    # Optionally project to [-1, 1]
    x_new = np.clip(x_new, -1.0, 1.0)
    
    # Re-project to sum constraint (clip may have changed the sum)
    diff = target_sum - np.sum(x_new)
    x_new += diff / n
    
    return x_new

def _quadratic_residue_sign(n, p):
    """
    Generate sign sequence based on quadratic residues mod p.
    """
    x = np.zeros(n)
    for i in range(n):
        if i == 0:
            x[i] = 1.0
        else:
            val = pow(i * i % p, (p - 1) // 2, p)
            if val == 1:
                x[i] = 1.0
            elif val == p - 1:
                x[i] = -1.0
            else:
                x[i] = 1.0
    return x

def _m_sequence(n, degree, poly):
    """
    Generate an m-sequence (maximal length LFSR sequence).
    """
    x = np.zeros(n)
    state = 1
    
    for i in range(n):
        x[i] = 1.0 if (state & 1) else -1.0
        feedback = (state & 1)
        for bit in range(degree - 1, -1, -1):
            if (poly >> bit) & 1:
                feedback ^= (state >> bit) & 1
        state = (state >> 1) | (feedback << (degree - 1))
        state &= (1 << degree) - 1
    
    return x

def _hadamard_row(n):
    """
    Generate a row of a Hadamard matrix.
    """
    if n <= 1:
        return np.array([1.0])
    
    h = 1
    while h < n:
        h *= 2
    
    H = np.array([[1.0]])
    for _ in range(int(np.log2(h))):
        H = np.block([[H, H], [H, -H]])
    
    return H[1, :n]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Phase 1: Generate structured candidates
    candidates = []
    
    # 1. Quadratic residue sequences
    primes = [401, 409, 419, 421, 431, 433, 439, 443, 449, 457]
    for p in primes:
        if p > N:
            x = _quadratic_residue_sign(N, p)
            candidates.append(x)
    
    # 2. M-sequences
    m_seq_configs = [
        (8, 0b10011),
        (8, 0b100011),
        (9, 0b1001011),
        (9, 0b1010101),
        (10, 0b10001101),
        (10, 0b10100101),
    ]
    for degree, poly in m_seq_configs:
        if (1 << degree) - 1 >= N:
            x = _m_sequence(N, degree, poly)
            candidates.append(x)
    
    # 3. Hadamard-based
    x_had = _hadamard_row(N)
    candidates.append(x_had)
    
    # 4. Random ±1 sequences
    rng = np.random.default_rng(42)
    for _ in range(50):
        x = rng.choice([-1.0, 1.0], size=N)
        candidates.append(x)
    
    # Evaluate all candidates
    x_best = np.ones(N, dtype=np.float64)
    best_score = _score(x_best)
    
    for x in candidates:
        if time.time() > deadline - 60:
            break
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            x_best = x.copy()
    
    # Phase 2: Continuous projected gradient descent
    # Start from the best binary candidate and relax to continuous
    x = x_best.copy().astype(np.float64)
    target_sum = np.sum(x)  # Keep sum fixed initially
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    # Gradient descent with adaptive learning rate
    lr = 0.01
    best_cont_score = best_score
    best_cont_x = x.copy()
    
    no_improve = 0
    max_no_improve = 100
    
    for step in range(2000):
        if time.time() > deadline - 20:
            break
        
        # Compute current autoconvolution
        ac = _autoconv(x)
        cur_score = _score(x)
        
        if cur_score < best_cont_score:
            best_cont_score = cur_score
            best_cont_x = x.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        
        # Gradient step
        x = _gradient_step(x, lr, target_sum, ac, fft_size)
        
        # Adapt learning rate
        if no_improve > 20:
            lr *= 0.9
        elif no_improve == 0:
            lr = min(lr * 1.05, 0.1)
    
    # Phase 3: Binary refinement from the continuous solution
    # Round the continuous solution to ±1 and do local search
    x_bin = np.sign(best_cont_x)
    x_bin[x_bin == 0] = 1.0
    sc_bin = _score(x_bin)
    
    if sc_bin < best_score:
        best_score = sc_bin
        x_best = x_bin.copy()
    
    # Local search on the best binary sequence
    x = x_best.copy()
    rng2 = np.random.default_rng(999)
    no_improve = 0
    max_no_improve = 50
    
    for _iter in range(5000):
        if time.time() > deadline - 5:
            break
        
        # Try flipping a random position
        idx = rng2.integers(0, N)
        x_trial = x.copy()
        x_trial[idx] = -x_trial[idx]
        
        sc_trial = _score(x_trial)
        
        if sc_trial < _score(x):
            x = x_trial
            if sc_trial < best_score:
                best_score = sc_trial
                x_best = x.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    # Phase 4: Final continuous refinement from the best binary solution
    x = x_best.copy().astype(np.float64)
    target_sum = np.sum(x)
    lr = 0.005
    
    for step in range(1000):
        if time.time() > deadline - 2:
            break
        
        ac = _autoconv(x)
        cur_score = _score(x)
        
        if cur_score < best_score:
            best_score = cur_score
            x_best = x.copy()
        
        x = _gradient_step(x, lr, target_sum, ac, fft_size)
        lr *= 0.99
    
    # Final validation
    final_score = _score(x_best)
    if final_score < best_score:
        best_score = final_score
    
    return x_best
