"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3_bound(f):
    """Compute the autoconvolution upper bound C3 for a signed step function.
    
    Returns (1/N^2) * ||R||_1 where R is the linear autocorrelation of f.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    R = np.fft.irfft(F * np.conj(F), n=n_fft)
    l1_norm = np.abs(R[0]) + 2.0 * np.sum(np.abs(R[1:N]))
    return l1_norm / (N * N)


def _compute_gradient(f):
    """Compute the gradient of the L1 autocorrelation objective.
    
    The objective is (1/N^2) * [R[0] + 2*sum_{k=1}^{N-1} |R[k]|]
    where R[k] = sum_{i=0}^{N-1-k} f[i] * f[i+k]
    
    For the differentiable part (ignoring |.|), the gradient of R[k] w.r.t. f[i]
    is f[i+k] for i < N-k and f[i-k] for i >= k.
    
    With the absolute value, the subgradient of |R[k]| is sign(R[k]) * dR[k]/df[i].
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    R = np.fft.irfft(F * np.conj(F), n=n_fft)
    
    # R[0..N-1] are the linear autocorrelation at lags 0..N-1
    # Gradient of the L1 norm w.r.t. f
    # d/d f[i] of R[k] for k >= 0: f[i+k] if i+k < N, plus f[i-k] if i-k >= 0 (from R[-k])
    # By symmetry, the full gradient contribution from |R[k]| is:
    # sign(R[k]) * (f[i+k] + f[i-k]) appropriately
    
    # More carefully: the objective is R[0] + 2*sum_{k=1}^{N-1} |R[k]|
    # dR[k]/df[i] = f[i+k] (for i+k < N) + f[i-k] (for i-k >= 0, from the -k lag term)
    # But since we're using the one-sided R[0..N-1] and accounting for symmetry with factor 2,
    # the gradient of 2*|R[k]| w.r.t. f[i] is 2*sign(R[k]) * (f[i+k] if i+k<N else 0) 
    #   plus 2*sign(R[k]) * (f[i-k] if i-k>=0 else 0)
    # Wait, this double counts. Let me think again.
    
    # The L1 norm = R[0] + 2*sum_{k=1}^{N-1} |R[k]|
    # R[k] = sum_{i=0}^{N-1-k} f[i]*f[i+k]
    # dR[k]/df[i] = f[i+k] if i <= N-1-k, else 0 (for the positive lag)
    #              + f[i-k] if i >= k, else 0 (this comes from R[-k] = R[k] by symmetry)
    # 
    # But in our formula, R[-k] is not explicitly in the sum; we have 2*|R[k]| which accounts
    # for both R[k] and R[-k]. Since R[-k] = R[k], the contribution to the gradient from
    # the pair (R[k], R[-k]) is:
    # d/d f[i] [2*|R[k]|] = 2*sign(R[k]) * dR[k]/df[i]
    # where dR[k]/df[i] only considers the positive lag definition.
    #
    # But wait, R[-k] = sum_{i=k}^{N-1} f[i]*f[i-k] = sum_{j=0}^{N-1-k} f[j+k]*f[j] = R[k]
    # So dR[-k]/df[i] = f[i-k] if i >= k, else 0
    # And since the objective has 2*|R[k]| = |R[k]| + |R[-k]|, the gradient is:
    # sign(R[k])*dR[k]/df[i] + sign(R[-k])*dR[-k]/df[i]
    # = sign(R[k])*(f[i+k] if i+k<N else 0) + sign(R[k])*(f[i-k] if i-k>=0 else 0)
    
    grad = np.zeros(N)
    # Contribution from R[0]: dR[0]/df[i] = f[i], and coefficient is 1 (no abs for R[0] since it's always positive)
    # Actually R[0] = sum f[i]^2 = N, so dR[0]/df[i] = 2*f[i]. But since f[i]^2 = 1 always for ±1,
    # in the continuous relaxation, dR[0]/df[i] = 2*f[i].
    # The objective term is R[0], so gradient contribution is 2*f[i].
    grad += 2.0 * f
    
    # Contributions from 2*|R[k]| for k=1..N-1
    # For each k, sign_k = sign(R[k])
    # grad[i] += sign_k * f[i+k] (if i+k < N) + sign_k * f[i-k] (if i-k >= 0)
    # This can be computed efficiently using correlations.
    
    # For the first part: sum_{k=1}^{N-1} sign(R[k]) * f[i+k] for i+k < N
    # = sum_{k=1}^{min(i+1, N-1)} sign(R[k]) * f[i+k]  -- no wait, for fixed i, k ranges 1..N-1-i
    # Let's use FFT-based approach.
    
    # Let s[k] = sign(R[k]) for k=0..N-1, s[0]=1 (since R[0]>0)
    s = np.sign(R[:N])
    s[0] = 1.0  # R[0] is always positive
    
    # Part 1: sum_{k=1}^{N-1} s[k] * f[i+k] for i+k < N
    # This is like a cross-correlation of f with s (reversed)
    # For fixed i: sum_{k=1}^{N-1-i} s[k] * f[i+k]
    # = (f * s_reversed)[N-1-i] where s_reversed[j] = s[N-1-j]... let me use direct FFT
    
    # Actually, let's compute this via FFT cross-correlation
    # g1[i] = sum_{k=1}^{N-1} s[k] * f[i+k] * (i+k < N)
    # Let's define s_ext = [0, s[1], s[2], ..., s[N-1]] (length N, s_ext[0]=0)
    # Then g1[i] = sum_{k=0}^{N-1} s_ext[k] * f[i+k] for i+k < N
    # This is a correlation. Using FFT:
    n2 = 2 * N
    s_ext = np.zeros(N)
    s_ext[1:] = s[1:]
    F_f = np.fft.rfft(f, n=n2)
    F_s = np.fft.rfft(s_ext, n=n2)
    # Cross-correlation: corr[i] = sum_k f[i+k] * s_ext[k] = sum_k f[i+k] * s_ext[k]
    # This is the cross-correlation of f and s_ext, which can be computed as:
    # IFFT(FFT(f) * conj(FFT(s_ext))) gives sum_k f[k] * s_ext[k-i]... 
    # Let me just use np.correlate or direct computation for clarity, but that's O(N^2).
    # For N=400, O(N^2) = 160000 operations per gradient, which is fine.
    
    # Direct computation (O(N^2) but N=400 is small)
    g1 = np.zeros(N)
    g2 = np.zeros(N)
    for k in range(1, N):
        sk = s[k]
        if sk == 0:
            continue
        # g1[i] += sk * f[i+k] for i = 0..N-1-k
        if k < N:
            g1[:N-k] += sk * f[k:]
        # g2[i] += sk * f[i-k] for i = k..N-1
        g2[k:] += sk * f[:N-k]
    
    grad += g1 + g2
    
    # Normalize by N^2
    grad /= (N * N)
    
    return grad


def _projected_gradient_descent(f_init, max_iters=2000, lr=None, time_limit=60.0):
    """Run projected gradient descent on the continuous relaxation.
    
    The continuous relaxation allows f[i] in [-1, 1] instead of {-1, 1}.
    The objective (1/N^2)*||R||_1 is convex in the continuous domain? 
    Actually, |R[k]| is convex in f, and R[k] is linear in f, so |R[k]| is convex.
    The sum of convex functions is convex, so the objective is convex!
    This means projected gradient descent should converge to the global minimum.
    
    Wait, is it truly convex? R[k] = sum f[i]*f[i+k] is NOT linear in f; it's quadratic!
    So |R[k]| is not necessarily convex. The problem is non-convex.
    
    However, projected gradient descent can still make progress. Let's try it.
    """
    start_time = time.time()
    f = f_init.astype(float).copy()
    
    if lr is None:
        lr = 0.1
    
    best_f = f.copy()
    best_val = _compute_c3_bound(f)
    
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
        
        # Also check continuous value occasionally
        if it % 50 == 0:
            val_cont = _compute_c3_bound(f)
            if val_cont < best_val:
                best_val = val_cont
                best_f = f.copy()
        
        # Adaptive learning rate
        if it > 100:
            lr *= 0.99
    
    return best_f, best_val


def _simulated_annealing(f_init, time_limit=40.0, block_sizes=None):
    """Simulated annealing with block-flip moves for discrete refinement."""
    start_time = time.time()
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    f = f_init.copy()
    current_val = _compute_c3_bound(f)
    best_f = f.copy()
    best_val = current_val
    
    T = 0.05  # Initial temperature
    cooling = 0.995
    
    np.random.seed(12345)
    
    while time.time() - start_time < time_limit:
        # Choose a random block size
        bs = block_sizes[np.random.randint(len(block_sizes))]
        
        # Choose a random start position
        start = np.random.randint(0, N - bs + 1)
        
        # Flip the block
        f[start:start+bs] *= -1.0
        new_val = _compute_c3_bound(f)
        delta = new_val - current_val
        
        # Accept with probability exp(-delta/T)
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_f = f.copy()
        else:
            # Revert
            f[start:start+bs] *= -1.0
        
        T *= cooling
        if T < 0.001:
            T = 0.001
    
    return best_f, best_val


def _local_refinement(f, time_limit=20.0):
    """Local refinement with single-bit and small block flips."""
    start_time = time.time()
    f = f.copy()
    current_val = _compute_c3_bound(f)
    best_f = f.copy()
    best_val = current_val
    
    np.random.seed(777)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        order = np.random.permutation(N)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            f[i] *= -1.0
            new_val = _compute_c3_bound(f)
            
            if new_val < current_val - 1e-15:
                current_val = new_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_f = f.copy()
            else:
                f[i] *= -1.0
        
        # Try small block flips
        if time.time() - start_time < time_limit:
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                bs = np.random.choice([2, 3, 4, 5])
                start = np.random.randint(0, N - bs + 1)
                f[start:start+bs] *= -1.0
                new_val = _compute_c3_bound(f)
                if new_val < current_val - 1e-15:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_f = f.copy()
                else:
                    f[start:start+bs] *= -1.0
    
    return best_f, best_val


def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_f = None
    best_val = np.inf
    
    # Phase 1: Projected gradient descent from multiple initializations
    pgd_time = 50.0
    pgd_deadline = start_time + pgd_time
    
    initializations = []
    
    # Random initialization
    np.random.seed(42)
    initializations.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Sine-based initializations
    for freq in [1, 2, 3, 5, 8, 13]:
        t = np.linspace(0, freq * 2 * np.pi, N)
        initializations.append(np.sign(np.sin(t)))
    
    # Structured patterns
    initializations.append(np.tile([1, -1], N // 2))
    initializations.append(np.tile([1, 1, -1, -1], N // 4))
    initializations.append(np.tile([1, -1, -1, 1], N // 4))
    
    for f_init in initializations:
        if time.time() > pgd_deadline:
            break
        
        f_pgd, val_pgd = _projected_gradient_descent(f_init, max_iters=1000, time_limit=10.0)
        if val_pgd < best_val:
            best_val = val_pgd
            best_f = f_pgd.copy()
    
    # Phase 2: Simulated annealing from best PGD result
    sa_time = 45.0
    sa_deadline = start_time + pgd_time + sa_time
    
    if time.time() < sa_deadline and best_f is not None:
        f_sa, val_sa = _simulated_annealing(best_f, time_limit=sa_time - (time.time() - start_time - pgd_time))
        if val_sa < best_val:
            best_val = val_sa
            best_f = f_sa.copy()
    
    # Also try SA from a random start
    if time.time() < sa_deadline:
        np.random.seed(999)
        f_rand = np.random.choice([-1.0, 1.0], size=N)
        f_sa2, val_sa2 = _simulated_annealing(f_rand, time_limit=sa_time - (time.time() - start_time - pgd_time))
        if val_sa2 < best_val:
            best_val = val_sa2
            best_f = f_sa2.copy()
    
    # Phase 3: Local refinement
    local_time = 25.0
    local_deadline = start_time + pgd_time + sa_time + local_time
    
    if time.time() < local_deadline and best_f is not None:
        f_local, val_local = _local_refinement(best_f, time_limit=local_time - (time.time() - start_time - pgd_time - sa_time))
        if val_local < best_val:
            best_val = val_local
            best_f = f_local.copy()
    
    # Final safety: ensure we return something valid
    if best_f is None:
        best_f = np.ones(N)
    
    # Ensure the result is a proper signed step function
    best_f = np.sign(best_f)
    best_f[best_f == 0] = 1.0
    
    return best_f
# EVOLVE-BLOCK-END
