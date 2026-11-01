import numpy as np
import time

N = 400

def _compute_autocorr(f):
    """Compute linear autocorrelation of f using FFT."""
    n_fft = 2 * N
    F = np.fft.fft(f, n=n_fft)
    R = np.fft.ifft(F * np.conj(F)).real
    return R

def _max_sidelobe(f):
    """Compute the maximum absolute sidelobe of the autocorrelation."""
    R = _compute_autocorr(f)
    # R[0] is main lobe, R[1:N] positive lags, R[N+1:2N] negative lags
    sidelobes = np.concatenate([R[1:N], R[N+1:2*N]])
    return np.max(np.abs(sidelobes))

def _local_search(f, max_iters=5000, time_limit=20.0):
    """Simulated annealing local search."""
    start_time = time.time()
    f = f.copy()
    current_score = _max_sidelobe(f)
    best_f = f.copy()
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Random flip
        i = np.random.randint(N)
        f[i] = -f[i]
        new_score = _max_sidelobe(f)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        else:
            f[i] = -f[i]  # revert
        
        T = max(T * cooling, T_min)
    
    return best_f, best_score

def _gradient_based_search(f0, time_limit=30.0):
    """
    Gradient-based search using a soft-sign parameterization.
    We represent f as sign(B @ c) where B is a low-rank DFT basis.
    Use soft-sign (tanh) for differentiability.
    """
    start_time = time.time()
    
    # Build DFT basis with K frequency components
    K = 30
    n = np.arange(N)
    k = np.arange(K)
    
    B = np.zeros((N, 2 * K))
    for kk in range(K):
        freq = 2 * np.pi * kk / N
        B[:, 2*kk] = np.cos(freq * n)
        B[:, 2*kk+1] = np.sin(freq * n)
    
    # Initialize c to match f0 as closely as possible
    # Solve least squares: B @ c ≈ f0
    try:
        c0, _, _, _ = np.linalg.lstsq(B, f0, rcond=None)
    except:
        c0 = np.random.randn(2 * K) * 0.1
    
    temp = 0.3
    
    def soft_autocorr_max(c):
        x = B @ c
        f_soft = np.tanh(x / temp)
        R = _compute_autocorr(f_soft)
        sidelobes = np.concatenate([R[1:N], R[N+1:2*N]])
        # Use soft-max for differentiability
        alpha = 20.0
        max_val = np.max(np.abs(sidelobes))
        soft_max = np.log(np.sum(np.exp(alpha * np.abs(sidelobes)))) / alpha
        return soft_max
    
    # Simple gradient descent with numerical gradients
    c = c0.copy()
    lr = 0.1
    best_c = c.copy()
    best_soft = soft_autocorr_max(c)
    
    for it in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Compute numerical gradient
        grad = np.zeros_like(c)
        eps = 1e-4
        for j in range(len(c)):
            c_plus = c.copy()
            c_plus[j] += eps
            c_minus = c.copy()
            c_minus[j] -= eps
            grad[j] = (soft_autocorr_max(c_plus) - soft_autocorr_max(c_minus)) / (2 * eps)
        
        # Gradient step
        c = c - lr * grad
        # Clamp to avoid overflow
        c = np.clip(c, -10, 10)
        
        current_soft = soft_autocorr_max(c)
        if current_soft < best_soft:
            best_soft = current_soft
            best_c = c.copy()
        
        # Adapt learning rate
        if it % 20 == 0:
            lr *= 0.9
    
    # Convert to hard sign
    f_candidate = np.sign(B @ best_c)
    f_candidate[f_candidate == 0] = 1.0
    
    return f_candidate

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    best_f = None
    best_score = float('inf')
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Random binary sequences
    for seed in range(5):
        np.random.seed(seed)
        f = np.random.choice([-1.0, 1.0], size=N)
        candidates.append(f)
    
    # Structured sequences: alternating patterns
    for pattern_len in [2, 4, 8, 16]:
        base = np.ones(pattern_len)
        base[::2] = -1.0
        f = np.tile(base, N // pattern_len)
        candidates.append(f)
    
    # DFT-based: sign of low-frequency DFT
    for K in [5, 10, 20]:
        n = np.arange(N)
        x = np.zeros(N)
        for kk in range(1, K+1):
            x += np.cos(2 * np.pi * kk * n / N)
        f = np.sign(x)
        f[f == 0] = 1.0
        candidates.append(f)
    
    # Evaluate all candidates
    for f in candidates:
        score = _max_sidelobe(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Phase 2: Gradient-based refinement on best candidate
    if time.time() < deadline - 60:
        f_grad = _gradient_based_search(best_f, time_limit=30.0)
        score_grad = _max_sidelobe(f_grad)
        if score_grad < best_score:
            best_score = score_grad
            best_f = f_grad.copy()
    
    # Phase 3: Simulated annealing local search
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        sa_time = min(remaining_time - 5, 40)
        f_sa, score_sa = _local_search(best_f, max_iters=10000, time_limit=sa_time)
        if score_sa < best_score:
            best_score = score_sa
            best_f = f_sa.copy()
    
    # Phase 4: If time remains, do another round of SA from a different start
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        # Start SA from a random candidate
        np.random.seed(123)
        f_rand = np.random.choice([-1.0, 1.0], size=N)
        sa_time = min(remaining_time - 5, 30)
        f_sa2, score_sa2 = _local_search(f_rand, max_iters=10000, time_limit=sa_time)
        if score_sa2 < best_score:
            best_score = score_sa2
            best_f = f_sa2.copy()
    
    # Phase 5: Final refinement with greedy hill climbing if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_f is not None:
        f = best_f.copy()
        current_score = _max_sidelobe(f)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            R = _compute_autocorr(f)
            sidelobes = np.concatenate([R[1:N], R[N+1:2*N]])
            max_sidelobe = np.max(np.abs(sidelobes))
            
            best_i = -1
            best_new_max = max_sidelobe
            
            for i in range(N):
                f_trial = f.copy()
                f_trial[i] = -f_trial[i]
                R_trial = _compute_autocorr(f_trial)
                sidelobes_trial = np.concatenate([R_trial[1:N], R_trial[N+1:2*N]])
                new_max = np.max(np.abs(sidelobes_trial))
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_i = i
            
            if best_i != -1:
                f[best_i] = -f[best_i]
                improved = True
        
        final_score = _max_sidelobe(f)
        if final_score < best_score:
            best_score = final_score
            best_f = f.copy()
    
    if best_f is not None:
        return best_f
    else:
        np.random.seed(0)
        return np.random.choice([-1.0, 1.0], size=N)
