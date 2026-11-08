"""Second autocorrelation inequality - optimized construction."""
import numpy as np
import time

N = 50

def _compute_score(h):
    """Compute the raw score value for a given height vector."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = float(np.sum(h**2))
    l1 = float(np.sum(h))
    if l1 < 1e-12 or l2_sq < 1e-12:
        return 0.0
    # Linear autoconvolution via FFT
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:N]
    # The autoconvolution has lags -N+1..N-1, but irfft gives 0..N-1
    # Actually for linear conv of length N with itself, result has length 2N-1
    # Let's use full linear convolution
    conv_full = np.fft.irfft(H * H, n=n_fft)
    # conv_full has length n_fft, but the valid part is 2N-1
    # The peaks can be anywhere in the convolution
    sup_norm = float(np.max(np.abs(conv_full)))
    if sup_norm < 1e-12:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _normalize(h):
    """Normalize so that L1 = 1 for consistent comparison."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s < 1e-12:
        return h
    return h / s

def _score_normalized(h):
    """Score with L1-normalized input."""
    h = _normalize(h)
    l2_sq = float(np.sum(h**2))
    l1 = float(np.sum(h))
    if l1 < 1e-12 or l2_sq < 1e-12:
        return 0.0
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv_full = np.fft.irfft(H * H, n=n_fft)
    sup_norm = float(np.max(np.abs(conv_full)))
    if sup_norm < 1e-12:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _random_search(deadline, best_h, best_score, n_restarts=200, steps_per_restart=200):
    """Random search with local refinement."""
    best_h_local = best_h.copy()
    best_score_local = best_score
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: random sparse or structured
        if restart % 3 == 0:
            # Sparse: few nonzero entries
            k = np.random.randint(1, N + 1)
            idx = np.random.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[idx] = np.random.rand(k) + 0.1
        elif restart % 3 == 1:
            # Block structure
            block_size = np.random.randint(1, N // 2 + 1)
            h = np.zeros(N)
            for start in range(0, N, block_size * 2):
                h[start:start+block_size] = np.random.rand(block_size) + 0.1
        else:
            # Random
            h = np.random.rand(N)
        
        h = _normalize(h)
        score = _score_normalized(h)
        if score > best_score_local:
            best_score_local = score
            best_h_local = h.copy()
        
        # Local refinement: coordinate descent / random perturbation
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Random coordinate perturbation
            i = np.random.randint(N)
            old_val = h[i]
            h[i] = max(0.0, old_val + np.random.normal(0, 0.05))
            
            h = _normalize(h)
            score = _score_normalized(h)
            
            if score > best_score_local:
                best_score_local = score
                best_h_local = h.copy()
            else:
                h[i] = old_val  # revert
                h = _normalize(h)
    
    return best_h_local, best_score_local

def _gradient_descent(h, deadline, n_steps=5000, lr=0.01):
    """Simple gradient-based optimization using finite differences."""
    h = _normalize(h).copy()
    best_score = _score_normalized(h)
    best_h = h.copy()
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Compute gradient via finite differences
        grad = np.zeros(N)
        eps = 1e-4
        base_score = best_score
        
        for i in range(N):
            h[i] += eps
            h = _normalize(h)
            s_plus = _score_normalized(h)
            h[i] -= 2 * eps
            h = _normalize(h)
            s_minus = _score_normalized(h)
            h[i] += eps
            h = _normalize(h)
            
            grad[i] = (s_plus - s_minus) / (2 * eps)
        
        # Update
        h = h + lr * grad
        h = np.maximum(h, 0.0)
        h = _normalize(h)
        
        score = _score_normalized(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.9
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _score_normalized(best_h)
    
    # Phase 1: Random search to find good starting points
    if time.time() < deadline - 30:
        best_h, best_score = _random_search(deadline - 30, best_h, best_score, 
                                             n_restarts=100, steps_per_restart=100)
    
    # Phase 2: Gradient refinement from best found
    if time.time() < deadline - 10:
        h_grad, score_grad = _gradient_descent(best_h, deadline - 5, n_steps=3000, lr=0.01)
        if score_grad > best_score:
            best_h = h_grad
            best_score = score_grad
    
    # Phase 3: More random search if time remains
    if time.time() < deadline - 5:
        best_h, best_score = _random_search(deadline - 2, best_h, best_score,
                                             n_restarts=50, steps_per_restart=150)
    
    # Final gradient polish
    if time.time() < deadline - 2:
        h_grad, score_grad = _gradient_descent(best_h, deadline - 1, n_steps=1000, lr=0.005)
        if score_grad > best_score:
            best_h = h_grad
            best_score = score_grad
    
    # Ensure nonnegative and return as list
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]
