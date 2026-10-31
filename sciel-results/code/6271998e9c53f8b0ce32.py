"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with all-ones (theoretical optimum for scale-invariant score)
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    def try_candidate(x):
        nonlocal best_x, best_score
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
            return True
        return False
    
    # Phase 1: Try structured real-valued sequences
    # Constant sequences with different scales (should give same score due to scale invariance)
    for scale in [0.1, 0.5, 1.0, 2.0, 10.0]:
        x = np.ones(N) * scale
        try_candidate(x)
    
    # Square wave: positive block then negative block
    # x = +1 for i < k, -1 for i >= k
    # sum = k - (N-k) = 2k - N
    # autoconv of square wave has structure
    for k in range(1, N):
        if time.time() > deadline - 10:
            break
        x = np.ones(N)
        x[k:] = -1.0
        try_candidate(x)
    
    # Phase 2: Gradient descent on real-valued sequence
    # Objective: minimize max|A| subject to sum(x) = S (or just minimize max|A|/sum²)
    # Use a smooth proxy: sum(exp(A_i/tau)) for the max, or just track the argmax
    
    x = np.ones(N).copy()
    current_sum = np.sum(x)
    size = 2 * N - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real[:size]
    
    # Gradient descent: minimize f(x) = max|A| - lambda * (sum(x) - target_sum)^2
    # Or directly: minimize max|A| / sum²
    # Use subgradient of max|A|
    
    T = 1.0
    lr = 0.1
    steps = 2000
    
    for step in range(steps):
        if time.time() > deadline - 10:
            break
        
        # Compute gradient of max|A| w.r.t. x
        # A[k] = sum_i x[i]*x[i+k] (with appropriate indexing for linear conv)
        # d A[k] / d x[j] = x[j+k] + x[j-k] (with boundary handling)
        # For the argmax of |A|, the subgradient points in direction of A[argmax]
        
        abs_A = np.abs(A)
        # Soft max with temperature for smoothness
        tau = max(T, 0.1)
        weights = np.exp(abs_A / tau - np.max(abs_A / tau))
        weights /= np.sum(weights)
        
        # Gradient of sum(w_i * |A_i|) w.r.t. x
        # |A_i| = sign(A_i) * A_i (for A_i != 0)
        # dA_i/dx_j = x_{j+i} + x_{j-i} (linear conv, with bounds)
        grad = np.zeros(N)
        for lag in range(size):
            w = weights[lag]
            if w < 1e-10:
                continue
            sgn = 1.0 if A[lag] >= 0 else -1.0
            # A[lag] = sum_{j=0}^{N-lag-1} x[j]*x[j+lag] (for lag >= 0)
            # Actually linear autoconv: A[lag] = sum_j x[j]*x[j+lag] for lag>=0
            # dA[lag]/dx[j] = x[j+lag] + (x[j-lag] if j>=lag else 0)
            # Simplified: for each j, contribution is x[j+lag] if j+lag < N, plus x[j-lag] if j >= lag
            j_range = np.arange(N)
            # Term 1: x[j+lag] for j+lag < N => j < N-lag
            mask1 = j_range < (N - lag)
            if np.any(mask1):
                grad[mask1] += w * sgn * x[mask1 + lag]
            # Term 2: x[j-lag] for j >= lag
            mask2 = j_range >= lag
            if np.any(mask2):
                grad[mask2] += w * sgn * x[mask2 - lag]
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-10:
            grad /= gnorm
        
        # Update x, keeping sum roughly constant
        # Project gradient to remove component along ones vector (to preserve sum)
        ones = np.ones(N)
        grad_proj = grad - np.mean(grad) * ones
        x -= lr * grad_proj
        
        # Renormalize: scale to keep sum = N (or any fixed value)
        new_sum = np.sum(x)
        if abs(new_sum) > 1e-10:
            x *= (current_sum / new_sum)
        
        # Update F and A
        F = fft(x, size2)
        A = ifft(F * np.conj(F)).real[:size]
        
        # Try this candidate
        try_candidate(x)
        
        # Decay temperature
        T *= 0.99
        
        # Adaptive learning rate
        if step % 100 == 0:
            lr *= 0.95
    
    # Phase 3: Local search - try flipping individual values (for real-valued, try small perturbations)
    x = best_x.copy()
    F = fft(x, size2)
    current_sum = np.sum(x)
    
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old_val = x[i]
            # Try setting to 0 or small values
            for new_val in [0.0, 0.5, -0.5, 1.0, -1.0, 0.1, -0.1]:
                if abs(new_val - old_val) < 1e-6:
                    continue
                x[i] = new_val
                new_sum = current_sum - old_val + new_val
                if abs(new_sum) < 1e-10:
                    x[i] = old_val
                    continue
                F_new = F + fft(np.concatenate([[0]*i, [new_val - old_val], [0]*(size2-i-1)]), size2) if i < size2 else F
                A_new = ifft(F_new * np.conj(F_new)).real[:size]
                m_new = np.max(np.abs(A_new))
                score_new = m_new / (new_sum * new_sum)
                if score_new < best_score - 1e-10:
                    improved = True
                    best_score = score_new
                    best_x = x.copy()
                    F = F_new
                    current_sum = new_sum
                    break
                x[i] = old_val
            if improved:
                break
    
    return best_x
# EVOLVE-BLOCK-END
