"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr(s):
    """Compute periodic autocorrelation of sequence s."""
    n = len(s)
    # Use FFT for efficient autocorrelation
    S = np.fft.fft(s)
    power = S * np.conj(S)
    acorr = np.fft.ifft(power).real
    return acorr

def _smooth_max_abs(x, beta=20.0):
    """Smooth approximation of max(|x|) using log-sum-exp."""
    abs_x = np.abs(x)
    # log-sum-exp for stability
    m = np.max(abs_x)
    if m == 0:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (abs_x - m)))) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    np.random.seed(42)
    
    # Start with random ±1 sequence
    s = np.random.choice([-1.0, 1.0], size=N)
    
    # Compute initial objective
    acorr = _compute_autocorr(s)
    # Exclude lag 0 (which is always N)
    acorr_nontrivial = np.concatenate([acorr[1:], acorr[:N-1]])  # circular
    # Actually, for periodic autocorrelation, lags 1..N-1
    acorr_nontrivial = acorr[1:]
    
    best_s = s.copy()
    best_score = np.max(np.abs(acorr_nontrivial))
    
    # Use continuous relaxation: x in R^N, s_i = tanh(x_i)
    # Optimize x to minimize smooth max of |autocorr|
    x = np.zeros(N)  # starts at 0, tanh(0)=0, but we want ±1
    
    # Actually, let's start from the random ±1 and map to x
    # tanh(x) ≈ ±1 when x is large
    x = np.zeros(N)
    for i in range(N):
        x[i] = 2.0 if s[i] > 0 else -2.0
    
    beta = 10.0  # temperature for smooth max
    lr = 0.1
    best_x = x.copy()
    best_obj = best_score
    
    # Gradient descent with adaptive learning rate
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        s_soft = np.tanh(x)
        acorr = _compute_autocorr(s_soft)
        acorr_nontriv = acorr[1:]
        
        obj = _smooth_max_abs(acorr_nontriv, beta)
        
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
        
        # Compute gradient of objective w.r.t. x
        # d obj / d s_i, then d s_i / d x_i = 1 - s_i^2
        abs_ac = np.abs(acorr_nontriv)
        m = np.max(abs_ac)
        if m > 0:
            weights = np.exp(beta * (abs_ac - m)) / np.sum(np.exp(beta * (abs_ac - m)))
        else:
            weights = np.ones_like(abs_ac) / len(abs_ac)
        
        # Gradient of max|acorr| w.r.t. acorr components
        # For each lag k, acorr[k] = sum_i s_i * s_{(i+k) mod N}
        # d|acorr[k]|/dacorr[k] = sign(acorr[k])
        grad_ac = np.zeros(N-1)
        for k in range(N-1):
            grad_ac[k] = weights[k] * beta * np.sign(acorr_nontriv[k])
        
        # Now backprop through autocorrelation to s
        # acorr[k] = sum_i s_i * s_{(i+k)%N}
        # d acorr[k] / d s_j = s_{(j-k)%N} + s_{(j+k)%N} (for periodic)
        grad_s = np.zeros(N)
        for k in range(1, N-1):
            grad_s += grad_ac[k-1] * (np.roll(s_soft, -k) + np.roll(s_soft, k))
        
        # d s_i / d x_i = 1 - tanh(x_i)^2
        grad_x = grad_s * (1.0 - s_soft ** 2)
        
        # Adaptive step size
        norm_grad = np.linalg.norm(grad_x)
        if norm_grad > 1e-12:
            step = lr * 0.01 / (norm_grad + 1e-8)
            x -= step * grad_x
        else:
            x -= lr * 0.01 * grad_x
        
        # Decay beta slowly to sharpen
        if iteration % 500 == 0 and beta < 50:
            beta *= 1.5
    
    # Final: round to ±1
    s_final = np.sign(best_x)
    s_final[s_final == 0] = 1.0
    
    # Local search: single-bit flips
    acorr = _compute_autocorr(s_final)
    acorr_nontriv = acorr[1:]
    current_score = np.max(np.abs(acorr_nontriv))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = s_final[i]
            s_final[i] = -old_val
            acorr = _compute_autocorr(s_final)
            acorr_nontriv = acorr[1:]
            new_score = np.max(np.abs(acorr_nontriv))
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                s_final[i] = old_val
    
    return s_final
# EVOLVE-BLOCK-END
