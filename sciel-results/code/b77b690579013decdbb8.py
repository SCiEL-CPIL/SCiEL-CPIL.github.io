"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_optimize(h_init, time_budget):
    """
    Optimize h in the frequency domain to minimize max(|H|^2) subject to sum(h)=1.
    This is equivalent to minimizing the peak power in the spectrum while keeping DC fixed.
    """
    # Normalize initial guess to sum 1
    s0 = np.sum(h_init)
    if abs(s0) < 1e-9:
        return h_init, _score(h_init)
    h = h_init / s0
    
    # Work with full complex FFT for symmetry handling
    n = len(h)
    fsize = 1
    while fsize < 2*n - 1:
        fsize *= 2
        
    # We optimize H = FFT(h, fsize)
    # Constraints:
    # 1. h is real -> H[k] = conj(H[fsize-k])
    # 2. sum(h) = 1 -> H[0] = 1 (since H[0] = sum(h))
    # Objective: min max_k |H[k]|^2
    
    # Initial spectrum
    H = np.fft.fft(h, fsize)
    
    # Mask for indices that are free to change.
    # Index 0 is fixed to 1.
    # Due to real-valued constraint, H[k] and H[fsize-k] are coupled.
    # We can parameterize the free variables.
    # For simplicity and speed, we will do gradient descent on the real-valued h,
    # but compute gradients in the frequency domain where the objective is smooth (soft-max).
    
    # Let's use a soft-max approximation for the objective:
    # J = (1/beta) * log(sum(exp(beta * |H[k]|^2)))
    # We want to minimize J.
    
    # Gradient of J w.r.t. h:
    # J = (1/beta) log(sum_k exp(beta |H_k|^2))
    # dJ/dH_k = 2 H_k * p_k, where p_k = exp(beta |H_k|^2) / sum_j exp(beta |H_j|^2)
    # dH_k/dh = (1/fsize) * conj(FFT_basis_k) ... actually dH/dh is the DFT matrix.
    # dJ/dh = (1/fsize) * IFFT(dJ/dH)
    
    beta = 50.0
    # Adaptive beta might help, but let's start fixed.
    
    # Precompute IFFT kernel? No, just use scipy or numpy fft.
    
    # Gradient Descent with momentum
    v = np.zeros_like(h)
    lr = 1.0 / fsize # Initial learning rate heuristic
    momentum = 0.9
    t_start = time.time()
    
    # We need to ensure sum(h) stays 1.
    # We can project h onto the hyperplane sum(h)=1 after each step.
    # h_proj = h - (sum(h)-1)/n * ones
    
    # To avoid the DC component changing, we can explicitly set H[0]=1 in the spectrum,
    # but since we are updating h, it's easier to project h.
    
    # Let's refine the objective function for the gradient.
    def compute_grad(h_curr):
        H = np.fft.fft(h_curr, fsize)
        # |H|^2
        power = np.abs(H)**2
        # Soft max weights
        # For numerical stability, subtract max
        max_p = np.max(power)
        if max_p < 1e-15:
             weights = np.ones(fsize)/fsize
        else:
             exp_terms = np.exp(beta * (power - max_p))
             sum_exp = np.sum(exp_terms)
             weights = exp_terms / sum_exp
             
        # dJ/dH = 2 * H * weights
        dJ_dH = 2.0 * H * weights
        # dJ/dh = IFFT(dJ_dH) / fsize  ? 
        # Recall: H = F h. dJ/dh = F^H (dJ/dH).
        # In numpy, ifft is 1/n * F^H. So dJ/dh = n * ifft(dJ_dH) / n = ifft(dJ_dH) ?
        # Let's check: h = ifft(H). dH/dh = F. dJ/dh = F^T dJ/dH (real).
        # F^T = n * ifft_matrix (conjugate).
        # Actually, for real h, we can just use:
        grad_h = np.fft.ifft(dJ_dH).real
        # Wait, the scaling.
        # h = (1/n) sum H_k exp(i 2 pi k n / n)
        # dJ/dh_i = sum_k (dJ/dH_k) (dH_k/dh_i)
        # dH_k/dh_i = exp(-i 2 pi k i / n)
        # So dJ/dh = IFFT(dJ_dH) * (1/n) ? No.
        # Standard IDFT: x[n] = 1/N sum X[k] e^{i 2 pi kn/N}
        # Gradient is the adjoint of the forward transform.
        # Forward: X[k] = sum x[n] e^{-i 2 pi kn/N}
        # Adjoint: x[n] = 1/N sum X[k] e^{i 2 pi kn/N}
        # So grad_h = ifft(dJ_dH).real
        
        return grad_h

    # Initialize
    h_curr = h.copy()
    best_h = h_curr.copy()
    best_score = _score(h_curr)
    
    # We want to minimize the score.
    # The spectral optimization minimizes max |H|^2.
    # Since sum(h)=1, H[0]=1.
    # The minimum possible max |H|^2 is 1 (if all other H[k]=0, which is impossible for finite support unless h is delta, but sum=1 means h=delta only if n=1).
    # Actually, if h is constant 1/N, H[0]=1, H[k>0]=0. Max |H|^2 = 1.
    # Score = max|ac| / (sum h)^2.
    # If H[0]=1, H[k]=0, ac is delta at 0 with value 1. max|ac|=1. Score = 1/1 = 1.
    # Wait, is it possible to get score < 1?
    # ac = ifft(|H|^2). sum(ac) = H[0]^2 = 1.
    # If ac is a delta function, max|ac| = 1.
    # If ac is spread out, max|ac| < 1?
    # Sum of ac is 1. If ac is non-negative and spread over 2N-1 points, max can be < 1.
    # But ac can be negative.
    # However, |H|^2 is non-negative. ac is the inverse FFT of a non-negative real even sequence.
    # Does ac have to be non-negative? No.
    # But max|ac| is at least max(ac).
    # If ac is spread out, its peak decreases.
    # So we want to spread |H|^2 as much as possible, subject to |H[0]|^2 = 1 and H[k]=conj(H[-k]).
    # This is equivalent to making the time-domain autocorrelation as flat as possible (Barker-like).
    
    # Let's run the optimization
    for step in range(5000):
        if time.time() > time_budget:
            break
            
        grad = compute_grad(h_curr)
        
        # Project gradient to zero-sum subspace?
        # We want to keep sum(h)=1.
        # If we move h -> h - lr * grad, sum changes by -lr * sum(grad).
        # We can subtract the mean of the gradient to ensure the step has zero mean.
        grad = grad - np.mean(grad)
        
        v = momentum * v - lr * grad
        h_curr = h_curr + v
        
        # Project back to sum=1
        h_curr = h_curr + (1.0 - np.sum(h_curr)) / n
        
        # Evaluate
        curr_score = _score(h_curr)
        if curr_score < best_score:
            best_score = curr_score
            best_h = h_curr.copy()
            
        # Adaptive learning rate
        # If score improves, keep lr. If not, decrease.
        # Simple heuristic: decrease lr slowly.
        lr *= 0.999
        
        # Occasionally, re-normalize H[0] exactly?
        # The projection handles sum(h).
        
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
        
    # Initialization strategies
    inits = []
    
    # 1. All ones (flat spectrum except DC)
    inits.append(np.ones(N))
    
    # 2. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(10):
        noise = rng.standard_normal(N)
        # Apply a low-pass filter to start with a smooth signal
        kernel_size = 5
        kernel = np.hanning(kernel_size)
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    # 3. Chirp
    x = np.arange(N)
    inits.append(np.cos(np.pi * x**2 / N))
    
    # 4. Barker-like (alternating)
    inits.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Optimization loop
    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
            
        # Spectral optimization
        try:
            h_opt, val_opt = _spectral_optimize(h_init, deadline - 5)
            if val_opt < best_score:
                best_score = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Final local search refinement (time domain flip) to polish
    if time.time() < deadline - 5:
        h_refined = best_h.copy()
        best_val = best_score
        # Simple flip search
        for _ in range(3):
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                h_refined[i] = -h_refined[i]
                new_val = _score(h_refined)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h_refined[i] = -h_refined[i]
            if not improved:
                break
        if best_val < best_score:
            best_score = best_val
            best_h = h_refined

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
