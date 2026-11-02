import numpy as np
import time
import scipy.optimize as opt

N = 100
W = 2.0 / N

def _compute_overlaps(h):
    """
    Compute the overlap for all shifts k.
    overlap[k] = W * (sum_h - autocorr[k])
    """
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    sum_h = np.sum(h)
    overlaps = W * (sum_h - autocorr)
    return overlaps

def _project_to_simplex_box(h, target_sum):
    """
    Project h onto the intersection of the box [0,1] and the hyperplane sum(h) = target_sum.
    """
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    lambda_opt = (lo + hi) / 2.0
    proj = np.clip(h - lambda_opt, 0.0, 1.0)
    return proj

def _soft_max(x, beta):
    """
    Smooth approximation of max(x) using soft-max.
    """
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _grad_soft_max(x, beta):
    """
    Gradient of soft_max(x, beta) w.r.t x.
    """
    x_max = np.max(x)
    shifted = beta * (x - x_max)
    exp_vals = np.exp(shifted)
    weights = exp_vals / np.sum(exp_vals)
    return weights

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy: Optimize h directly using a differentiable soft-max objective.
    # We want to minimize max(overlaps) = max(W * (sum_h - autocorr)).
    # Since W and sum_h are constant (sum_h = N/2 = 50), minimizing max overlap
    # is equivalent to maximizing min(autocorr) or minimizing max(-autocorr).
    # Let f(h) = -autocorr. We want to minimize max(f(h)).
    # Using soft-max: obj = soft_max(f(h), beta).
    # We perform gradient descent on h, projecting to the feasible set [0,1] with sum N/2.
    
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    # Initialize with a random feasible point
    np.random.seed(42)
    h = np.full(N, 0.5) + np.random.normal(0, 0.1, N)
    h = _project_to_simplex_box(h, N / 2.0)
    
    beta = 20.0
    beta_max = 100.0
    lr = 0.02
    max_iters = 2000
    patience = 50
    no_improve = 0
    
    for i in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Compute autocorrelation
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(H * np.conj(H)).real
        sum_h = np.sum(h)
        overlaps = W * (sum_h - autocorr)
        
        # Check validity
        if not np.all(overlaps >= 0):
            # Numerical error, skip update
            continue
            
        max_ov = np.max(overlaps)
        
        if max_ov < best_score - 1e-8:
            best_score = max_ov
            best_h = h.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > patience:
            # Restart with a new random perturbation if stuck
            h = np.full(N, 0.5) + np.random.normal(0, 0.2, N)
            h = _project_to_simplex_box(h, N / 2.0)
            no_improve = 0
            beta = 20.0 # Reset beta on restart
        
        # Compute gradient of soft-max overlap
        # obj = soft_max(overlaps, beta)
        # grad_obj = weights * W * (- grad_autocorr)
        # autocorr = IFFT(|FFT(h)|^2)
        # d(autocorr)/d(h) = ?
        # Let's use the spectral gradient derived earlier or finite differences?
        # Finite differences are slow (N=100, 2000 iters -> 20k FFTs, okay).
        # But analytical gradient is better.
        # From previous logic:
        # autocorr_k = sum_j h_j h_{j+k}
        # d(autocorr_k)/d(h_i) = h_{i-k} + h_{i+k} (circular)
        # This is the convolution of h with [0..1..0] (impulse at -k and k).
        # Specifically, d(autocorr)/d(h) = h_conv where h_conv is the circular cross-correlation of h with a delta pair?
        # Actually, let's use the FFT property.
        # autocorr = IFFT(H * conj(H)).
        # d(autocorr)/d(H) = conj(H) + H (since d(H conj(H))/dH = conj(H) and d/d conj(H) = H, but h is real so H is symmetric).
        # For real h, d(autocorr)/d(h) = IFFT(2 * conj(H)) * ...? No.
        # Let's stick to the subgradient of the soft-max directly on h using the chain rule via FFT.
        
        # Objective: J = soft_max(W * (S - autocorr), beta)
        # dJ/d(autocorr) = -W * weights
        # d(autocorr)/d(h) is the circular convolution of h with a specific kernel?
        # autocorr = IFFT(H * H*)
        # Let A = H * H*.
        # dA/dH = H* (treating H* as constant)
        # dA/dH* = H
        # Since h is real, H* = conj(H).
        # This is tricky to differentiate directly in h-space without care.
        
        # Alternative: Use finite differences for gradient?
        # N=100. 100 extra FFTs per step. 2000 steps -> 200,000 FFTs.
        # FFT of size 100 is very fast. ~10us. 200k * 10us = 2s. This is feasible.
        
        # Let's implement numerical gradient to be safe and robust.
        grad = np.zeros(N)
        eps = 1e-5
        h_orig = h.copy()
        for j in range(N):
            h_plus = h_orig.copy()
            h_plus[j] += eps
            h_plus = _project_to_simplex_box(h_plus, N / 2.0) # This breaks the local gradient structure!
            # Projection is non-differentiable.
            # It is better to optimize in the spectral domain or use a penalty method.
            
            # Given the complexity of the projection, let's switch to a different strategy:
            # Optimize the power spectrum P_k directly, subject to P_k >= 0 and symmetry.
            # Reconstruct h from P_k (assuming phase 0 for simplicity, or optimizing phase too?).
            # If we fix phase to 0, h is symmetric and even.
            # h = IFFT(sqrt(P)).
            # We can optimize log_P.
            
            break # Stop numerical gradient idea
            
        # Let's implement the spectral optimization strategy (Inspiration 2/3 style)
        # Variables: log_P_sym (length N//2 + 1)
        # h = IFFT(sqrt(P_sym))
        # Constraint: h in [0,1], sum(h)=N/2.
        # This is hard to enforce.
        
        # Let's go back to direct h optimization but handle the projection issue.
        # Use a penalty method for the bounds and sum constraint?
        # Or, use the fact that the optimal solution likely has many values at 0 or 1?
        # The Erdos problem suggests a binary-like solution might be optimal or near-optimal.
        
        # New Strategy: Simulated Annealing on h with soft-max objective.
        # We already have a projection step.
        # Let's compute the gradient of the *unprojected* soft-max overlap and use it as a direction.
        # Then project the new point. This is a projected gradient method (approximate).
        
        # Compute weights for soft-max
        max_ov = np.max(overlaps)
        shifted = beta * (overlaps - max_ov)
        exp_vals = np.exp(shifted)
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of overlaps w.r.t h
        # overlaps = W * (S - autocorr)
        # grad_overlaps = -W * grad_autocorr
        # grad_autocorr is the circular convolution of h with [1, 0, ..., 0, 1]?
        # No.
        # autocorr_k = sum_i h_i h_{i+k}
        # d(autocorr_k)/d(h_j) = h_{j-k} + h_{j+k}
        # Let K be a vector where K[0]=1, K[k]=1, K[-k]=1.
        # Then grad_autocorr_j = sum_k (d(autocorr_k)/d(h_j)) * dJ/d(autocorr_k)
        # This is a convolution of weights with h?
        # Let's compute grad_autocorr directly via FFT.
        # d(autocorr)/d(h) = IFFT(2 * conj(H) * d(autocorr)/d(H)?)
        # Actually, there is a known result:
        # If y = x * x (circular autocorr), then d(y)/d(x) = 2 * (x * delta) where delta is symmetric?
        # Let's use the FFT derivative:
        # Y = H * conj(H)
        # dY/dH = conj(H)
        # dY/dconj(H) = H
        # dy = IFFT(dY)
        # dh = IFFT(dH)
        # This is getting messy.
        
        # Let's use a simpler heuristic:
        # The gradient of the autocorrelation at shift k is h shifted by k plus h shifted by -k.
        # So the gradient of the total objective (weighted sum of autocorrs) is:
        # grad_h = -W * (sum_k weights_k * (roll(h, k) + roll(h, -k)))
        
        grad_h = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-6:
                grad_h -= W * weights[k] * (np.roll(h, k) + np.roll(h, -k))
        
        # Update h
        h_new = h - lr * grad_h
        h_new = _project_to_simplex_box(h_new, N / 2.0)
        
        # Accept if better or with probability for annealing
        overlaps_new = _compute_overlaps(h_new)
        max_ov_new = np.max(overlaps_new)
        
        if max_ov_new < max_ov:
            h = h_new
        else:
            # Annealing
            delta = max_ov_new - max_ov
            if np.random.random() < np.exp(-delta / (0.01 + 0.01 * np.exp(-i/500.0))):
                h = h_new
        
        # Increase beta
        beta = min(beta * 1.02, beta_max)
        # Decay lr
        lr *= 0.999
        
    best_h = _project_to_simplex_box(best_h, N / 2.0)
    if not np.isclose(np.sum(best_h), N/2.0, atol=1e-5):
        best_h = np.full(N, 0.5)
        
    return best_h
