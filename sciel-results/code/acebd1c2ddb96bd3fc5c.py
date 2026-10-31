import numpy as np
from scipy import optimize

N = 600
FFT_SIZE = 2048  # Next power of 2 >= 2*N-1 = 1199

def _fft_convolve(h):
    """Compute linear autoconvolution using FFT for speed."""
    H = np.fft.rfft(h, FFT_SIZE)
    P = H * np.conj(H)
    conv = np.fft.irfft(P, FFT_SIZE)
    return conv[:2*N-1]

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total == 0:
        return 0.0
    return np.max(conv) / (total * total)

def _soft_max_value(conv, T):
    """Compute soft max of conv with temperature T using log-sum-exp."""
    conv_max = np.max(conv)
    shifted = (conv - conv_max) / T
    shifted = np.clip(shifted, -50, 50)
    return conv_max + T * np.log(np.sum(np.exp(shifted)))

def _soft_objective(x, T):
    """Smooth surrogate objective using log-sum-exp."""
    h = np.abs(x)
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total < 1e-10:
        return 1e10
    sm = _soft_max_value(conv, T)
    return sm / (total * total)

def _soft_gradient(x, T):
    """Gradient of the soft objective."""
    h = np.abs(x)
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total < 1e-10:
        return np.zeros(N)
    
    conv_max = np.max(conv)
    shifted = (conv - conv_max) / T
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # softmax weights
    
    # d(soft_max)/d(conv_k) = weights[k]
    # conv = h * h, so d(conv)/d(h_i) involves h_j for all j
    # Using FFT: conv_k = sum_i h_i * h_{k-i}
    # d(conv_k)/d(h_i) = h_{k-i} + h_{k-i} (for autoconv, symmetric)
    # Actually for autoconv: conv[k] = sum_i h[i]*h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] + h[j-k] if we consider both terms
    # But since conv is autoconvolution, it's symmetric.
    # More precisely: d(conv[k])/d(h[j]) = h[k-j] if k-j is in range, plus h[j-k] if j-k is in range
    # For full convolution of h with itself:
    # conv[k] = sum_i h[i] * h[k-i] where i ranges such that both indices valid
    # d(conv[k])/d(h[j]) = h[k-j] (from first factor) + h[k-j] (from second factor when i=j in the other sum)
    # Wait, let me think more carefully.
    # conv = h * h (full convolution)
    # conv[n] = sum_i h[i] * h[n-i]
    # d(conv[n])/d(h[j]) = h[n-j] + h[j-n] if both n-j and j-n are valid indices
    # Since h is defined on [0, N-1], n-j is valid if 0 <= n-j <= N-1
    # and j-n is valid if 0 <= j-n <= N-1, i.e., n <= j <= n+N-1
    # So d(conv[n])/d(h[j]) = h[n-j] * (1 if 0<=n-j<N else 0) + h[j-n] * (1 if 0<=j-n<N else 0)
    # But h[j-n] is only non-zero if 0 <= j-n < N
    # This is getting complex. Let me use a different approach.
    # 
    # Alternative: use the fact that conv = ifft(fft(h)^2)
    # d(conv)/d(h) can be computed via FFT as well
    # But for simplicity, let's use a numerical gradient or a simplified analytical form.
    #
    # Actually, for autoconvolution, we can write:
    # conv = h ⊛ h (cross-correlation-like)
    # The gradient of sum_k w_k * conv[k] w.r.t. h is:
    # grad[j] = sum_k w_k * (d(conv[k])/d(h[j]))
    # d(conv[k])/d(h[j]) = h[k-j] + h[j-k] (with boundary handling)
    #
    # Let me use a simpler approach: compute the gradient numerically for the key terms.
    # Or better: use the relationship that the gradient of the LSE soft-max of autoconv
    # can be computed as: grad = 2 * (weights * h) convolved appropriately
    #
    # Actually, the cleanest way: 
    # Let g = weights (shape 2N-1)
    # We want d/dh [sum_k g[k] * conv[k]] / (total^2) - 2*sum(h)*sum_k g[k]*conv[k] / total^3 * h
    # 
    # First term: d/dh [sum_k g[k] * conv[k]]
    # conv[k] = sum_i h[i] * h[k-i]
    # d/dh[j] sum_k g[k] conv[k] = sum_k g[k] * (h[k-j] + h[j-k]) [with bounds]
    # = sum_k g[k] h[k-j] + sum_k g[k] h[j-k]
    # First sum: let m = k-j, = sum_m g[m+j] h[m] = (g_shifted * h) at position... 
    # This is a cross-correlation of g with h.
    #
    # Let me just use a numerical gradient approach for robustness, or use a simpler
    # analytical form. Given the complexity, let me use a semi-analytical approach:
    #
    # The gradient of sum_k w_k conv[k] w.r.t. h[j] is:
    # = sum_k w_k [h[k-j] * I(0<=k-j<N) + h[j-k] * I(0<=j-k<N)]
    # = sum_k w_k h[k-j] * I(0<=k-j<N) + sum_k w_k h[j-k] * I(0<=j-k<N)
    #
    # First term: for fixed j, k-j ranges from 0 to N-1, so k from j to j+N-1
    #   sum_{k=j}^{j+N-1} w_k h[k-j] = sum_{m=0}^{N-1} w_{j+m} h[m]
    # Second term: for fixed j, j-k ranges from 0 to N-1, so k from j-N+1 to j
    #   sum_{k=max(0,j-N+1)}^{j} w_k h[j-k] = sum_{m=0}^{min(j,N-1)} w_{j-m} h[m]
    #
    # This is O(N^2) which is too slow. Let me use FFT.
    #
    # Actually, let me just use a simpler approach: compute the gradient using
    # the fact that for autoconvolution, the gradient can be expressed as
    # a convolution of the weights with h.
    #
    # Let me use a practical approach: compute the gradient numerically for a
    # subset or use an approximation. Given time constraints, let me use
    # L-BFGS-B with the FFT-accelerated objective, which computes gradients
    # numerically but with fast function evaluations.

    # For now, let's use L-BFGS-B which handles gradients internally
    # The key speedup is from FFT in the objective function
    raise NotImplementedError("Use L-BFGS-B instead")

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s margin
    
    # Initial guess: all ones
    h = np.ones(N)
    best_h = h.copy()
    best_score = compute_autoconv_max(h)
    
    np.random.seed(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Cosine perturbation
    t = np.linspace(0, 1, N, endpoint=False)
    initializations.append(np.maximum(1.0 + 0.15 * np.cos(2 * np.pi * t), 0.01))
    initializations.append(np.maximum(1.0 + 0.15 * np.sin(2 * np.pi * t), 0.01))
    
    # Two-frequency
    initializations.append(np.maximum(1.0 + 0.1 * np.cos(2 * np.pi * t) + 0.05 * np.cos(4 * np.pi * t), 0.01))
    initializations.append(np.maximum(1.0 + 0.05 * np.cos(2 * np.pi * t) + 0.1 * np.cos(3 * np.pi * t), 0.01))
    
    # Random perturbations
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        h0 = np.ones(N) + 0.1 * np.random.randn(N)
        initializations.append(np.maximum(h0, 0.01))
    
    # Edge-enhanced
    initializations.append(np.maximum(1.0 + 0.3 * np.abs(np.linspace(-1, 1, N)), 0.01))
    
    # Center-enhanced  
    t2 = np.linspace(-1, 1, N)
    initializations.append(np.maximum(1.0 + 0.3 * np.exp(-t2**2 * 5), 0.01))
    
    # Asymmetric shapes (from knowledge: reduces overlap of f(t) and f(-t))
    initializations.append(np.maximum(1.0 + 0.2 * np.linspace(0, 1, N), 0.01))
    initializations.append(np.maximum(1.0 - 0.2 * np.linspace(0, 1, N), 0.01))
    initializations.append(np.maximum(1.0 + 0.25 * np.sin(np.pi * np.linspace(0, 1, N)), 0.01))
    
    # Random with normalization
    for seed in [100, 200, 300]:
        np.random.seed(seed)
        h0 = np.random.rand(N) + 0.05
        h0 = h0 / np.mean(h0)
        initializations.append(h0)
    
    # Optimize each initialization
    T_values = [0.02, 0.01, 0.005]  # Lower temperatures for better approximation
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        h0 = np.abs(h0)
        
        # Multi-stage optimization with decreasing temperature
        current_h = h0.copy()
        
        for T in T_values:
            if time.time() > deadline - 25:
                break
            
            try:
                # Use L-BFGS-B with FFT-accelerated objective
                def obj(x):
                    return _soft_objective(x, T)
                
                result = optimize.minimize(
                    obj, current_h,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
                )
                
                if result.fun < 1e9:  # Valid result
                    current_h = np.abs(result.x)
                    
            except Exception:
                pass
        
        # Evaluate with exact objective
        actual_score = compute_autoconv_max(current_h)
        if actual_score < best_score:
            best_score = actual_score
            best_h = current_h.copy()
    
    # Final polish: high-precision optimization from best solution
    if time.time() < deadline - 15:
        try:
            # Very low temperature for near-exact optimization
            def obj_fine(x):
                return _soft_objective(x, 0.001)
            
            result = optimize.minimize(
                obj_fine, best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
            )
            
            h_new = np.abs(result.x)
            actual_score = compute_autoconv_max(h_new)
            if actual_score < best_score:
                best_score = actual_score
                best_h = h_new.copy()
                
        except Exception:
            pass
    
    # Additional polish with local perturbations
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            try:
                h_perturbed = best_h + 0.005 * np.random.randn(N)
                h_perturbed = np.maximum(h_perturbed, 0)
                
                def obj_local(x):
                    return _soft_objective(x, 0.001)
                
                result = optimize.minimize(
                    obj_local, h_perturbed,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                h_new = np.abs(result.x)
                actual_score = compute_autoconv_max(h_new)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_new.copy()
                    
            except Exception:
                pass
    
    return list(best_h)
