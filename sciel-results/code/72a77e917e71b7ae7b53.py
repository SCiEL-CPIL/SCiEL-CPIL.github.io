import numpy as np
import time
import random
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        # Autoconvolution via FFT
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)

    def spectral_optimize(h_init, max_iters=5000, flatten_power=0.99):
        """Iterative projection: flatten power spectrum + enforce nonnegativity."""
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        
        size = 2 * N
        
        for it in range(max_iters):
            if time_left() < 10:
                break
            
            # FFT
            H = np.fft.rfft(h, n=size)
            
            # Power spectrum
            power = np.abs(H) ** 2
            
            # Flatten: reduce peaks in power spectrum
            mean_power = np.mean(power)
            threshold = mean_power * 1.5
            excess = np.maximum(power - threshold, 0)
            power_new = power - flatten_power * excess
            power_new = np.maximum(power_new, 0)
            
            # Reconstruct H with flattened power (keep phase)
            phase = np.angle(H)
            H_new = np.sqrt(power_new) * np.exp(1j * phase)
            
            # IFFT back
            h_new = np.fft.irfft(H_new, n=size)[:N]
            
            # Project onto nonnegative
            h_new = np.maximum(h_new, 0.0)
            
            # Ensure not all zero
            if np.all(h_new == 0):
                h_new = h.copy()
            
            m_new = compute_metric(h_new)
            
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            
            h = h_new
        
        return best_h, best_m

    def local_refine(h_init, max_iters=50000):
        """Random perturbation local search to refine the result."""
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        
        np.random.seed(42)
        
        for it in range(max_iters):
            if time_left() < 5:
                break
            
            # Pick two random indices and swap some mass
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            # Move small fraction of mass
            frac = np.random.uniform(0.01, 0.1)
            delta = h[j] * frac
            if delta < 1e-15:
                continue
            
            new_h = h.copy()
            new_h[j] -= delta
            new_h[i] += delta
            
            if new_h[j] < -1e-15:
                continue
            
            m_new = compute_metric(new_h)
            
            if m_new < best_m:
                best_m = m_new
                best_h = new_h.copy()
                h = new_h
        
        return best_h, best_m

    def gradient_descent_simplex(h_init, max_iters=2000, lr=0.01):
        """
        Gradient descent with projection onto probability simplex.
        Model as discrete probability distribution, minimize peak of autoconvolution.
        Uses subgradient of the max function.
        """
        h = h_init.copy().astype(np.float64)
        # Normalize to probability simplex
        total = np.sum(h)
        if total > 0:
            h = h / total
        
        best_h = h.copy()
        best_m = compute_metric(h)
        
        size = 2 * N
        
        for it in range(max_iters):
            if time_left() < 10:
                break
            
            # Compute autoconvolution
            H = np.fft.rfft(h, n=size)
            acorr = np.fft.irfft(H * np.conj(H), n=size)
            
            # Find peak location
            peak_idx = np.argmax(acorr)
            peak_val = acorr[peak_idx]
            
            # Subgradient with respect to h
            # d/dh_k max(acorr) = sum_j acorr[peak_idx - (j-k)] for j where acorr is max
            # For simplicity, use the gradient of acorr[peak_idx] w.r.t. h
            # acorr[m] = sum_k h[k] * h[k+m] (for m >= 0, with appropriate indexing)
            # Gradient w.r.t. h[k] is h[k+peak_idx] + h[k-peak_idx] (with wraparound handling)
            
            # Compute gradient numerically via FFT derivative
            # grad_h[k] = sum_j h[j] * delta(j - k + peak_idx) + sum_j h[j] * delta(j - k - peak_idx)
            # = h[(k - peak_idx) % N] + h[(k + peak_idx) % N] (approximately, for linear conv)
            
            # More carefully: acorr[m] = sum_{k=0}^{N-1} h[k] * h[k+m] where h is zero-padded
            # For the peak at index peak_idx in the circular convolution of size 2N:
            # The gradient w.r.t. h[k] involves h at positions shifted by peak_idx
            
            # Let's compute gradient via FFT
            # d(acorr[m])/d(h[k]) = h[(k-m) mod N] + h[(k+m) mod N] for the non-circular case
            # But we're using circular convolution of size 2N, so it's more complex.
            
            # Simpler approach: compute gradient as the cross-correlation of h with itself at the peak lag
            # For linear convolution: grad[k] = h[k - peak_idx] + h[k + peak_idx] (with zero padding)
            
            grad = np.zeros(N)
            for k in range(N):
                # Contribution from h[k] * h[k + peak_idx]
                if k + peak_idx < N:
                    grad[k] += h[k + peak_idx]
                # Contribution from h[k - peak_idx] * h[k]
                if k - peak_idx >= 0:
                    grad[k] += h[k - peak_idx]
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            # Update: move in negative gradient direction
            h_new = h - lr * grad
            
            # Project onto probability simplex (nonnegative, sum=1)
            h_new = np.maximum(h_new, 0)
            total = np.sum(h_new)
            if total > 0:
                h_new = h_new / total
            else:
                h_new = h.copy()
            
            m_new = compute_metric(h_new)
            
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
                h = h_new
                # Decrease learning rate slowly
                lr *= 0.999
            else:
                # Keep h, maybe decrease lr
                lr *= 0.99
        
        return best_h, best_m

    def gaussian_basis_optimize(h_init, K=8, max_iters=300):
        """
        Optimize using Gaussian basis representation.
        h(x) = sum_{j=1}^{K} w_j * exp(-((x - c_j)^2) / (2 * s_j^2))
        Then project to nonnegative and normalize.
        """
        i = np.arange(N, dtype=float)
        
        # Initialize parameters from h_init
        # Simple initialization: spread centers evenly
        centers = np.linspace(0, N-1, K)
        weights = np.ones(K) * (np.sum(h_init) / K)
        sigmas = np.ones(K) * (N / 4)
        
        params = np.concatenate([weights, centers, sigmas])
        
        def params_to_h(params):
            h = np.zeros(N)
            for j in range(K):
                w_j = max(params[3*j], 0)
                c_j = params[3*j+1]
                s_j = max(params[3*j+2], 1e-6)
                h += w_j * np.exp(-((i - c_j)**2) / (2 * s_j**2))
            return h
        
        def objective(params):
            if time_left() < 5:
                return 1e10
            h = params_to_h(params)
            total = np.sum(h)
            if total < 1e-10:
                return 1e10
            h_norm = h / total
            return compute_metric(h_norm)
        
        # L-BFGS-B optimization
        bounds = []
        for j in range(K):
            bounds.append((0.01, 10.0))   # w_j
            bounds.append((0.0, N-1))     # c_j
            bounds.append((1.0, N/2))     # s_j
        
        try:
            res = minimize(
                objective,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = params_to_h(res.x)
            total = np.sum(h_opt)
            if total > 1e-10:
                h_opt = h_opt / total
                m_opt = compute_metric(h_opt)
                return h_opt, m_opt
        except Exception:
            pass
        
        return h_init, compute_metric(h_init)

    # Multiple initializations for spectral method
    i_arr = np.arange(N)
    mid = N / 2.0
    
    initializations = [
        np.ones(N),
        0.5 * (1.0 + np.cos(np.pi * (i_arr - mid) / (N / 2.0))),
        np.exp(-2.0 * ((i_arr - mid) / (N / 4.0)) ** 2),
        1.0 - 0.3 * np.cos(2 * np.pi * i_arr / N),
        np.maximum(np.cos(np.pi * (i_arr - mid) / N), 0),
        # Bimodal initialization
        np.concatenate([np.exp(-((i_arr[:N//2] - N//8) / (N//8))**2),
                        np.exp(-((i_arr[N//2:] - 7*N//8) / (N//8))**2)]),
    ]
    
    best_h = None
    best_m = float('inf')
    
    # Phase 1: Spectral optimization from multiple starts
    for init in initializations:
        if time_left() < 20:
            break
        init = np.maximum(init, 0.0)
        h_opt, m_opt = spectral_optimize(init, max_iters=2000)
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt
    
    # Phase 2: Gradient descent with simplex projection from best
    if best_h is not None and time_left() > 30:
        h_grad, m_grad = gradient_descent_simplex(best_h, max_iters=1500, lr=0.01)
        if m_grad < best_m:
            best_m = m_grad
            best_h = h_grad
    
    # Phase 3: Gaussian basis refinement
    if best_h is not None and time_left() > 25:
        h_gauss, m_gauss = gaussian_basis_optimize(best_h, K=8, max_iters=200)
        if m_gauss < best_m:
            best_m = m_gauss
            best_h = h_gauss
    
    # Phase 4: Local refinement
    if best_h is not None and time_left() > 15:
        h_ref, m_ref = local_refine(best_h, max_iters=20000)
        if m_ref < best_m:
            best_m = m_ref
            best_h = h_ref
    
    # Phase 5: Final spectral polish if time remains
    if best_h is not None and time_left() > 10:
        h_more, m_more = spectral_optimize(best_h, max_iters=2000, flatten_power=0.95)
        if m_more < best_m:
            best_m = m_more
            best_h = h_more
    
    return list(best_h)

# EVOLVE-BLOCK-END
