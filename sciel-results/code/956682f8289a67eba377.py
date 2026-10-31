<<<<<<< SEARCH
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s limit
    
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
            if time.time() > deadline:
                break
            
            # FFT
            H = np.fft.rfft(h, n=size)
            
            # Power spectrum
            power = np.abs(H) ** 2
            
            # Flatten: reduce peaks in power spectrum
            # Target: make power spectrum more uniform
            # Use a soft clipping: if power > threshold, reduce it
            mean_power = np.mean(power)
            # Only flatten the peaks (above some multiple of mean)
            threshold = mean_power * 1.5
            excess = np.maximum(power - threshold, 0)
            # Reduce the excess proportionally
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
            if time.time() > deadline:
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
    
    # Multiple initializations for spectral method
    i_arr = np.arange(N)
    mid = N / 2.0
    
    initializations = [
        np.ones(N),
        0.5 * (1.0 + np.cos(np.pi * (i_arr - mid) / (N / 2.0))),
        np.exp(-2.0 * ((i_arr - mid) / (N / 4.0)) ** 2),
        1.0 - 0.3 * np.cos(2 * np.pi * i_arr / N),
        np.cos(np.pi * (i_arr - mid) / N),
    ]
    
    best_h = None
    best_m = float('inf')
    
    # Phase 1: Spectral optimization from multiple starts
    for init in initializations:
        if time.time() > deadline:
            break
        init = np.maximum(init, 0.0)
        h_opt, m_opt = spectral_optimize(init, max_iters=3000)
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt
    
    # Phase 2: Local refinement
    if best_h is not None and time.time() < deadline - 10:
        h_ref, m_ref = local_refine(best_h, max_iters=30000)
        if m_ref < best_m:
            best_m = m_ref
            best_h = h_ref
    
    # Phase 3: If still have time, try more spectral iterations from best
    if best_h is not None and time.time() < deadline - 5:
        h_more, m_more = spectral_optimize(best_h, max_iters=5000, flatten_power=0.95)
        if m_more < best_m:
            best_m = m_more
            best_h = h_more
    
    return list(best_h)

# EVOLVE-BLOCK-END
=======
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)
    
    def cosine_series_to_heights(c, n_terms):
        """Convert cosine series coefficients to step function heights.
        h[k] = c[0] + sum_{m=1}^{n_terms} c[m] * cos(2*pi*m*k/N)
        """
        k = np.arange(N)
        h = np.full(N, c[0])
        for m in range(1, n_terms + 1):
            h += c[m] * np.cos(2.0 * np.pi * m * k / N)
        return h
    
    def compute_convolution_from_coeffs(c, n_terms):
        """Compute autoconvolution from cosine series coefficients.
        For a function with Fourier coefficients a_m, the autoconvolution
        at lag 0 involves sum of |a_m|^2, and at other lags involves products.
        We compute via FFT for accuracy.
        """
        h = cosine_series_to_heights(c, n_terms)
        h = np.maximum(h, 0.0)
        total = np.sum(h)
        if total == 0:
            return np.zeros(2*N), 0.0
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        return acorr, total
    
    def remez_optimize_cosine(n_terms, max_outer=50, max_inner=200, tol=1e-10):
        """Remez-style algorithm for minimizing max autoconvolution
        over cosine-series parameterized nonnegative functions.
        
        The key insight: if h[k] = c0 + sum c_m cos(2*pi*m*k/N),
        then the DFT of h is sparse (only at frequencies 0 and ±m),
        which makes the autoconvolution structure simple.
        
        We use an iterative equioscillation approach:
        1. Find current peak locations in autoconvolution
        2. Adjust coefficients to reduce peaks while maintaining nonnegativity
        """
        # Initialize with uniform (c0=1, all others 0)
        c = np.zeros(n_terms + 1)
        c[0] = 1.0
        best_c = c.copy()
        best_metric = float('inf')
        
        # Precompute basis functions
        k = np.arange(N)
        basis = np.zeros((n_terms + 1, N))
        basis[0] = 1.0
        for m in range(1, n_terms + 1):
            basis[m] = np.cos(2.0 * np.pi * m * k / N)
        
        # Precompute autoconvolution of each basis pair
        # acorr[m1, m2, lag] for the product basis[m1]*basis[m2]
        # This is expensive, so we compute on-the-fly via FFT
        
        for outer in range(max_outer):
            if time.time() > deadline - 5:
                break
            
            h = np.maximum(basis @ c, 0.0)
            total = np.sum(h)
            if total < 1e-15:
                break
            
            size = 2 * N
            H = np.fft.rfft(h, n=size)
            acorr = np.fft.irfft(H * np.conj(H), n=size)
            peak = np.max(acorr)
            metric = peak / (total * total)
            
            if metric < best_metric:
                best_metric = metric
                best_c = c.copy()
            
            # Find top peak locations (for equioscillation)
            # We need at least n_terms+1 alternating extrema for Remez
            # Use top peaks
            n_peaks = min(n_terms + 2, size)
            peak_idx = np.argsort(acorr)[-n_peaks:]
            peak_vals = acorr[peak_idx]
            peak_signs = np.ones(n_peaks)  # all positive for max
            
            # Compute gradient of metric w.r.t. coefficients
            # d(acorr)/dc_j involves the convolution of h with basis[j]
            # and the derivative of the normalization
            
            # Simpler approach: use finite differences for gradient
            # But that's expensive. Instead, use a steepest-descent-like update.
            
            # For each coefficient c_j, compute how changing it affects the peak
            # d(h)/dc_j = basis[j]
            # d(acorr)/dc_j = h * basis[j] (convolution) + basis[j] * h (convolution)
            #                 = 2 * fft(h) * fft(basis[j]) (in frequency domain)
            
            # Compute the response of each peak to each coefficient
            # response[p, j] = d(acorr[peak_p])/dc_j
            H_basis = np.fft.rfft(basis.T, n=size)  # (n_terms+1, size//2+1)
            H_h = H  # (size//2+1,)
            
            # acorr = ifft(H * conj(H))
            # d(acorr)/dc_j = ifft(2 * H * conj(dH/dc_j)) where dH/dc_j = fft(basis[j])
            # But h = max(basis@c, 0), so dH/dc_j is more complex at kinks.
            # For simplicity, ignore the max() nonlinearity (assume h > 0 everywhere)
            
            # d(acorr)/dc_j = ifft(2 * H * conj(H_basis[j]))
            # But we only need it at peak locations.
            
            # Compute response matrix
            # response[p, j] = real(ifft(2 * H * conj(H_basis[j]))[peak_idx[p]])
            
            # Efficient: compute ifft for each j
            responses = np.zeros((n_peaks, n_terms + 1))
            for j in range(n_terms + 1):
                d_acorr = np.fft.irfft(2.0 * H * np.conj(H_basis[j]), n=size)
                responses[:, j] = d_acorr[peak_idx]
            
            # We want to minimize max(acorr) / total^2
            # Use a weighted sum of peaks as surrogate
            # Or: minimize the maximum by making all top peaks equal (equioscillation)
            
            # Simple approach: move in direction that reduces the max peak
            # while trying to equalize peaks
            # d_metric ~ (1/total^2) * sum_p w_p * d(acorr_p)/dc
            # where w_p emphasizes the largest peaks
            
            # Weight by peak value (emphasize larger peaks)
            weights = peak_vals / np.sum(peak_vals) if np.sum(peak_vals) > 0 else np.ones(n_peaks)/n_peaks
            
            # Gradient of weighted peak sum w.r.t. c
            grad = responses.T @ weights  # (n_terms+1,)
            
            # Also account for the normalization: d(1/total^2)/dc_j = -2/total^3 * d(total)/dc_j
            # d(total)/dc_j = sum(basis[j]) (for positive part)
            # For uniform c, sum(basis[j]) = 0 for j>0, = N for j=0
            d_total_dc = np.zeros(n_terms + 1)
            for j in range(n_terms + 1):
                d_total_dc[j] = np.sum(np.maximum(basis[j], 0.0))  # approximate
            
            # Full gradient of metric
            metric_grad = grad / (total * total) - 2.0 * metric * d_total_dc / total
            
            # Line search
            step_size = 0.1
            for ls in range(20):
                c_new = c - step_size * metric_grad
                h_new = np.maximum(basis @ c_new, 0.0)
                total_new = np.sum(h_new)
                if total_new < 1e-15:
                    step_size *= 0.5
                    continue
                H_new = np.fft.rfft(h_new, n=size)
                acorr_new = np.fft.irfft(H_new * np.conj(H_new), n=size)
                metric_new = np.max(acorr_new) / (total_new * total_new)
                
                if metric_new < metric:
                    c = c_new
                    break
                else:
                    step_size *= 0.5
            
            if step_size < 1e-15:
                break
        
        return best_c, best_metric
    
    def full_optimize_from_cosine(n_terms):
        """Optimize using cosine series parameterization."""
        c_opt, m_opt = remez_optimize_cosine(n_terms, max_outer=100, max_inner=100)
        h_opt = np.maximum(cosine_series_to_heights(c_opt, n_terms), 0.0)
        return h_opt, m_opt
    
    def spectral_optimize(h_init, max_iters=3000, flatten_power=0.99):
        """Iterative projection: flatten power spectrum + enforce nonnegativity."""
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        
        size = 2 * N
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            H = np.fft.rfft(h, n=size)
            power = np.abs(H) ** 2
            mean_power = np.mean(power)
            threshold = mean_power * 1.5
            excess = np.maximum(power - threshold, 0)
            power_new = power - flatten_power * excess
            power_new = np.maximum(power_new, 0)
            phase = np.angle(H)
            H_new = np.sqrt(power_new) * np.exp(1j * phase)
            h_new = np.fft.irfft(H_new, n=size)[:N]
            h_new = np.maximum(h_new, 0.0)
            
            if np.all(h_new == 0):
                h_new = h.copy()
            
            m_new = compute_metric(h_new)
            
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            
            h = h_new
        
        return best_h, best_m
    
    def local_refine(h_init, max_iters=20000):
        """Random perturbation local search to refine the result."""
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        
        np.random.seed(42)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            frac = np.random.uniform(0.01, 0.05)
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
    
    best_h = None
    best_m = float('inf')
    
    # Phase 1: Cosine series Remez optimization with increasing order
    for n_terms in [4, 8, 16, 24]:
        if time.time() > deadline - 10:
            break
        h_opt, m_opt = full_optimize_from_cosine(n_terms)
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt.copy()
    
    # Phase 2: Spectral refinement from best cosine solution
    if best_h is not None and time.time() < deadline - 10:
        h_spec, m_spec = spectral_optimize(best_h, max_iters=3000, flatten_power=0.99)
        if m_spec < best_m:
            best_m = m_spec
            best_h = h_spec
    
    # Phase 3: Local refinement
    if best_h is not None and time.time() < deadline - 5:
        h_ref, m_ref = local_refine(best_h, max_iters=20000)
        if m_ref < best_m:
            best_m = m_ref
            best_h = h_ref
    
    # Phase 4: Final spectral polish
    if best_h is not None and time.time() < deadline - 2:
        h_polish, m_polish = spectral_optimize(best_h, max_iters=2000, flatten_power=0.95)
        if m_polish < best_m:
            best_m = m_polish
            best_h = h_polish
    
    if best_h is None:
        best_h = np.ones(N)
    
    return list(best_h)

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
