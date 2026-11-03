"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    dx = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, N intervals

    def autoconv_max_ratio(h):
        """Compute max(autoconv) / ||h||_2^2 for step function h on N equal intervals."""
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        l2_sq = np.sum(h**2) * dx
        if l2_sq < 1e-15:
            return float('inf')
        
        # Use FFT for efficient autoconvolution
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        
        # The linear autoconvolution has support from -(N-1)*dx to (N-1)*dx
        # With circular conv of size 2N, indices 0..N-1 correspond to lags 0..(N-1)
        # and indices N..2N-1 correspond to lags -(N-1)..-1
        autoconv_vals = autoconv_circ * dx**2
        max_autoconv = np.max(autoconv_vals)
        
        return max_autoconv / l2_sq

    def fourier_to_heights(coeffs, base=0.0):
        """Convert Fourier coefficients (cosine and sine) to heights on the grid."""
        centers = np.arange(N) - N // 2
        x = centers * dx  # positions in [-1/4, 1/4]
        h = np.full(N, base)
        for k, (a, b) in enumerate(coeffs, start=1):
            freq = 2.0 * np.pi * k
            h += a * np.cos(freq * x) + b * np.sin(freq * x)
        return np.maximum(h, 0)

    def optimize_fourier_coeffs(init_coeffs, base, max_iter=5000):
        """Optimize Fourier coefficients using Nelder-Mead to minimize autoconv_max_ratio."""
        from scipy.optimize import minimize
        
        def objective(params):
            coeffs = list(zip(params[::2], params[1::2]))
            h = fourier_to_heights(coeffs, base)
            if np.sum(h) < 1e-10:
                return 1e10
            return autoconv_max_ratio(h)
        
        x0 = np.array(init_coeffs).flatten()
        
        # Nelder-Mead options
        options = {
            'maxiter': max_iter,
            'maxfev': max_iter,
            'xatol': 1e-4,
            'fatol': 1e-6,
            'adaptive': True
        }
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', options=options)
            if result.success or result.fun < 1e10:
                coeffs = list(zip(result.x[::2], result.x[1::2]))
                h = fourier_to_heights(coeffs, base)
                return h, result.fun
        except Exception:
            pass
        
        return None, float('inf')

    def coordinate_descent_refine(h, max_iters=200, step_decay=0.9):
        """Refine h using coordinate descent to minimize autoconv_max_ratio."""
        h = np.asarray(h, dtype=np.float64).copy()
        current_score = autoconv_max_ratio(h)
        step_size = 0.05
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try increasing and decreasing h[i]
                for direction in [1, -1]:
                    h_trial = h.copy()
                    h_trial[i] = max(0, h[i] + direction * step_size)
                    trial_score = autoconv_max_ratio(h_trial)
                    if trial_score < current_score:
                        h = h_trial
                        current_score = trial_score
                        improved = True
                        break
                if not improved and direction == -1:
                    step_size *= step_decay
            if not improved:
                break
        
        return h, current_score

    best_score = float('inf')
    best_h = None

    centers = np.arange(N) - N // 2
    x = centers * dx

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Gaussian peaks with various widths
    for sigma in [1, 2, 5, 10, 20, 50, 100]:
        h = np.exp(-centers**2 / (2 * sigma**2))
        candidates.append(h)
    
    # 2. Triangular peaks
    for width in [10, 50, 100, 200, 300]:
        h = np.maximum(0, 1 - np.abs(centers) / width)
        candidates.append(h)
    
    # 3. Box functions (constant on central region)
    for half_width in [10, 50, 100, 200, 300]:
        h = np.zeros(N)
        mask = np.abs(centers) <= half_width
        h[mask] = 1.0
        candidates.append(h)
    
    # 4. Gaussian + base
    for sigma in [5, 20, 50]:
        for base in [0.1, 0.3, 0.5]:
            h = base + (1 - base) * np.exp(-centers**2 / (2 * sigma**2))
            candidates.append(h)
    
    # 5. Inverted Gaussian (valley in center, high at edges)
    for sigma in [50, 100, 200]:
        h = 1.0 - np.exp(-centers**2 / (2 * sigma**2))
        h = np.maximum(h, 0)
        candidates.append(h)
    
    # 6. Random smooth functions
    np.random.seed(42)
    for trial in range(5):
        freqs = np.random.uniform(0.5, 5.0, size=5)
        phases = np.random.uniform(0, 2*np.pi, size=5)
        amps = np.random.uniform(0.5, 2.0, size=5)
        h = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            h += a * np.cos(f * centers / N * 2 * np.pi + p)
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-10:
            candidates.append(h)

    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        if np.sum(h) < 1e-10:
            continue
        score = autoconv_max_ratio(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Fourier-domain optimization: represent via low-frequency harmonics
    # Try different numbers of harmonics and base levels
    n_harmonics_options = [5, 10, 15, 20]
    base_options = [0.0, 0.05, 0.1, 0.2]
    
    for n_harm in n_harmonics_options:
        if time.time() > deadline:
            break
        for base in base_options:
            if time.time() > deadline:
                break
            
            # Initialize with Gaussian-like spectrum
            # Low frequencies have higher amplitude
            init_coeffs = []
            for k in range(1, n_harm + 1):
                # Amplitude decays with frequency
                amp = 1.0 / (1.0 + 0.5 * k)
                # Cosine term, sine term (start with zero sine for symmetry)
                init_coeffs.append([amp, 0.0])
            
            h_fourier, score_fourier = optimize_fourier_coeffs(init_coeffs, base, max_iter=2000)
            
            if h_fourier is not None and score_fourier < best_score:
                best_score = score_fourier
                best_h = h_fourier.copy()

    # Refine the best candidate with coordinate descent
    if best_h is not None and time.time() < deadline:
        refined_h, refined_score = coordinate_descent_refine(best_h, max_iters=300, step_decay=0.9)
        if refined_score < best_score:
            best_score = refined_score
            best_h = refined_h
    
    # Additional refinement passes with smaller steps
    if best_h is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            refined_h, refined_score = coordinate_descent_refine(best_h, max_iters=100, step_decay=0.9)
            if refined_score < best_score:
                best_score = refined_score
                best_h = refined_h
            else:
                break

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
