import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    # Initialize with the seed
    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    rng = np.random.default_rng(123)
    
    # Strategy: Optimize in the frequency domain or use structured signals
    # The value is V = 2N * max(corr) / (sum h)^2
    # Let h = a * u, where sum(u) = 1. Then V = 2N * max(conv(u,u))
    # We want to minimize max(conv(u,u)) subject to sum(u)=1.
    # Note that conv(u,u)[k] = sum_i u[i] u[i+k].
    # max(conv(u,u)) >= (sum u)^2 / N = 1/N. So V >= 2.
    # To get V < 2, we need max(conv(u,u)) < 1/N.
    # This is possible if the "energy" of u is spread out such that the peak
    # of the autoconvolution is below the average.
    
    # Candidate families:
    candidates = []
    
    # 1. Cosine window (tapered ends)
    # h[i] = cos(pi * i / (N-1))
    # This reduces the edge effects which often dominate the max of the convolution
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        t = np.linspace(0, 1, N)
        h = np.cos(np.pi * alpha * t)
        # Ensure non-negative? No, real numbers allowed.
        # But if we use cos, it becomes negative. 
        # The problem allows negative heights.
        # However, if h has negative parts, sum(h) might be small or zero.
        # Let's try h = 1 + eps * (cos - mean)
        mean_h = np.mean(h)
        h_centered = h - mean_h
        for eps in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]:
            h_test = 1.0 + eps * h_centered
            if np.sum(h_test) == 0:
                continue
            val = compute_value(h_test)
            if val < val_best:
                val_best = val
                h_best = h_test.copy()
            candidates.append((val, h_test.copy()))

    # 2. Sinc-like functions (bandlimited)
    # h[i] = sin(2*pi*f*i) / (2*pi*f*i)
    # These have very flat spectra, potentially leading to lower peak correlation?
    # Actually, sinc has high sidelobes in time, but in frequency it's a box.
    # Let's try low frequency components.
    for f in [0.1, 0.2, 0.5, 1.0, 2.0]:
        t = np.arange(N)
        # Avoid division by zero at i=0
        h = np.sinc(f * t / N) # np.sinc(x) = sin(pi x)/(pi x)
        # np.sinc(0) = 1
        mean_h = np.mean(h)
        h_centered = h - mean_h
        for eps in [0.1, 0.3, 0.5, 0.7, 0.9]:
            h_test = 1.0 + eps * h_centered
            if np.sum(h_test) == 0:
                continue
            val = compute_value(h_test)
            if val < val_best:
                val_best = val
                h_best = h_test.copy()
            candidates.append((val, h_test.copy()))

    # 3. Random Fourier Series
    # h(t) = 1 + sum_{k=1}^K a_k cos(2*pi*k*t/N + phi_k)
    # This allows us to shape the spectrum directly.
    K = 10
    for trial in range(200):
        if time.time() > deadline:
            break
        # Random amplitudes and phases
        amps = rng.normal(0, 0.1, K)
        phases = rng.uniform(0, 2*np.pi, K)
        h_test = np.ones(N)
        t = np.arange(N)
        for k in range(1, K+1):
            h_test += amps[k-1] * np.cos(2*np.pi*k*t/N + phases[k-1])
        
        # Normalize to keep sum close to N? 
        # The value is scale invariant in h? 
        # V = 2N * max(conv(h,h)) / (sum h)^2
        # If h -> c*h, conv -> c^2 conv, sum -> c sum.
        # V -> 2N * c^2 max / (c^2 sum^2) = V.
        # So scaling doesn't matter.
        # But we must ensure sum(h) != 0.
        if abs(np.sum(h_test)) < 1e-6:
            continue
            
        val = compute_value(h_test)
        if val < val_best:
            val_best = val
            h_best = h_test.copy()
        candidates.append((val, h_test.copy()))
        
        # Local refinement around this candidate
        if val < 2.0: # Only refine if we are already beating the seed
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Perturb amplitudes
                new_amps = amps + rng.normal(0, 0.01, K)
                new_h = np.ones(N)
                for k in range(1, K+1):
                    new_h += new_amps[k-1] * np.cos(2*np.pi*k*t/N + phases[k-1])
                if abs(np.sum(new_h)) < 1e-6:
                    continue
                new_val = compute_value(new_h)
                if new_val < val:
                    val = new_val
                    amps = new_amps
                    h_test = new_h
                    if val < val_best:
                        val_best = val
                        h_best = h_test.copy()
                    candidates.append((val, h_test.copy()))

    # 4. Gradient-free optimization (Nelder-Mead) on a small set of parameters
    # Define a parameterization h(params)
    # Use a polynomial or trigonometric basis
    # Let's try a polynomial basis h[i] = p(i/N)
    # Degree 4 polynomial: h(x) = 1 + a*x + b*x^2 + c*x^3 + d*x^4
    
    def make_h_poly(params):
        a, b, c, d = params
        x = np.linspace(0, 1, N)
        return 1.0 + a*x + b*x**2 + c*x**3 + d*x**4

    # Initial guess: symmetric bump
    # h(x) = 1 + a*(x-0.5)^2
    # Let's start with a few random polynomial candidates
    for trial in range(100):
        if time.time() > deadline:
            break
        params = rng.normal(0, 0.5, 4)
        h_test = make_h_poly(params)
        if abs(np.sum(h_test)) < 1e-6:
            continue
        val = compute_value(h_test)
        if val < val_best:
            val_best = val
            h_best = h_test.copy()
        candidates.append((val, h_test.copy()))
        
        # Quick local search
        for _ in range(20):
            if time.time() > deadline:
                break
            new_params = params + rng.normal(0, 0.05, 4)
            new_h = make_h_poly(new_params)
            if abs(np.sum(new_h)) < 1e-6:
                continue
            new_val = compute_value(new_h)
            if new_val < val:
                val = new_val
                params = new_params
                h_test = new_h
                if val < val_best:
                    val_best = val
                    h_best = h_test.copy()
                candidates.append((val, h_test.copy()))

    return h_best
