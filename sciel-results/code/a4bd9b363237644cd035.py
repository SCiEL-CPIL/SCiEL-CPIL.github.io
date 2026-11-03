import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.optimize import linprog

def _autoconv(f):
    """Compute autoconvolution (convolution f*f) using FFT."""
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    conv_full = np.fft.ifft(F_full * F_full)[:2*n-1].real
    return conv_full

def _compute_score(f):
    """Compute the normalized max of autoconvolution."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    conv = _autoconv(f)
    max_conv = np.max(conv)
    sum_f = np.sum(f)
    norm = sum_f * sum_f
    if norm == 0:
        return 0.0
    return max_conv / norm

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Initialize with a smooth bump function
    x = np.linspace(-0.25, 0.25, n, endpoint=False)
    # Gaussian-like bump
    f = np.exp(-x**2 / (2 * 0.08**2))
    # Normalize
    f = np.maximum(f, 0)
    
    def time_remaining():
        return deadline - time.time()
    
    # Phase 1: LP-based approach using Radon-Haar wavelet coefficients
    # We'll use a simpler approach: direct optimization in time domain with 
    # smooth max surrogate and then refine
    
    # First, let's try a cosine basis parameterization for smoothness
    # f(x) = sum_k a_k * cos(2*pi*k*x/L) for k=0..K, where L=0.5
    # This ensures smoothness and we can enforce non-negativity
    
    K_max = 30  # number of cosine basis functions
    
    def cosine_basis(x, K):
        """Compute cosine basis values at points x."""
        L = 0.5
        k = np.arange(K + 1)[:, None]
        m = np.arange(len(x))[None, :]
        # cos(2*pi*k*m/n) for k=0..K
        basis = np.cos(2 * np.pi * k * m / n)
        # Scale DC term
        basis[0, :] = 1.0
        return basis
    
    def params_to_f(a, K):
        """Convert cosine coefficients to step function values."""
        B = cosine_basis(x, K)
        f = B.T @ a
        return np.maximum(f, 0.0)
    
    def smooth_max(conv, temperature=0.01):
        """Soft maximum approximation."""
        # Use log-sum-exp for numerical stability
        max_val = np.max(conv)
        return max_val + temperature * np.log(np.sum(np.exp((conv - max_val) / temperature)))
    
    def objective_from_params(a, K, temperature=0.01):
        """Objective: normalized smooth max of autoconvolution."""
        f = params_to_f(a, K)
        if np.sum(f) < 1e-10:
            return 1.0
        conv = _autoconv(f)
        smax = smooth_max(conv, temperature)
        return smax / (np.sum(f) ** 2)
    
    # Start with initial parameters
    # Initialize with a low-frequency dominant spectrum
    a = np.zeros(K_max + 1)
    a[0] = 1.0  # DC component
    # Add some low frequency components
    for k in range(1, min(10, K_max + 1)):
        a[k] = 0.1 * np.exp(-k / 5)
    
    best_score = _compute_score(params_to_f(a, K_max))
    best_a = a.copy()
    
    # Coordinate descent on cosine coefficients
    rng = np.random.default_rng(42)
    
    max_passes = 30
    for pass_num in range(max_passes):
        if time_remaining() < 15.0:
            break
        
        improved = False
        
        # Coordinate descent on each cosine coefficient
        for i in range(K_max + 1):
            if time_remaining() < 15.0:
                break
            
            orig_val = best_a[i]
            scale = max(abs(orig_val), 0.01)
            
            def obj_1d(val):
                test_a = best_a.copy()
                test_a[i] = val
                return objective_from_params(test_a, K_max, temperature=0.005)
            
            res = minimize_scalar(
                obj_1d,
                bounds=(orig_val - 3*scale, orig_val + 3*scale),
                method='bounded',
                options={'xatol': 1e-5, 'maxiter': 15}
            )
            
            if res.fun < best_score - 1e-8:
                best_a[i] = res.x
                best_score = res.fun
                improved = True
        
        if not improved:
            # Try random restarts in parameter space
            escaped = False
            for _ in range(10):
                if time_remaining() < 15.0:
                    break
                test_a = best_a + rng.normal(0, 0.02, K_max + 1)
                test_score = objective_from_params(test_a, K_max, temperature=0.005)
                if test_score < best_score:
                    best_a = test_a
                    best_score = test_score
                    escaped = True
            if not escaped:
                break
    
    # Convert to time-domain step function
    f = params_to_f(best_a, K_max)
    
    # Phase 2: Refine in time domain using coordinate descent
    # With a smooth max surrogate to handle the nonsmooth max
    
    current_score = _compute_score(f)
    
    if time_remaining() > 20.0:
        # Time-domain coordinate descent with smooth max
        temperature = 0.002
        
        def time_obj(f_vec, temperature=temperature):
            if np.sum(f_vec) < 1e-10:
                return 1.0
            conv = _autoconv(f_vec)
            smax = smooth_max(conv, temperature)
            return smax / (np.sum(f_vec) ** 2)
        
        max_td_passes = 20
        for td_pass in range(max_td_passes):
            if time_remaining() < 15.0:
                break
            
            improved = False
            
            # Random subset of coordinates for speed
            idx = rng.choice(n, size=min(n, 200), replace=False)
            idx = np.sort(idx)
            
            for i in idx:
                if time_remaining() < 15.0:
                    break
                
                orig_val = f[i]
                scale = max(orig_val, 0.01)
                
                def obj_td(val):
                    test_f = f.copy()
                    test_f[i] = max(val, 0.0)
                    return time_obj(test_f, temperature)
                
                res = minimize_scalar(
                    obj_td,
                    bounds=(0.0, orig_val + 2*scale),
                    method='bounded',
                    options={'xatol': 1e-4, 'maxiter': 10}
                )
                
                if res.fun < current_score - 1e-8:
                    f[i] = max(res.x, 0.0)
                    current_score = res.fun
                    improved = True
            
            if not improved:
                # Try a few random perturbations
                escaped = False
                for _ in range(5):
                    if time_remaining() < 15.0:
                        break
                    test_f = f + rng.normal(0, 0.005, n)
                    test_f = np.maximum(test_f, 0.0)
                    test_score = time_obj(test_f, temperature)
                    if test_score < current_score:
                        f = test_f
                        current_score = test_score
                        escaped = True
                if not escaped:
                    break
    
    # Final: compute exact score and return
    f = np.maximum(f, 0.0)
    return list(f)
