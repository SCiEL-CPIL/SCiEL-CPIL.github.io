import numpy as np
import time

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def time_remaining():
        return deadline - time.time()

    # Precompute FFT indices for efficiency
    freq_idx = np.fft.fftfreq(n, d=1.0/n)
    
    # Initialize with a smooth bump in time domain, then transform to frequency
    x = np.linspace(-0.25, 0.25, n, endpoint=False)
    f_init = np.exp(-x**2 / (2 * 0.08**2))
    F = np.fft.fft(f_init)
    
    # We optimize over the real and imaginary parts of F separately
    # But to maintain Hermitian symmetry for real f, we can optimize over
    # the first n//2 + 1 complex coefficients and reconstruct, OR
    # simpler: optimize over full complex F but project to real f at each step.
    # Actually, for real f, F must satisfy F[k] = conj(F[n-k]).
    # To keep it simple and robust, we optimize over the full complex F
    # but enforce realness by projecting f = ifft(F) to real and non-negative.
    
    # Better approach: parameterize by the magnitude and phase? 
    # No, let's stick to optimizing the complex coefficients directly 
    # but use the structure: we only need to optimize the independent coefficients.
    # However, for simplicity and speed, let's optimize the full vector F 
    # and in the projection step, take f = np.maximum(np.real(np.fft.ifft(F)), 0)
    # and then recompute F = np.fft.fft(f). This ensures consistency.
    
    # Objective: minimize max(autoconv(f)) / (sum(f))^2
    # Since sum(f) is scale invariant in the ratio, we can fix sum(f) or just
    # compute the ratio. The ratio is invariant to scaling f by a constant c:
    # max(c*f * c*f) / (sum(c*f))^2 = c^2 max(f*f) / (c sum(f))^2 = max(f*f)/(sum(f))^2.
    # So we can fix sum(f) = 1 to simplify, or just use the ratio.
    # Using the ratio directly is safer.
    
    def compute_score_from_F(F):
        f = np.fft.ifft(F).real
        f = np.maximum(f, 0.0)
        if np.sum(f) < 1e-10:
            return 1.0, f
        # Autoconvolution via FFT
        F_f = np.fft.fft(f, n=2*n)
        conv = np.fft.ifft(F_f * F_f)[:2*n-1].real
        max_conv = np.max(conv)
        sum_f = np.sum(f)
        score = max_conv / (sum_f * sum_f)
        return score, f

    def smooth_max(conv, temperature=0.005):
        max_val = np.max(conv)
        # Use log-sum-exp for stability
        shifted = (conv - max_val) / temperature
        # Clip to avoid overflow in exp
        shifted = np.clip(shifted, -100, 100)
        return max_val + temperature * np.log(np.sum(np.exp(shifted)))

    def objective_from_F(F, temperature=0.005):
        f = np.fft.ifft(F).real
        f = np.maximum(f, 0.0)
        if np.sum(f) < 1e-10:
            return 1.0
        F_f = np.fft.fft(f, n=2*n)
        conv = np.fft.ifft(F_f * F_f)[:2*n-1].real
        smax = smooth_max(conv, temperature)
        sum_f = np.sum(f)
        return smax / (sum_f * sum_f)

    # Projected Gradient Descent on F
    # Step size needs to be tuned. Let's use a backtracking line search or fixed step.
    # Given the nonlinearity, a fixed step with decay might work.
    
    best_F = F.copy()
    best_score, best_f = compute_score_from_F(best_F)
    
    rng = np.random.default_rng(42)
    
    # Gradient estimation: we need the gradient of the objective w.r.t. F.
    # This is complex. Let's use a simpler approach: 
    # Since we are projecting to real non-negative f at each step, 
    # we can use a subgradient or finite difference.
    # Alternatively, we can optimize in the time domain but using spectral smoothing.
    
    # Let's try a different structural approach: 
    # Optimize the power spectrum |F|^2 directly? 
    # No, the phase matters for the peak.
    
    # Let's use a simpler heuristic: 
    # 1. Start with a good initial F.
    # 2. Perform gradient descent in the time domain f, but use a 
    #    "spectral filter" to smooth updates, ensuring we stay in a 
    #    low-dimensional manifold of smooth functions.
    
    # Actually, the directive suggests "projected gradient descent on the frequency domain".
    # Let's implement a finite-difference gradient for F.
    # This is expensive (2n evaluations per step). 
    # n=600, so 1200 evaluations. Each evaluation is O(n log n). 
    # 1200 * 600 * log(600) ~ 1200 * 600 * 9.4 ~ 6.7 million ops. 
    # This might be too slow for many iterations.
    
    # Alternative: Use the fact that the objective is differentiable 
    # if we ignore the non-negativity constraint (smooth max).
    # The gradient of smooth_max(conv) w.r.t. f can be computed.
    # Then chain rule to F.
    
    # Let's try a simpler, faster approach: 
    # Coordinate descent on the frequency coefficients, but with a 
    # smarter initialization and update rule.
    
    # Wait, the inspiration 1 (score 1.2488) used spectral-domain optimization.
    # Let's mimic that: optimize the real and imaginary parts of the 
    # independent frequency coefficients.
    
    # Independent coefficients: k=0 to n//2.
    # For k=0 and k=n//2 (if n even), they are real.
    # For others, complex.
    
    # Let's define a function to pack/unpack the independent coefficients.
    def pack_F(F):
        # F is length n. 
        # We take F[0] (real), F[1:n//2] (complex), F[n//2] (real if n even)
        # n=600 is even.
        # F[0] is real.
        # F[1:300] are complex (600 real dims).
        # F[300] is real.
        # F[301:600] are conjugates of F[299:0:-1].
        real_parts = np.concatenate([
            [F[0].real],
            F[1:n//2].real,
            F[1:n//2].imag,
            [F[n//2].real]
        ])
        return real_parts

    def unpack_F(params):
        F = np.zeros(n, dtype=np.complex128)
        F[0] = params[0]
        # params[1:301] are real parts of F[1:301]
        # params[301:601] are imag parts of F[1:301]
        real_part = params[1:1 + (n//2 - 1)]
        imag_part = params[1 + (n//2 - 1): 2 + 2*(n//2 - 1)]
        F[1:n//2] = real_part + 1j * imag_part
        F[n//2] = params[2 + 2*(n//2 - 1)]
        # Fill in the conjugates
        for k in range(1, n//2):
            F[n-k] = np.conj(F[k])
        return F

    # Initial parameters
    p = pack_F(best_F)
    
    # Objective in terms of p
    def obj_p(p, temperature=0.005):
        F = unpack_F(p)
        return objective_from_F(F, temperature)

    # Gradient via finite differences (central difference)
    # This is expensive. Let's use forward difference with a small step.
    # Or better, use analytic gradient if possible.
    
    # Let's try a different strategy: 
    # Use scipy.optimize.minimize with method='L-BFGS-B' on the packed parameters.
    # This uses gradient estimation internally, which might be faster 
    # if we provide a good gradient.
    
    # But we don't have an analytic gradient. 
    # Let's use 'Nelder-Mead' or 'Powell' which don't require gradients.
    # Powell is good for high dimensions but might be slow.
    
    # Let's try 'L-BFGS-B' with finite difference gradients.
    # The dimension is ~600. L-BFGS-B might struggle.
    
    # Let's go back to coordinate descent on the frequency coefficients, 
    # but use a better initialization and step size.
    
    # Initialize with a Gaussian in frequency domain?
    # A Gaussian in time is a Gaussian in frequency.
    # Let's try a flat spectrum (white noise) but with low-pass filter.
    
    # Let's use the current best_F as start.
    
    # Coordinate descent on p
    dim = len(p)
    best_p = p.copy()
    best_score_p = obj_p(best_p)
    
    max_passes = 20
    for pass_num in range(max_passes):
        if time_remaining() < 20.0:
            break
        
        improved = False
        
        # Randomize the order of coordinates
        order = rng.permutation(dim)
        
        for i in order:
            if time_remaining() < 20.0:
                break
            
            orig_val = best_p[i]
            scale = max(abs(orig_val), 0.01)
            
            def obj_1d(val):
                test_p = best_p.copy()
                test_p[i] = val
                return obj_p(test_p, temperature=0.005)
            
            # Use minimize_scalar
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(
                obj_1d,
                bounds=(orig_val - 5*scale, orig_val + 5*scale),
                method='bounded',
                options={'xatol': 1e-4, 'maxiter': 10}
            )
            
            if res.fun < best_score_p - 1e-8:
                best_p[i] = res.x
                best_score_p = res.fun
                improved = True
        
        if not improved:
            break
    
    # Convert back to f
    best_F_final = unpack_F(best_p)
    f_final = np.fft.ifft(best_F_final).real
    f_final = np.maximum(f_final, 0.0)
    
    # Final refinement in time domain with a few steps
    if time_remaining() > 10.0:
        # Simple coordinate descent on f
        temperature = 0.002
        current_score = compute_score_from_F(best_F_final)[0]
        
        for _ in range(5):
            if time_remaining() < 10.0:
                break
            improved = False
            idx = rng.choice(n, size=100, replace=False)
            for i in idx:
                if time_remaining() < 10.0:
                    break
                orig_val = f_final[i]
                scale = max(orig_val, 0.01)
                
                def obj_td(val):
                    test_f = f_final.copy()
                    test_f[i] = max(val, 0.0)
                    F_test = np.fft.fft(test_f, n=2*n)
                    conv = np.fft.ifft(F_test * F_test)[:2*n-1].real
                    smax = smooth_max(conv, temperature)
                    sum_f = np.sum(test_f)
                    if sum_f < 1e-10:
                        return 1.0
                    return smax / (sum_f * sum_f)
                
                res = minimize_scalar(
                    obj_td,
                    bounds=(0.0, orig_val + 2*scale),
                    method='bounded',
                    options={'xatol': 1e-4, 'maxiter': 10}
                )
                
                if res.fun < current_score - 1e-8:
                    f_final[i] = max(res.x, 0.0)
                    current_score = res.fun
                    improved = True
            if not improved:
                break
    
    f_final = np.maximum(f_final, 0.0)
    return list(f_final)
