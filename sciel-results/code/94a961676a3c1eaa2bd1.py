import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 is defined here as the maximum absolute value of the autoconvolution.
    """
    # signed_step is an array of N values, typically in {-1, 1} or continuous
    # The function f(x) = signed_step[i] for x in [-1/4 + i/N, -1/4 + (i+1)/N)
    
    # For efficiency, we compute the autoconvolution via FFT.
    # The autoconvolution (f * f)(t) = integral f(s) f(t-s) ds.
    # Since f is a step function, we can approximate the integral by summing over the intervals.
    # Let h = 1/N be the interval width.
    # f(s) is constant on each interval.
    # The convolution of two step functions is a piecewise linear function.
    # The peak usually occurs at t=0 for symmetric functions, but we need max over all t.
    
    # Using FFT for discrete convolution:
    # We treat the step heights as a discrete sequence.
    # The discrete autoconvolution c[k] = sum_j f[j] f[j+k] * h^2 ? 
    # Actually, let's stick to the numerical integration approach which is robust.
    
    # A faster way:
    # The autoconvolution of a step function can be computed exactly if we consider the overlap.
    # However, FFT is O(N log N).
    
    # Let's use the FFT method on the sampled values.
    # We need to be careful with the scaling.
    # f(s) ~ signed_step[i] for s in [i*h, (i+1)*h).
    # (f*f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds.
    
    # If we sample f at N points, the discrete convolution approximates the integral by h * sum.
    
    # To get the true maximum, we might need to check the vertices of the piecewise linear convolution.
    # But for optimization purposes, the discrete max is a good proxy, and we can refine at the end.
    
    # For the continuous optimization, we need a differentiable proxy.
    # max |autoconv| is not differentiable.
    # We can use sum(autoconv^2) or max(autoconv^2) with smoothing, or just optimize for the L2 norm of the autoconvolution to reduce peaks?
    # No, we want to minimize the peak.
    # A common trick: minimize the L2 norm of the autoconvolution, which tends to reduce peaks, or use a soft-max approximation.
    
    # However, the task asks for a specific metric. "Paper and published result notebook place the absolute value differently".
    # The seed uses np.ones(N).
    # If f=1, autoconv is a triangle with peak 0.25 (area 0.5 * 0.5 = 0.25? No, integral of 1*1 over overlap. Max overlap is 0.5. So peak is 0.5? 
    # Domain [-0.25, 0.25]. Length 0.5.
    # f=1. (f*f)(0) = integral_{-0.25}^{0.25} 1 * 1 ds = 0.5.
    # So C3(seed) = 0.5.
    
    # If we flip signs, we can make the autoconvolution smaller.
    # E.g., alternating 1, -1, 1, -1...
    # Then f(s) f(t-s) will have cancellations.
    
    # Let's compute C3 for a given signed_step array.
    # We will use a fine grid to be accurate.
    
    M = 4 * N
    dx = 0.5 / M
    
    f_fine = np.zeros(M)
    # Map N steps to M fine steps
    # Each step covers M/N fine steps
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    # Autoconvolution via FFT
    # We need linear convolution, so pad to 2*M
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-0.5, 0.5] roughly.
    # We want max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _compute_c3_continuous(x):
    """
    Compute a differentiable proxy for C3 for continuous optimization.
    x is a vector of real-valued heights.
    We use the L2 norm of the autoconvolution as a proxy, or a smoothed max.
    Actually, minimizing the L2 norm of the autoconvolution is equivalent to minimizing the energy,
    which by Parseval's theorem is related to the sum of squares of the FFT coefficients.
    
    Let's try minimizing the L2 norm of the autoconvolution first, as it is smooth and easy to compute.
    If that doesn't work well, we can switch to a differentiable max approximation.
    
    Wait, the goal is to minimize the *peak*. Minimizing L2 might not minimize the peak if the peak is narrow.
    However, for step functions, the autoconvolution is piecewise linear.
    
    Let's use the discrete autoconvolution energy.
    """
    # x is continuous heights
    # Compute FFT of x
    # The L2 norm of the autoconvolution in time domain is equal to (1/2pi) * integral |F(f)|^4 ? No.
    # (f*f) in time is F(f)^2 in freq.
    # L2 norm of (f*f) is L2 norm of F(f)^2.
    
    # Let's just compute the discrete autoconvolution and its L2 norm.
    # This is differentiable with respect to x.
    
    # Use FFT for speed
    fft_x = np.fft.fft(x)
    # Autoconvolution in freq domain is fft_x * conj(fft_x) ? No, that's power spectrum.
    # Convolution theorem: FFT(f*g) = FFT(f) * FFT(g).
    # So FFT(f*f) = FFT(f) * FFT(f).
    # Wait, autoconvolution is usually defined with conjugate?
    # (f*f)(t) = int f(s) f(t-s) ds.
    # If f is real, FFT(f*f) = FFT(f) * FFT(f).
    # If it were correlation, it would be FFT(f) * conj(FFT(f)).
    # The task says "autoconvolution". Usually in signal processing, autocorrelation uses conjugate.
    # "Autoconvolution" strictly means convolution with itself.
    # For real signals, convolution with itself is not the same as autocorrelation unless the signal is even.
    # Given "signed step function", it's likely convolution.
    
    # Let's assume standard convolution: c = f * f.
    # In freq domain: C = F * F.
    
    # We want to minimize max |c(t)|.
    # Proxy: minimize sum(c(t)^2) which is ||c||_2^2.
    
    # ||c||_2^2 = (1/2pi) * integral |C(w)|^2 dw ?
    # By Parseval, sum |c[n]|^2 = (1/N) sum |C[k]|^2 (with proper scaling).
    
    # Let's just compute c in time domain via IFFT and sum squares.
    # c = IFFT(F * F)
    # Note: We need to handle the scaling correctly for the integral approximation.
    
    # For the optimizer, we just need the gradient direction, so scaling doesn't matter for the minimization path,
    # but we want to ensure we are minimizing the right thing.
    
    # Let's compute the discrete autoconvolution.
    # To avoid O(N^2), use FFT.
    n_fft = 2 * len(x)
    F = np.fft.fft(x, n=n_fft)
    C = np.fft.ifft(F * F)
    
    # C is the discrete convolution.
    # We want to minimize the peak.
    # Using L2 norm as a proxy:
    return np.sum(np.abs(C)**2)

def construct():
    start_time = time.time()
    
    # Initialize with random continuous values? Or start from seed (all ones) and perturb?
    # Starting from all ones (the seed) is a good baseline.
    # But we want to explore the space.
    # Let's start with a random signed step function, then optimize continuously.
    
    # Strategy:
    # 1. Generate a few random initial points (signed steps).
    # 2. For each, run L-BFGS-B on the continuous relaxation.
    # 3. Discretize the result (round to -1/1).
    # 4. Evaluate the true discrete C3.
    # 5. Keep the best.
    
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    np.random.seed(42)
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10.0:
            break
        
        # Initialize with random signs
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        # Run continuous optimization
        # We minimize the L2 norm of the autoconvolution.
        # This is a proxy for reducing the peak.
        
        try:
            result = minimize(_compute_c3_continuous, x0, method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4})
            
            # Discretize the result
            # Round to nearest sign
            x_opt = result.x
            step_candidate = np.sign(x_opt).astype(np.int8)
            # Handle zeros (should be rare with L-BFGS-B starting from +-1)
            step_candidate[step_candidate == 0] = 1
            
            # Evaluate true C3
            c3_candidate = _compute_c3(step_candidate)
            
            if c3_candidate < best_c3:
                best_c3 = c3_candidate
                best_step = step_candidate
                
        except Exception as e:
            pass
            
    # Final local refinement on the best discrete solution
    # Use simulated annealing or simple hill climbing to fine-tune
    # Since we have a good continuous optimum, the discrete version should be close.
    # Let's do a quick local search.
    
    current_step = best_step.copy()
    current_c3 = best_c3
    no_improve = 0
    patience = 2000
    
    while time.time() - start_time < DEADLINE - 5.0 and no_improve < patience:
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3(current_step)
        
        if new_c3 < current_c3:
            current_c3 = new_c3
            no_improve = 0
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            current_step[idx] = old_val
            no_improve += 1
            
    return best_step
