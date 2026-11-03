"""Rewritten program for signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # We want to minimize the maximum absolute value of the autoconvolution.
    # The autoconvolution of a sequence h is (h * h)_k = sum_i h_i h_{i+k}.
    # Note: The task says "autoconvolution" but often in these contexts it implies autocorrelation 
    # or circular convolution. Let's assume linear convolution or circular?
    # "Signed autocorrelation maximum absolute-value variant"
    # "Signed equal-width step functions minimizing the maximum absolute autoconvolution divided by squared integral."
    # Usually, for a step function, the "integral" is the sum of heights.
    # Let's assume the metric is: max_k |sum_i h_i h_{i+k}| / (sum_i h_i)^2
    # Wait, the task says "score is the upper bound on C3' (maximum absolute autoconvolution)".
    # "The evaluator measures the raw value (lower is better)."
    # "Signed autocorrelation maximum absolute-value variant"
    
    # Let's look at the standard C3 problem.
    # If we use time-reversal symmetric vectors h_i = h_{N-1-i}, the Fourier transform is real.
    # The autoconvolution in time domain corresponds to the square of the Fourier transform in frequency domain.
    # Specifically, if H is the DFT of h, then the circular autoconvolution c is the IDFT of H^2.
    # If h is symmetric, H is real, so H^2 is non-negative real.
    # Then c is real and symmetric.
    # The peak of c is related to the peak of H^2? No, c is the time-domain signal.
    
    # Let's implement a simple optimization.
    # Objective: Minimize max_k | (h * h)_k | subject to sum(h)^2 = 1 (or fixed sum).
    # Actually, the ratio is invariant to scaling. So we can fix sum(h) = 1 or sum(h^2)=1.
    # Let's fix sum(h) = 1.
    
    # Strategy:
    # 1. Initialize h as symmetric random or ones.
    # 2. Use gradient descent on a smooth approximation of the max absolute value.
    #    f(h) = log(sum_k exp(|c_k| / tau)) * tau  (Soft-max approximation)
    #    where c = convolve(h, h).
    
    # However, N=400 is small. We can use FFT for convolution.
    # Circular convolution is easier with FFT.
    # Let's assume circular convolution as it's standard in such spectral problems.
    
    h = np.ones(N)
    h = h / np.sum(h)  # Normalize sum to 1
    
    # Enforce symmetry: h[i] = h[N-1-i]
    # We only optimize the first N//2 elements.
    M = N // 2
    h_half = h[:M].copy()
    
    # We need to reconstruct h from h_half
    def reconstruct(h_half):
        h = np.zeros(N)
        h[:M] = h_half
        h[M:] = h_half[::-1]
        return h
    
    # Compute circular autoconvolution
    def autoconv(h):
        H = np.fft.fft(h)
        C = np.fft.ifft(H * H)
        return np.real(C)  # Should be real if h is real
        
    # Objective: max |C_k|
    # Soft-max approximation: tau * log(sum(exp(|C_k|/tau)))
    
    tau = 0.1
    lr = 0.01
    iterations = 5000
    
    best_h = h.copy()
    best_score = np.max(np.abs(autoconv(h)))
    
    for i in range(iterations):
        if time.time() > deadline:
            break
        
        h = reconstruct(h_half)
        C = autoconv(h)
        abs_C = np.abs(C)
        
        # Soft-max gradient
        # f = tau * log(sum(exp(abs_C/tau)))
        # df/dC_k = exp(abs_C_k/tau) / sum(exp(abs_C/tau)) * sign(C_k)
        
        exp_abs = np.exp(abs_C / tau)
        sum_exp = np.sum(exp_abs)
        weights = exp_abs / sum_exp
        grad_C = weights * np.sign(C)
        
        # Gradient of C with respect to h
        # C = IFFT(FFT(h) * FFT(h))
        # dC/dh is a circulant matrix with first column equal to h (reversed for convolution?)
        # Actually, d(IFFT(H^2))/dh = IFFT(2 * H * dH/dh)
        # dH/dh is the DFT matrix.
        # This is complex to compute directly.
        
        # Alternative: Use automatic differentiation logic or finite differences?
        # Finite differences are too slow for N=400 and 5000 iters.
        
        # Let's use the property that the gradient of the circular convolution c = h * h
        # is 2 * h * dh (where * is circular convolution)
        # So grad_h = 2 * convolve(h, grad_C)
        
        grad_h = 2 * np.fft.ifft(np.fft.fft(h) * np.fft.fft(grad_C)).real
        
        # Gradient with respect to h_half
        # h_half[0:M] = h[0:M]
        # h_half[M-1-i] = h[M+i] (for i=0..M-1)
        # Actually, h[N-1-i] = h[i].
        # h_half[i] contributes to h[i] and h[N-1-i].
        # So grad_h_half[i] = grad_h[i] + grad_h[N-1-i]
        
        grad_h_half = grad_h[:M] + grad_h[M:]  # Note: h[M:] is reverse of h[:M]
        # Wait, h[M:] corresponds to indices M..N-1.
        # h[N-1-i] for i in 0..M-1 are indices N-1, N-2, ..., N-M.
        # h[M:] are indices M, M+1, ..., N-1.
        # If N is even, M=N/2.
        # h[N-1] = h[0]
        # h[N-2] = h[1]
        # ...
        # h[N-M] = h[M-1]
        # So h[M:] is [h[M], ..., h[N-1]] which is [h[N-M], ..., h[N-1]] reversed?
        # h[N-1-i] is the element at index N-1-i.
        # The slice h[M:] is h[M], h[M+1], ..., h[N-1].
        # The reverse of h[:M] is h[M-1], ..., h[0].
        # So h[M:] == h[:M][::-1] is NOT true directly by index, but by value if symmetric.
        # h[N-1-i] = h[i].
        # Index N-1-i ranges from N-1 down to N-M.
        # So h[N-1-i] is the (M-1-i)-th element of h[M:]?
        # Let's just construct the gradient properly.
        
        grad_h_half = np.zeros(M)
        for j in range(M):
            idx1 = j
            idx2 = N - 1 - j
            grad_h_half[j] = grad_h[idx1] + grad_h[idx2]
        
        # Update h_half
        h_half -= lr * grad_h_half
        
        # Normalize to keep sum(h) = 1
        # sum(h) = 2 * sum(h_half) (if N even and no center element)
        # If N is odd, there is a center element h[M].
        # Here N=400, so no center.
        # sum(h) = 2 * sum(h_half)
        # We want sum(h) = 1, so sum(h_half) = 0.5
        # Project back to the simplex? Or just normalize?
        # Normalizing changes the scale, but the objective is scale-invariant.
        # However, gradient descent assumes fixed scale.
        # Let's just normalize h_half such that sum(h_half) = 0.5
        
        current_sum = np.sum(h_half)
        if current_sum != 0:
            h_half = h_half * (0.5 / current_sum)
            
        # Compute score
        h = reconstruct(h_half)
        C = autoconv(h)
        score = np.max(np.abs(C))
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    # Return the best h found
    # The task asks for "sequence of signed step heights".
    # The score is the value.
    # The function returns the sequence.
    return best_h
# EVOLVE-BLOCK-END
