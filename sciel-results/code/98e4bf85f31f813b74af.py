"""Program for minimizing the normalized maximum of autoconvolution of a step function."""
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Define the autoconvolution (circular) via FFT for speed
    def autoconv(h):
        # h is a 1D array of length N
        # Circular convolution: (h * h)(k) = sum_i h[i] h[k-i mod N]
        H = np.fft.fft(h)
        C = np.fft.ifft(H * H)
        return C.real

    # Objective: max |C|
    # We want to minimize this.
    # Note: The problem says "normalized maximum". Let's assume normalization is by sum of h (which is 1) or similar.
    # Since h is non-negative and we can scale, usually we normalize sum(h) = 1 or max(h) = 1.
    # The seed returns [1.0]*N. Sum = 600.
    # Let's stick to the convention that the output is the raw heights, and the evaluator handles normalization.
    # However, the "normalized maximum" likely means max(autoconv) / (sum(h)^2) or similar, or just max(autoconv) assuming sum=1.
    # If we return [1.0]*N, autoconv is a triangle with peak 600.
    # If we return a delta function [1, 0, ..., 0], autoconv is a delta with peak 1.
    # If we normalize sum(h)=1, then [1/N]*N gives peak 1/N * N * (1/N) = 1/N? No.
    # Let's analyze the seed score 1.0.
    # If the evaluator computes `max(autoconv(h)) / (sum(h)**2)`, then for constant h:
    # autoconv(k) = sum_i h[i]h[k-i] = N * (1.0*1.0) = N? No, for constant 1, h[i]=1.
    # (h*h)(k) = sum_{i=0}^{N-1} 1*1 = N.
    # sum(h) = N. sum(h)^2 = N^2.
    # Ratio = N / N^2 = 1/N. This would be small.
    # Maybe the normalization is different.
    # Perhaps the "normalized maximum" is just `max(autoconv(h))` and the input is assumed to be normalized to sum=1?
    # But the seed returns `[1.0]*N`. If the evaluator doesn't normalize the input, the max is N=600.
    # If the evaluator normalizes the input to sum=1 before computing, then the seed becomes `[1/600]*600`.
    # Then max(autoconv) = 600 * (1/600 * 1/600) = 1/600.
    # Let's assume the evaluator does: `h_norm = h / sum(h)` (if sum>0), then `score = max(ifft(fft(h_norm)^2).real)`.
    # In this case, minimizing `max(autoconv(h))` is equivalent to minimizing `max(autoconv(h_norm))`.
    # And since `h` is non-negative, we can just work with `h` and normalize at the end or during optimization.
    # Let's try to minimize `max(autoconv(h))` subject to `h >= 0` and `sum(h) = 1`.
    # Actually, since the objective is homogeneous of degree 2 (autoconv scales with square of amplitude),
    # and we are minimizing, the scale matters. But if the evaluator normalizes, scale doesn't matter for the relative shape.
    # Let's assume the evaluator normalizes the input to have sum=1. Then we should minimize the peak of the autoconvolution of the normalized vector.
    # This is equivalent to finding a probability distribution on the circle that minimizes the peak of its self-convolution.
    # The uniform distribution gives a constant autoconvolution (peak = 1/N if sum=1? No.
    # If h_i = 1/N, then (h*h)(k) = sum (1/N)(1/N) = N * (1/N^2) = 1/N.
    # So uniform gives peak 1/600.
    # A delta function gives peak 1.
    # We want to go as low as possible. The theoretical minimum for a non-negative function is 0? No, by Parseval, sum |C_k|^2 = sum |H_k|^4.
    # We want to minimize max C_k.
    # It is known that spreading out reduces the peak. The uniform distribution is a strong candidate.
    # Can we do better than uniform?
    # If we use a Gaussian, the convolution is a wider Gaussian, lower peak.
    # On a discrete circle, a Gaussian-like shape should have a lower peak than uniform?
    # Let's check. Uniform: all values are 1/N. Peak 1/N.
    # Gaussian: centered peak is higher, tails are lower. The peak of the convolution will be at 0.
    # Is the peak of the convolution of a Gaussian lower than the uniform peak?
    # For continuous case: Uniform on [-1,1] (density 1/2). Autoconv is triangle, peak 1/2 (at 0), base 2. Area 1.
    # Gaussian with sigma=1. Autoconv is Gaussian with sigma=sqrt(2). Peak = 1/(2*pi*sqrt(2))*exp(0) = 1/(2*pi*1.414) ~ 0.112.
    # Uniform peak 0.5. Gaussian peak 0.112. So Gaussian is better!
    # So we should aim for a Gaussian-like shape.
    
    # Strategy:
    # 1. Initialize with a Gaussian or Cosine shape.
    # 2. Use Gradient Descent to minimize the max of the autoconvolution.
    #    Since max is non-differentiable, we can use a smooth approximation or just use the top few values.
    #    Alternatively, we can use the fact that we want to flatten the autoconvolution.
    #    We can minimize the sum of squares of the autoconvolution values? No, that minimizes energy, not peak.
    #    We want to minimize the peak.
    #    Let's use gradient descent on `max(C)` by tracking the indices of the max and updating based on their gradients.
    #    Gradient of C[k] w.r.t h[i] is h[k-i].
    #    So d/dh[i] max(C) = sum_{k in ArgMax(C)} h[k-i].
    
    # Let's implement this.
    
    def optimize(h0, steps=1000, lr=0.01):
        h = h0.copy()
        h = h / np.sum(h) # normalize
        
        for step in range(steps):
            C = autoconv(h)
            max_val = np.max(C)
            max_indices = np.where(C == max_val)[0]
            
            # Gradient for the max indices
            grad = np.zeros(N)
            for k in max_indices:
                # dC[k]/dh[i] = h[k-i]
                # grad[i] += h[k-i]
                # h[k-i] is just a shift of h.
                # grad += np.roll(h, k) ? 
                # h[k-i] for i=0..N-1 is h[k], h[k-1], ...
                # This is np.roll(h, k)?
                # np.roll(h, 1) -> [h[N-1], h[0], ...]. Index i gets h[i-1].
                # We want h[k-i].
                # Let j = k-i => i = k-j.
                # grad[i] = h[k-i].
                # This is the vector h reversed and shifted?
                # Let's just compute it directly for simplicity, or use FFT?
                # Direct computation is O(N * len(max_indices)). len(max_indices) is usually small.
                for i in range(N):
                    grad[i] += h[(k - i) % N]
            
            # Update
            h -= lr * grad
            
            # Project to non-negative
            h = np.clip(h, 0, None)
            
            # Normalize
            h_sum = np.sum(h)
            if h_sum > 0:
                h /= h_sum
            else:
                # Fallback to uniform
                h = np.ones(N) / N
                
        return h

    # Initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 / (2 * 0.1**2))
    inits.append(gauss)
    
    # 3. Cosine (raised cosine)
    cos = np.cos(np.pi * x) ** 2 # Smooth bump
    inits.append(cos)
    
    best_h = None
    best_score = np.inf
    
    for h0 in inits:
        h_opt = optimize(h0, steps=500, lr=0.05)
        C = autoconv(h_opt)
        score = np.max(C)
        if score < best_score:
            best_score = score
            best_h = h_opt
            
    # Return the best one
    # The interface expects a sequence of nonnegative step heights.
    # We return best_h.
    return list(best_h)
# EVOLVE-BLOCK-END
