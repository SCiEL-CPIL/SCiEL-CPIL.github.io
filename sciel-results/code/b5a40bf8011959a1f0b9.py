import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initialize h
    h = np.full(N, 0.5)
    # Add small random noise to break symmetry
    h += np.random.uniform(-0.1, 0.1, N)
    h = np.clip(h, 0, 1)
    
    # Normalize to sum N/2
    # Simple projection: h = h - mean(h) + 0.5
    # But this might violate bounds.
    
    best_h = h.copy()
    best_score = np.inf
    
    # We want to minimize max_overlap
    # max_overlap = max( sum h_i(1-h_i), max_k sum h_i(1-h_{i+k}) )
    
    def get_score(h):
        # Compute overlaps
        # O_0
        o0 = np.sum(h * (1 - h))
        # O_k for k != 0
        # Use FFT for autocorrelation?
        # A_k = sum h_i h_{i+k}
        # O_k = N/2 - A_k
        # We want min A_k to be as large as possible.
        # Equivalent to max O_k to be as small as possible.
        
        # Circular autocorrelation
        # FFT method
        H = np.fft.fft(h)
        A = np.fft.ifft(H * np.conj(H)).real
        A = np.real_if_close(A)
        
        # A[0] is sum h_i^2
        # A[k] is sum h_i h_{i+k}
        
        # Overlaps
        overlaps = N/2 - A
        # overlaps[0] is N/2 - sum h_i^2 = sum h_i(1-h_i) = o0
        # Check consistency
        # o0_calc = np.sum(h * (1-h))
        # assert np.allclose(overlaps[0], o0_calc)
        
        return np.max(overlaps)

    current_score = get_score(h)
    if current_score < best_score:
        best_score = current_score
        best_h = h.copy()
        
    # Simple gradient step / subgradient
    # We want to minimize max(overlaps).
    # This is a convex problem in h?
    # Overlaps are linear in h?
    # O_k = N/2 - sum h_i h_{i+k}. This is quadratic.
    # So it's not convex.
    
    # Let's use a simple local search or gradient descent on the quadratic form.
    # Or just random restarts with local improvement.
    
    # Given time, let's try a few random initializations and a simple iterative improvement.
    
    for _ in range(50):
        if time.time() > deadline:
            break
            
        h = np.full(N, 0.5)
        h += np.random.uniform(-0.5, 0.5, N)
        h = np.clip(h, 0, 1)
        
        # Iterate
        for _ in range(200):
            if time.time() > deadline:
                break
                
            score = get_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
            # Find the shift k with max overlap
            H = np.fft.fft(h)
            A = np.fft.ifft(H * np.conj(H)).real
            overlaps = N/2 - A
            
            k_max = np.argmax(overlaps)
            
            # If k_max == 0, we want to increase sum h_i(1-h_i) is wrong.
            # We want to DECREASE overlaps.
            # If k_max == 0, O_0 = sum h_i(1-h_i) is too high?
            # No, we want to MINIMIZE the max.
            # If O_0 is the max, we want to decrease O_0.
            # O_0 = sum h_i - sum h_i^2. To decrease O_0, we need to INCREASE sum h_i^2.
            # This means making h_i more 0 or 1.
            
            # If k_max != 0, O_k is too high.
            # O_k = N/2 - A_k. To decrease O_k, we need to INCREASE A_k.
            # A_k = sum h_i h_{i+k}.
            
            # Gradient step
            # Let's just do a simple random perturbation and accept if better.
            # This is slow.
            
            # Better: Use the fact that we want to flatten the overlap spectrum.
            # We can try to adjust h to make overlaps more uniform.
            
            # For now, let's just stick to the seed if we can't beat it quickly.
            # But the seed is 1.0. We need to beat it.
            
            # Let's try a specific construction:
            # h_i = 0.5 + A * cos(2*pi*i/N + phi)
            # This has low frequency.
            # Autocorrelation of cos is cos.
            # B_k = A^2/2 * cos(2*pi*k/N).
            # Min B_k = -A^2/2.
            # Max O_k = N/4 - (-A^2/2) = N/4 + A^2/2.
            # This is WORSE than N/4.
            
            # We need B_k > 0.
            # This requires a signal with positive autocorrelation.
            # This is hard.
            
            # What if we use a "spiky" signal?
            # h_i = 1 for i in S, 0 otherwise.
            # Then O_0 = 0.
            # O_k = N/2 - |S \cap (S-k)|.
            # We want to minimize max O_k, i.e., maximize min |S \cap (S-k)|.
            # This is related to difference sets.
            # A (v, k, lambda) difference set has |S \cap (S-k)| = lambda for all k != 0.
            # If we can find a difference set with k=N/2, lambda close to N/4, we win.
            # For N=100, k=50.
            # Does a (100, 50, lambda) difference set exist?
            # Necessary condition: k(k-1) = lambda(v-1)
            # 50*49 = lambda * 99
            # 2450 = 99 lambda
            # lambda = 24.74... Not integer.
            # So no perfect difference set.
            
            # But we can approximate.
            # Let's try to construct a binary sequence that approximates a difference set.
            
            # Or use real values to smooth it out.
            
            # Let's try to optimize a binary sequence.
            # Start with random binary.
            # Flip bits to reduce max overlap.
            
            h_bin = (np.random.rand(N) > 0.5).astype(float)
            # Ensure sum is N/2
            # If sum is not N/2, adjust.
            # For now, let's just use the best real-valued found.
            
            pass

    # If we haven't found anything better, return the seed?
    # The seed is 0.5 everywhere.
    # Let's check if a simple alternating pattern 0.5, 0.5... is different? No, same.
    # What about 0, 1, 0, 1...?
    # h = [0, 1, 0, 1, ...]
    # O_0 = 0.
    # A_k:
    # k=1: sum h_i h_{i+1} = 0. O_1 = N/2.
    # Max overlap = N/2.
    # Seed max overlap = N/4.
    # So alternating is worse.
    
    # What about blocks of 0s and 1s?
    # If we use a random binary sequence, the autocorrelation fluctuates.
    # The max overlap will be larger than N/4.
    
    # It seems the constant sequence 0.5 is a local minimum for the max overlap?
    # Let's check the derivative.
    # If we perturb 0.5 by epsilon, the overlaps change by second order.
    # O_k = N/4 - B_k.
    # B_k is sum delta_i delta_{i+k}.
    # If delta is random, B_k is small.
    # So O_k is close to N/4.
    # But some B_k will be negative, making O_k > N/4.
    # So max O_k > N/4.
    # Thus, the constant sequence is better than random perturbations.
    
    # Is there a way to make B_k positive for all k?
    # If we can, then O_k < N/4 for all k != 0.
    # And O_0 = N/4 - S. If S is small, O_0 is close to N/4.
    # If we can make B_k > 0, we can beat N/4.
    
    # Consider h_i = 0.5 + c * exp(-alpha i^2) (Gaussian bump, wrapped).
    # Autocorrelation of Gaussian is Gaussian.
    # B_k = c^2 * exp(-alpha k^2 / 2) (approx).
    # This is always positive!
    # So O_k = N/4 - c^2 * exp(...) < N/4.
    # And O_0 = N/4 - S.
    # S = sum delta_i^2.
    # If we choose c small enough, S is small, so O_0 is close to N/4.
    # But O_k is slightly less than N/4.
    # So the max is O_0, which is slightly less than N/4.
    # This should beat the seed!
    
    # Let's implement this.
    
    h_gauss = np.full(N, 0.5)
    # Create a Gaussian bump
    x = np.arange(N)
    # Center at 0
    # Wrap around
    dist = np.minimum(x, N - x)
    sigma = 5.0
    delta = np.exp(-0.5 * (dist / sigma) ** 2)
    # Normalize delta to have mean 0?
    # delta is positive, so mean is positive.
    # We need mean 0.
    # Subtract mean.
    delta = delta - np.mean(delta)
    # Scale delta
    # We want to keep h in [0,1].
    # Max delta is around 1. Min is negative.
    # Let's scale it down.
    scale = 0.1
    h_gauss += scale * delta
    
    # Clip to [0,1]
    h_gauss = np.clip(h_gauss, 0, 1)
    
    # Re-normalize to mean 0.5
    # This step might distort the Gaussian shape.
    # But let's try.
    h_gauss = h_gauss - np.mean(h_gauss) + 0.5
    h_gauss = np.clip(h_gauss, 0, 1)
    
    score_gauss = get_score(h_gauss)
    
    if score_gauss < best_score:
        best_score = score_gauss
        best_h = h_gauss.copy()
        
    # Try different sigmas
    for sigma in [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]:
        if time.time() > deadline:
            break
        h_test = np.full(N, 0.5)
        dist = np.minimum(x, N - x)
        delta = np.exp(-0.5 * (dist / sigma) ** 2)
        delta = delta - np.mean(delta)
        # Try different scales
        for scale in [0.05, 0.1, 0.2, 0.3]:
            if time.time() > deadline:
                break
            h_test += scale * delta
            h_test = np.clip(h_test, 0, 1)
            h_test = h_test - np.mean(h_test) + 0.5
            h_test = np.clip(h_test, 0, 1)
            
            s = get_score(h_test)
            if s < best_score:
                best_score = s
                best_h = h_test.copy()
                
    # Try a "comb" or other patterns?
    # The Gaussian bump seems promising because it has positive autocorrelation.
    
    return best_h

# EVOLVE-BLOCK-END
