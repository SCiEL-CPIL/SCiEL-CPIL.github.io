import numpy as np
import time

N = 100

def _max_overlap(h):
    """Compute max over shifts of sum(h[i]*(1-h[(i+s)%N]))."""
    N = len(h)
    inv_h = 1.0 - h
    # Compute all shifts efficiently using FFT or direct loop
    # Direct loop is O(N^2) = 10000 ops, very fast
    best = 0.0
    for s in range(N):
        # overlap for shift s: sum of h[i] * inv_h[(i+s)%N]
        # np.roll(inv_h, -s) aligns inv_h such that index i corresponds to (i+s)%N
        ov = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
        if ov > best:
            best = ov
    return best

def _score(h):
    s = h.sum()
    if abs(s - N/2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-8 or h.max() > 1.0 + 1e-8:
        return float('inf')
    return _max_overlap(h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # The constant solution h=0.5 is a local minimum with value 25.
    # The global minimum is likely lower.
    # We use a spectral approach: h = 0.5 + sum c_k * exp(i * 2*pi*k*n/N)
    # Since h is real, c_{-k} = conj(c_k).
    # We optimize the real and imaginary parts of c_k for k=1..N/2-1 and c_{N/2} (real).
    # To ensure h in [0,1], we can use a soft constraint or clipping, but clipping breaks gradients.
    # Instead, we can parameterize h via a sigmoid or tanh to enforce bounds, but that might be slow.
    # Given the short time, we will use a simple gradient descent on the coefficients
    # and project back to [0,1] if needed, or just rely on the small magnitude of perturbations.
    
    # Actually, the previous "L-BFGS-B" inspiration mentioned a score of 1.076 -> 1.15.
    # This suggests that moving away from 0.5 can improve the score.
    # The score is seed/value. If seed=25, and we get 24, score = 25/24 = 1.04.
    # If we get 20, score = 25/20 = 1.25.
    
    # Let's try a specific structured perturbation.
    # The overlap is O_s = sum h_i (1-h_{i+s}).
    # O_s = sum h_i - sum h_i h_{i+s} = N/2 - C_s.
    # Minimizing max O_s is equivalent to maximizing min C_s.
    # C_s = sum h_i h_{i+s}.
    # In Fourier domain:
    # h_n = 1/2 + sum_{k=1}^{N/2} a_k cos(2*pi*k*n/N) + b_k sin(2*pi*k*n/N)
    # C_s = sum_n h_n h_{n+s}.
    # This is the autocorrelation.
    # The average of C_s is (1/N) sum_s C_s = (1/N) sum_n h_n^2 * N? No.
    # Parseval: sum_n h_n^2 = N/2 * (|H_0|^2 + 2 sum_{k=1}^{N/2} |H_k|^2).
    # sum_s C_s = sum_n h_n^2.
    # So average C_s = (1/N) sum_n h_n^2.
    # We want to maximize min C_s.
    # If C_s is constant, min C_s = average C_s = (1/N) sum h_n^2.
    # To maximize this, we should maximize sum h_n^2 subject to sum h_n = N/2 and 0<=h_n<=1.
    # The maximum of sum h_n^2 is achieved by making h_n as binary as possible (0 or 1).
    # If h is binary (50 ones, 50 zeros), sum h_n^2 = 50.
    # Average C_s = 50/100 = 0.5? No, C_s is sum, so average C_s = 50.
    # Wait, C_s = sum h_n h_{n+s}.
    # If h is binary, C_s is the number of 1s at n such that n+s is also 1.
    # This is the number of 1-1 pairs at shift s.
    # For a random binary sequence, C_s ~ 25.
    # We want to maximize the MINIMUM C_s.
    # This is related to difference sets or balanced sequences.
    
    # Let's try a gradient descent on the Fourier coefficients to maximize min C_s.
    # Variables: a_k, b_k for k=1..49, a_50.
    # h_n = 0.5 + sum ...
    
    # To be safe and fast, let's use a random restart strategy with local optimization.
    # We will try to optimize a small number of Fourier modes.
    
    best_h = np.full(N, 0.5)
    best_val = _score(best_h)
    
    # Try a few random Fourier-based initializations
    # and perform a simple coordinate descent or gradient step.
    
    def h_from_fourier(coeffs):
        # coeffs: [a1, b1, a2, b2, ..., a49, b49, a50]
        # length 99
        n = np.arange(N)
        h = np.full(N, 0.5)
        for k in range(1, N//2):
            a = coeffs[2*(k-1)]
            b = coeffs[2*(k-1)+1]
            h += a * np.cos(2*np.pi*k*n/N) + b * np.sin(2*np.pi*k*n/N)
        a50 = coeffs[2*(N//2 - 1)]
        h += a50 * np.cos(2*np.pi*(N//2)*n/N)
        return h
    
    def score_from_coeffs(coeffs):
        h = h_from_fourier(coeffs)
        # Project to [0,1] if out of bounds? 
        # For small coeffs, h stays in [0,1].
        # We will keep coeffs small.
        if h.min() < -1e-6 or h.max() > 1.0 + 1e-6:
            return float('inf')
        return _score(h)
    
    # Initialize with small random coeffs
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Start with small random coefficients
        num_coeffs = 2*(N//2 - 1) + 1
        coeffs = np.random.randn(num_coeffs) * 0.1
        
        # Local search: Gradient Descent
        # We want to maximize min C_s, which is minimizing max O_s.
        # Let f(coeffs) = max_s O_s(coeffs).
        # We can use a simple subgradient method or just random hill climbing.
        # Given the complexity, let's try a few steps of random perturbation.
        
        current_coeffs = coeffs.copy()
        current_val = score_from_coeffs(current_coeffs)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h_from_fourier(current_coeffs)
        
        # Simple hill climbing
        for step in range(100):
            if time.time() > deadline:
                break
            
            # Random direction
            direction = np.random.randn(num_coeffs)
            lr = 0.01
            new_coeffs = current_coeffs + lr * direction
            new_val = score_from_coeffs(new_coeffs)
            
            if new_val < current_val:
                current_coeffs = new_coeffs
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h_from_fourier(current_coeffs)
            else:
                # Try smaller step
                lr = 0.001
                new_coeffs = current_coeffs + lr * direction
                new_val = score_from_coeffs(new_coeffs)
                if new_val < current_val:
                    current_coeffs = new_coeffs
                    current_val = new_val
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h_from_fourier(current_coeffs)
        
        # If we haven't improved in a while, break
        if best_val <= 25.0: # Can't beat 25 with this simple method? 
             # Actually, we proved 25 is the lower bound for the AVERAGE, but not the MAX.
             # Wait, my proof was: Average O_s = 25. Max O_s >= Average O_s = 25.
             # So 25 IS the global minimum.
             # If this is true, then no solution can beat 25.
             # Let me re-verify the proof.
             # O_s = sum h_i (1-h_{i+s}).
             # Sum_s O_s = sum_i h_i (N - sum_j h_j) = sum_i h_i (N - N/2) = sum_i h_i (N/2) = (N/2)(N/2) = N^2/4.
             # Average O_s = (N^2/4) / N = N/4 = 25.
             # Max O_s >= Average O_s.
             # So Min(Max O_s) >= 25.
             # Constant h=0.5 gives O_s = 25 for all s.
             # So 25 is the global minimum.
             
             # Why did the inspiration say 1.076 -> 1.15?
             # Maybe the seed value is NOT 25?
             # Or maybe my definition of overlap is wrong?
             # "mean(h[i]*(1-h[(i+s)%N]))"
             # If the score is the MEAN, then for h=0.5, mean is 0.25.
             # Sum_s Mean_O_s = (1/N) Sum_s O_s = 25.
             # Average Mean_O_s = 25/N = 0.25.
             # Max Mean_O_s >= 0.25.
             # Constant h=0.5 gives Mean_O_s = 0.25.
             # So 0.25 is the global minimum.
             
             # The code `_max_overlap` returns the SUM.
             # `_score` returns `_max_overlap`.
             # So the raw value is 25.
             
             # If the seed value is 25, and I return 25, score is 1.0.
             # If I return 24, score is 25/24 = 1.04.
             # But 24 is impossible.
             
             # Is it possible that the evaluator uses a different N?
             # "N = 100" is in the code.
             
             # Is it possible that the "seed" is a different program that scored > 25?
             # "CURRENT PROGRAM (score 1.0000)".
             # This implies Current Value = Seed Value.
             # If Current Value is 25, Seed Value is 25.
             
             # I will stick to the constant solution.
             pass

    return best_h
# EVOLVE-BLOCK-END
