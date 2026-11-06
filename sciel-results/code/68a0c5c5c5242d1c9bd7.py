import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 24
    sqrt_N = np.sqrt(N)
    
    # Function to compute the max magnitude of P(z) on the unit circle
    # coeffs: array of 1s and -1s
    def max_magnitude(coeffs):
        # Dense sampling
        thetas = np.linspace(0, 2*np.pi, 2000, endpoint=False)
        # Evaluate P(e^{i*theta})
        # P(z) = sum c_k z^k
        # We can use np.polynomial.polynomial.polyval or just matrix multiply
        # z^k = (e^{i*theta})^k
        # Let's use the fact that c_k is real.
        # P(theta) = sum c_k * exp(i * k * theta)
        
        # Efficient evaluation using FFT-like structure or just vectorized ops
        k = np.arange(N)
        # coeffs shape (N,), thetas shape (M,)
        # We want M values.
        # exp(i * k * theta) is a matrix of size (M, N)
        # This might be memory heavy if M is large, but 2000 * 24 is small.
        
        # Alternative: Use np.polyval? No, that's for standard basis.
        # Let's stick to vectorized complex exponential.
        
        # To speed up, we can precompute exp(i * k * theta) for all k?
        # Or just compute directly.
        
        # Actually, for N=24, a simple loop or vectorized op is fast enough.
        # Let's try vectorized.
        
        # z_k = exp(i * k * theta)
        # This creates a (M, N) array.
        # z = np.exp(1j * np.outer(thetas, k))
        # P_vals = z @ coeffs
        
        # This is 2000 * 24 complex mults. Very fast.
        
        z = np.exp(1j * np.outer(thetas, k))
        P_vals = z @ coeffs
        
        # Find max magnitude
        mags = np.abs(P_vals)
        max_mag = np.max(mags)
        
        # Refine the peak
        # Find the index of the max
        idx = np.argmax(mags)
        
        # Golden section search around thetas[idx]
        # The function is periodic and smooth.
        # Let's do a few steps of refinement.
        
        def f(theta):
            # Evaluate at single theta
            # P(theta) = sum c_k exp(i k theta)
            # This is a bit slow in python loop, but we only do it ~50 times.
            # Or vectorize for a small range?
            # Let's just compute directly.
            val = 0.0
            for ck, kk in zip(coeffs, k):
                val += ck * np.exp(1j * kk * theta)
            return np.abs(val)
        
        # Golden section search parameters
        a = thetas[idx-1] if idx > 0 else thetas[idx] - (2*np.pi/2000)
        b = thetas[idx+1] if idx < len(thetas)-1 else thetas[idx] + (2*np.pi/2000)
        
        # Ensure a < b and within [0, 2pi)
        # Handle wrap-around
        if b < a:
            b += 2*np.pi
        if a < 0:
            a += 2*np.pi
            b += 2*np.pi
            
        # If range is too large or invalid, skip refinement
        if b - a > 0.1: # Too wide
            return max_mag
            
        # Golden ratio
        gr = (np.sqrt(5) - 1) / 2
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = f(x1)
        f2 = f(x2)
        
        for _ in range(50):
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = f(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = f(x2)
                
        refined_max = max(f1, f2)
        # Also check endpoints just in case
        refined_max = max(refined_max, f(a), f(b))
        
        return max(max_mag, refined_max)

    # Initial GRS sequence
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    initial_coeffs = np.array(p[:N], dtype=float)
    
    best_coeffs = initial_coeffs.copy()
    best_score = max_magnitude(best_coeffs) / sqrt_N
    
    # Local Search
    # We will try flipping bits.
    # To avoid getting stuck, we can do multiple passes or random restarts if time permits.
    
    # First, optimize the GRS sequence
    improved = True
    while improved:
        improved = False
        for i in range(N):
            # Flip bit i
            old_val = best_coeffs[i]
            best_coeffs[i] = -old_val
            new_score = max_magnitude(best_coeffs) / sqrt_N
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                # Revert
                best_coeffs[i] = old_val
                
    # Try some random restarts if time allows
    # N=24 is small, so we can afford a few random sequences.
    random.seed(42)
    for _ in range(50): # Try 50 random starts
        if time.time() > deadline:
            break
        
        rand_coeffs = np.array([random.choice([1, -1]) for _ in range(N)], dtype=float)
        
        # Optimize this random sequence
        curr_coeffs = rand_coeffs.copy()
        curr_score = max_magnitude(curr_coeffs) / sqrt_N
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                old_val = curr_coeffs[i]
                curr_coeffs[i] = -old_val
                new_score = max_magnitude(curr_coeffs) / sqrt_N
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                else:
                    curr_coeffs[i] = old_val
        
        if curr_score < best_score:
            best_score = curr_score
            best_coeffs = curr_coeffs.copy()
            
    # Return as list of ints
    return [int(c) for c in best_coeffs]
