"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_val = np.inf
    best_x = np.ones(N)
    
    # Helper to compute the score: max |autocorr| / (sum x)^2
    # Note: autocorrelation here is likely the circular or linear? 
    # "autoconvolution" usually means linear convolution of x with itself.
    # "maximum absolute autoconvolution" = max |x * x|
    # "squared integral" = (sum x)^2 or (integral x)^2? 
    # In discrete context, "integral" is sum. "Squared integral" is (sum x)^2.
    # Wait, the task says "divided by squared integral". 
    # If x is a step function, integral is sum(x_i * width). Width is 1. So sum(x).
    # Score = max |convolve(x, x)| / (sum(x))^2
    
    def score(x):
        # Ensure x is real
        x = np.real(x)
        s = np.sum(x)
        if s == 0:
            return np.inf
        # Linear autocorrelation (autoconvolution)
        # fftconvolve is faster for large N
        ac = fftconvolve(x, x, mode='full')
        max_ac = np.max(np.abs(ac))
        return max_ac / (s * s)
    
    # Initial candidates
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Alternating 1, -1
    alt = np.ones(N)
    alt[1::2] = -1
    candidates.append(alt)
    
    # 3. Random signs
    for _ in range(5):
        candidates.append(np.random.choice([-1, 1], N))
    
    # 4. Block signs (random blocks)
    for _ in range(5):
        x = np.ones(N)
        block_len = 10
        for i in range(0, N, block_len):
            if np.random.rand() < 0.5:
                x[i:i+block_len] = -1
        candidates.append(x)
    
    # Evaluate initial candidates
    for x in candidates:
        val = score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        if time.time() > deadline:
            break
    
    # Local Search / Gradient Optimization
    # We can use scipy minimize on a smoothed version or direct gradient?
    # Direct gradient of max|ac| is tricky due to max and abs.
    # Let's try a simple local search first: flip individual bits or blocks.
    
    # Strategy: Hill Climbing with random restarts and block flips
    # Since N=400, flipping a block of size B changes O(B*N) in convolution? 
    # Actually, convolution is O(N^2) naive, O(N log N) FFT.
    # FFT is fast. Let's use FFT for evaluation.
    
    # Let's try a more structured optimization:
    # The objective is min max_k |sum_i x_i x_{i+k}| / (sum x)^2
    # This is related to minimizing peak sidelobe levels.
    # Known good sequences for low autocorrelation peak: m-sequences, Golay, etc.
    # But we can choose any signed step heights (presumably +/-1 or continuous?).
    # "signed step heights" usually implies discrete levels, often +/-1.
    # Let's assume we can use continuous values in [-1, 1] to help optimization, then snap?
    # Or just stick to +/-1.
    
    # Let's try a gradient-free local search on the +/-1 space.
    # Move: flip a contiguous block of length L.
    
    current_x = best_x.copy()
    current_val = best_val
    
    # Precompute FFT of x for faster updates? 
    # Updating FFT after a flip is O(N). 
    # But we need the convolution in time domain to find the max.
    # If we keep x in time domain, FFT(x) is O(N log N). 
    # Convolution in freq domain: |FFT(x)|^2 -> IFFT -> ac.
    # So each eval is O(N log N).
    
    def fast_score(x):
        s = np.sum(x)
        if s == 0:
            return np.inf
        X = np.fft.fft(x)
        ac = np.fft.ifft(np.abs(X)**2).real
        # ac is circular autocorrelation? 
        # fftconvolve(x, x, 'full') is linear.
        # Circular autocorrelation is different.
        # The task says "autoconvolution". Usually linear.
        # Let's stick to fftconvolve for correctness, it's fast enough for N=400.
        ac = fftconvolve(x, x, mode='full')
        max_ac = np.max(np.abs(ac))
        return max_ac / (s * s)
    
    # Optimized local search
    max_iters = 2000
    iters = 0
    no_improve_count = 0
    patience = 100
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        
        # Random block flip
        block_len = np.random.choice([1, 2, 3, 5, 10, 20])
        if block_len > N:
            block_len = N
        start = np.random.randint(0, N - block_len + 1)
        
        new_x = current_x.copy()
        new_x[start:start+block_len] *= -1
        
        new_val = fast_score(new_x)
        
        if new_val < current_val:
            current_x = new_x
            current_val = new_val
            no_improve_count = 0
            
            if current_val < best_val:
                best_val = current_val
                best_x = current_x.copy()
        else:
            no_improve_count += 1
            if no_improve_count > patience:
                # Restart with a perturbation of best_x
                current_x = best_x.copy()
                current_val = best_val
                # Perturb
                for _ in range(np.random.randint(1, 10)):
                    bl = np.random.choice([1, 2, 5])
                    st = np.random.randint(0, N - bl + 1)
                    current_x[st:st+bl] *= -1
                current_val = fast_score(current_x)
                no_improve_count = 0
    
    # Try a few random restarts with different initial structures
    # e.g., periodic patterns
    for period in [2, 3, 4, 5, 7]:
        if time.time() > deadline:
            break
        x = np.zeros(N)
        for i in range(N):
            x[i] = 1 if (i % period) < period//2 else -1
        val = fast_score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
        # Local search from this start
        current_x = x.copy()
        current_val = val
        no_improve_count = 0
        local_iters = 0
        local_max = 500
        
        while time.time() < deadline and local_iters < local_max and no_improve_count < 50:
            local_iters += 1
            block_len = np.random.choice([1, 2, 3, 5])
            if block_len > N:
                block_len = N
            start = np.random.randint(0, N - block_len + 1)
            
            new_x = current_x.copy()
            new_x[start:start+block_len] *= -1
            new_val = fast_score(new_x)
            
            if new_val < current_val:
                current_x = new_x
                current_val = new_val
                no_improve_count = 0
                if current_val < best_val:
                    best_val = current_val
                    best_x = current_x.copy()
            else:
                no_improve_count += 1
    
    return best_x

# EVOLVE-BLOCK-END
