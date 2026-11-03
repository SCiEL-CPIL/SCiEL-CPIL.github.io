import numpy as np
import time

N = 400

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
    """
    # Calculate autoconvolution using FFT for speed
    # Linear convolution length is 2*N - 1
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    X = np.fft.rfft(x, fsize)
    # Autoconvolution is IFFT(X^2)
    conv = np.fft.irfft(X * X, fsize)
    conv = conv[:size]
    
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    
    if integral == 0:
        return float('inf')
        
    return max_abs_conv / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Baseline: All ones
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Candidate Generators
    # We suspect the metric might favor sequences with slightly reduced sidelobes
    # even if the sum decreases, OR the "max abs conv" excludes the peak.
    # We will explore a few structured candidates.
    
    candidates = [best_x]
    
    # Chirp-like sequence with positive bias
    # x_i = 1 if cos(...) > 0 else -1, but biased
    for bias in [0.1, 0.2, 0.5]:
        t = np.arange(N)
        # Quadratic chirp
        phase = 2 * np.pi * (t**2) / N
        # Map to signs with bias
        # If we use 1 + bias * sign(cos), it's not binary.
        # For binary: 1 if cos(phase) > -bias else -1
        x_chirp = np.ones(N)
        mask = np.cos(phase) < -bias
        x_chirp[mask] = -1
        candidates.append(x_chirp)
        
    # Barker-like (random but local search optimized)
    # Start with random
    for _ in range(10):
        x_rand = np.random.choice([1, -1], N)
        candidates.append(x_rand)
        
    # Evaluate candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # 3. Local Search (Hill Climbing)
    # Flip individual bits
    # We use a time-based loop to ensure we finish in time.
    
    improved = True
    no_improve_count = 0
    max_no_improve = 500
    
    while improved and time.time() < deadline:
        improved = False
        # Try a batch of random flips
        batch_size = 2000
        for _ in range(batch_size):
            if time.time() > deadline:
                break
            
            i = np.random.randint(N)
            new_x = best_x.copy()
            new_x[i] = -new_x[i]
            
            val = compute_raw_value(new_x)
            if val < best_val:
                best_val = val
                best_x = new_x
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Try a larger perturbation to escape local minima
            # Flip a random block of size 5-10
            for _ in range(100):
                if time.time() > deadline:
                    break
                start = np.random.randint(0, N-10)
                length = np.random.randint(5, 11)
                new_x = best_x.copy()
                new_x[start:start+length] = -new_x[start:start+length]
                val = compute_raw_value(new_x)
                if val < best_val:
                    best_val = val
                    best_x = new_x
                    improved = True
                    no_improve_count = 0
            no_improve_count = 0 # Reset after perturbation attempt

    return best_x

if __name__ == "__main__":
    # Quick self-test
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
