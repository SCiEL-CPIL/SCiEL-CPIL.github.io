import numpy as np
import time
import math
import random
import sys

_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin before 120s limit

N = 24
FFT_SIZE = 1 << 17  # 131072, large enough for accurate sup-norm estimation
SQRTN = math.sqrt(N)

def _time_left():
    return _DEADLINE - time.time()

def _supnorm_fast(coeffs):
    """
    Fast sup-norm estimate via FFT on a dense grid.
    coeffs: list or array of length N
    """
    # Ensure coeffs is a numpy array for speed
    c = np.asarray(coeffs, dtype=float)
    padded = np.zeros(FFT_SIZE, dtype=float)
    padded[:N] = c
    vals = np.fft.fft(padded)
    return float(np.max(np.abs(vals)))

def _supnorm_precise(coeffs):
    """
    More precise sup-norm using refinement of peaks from FFT.
    """
    c = np.asarray(coeffs, dtype=float)
    # First, get a coarse grid
    coarse_size = 1 << 16
    padded = np.zeros(coarse_size, dtype=float)
    padded[:N] = c
    vals = np.fft.fft(padded)
    mags = np.abs(vals)
    
    # Find the top few peaks in the coarse grid
    # We only need to refine the highest ones
    top_k = 10
    indices = np.argpartition(mags, -top_k)[-top_k:]
    
    max_val = 0.0
    for idx in indices:
        # Refine around idx using golden section or just a finer grid
        # The frequency is idx / coarse_size
        # Let's do a small local search on the continuous variable theta
        # P(e^{i theta}) = sum c_k e^{i k theta}
        # We can use scipy.optimize.minimize_scalar or a simple grid
        
        # Simple fine grid around the peak
        freq = idx / coarse_size
        # Search in a small window
        window = 1.0 / coarse_size
        theta0 = 2 * math.pi * freq
        # Sample 100 points around theta0
        thetas = np.linspace(theta0 - window, theta0 + window, 100)
        # Evaluate P(z) at these points
        # P(z) = sum c_k z^k
        # z = exp(i theta)
        # This is expensive if done naively for each theta.
        # We can use the FFT property: the value at theta is the inverse FFT of the spectrum?
        # No, P(e^{i theta}) is just the polynomial evaluated at that point.
        # For N=24, direct evaluation is fast.
        zs = np.exp(1j * thetas)
        # Compute P(z) for all thetas at once using matrix multiplication
        # coeffs shape (N,), zs shape (M, 1) -> need powers
        # powers[k, m] = zs[m]^k
        # This is (N, M) matrix.
        # For M=100, N=24, this is tiny.
        k_indices = np.arange(N)
        # zs[:, None] ** k_indices[None, :]
        powers = np.power(zs[:, None], k_indices[None, :])
        P_vals = powers @ c
        local_max = np.max(np.abs(P_vals))
        if local_max > max_val:
            max_val = local_max
            
    return max_val

def _local_search(coeffs):
    """
    Perform local search with 1-bit and 2-bit flips.
    Returns the best coefficients found.
    """
    coeffs = list(coeffs)
    current_val = _supnorm_fast(coeffs)
    
    # We use a loop that runs until time is low or no improvement is found
    # To avoid getting stuck in 1-bit minima, we always try 2-bit flips.
    
    improved = True
    while improved and _time_left() > 2.0:
        improved = False
        
        # 1. Try all 1-bit flips
        best_1_val = current_val
        best_1_idx = -1
        for k in range(N):
            coeffs[k] = -coeffs[k]
            v = _supnorm_fast(coeffs)
            coeffs[k] = -coeffs[k] # revert
            if v < best_1_val - 1e-12:
                best_1_val = v
                best_1_idx = k
                
        if best_1_idx != -1:
            coeffs[best_1_idx] = -coeffs[best_1_idx]
            current_val = best_1_val
            improved = True
            continue
            
        # 2. Try all 2-bit flips if no 1-bit improvement
        best_2_val = current_val
        best_2_i = -1
        best_2_j = -1
        for i in range(N):
            for j in range(i + 1, N):
                coeffs[i] = -coeffs[i]
                coeffs[j] = -coeffs[j]
                v = _supnorm_fast(coeffs)
                coeffs[i] = -coeffs[i]
                coeffs[j] = -coeffs[j]
                if v < best_2_val - 1e-12:
                    best_2_val = v
                    best_2_i = i
                    best_2_j = j
                    
        if best_2_i != -1:
            coeffs[best_2_i] = -coeffs[best_2_i]
            coeffs[best_2_j] = -coeffs[best_2_j]
            current_val = best_2_val
            improved = True
            continue
            
        # 3. If no improvement, try a random 3-bit flip to escape local minimum
        # This is a heuristic to jump to a different basin
        if _time_left() > 5.0:
            flips = random.sample(range(N), 3)
            for k in flips:
                coeffs[k] = -coeffs[k]
            v = _supnorm_fast(coeffs)
            # Accept if not too bad? Or just keep going?
            # If we accept a worse state, we need to track the best seen so far.
            # For simplicity, we revert if it's worse, but we can also use simulated annealing.
            # Given the small N, let's just revert and try a different random perturbation next time.
            # Actually, if we are stuck, we should probably restart from a new random point
            # or accept a slightly worse state to explore.
            # Let's try accepting if it's within 10% of current best, to explore.
            if v < current_val * 1.1:
                current_val = v
                improved = True # Force another iteration to try and climb from here
            else:
                # Revert
                for k in flips:
                    coeffs[k] = -coeffs[k]
                # If we can't improve even with random jumps, we are likely in a deep local min
                # Break out and let the outer loop try a new start
                break

    return coeffs, current_val

def _golay_rudin_shapiro(n):
    """
    Generate Golay-Rudin-Shapiro sequence of length n.
    c_k = (-1)^{popcount(k)}
    """
    coeffs = []
    for k in range(n):
        if bin(k).count('1') % 2 == 0:
            coeffs.append(1)
        else:
            coeffs.append(-1)
    return coeffs

def construct():
    """
    Return 24 coefficients in {+1, -1} minimizing max_{|z|=1} |P(z)| / sqrt(24).
    """
    random.seed(42)
    np.random.seed(42)
    
    best_coeffs = None
    best_val = float('inf')
    
    # List of initializers
    initializers = []
    
    # 1. Golay-Rudin-Shapiro
    initializers.append(_golay_rudin_shapiro(N))
    
    # 2. Random sequences
    for _ in range(5):
        initializers.append([random.choice([1, -1]) for _ in range(N)])
        
    # 3. "Spectral" initialization: 
    # Start with random, then do a few steps of gradient descent on the phase?
    # Or simply: minimize the variance of the power spectrum.
    # Let's add a few more random starts to be safe.
    for _ in range(5):
        initializers.append([random.choice([1, -1]) for _ in range(N)])

    # Process each initializer
    for init in initializers:
        if _time_left() < 5.0:
            break
            
        coeffs, val = _local_search(init)
        
        # Refine the best candidate with a precise sup-norm check
        # Only do this if it's a new best or close to best
        if val < best_val:
            precise_val = _supnorm_precise(coeffs)
            if precise_val < best_val:
                best_val = precise_val
                best_coeffs = coeffs
        
    # If we have time left, do more random restarts
    while _time_left() > 10.0:
        init = [random.choice([1, -1]) for _ in range(N)]
        coeffs, val = _local_search(init)
        
        if val < best_val:
            precise_val = _supnorm_precise(coeffs)
            if precise_val < best_val:
                best_val = precise_val
                best_coeffs = coeffs
                
    # Final check: ensure we return a valid list of ints
    if best_coeffs is None:
        best_coeffs = _golay_rudin_shapiro(N)
        
    return [int(c) for c in best_coeffs]

if __name__ == "__main__":
    result = construct()
    # Print score for debugging
    val = _supnorm_precise(result)
    print(f"Score: {val / SQRTN}")
    print(result)
