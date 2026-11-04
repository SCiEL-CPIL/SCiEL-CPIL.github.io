"""
Program to minimize the peak autoconvolution of a signed step function.
Uses van der Corput and Thue-Morse initializations with local search.
"""

import numpy as np
import time

N = 400

def _autoconv_peak(f):
    """
    Compute the maximum absolute value of the autoconvolution (f * f)
    for non-zero lags.
    """
    n = len(f)
    if n == 0:
        return 0.0
    
    # Use FFT for efficient convolution
    # We need the full convolution, so length 2n-1
    fft_len = 1
    while fft_len < 2 * n - 1:
        fft_len *= 2
        
    F = np.fft.rfft(f, n=fft_len)
    conv = np.fft.irfft(F * F, n=fft_len)
    
    # The autoconvolution of a sequence of length n has length 2n-1.
    # The lag 0 is at index n-1 in the linear convolution if we consider 
    # standard indexing, but np.convolve or FFT based conv usually places 
    # the zero lag at n-1 for length 2n-1 output? 
    # Let's verify: 
    # np.convolve([1,2],[1,2]) -> [1, 4, 4]
    # Lag 0 (index 0 in correlation sense) is the sum of products f[i]f[i].
    # In convolution c[k] = sum f[i] f[k-i].
    # For autoconvolution, c[k] = sum_{i} f[i] f[k-i].
    # The "zero lag" in autocorrelation R[0] = sum f[i]^2.
    # In the convolution vector of length 2n-1, the peak is at index n-1.
    # We want the max absolute value for lags != 0.
    # So we exclude the center index n-1.
    
    center = n - 1
    # Exclude the center element
    left = conv[:center]
    right = conv[center+1:]
    
    # Note: The convolution is symmetric for real inputs, so max(abs(left)) == max(abs(right))
    # But due to numerical precision, we check both.
    max_val = 0.0
    if len(left) > 0:
        max_val = max(max_val, np.max(np.abs(left)))
    if len(right) > 0:
        max_val = max(max_val, np.max(np.abs(right)))
        
    return float(max_val)

def _van_der_corput_signs(n):
    """
    Generate a signed sequence based on the van der Corput sequence.
    The van der Corput sequence is a low-discrepancy sequence in [0,1].
    We map values < 0.5 to -1 and >= 0.5 to 1.
    """
    vdc = np.zeros(n)
    for i in range(n):
        x = 0.0
        denom = 1.0
        num = i + 1
        while num > 0:
            x += (num % 2) / denom
            num //= 2
            denom *= 2.0
        vdc[i] = x
        
    # Map to signs
    signs = np.where(vdc < 0.5, -1.0, 1.0)
    return signs

def _thue_morse_signs(n):
    """
    Generate a Thue-Morse sequence mapped to signs.
    The Thue-Morse sequence is defined by the parity of the number of 1s in the binary representation of the index.
    """
    signs = np.zeros(n)
    for i in range(n):
        # Count bits set in i
        count = bin(i).count('1')
        signs[i] = 1.0 if count % 2 == 0 else -1.0
    return signs

def _local_search(f, deadline, max_iter_per_call=10000):
    """
    Perform local search (Hill Climbing with random restarts within the function)
    to minimize the peak autoconvolution.
    """
    current_f = f.copy()
    current_score = _autoconv_peak(current_f)
    best_f = current_f.copy()
    best_score = current_score
    
    rng = np.random.RandomState(42)
    
    # Use a simple hill climbing with occasional jumps (SA-like)
    temp = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Check time frequently
        if time.time() > deadline:
            break
            
        # Try a random flip
        pos = rng.randint(0, N)
        old_val = current_f[pos]
        current_f[pos] = -old_val
        
        new_score = _autoconv_peak(current_f)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = current_f.copy()
                # Optionally cool down faster if we found a new best
                temp *= 0.95
        else:
            # Accept worse move with probability
            if temp > 1e-5 and np.random.random() < np.exp(-delta / temp):
                current_score = new_score
            else:
                # Revert
                current_f[pos] = old_val
        
        temp *= cooling
        
        # Prevent infinite loop if time is tight
        # The loop condition handles the deadline, but we ensure we don't hang
        # on a single slow computation.
        
    return best_f, best_score

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    autoconvolution upper bound.
    """
    start_time = time.time()
    # Safety margin: 150s limit, use 110s for search
    deadline = start_time + 110.0
    
    # Generate candidate initializations
    candidates = []
    
    # 1. Van der Corput
    candidates.append(_van_der_corput_signs(N))
    
    # 2. Thue-Morse
    candidates.append(_thue_morse_signs(N))
    
    # 3. Random
    rng = np.random.RandomState(123)
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 4. Another Random with different seed
    rng2 = np.random.RandomState(456)
    candidates.append(rng2.choice([-1.0, 1.0], size=N))
    
    # 5. Legendre (from original, known to be decent)
    # We'll re-implement a quick Legendre generator if time permits, 
    # but for now, let's stick to the new structural changes.
    
    best_f = None
    best_score = float('inf')
    
    # Allocate time for each candidate
    # We have ~110s. 4 candidates. ~27s each.
    # But we should prioritize the better initializations.
    # Let's do 2 full passes if time allows, or just run them sequentially
    # and stop if we hit the deadline.
    
    for i, f_init in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Run local search
        # Adjust deadline for this specific run to ensure we don't overrun
        # in the next iteration.
        # Actually, the local search has its own deadline check.
        # We just pass the global deadline.
        
        f_opt, score = _local_search(f_init, deadline)
        
        if score < best_score:
            best_score = score
            best_f = f_opt
            
        # If we have time left, we might restart the best one found so far
        # with a higher temperature or different random seed?
        # For simplicity and robustness, we'll just ensure we return the best found.
        
    if best_f is None:
        # Fallback
        best_f = candidates[0]
        
    return best_f

# EVOLVE-BLOCK-END
