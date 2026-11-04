import numpy as np
import time

N = 400
N_FFT = 1024  # Next power of 2 >= 2*N - 1

def _calc_max_autoconv(x):
    """
    Calculate the maximum absolute value of the aperiodic autocorrelation.
    x: numpy array of length N with values 1 or -1.
    """
    X = np.fft.fft(x, n=N_FFT)
    P = X * np.conj(X)
    r = np.fft.ifft(P).real
    # Extract lags 1 to N-1
    return np.max(np.abs(r[1:N]))

def _calc_full_autoconv(x):
    """
    Calculate the full aperiodic autocorrelation vector for lags 1 to N-1.
    """
    X = np.fft.fft(x, n=N_FFT)
    P = X * np.conj(X)
    r = np.fft.ifft(P).real
    return r[1:N]

def construct():
    start_time = time.time()
    deadline = 140.0 # Keep 10s margin from 150s limit
    
    best_x = np.random.choice([-1, 1], N)
    best_val = _calc_max_autoconv(best_x)
    
    # We will perform multiple restarts. 
    # Each restart:
    # 1. Initialize sequence (random or structured)
    # 2. Perform Greedy Hill Climbing until local optimum
    # 3. Apply "Error Repair" if stagnation occurs or as a specific step
    
    current_x = best_x.copy()
    current_val = best_val
    
    # Helper to check time
    def time_ok():
        return time.time() - start_time < deadline

    # Structured initialization often helps avoid deep local minima in binary sequences
    # e.g., starting with a known good pattern like Barker or random
    # But random is fine with enough restarts. Let's mix.
    
    restarts = 0
    while time_ok():
        restarts += 1
        # Initialize current sequence
        if restarts % 3 == 0:
            # Try a structured start: e.g., alternating or random blocks
            # Let's stick to random for simplicity but maybe seed it differently
            current_x = np.random.choice([-1, 1], N)
        else:
            # Random restart
            current_x = np.random.choice([-1, 1], N)
        
        current_val = _calc_max_autoconv(current_x)
        
        # Greedy Hill Climbing Loop
        improved = True
        stagnation_count = 0
        max_stagnation = 5000 # Max single-bit flips to try without improvement before considering repair/restart
        
        while improved and time_ok():
            improved = False
            # We will scan indices. To make it faster, we can shuffle indices each pass.
            # Or just iterate 0 to N-1. 
            # N=400 is small. A full pass is 400 evaluations.
            # Each evaluation is an FFT of size 1024.
            # 400 * 1024^2 ~ 4e8 ops? No, FFT is O(N log N).
            # 1024 * log2(1024) = 1024 * 10 = 10k ops per FFT.
            # 400 * 10k = 4e6 ops per pass.
            # 100 passes = 4e8 ops. Should be well within 1 second.
            
            indices = np.random.permutation(N)
            
            for i in indices:
                if not time_ok():
                    break
                
                # Try flipping bit i
                old_val = current_x[i]
                current_x[i] *= -1
                
                new_val = _calc_max_autoconv(current_x)
                
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    # If this is better than global best, update it
                    if current_val < best_val:
                        best_val = current_val
                        best_x = current_x.copy()
                else:
                    # Revert
                    current_x[i] = old_val
                
                # Early exit if we hit a perfect sequence (max autoconv 0 is unlikely for binary, 
                # but min possible is low)
                # Actually, for N=400, the theoretical lower bound is not 0.
                # But if we stop improving, we break the inner loop.
            
            # If no improvement in a full pass, we are at a local optimum
            if not improved:
                # Apply "Error Repair" strategy
                # The idea: Identify the lag k that contributes most to the max autocorrelation.
                # Then, try to flip bits in a way that reduces the correlation at that specific lag.
                # This is a targeted move, different from random hill climbing.
                
                r = _calc_full_autoconv(current_x)
                abs_r = np.abs(r)
                max_lag = np.argmax(abs_r)
                
                # The correlation at lag k is sum x[i] x[i+k].
                # To reduce it, we should flip bits where x[i] and x[i+k] have the same sign 
                # (contributing +1 to sum) if the sum is positive, or opposite if negative.
                # Since we minimize |sum|, we want to change the sign of the sum.
                
                sign_r = np.sign(r[max_lag])
                if sign_r == 0:
                    sign_r = 1 # Arbitrary
                
                # Identify indices i such that flipping x[i] or x[i+k] changes the correlation.
                # Flipping x[i] changes the term x[i]x[i+k] by -2 * x[i]x[i+k].
                # If x[i]x[i+k] has the same sign as r[max_lag], flipping reduces |r|.
                
                # Let's try a few targeted flips
                repaired = False
                # Sample a few candidate indices
                candidates = np.random.choice(N, size=min(100, N), replace=False)
                
                for i in candidates:
                    if not time_ok():
                        break
                    
                    # Check contribution at max_lag
                    j = i + max_lag
                    if j < N:
                        # Current product
                        prod = current_x[i] * current_x[j]
                        # If prod has same sign as r[max_lag], flipping i reduces magnitude
                        if np.sign(prod) == sign_r:
                            old_val = current_x[i]
                            current_x[i] *= -1
                            new_val = _calc_max_autoconv(current_x)
                            if new_val < current_val:
                                current_val = new_val
                                repaired = True
                                if current_val < best_val:
                                    best_val = current_val
                                    best_x = current_x.copy()
                                # Keep the flip
                            else:
                                current_x[i] = old_val
                    else:
                        # Check negative lag? Autocorrelation is symmetric for real signals in magnitude.
                        # r[k] = r[-k].
                        # If i - max_lag >= 0
                        j_neg = i - max_lag
                        if j_neg >= 0:
                            prod = current_x[i] * current_x[j_neg]
                            if np.sign(prod) == sign_r:
                                old_val = current_x[i]
                                current_x[i] *= -1
                                new_val = _calc_max_autoconv(current_x)
                                if new_val < current_val:
                                    current_val = new_val
                                    repaired = True
                                    if current_val < best_val:
                                        best_val = current_val
                                        best_x = current_x.copy()
                                else:
                                    current_x[i] = old_val
                
                if not repaired:
                    # If repair didn't help, we are stuck in a local optimum that repair can't fix.
                    # Break to restart.
                    break

    return best_x

# EVOLVE-BLOCK-END
