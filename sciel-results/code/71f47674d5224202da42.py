"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _spectral_shaping():
    """
    Uses frequency domain optimization to shape the autocorrelation sidelobes.
    1. Start with a random spectral mask.
    2. Iteratively update the mask to reduce peak sidelobe energy.
    3. Convert back to time domain and threshold to ±1.
    """
    rng = np.random.default_rng(42)
    
    # Initialize with a random binary sequence (or its spectral representation)
    # We will optimize the spectral magnitude directly.
    # The autocorrelation in time domain corresponds to |S(f)|^2 in frequency domain.
    # We want to minimize the max of |S(f)|^2 for f != 0 (sidelobes).
    
    # Start with a random phase/magnitude
    S = np.zeros(N, dtype=complex)
    # Random initial signal
    x = rng.choice([-1, 1], size=N)
    S = np.fft.fft(x)
    
    # We want to flatten the spectrum (except DC) to minimize peak sidelobes.
    # This is similar to designing a low-peak-sidelobe sequence.
    
    # Iterative optimization in frequency domain
    # Target: constant magnitude for all non-DC bins (ideally 1/sqrt(N-1) if normalized, but relative matters)
    # Let's try to push energy out of high peaks in |S|^2.
    
    S_mag = np.abs(S)
    
    # We keep the DC component fixed (sum of x) or let it vary? 
    # For autocorrelation, DC is the main lobe. We want to minimize other lobes.
    # Actually, the "upper bound on C3" likely refers to the maximum autocorrelation value excluding the zero lag, 
    # or perhaps the integral. The prompt says "autoconvolution upper bound". 
    # Usually, for binary sequences, we minimize the peak sidelobe level (PSL).
    
    # Let's use an iterative projection method.
    # 1. Compute current x
    # 2. Compute X = FFT(x)
    # 3. Modify X to reduce peaks in |X|^2 (for k != 0)
    # 4. x_new = IFFT(X_modified)
    # 5. Threshold x_new to ±1
    # 6. Repeat
    
    x = rng.choice([-1, 1], size=N)
    
    # We need to evaluate the "score". The prompt says "score is the upper bound on C3".
    # And "evaluator measures the raw value (lower is better)".
    # For a signed step function, the autoconvolution at lag 0 is N.
    # The sidelobes are the values at other lags.
    # Let's assume the metric is the maximum absolute autocorrelation value at non-zero lags.
    
    def _max_sidelobe(x):
        # Compute autocorrelation
        # Use FFT for speed
        X = np.fft.fft(x)
        acf = np.fft.ifft(np.abs(X)**2).real
        # acf is circular autocorrelation. 
        # For a step function on [-1/4, 1/4], the domain is discrete.
        # The "upper bound" likely refers to the max |acf[k]| for k != 0.
        # Note: acf[0] is N.
        max_side = np.max(np.abs(acf[1:]))
        return max_side

    current_x = x.copy()
    current_score = _max_sidelobe(current_x)
    
    # Optimization loop
    # Use a gradient-free or simple heuristic update in frequency domain.
    # We want to minimize max |acf[k]| for k!=0.
    # This is equivalent to minimizing max |X[k]|^2 for k!=0? 
    # No, acf is the inverse FFT of |X|^2. So minimizing the peak of acf (sidelobes) is not simply flattening |X|.
    # However, flattening |X| tends to reduce peak sidelobes (like a Barker code or Frank code).
    
    # Let's try an iterative algorithm:
    # 1. Compute X = FFT(x)
    # 2. Identify bins with high |X|^2 (excluding DC)
    # 3. Reduce their magnitude slightly, or set them to a target level.
    # 4. IFFT and threshold.
    
    # A better approach: Use the "Frequency Domain Optimization" for low PSL sequences.
    # We can use a simple gradient descent on the spectral magnitude.
    
    # However, we have a tight time limit. Let's use a robust heuristic:
    # Random restarts + Local Search is often strong for N=400.
    # But the directive asks for a STRUCTURAL change. Spectral shaping is that change.
    
    # Let's implement a simple iterative spectral flattening.
    
    best_x = current_x
    best_score = current_score
    
    deadline = time.time() + 140  # 10s margin
    
    for i in range(5000): # Max iterations
        if time.time() > deadline:
            break
            
        # Compute FFT
        X = np.fft.fft(best_x)
        
        # We want to reduce the peaks in the time-domain autocorrelation sidelobes.
        # This is hard to do directly in frequency domain without calculating the full acf.
        # Let's use a heuristic: minimize the variance of |X|^2 (excluding DC).
        # A flat spectrum corresponds to a "white" sequence, which has low autocorrelation sidelobes on average.
        
        X_mag = np.abs(X)
        # Exclude DC (index 0)
        X_mag_side = X_mag.copy()
        X_mag_side[0] = 0
        
        # Calculate target magnitude
        # Ideally, we want |X[k]| to be constant for all k != 0.
        # Target = mean(|X[k]| for k != 0)
        target_mag = np.mean(X_mag_side)
        
        # Blend current magnitude towards target
        # This is a projection onto the set of flat-spectrum signals.
        alpha = 0.1  # Step size
        
        # Only modify bins that are significantly above target?
        # Or all? Let's try all.
        X_mag_new = X_mag_side * (1 - alpha) + target_mag * alpha
        
        # Reconstruct X with new magnitudes and original phases
        X_phase = X / (np.abs(X) + 1e-12)
        X_new = X_mag_new * X_phase
        X_new[0] = X[0]  # Keep DC fixed
        
        # Inverse FFT
        x_real = np.fft.ifft(X_new).real
        
        # Threshold to ±1
        x_new = np.sign(x_real)
        # Handle zeros
        x_new[x_new == 0] = 1
        
        # Evaluate
        score = _max_sidelobe(x_new)
        
        if score < best_score:
            best_score = score
            best_x = x_new
        else:
            # If no improvement, try a smaller step or random perturbation?
            # For now, keep best_x as is for the next iteration?
            # Or restart?
            # Let's stick to best_x to maintain progress.
            pass
            
    return best_x

def construct():
    # Use spectral shaping
    x = _spectral_shaping()
    return x
# EVOLVE-BLOCK-END
