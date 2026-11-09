"""DFT-based spectral root-finder with Newton refinement."""
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy:
      1. Quick check if x0 is already a root.
      2. Sample f at a small number of equally-spaced points around x0.
      3. Use a DFT to estimate the dominant frequency and phase of the oscillation.
      4. Predict the nearest zero-crossing from the fitted sinusoid.
      5. Refine with a secant/Newton hybrid.
    """
    x0 = float(x0)
    
    # Quick check
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0
    
    # --- Step 1: Initial sampling to detect oscillation characteristics ---
    # We'll sample at N points over a window around x0.
    # Start with a modest window; adjust if needed.
    N = 64
    # Initial window: try to capture at least one full period if possible.
    # For oscillatory functions, a window of ~2*pi is a reasonable start.
    # But we don't know the frequency. Let's try a few window sizes.
    
    best_x = x0
    best_fx = fx0
    
    # Try a few window scales
    for scale in [0.5, 1.0, 2.0, 4.0, 8.0]:
        window = scale * 2.0 * math.pi
        # Sample points
        xs = x0 + window * (np.linspace(-0.5, 0.5, N))
        fs = np.array([f(x) for x in xs])
        
        # Check if any sample is already a root
        min_idx = np.argmin(np.abs(fs))
        if abs(fs[min_idx]) <= 1e-8:
            return xs[min_idx]
        
        # DFT to find dominant frequency
        # Remove mean
        fs_centered = fs - np.mean(fs)
        fft_vals = np.fft.rfft(fs_centered)
        # Power spectrum
        power = np.abs(fft_vals)**2
        # Exclude DC component (index 0)
        if len(power) > 1:
            freq_idx = np.argmax(power[1:]) + 1
            # Frequency in cycles per window
            freq = freq_idx / N
            # Wavelength in x-units
            wavelength = window / freq if freq > 0 else window
            
            # Phase at x0
            # The DFT gives us the complex amplitude at the dominant frequency
            A = fft_vals[freq_idx]
            # The signal is approximately: Re[A * exp(i * 2*pi * freq * t)]
            # where t is the normalized position in [0, 1]
            # At x0, t = 0.5 (center of the window)
            # Actually, xs[0] corresponds to t=0, xs[N-1] to t=1
            # x0 corresponds to t=0.5
            # The phase at x0: phi = angle(A) + 2*pi*freq*0.5
            # Wait, the DFT convention: X[k] = sum x[n] exp(-i 2 pi k n / N)
            # So x[n] ≈ Re[A exp(i 2 pi k n / N)]
            # At x0, n = N/2 = 32
            # phase_at_x0 = angle(A) + 2*pi*freq*(N/2)/N = angle(A) + pi*freq
            
            # Actually, let's be more careful.
            # x[n] = A cos(2*pi*freq*n/N + phi)
            # The DFT coefficient A_k relates to the complex amplitude.
            # For a cosine: x[n] = C cos(2*pi*k0*n/N + phi)
            # The DFT at k0 gives: C/2 * exp(i*phi) (approximately, for large N)
            # So A = fft_vals[freq_idx] ≈ C/2 * exp(i*phi)
            # Thus phi = angle(A)
            
            # The zero crossings of C cos(2*pi*freq*t + phi) = 0
            # occur when 2*pi*freq*t + phi = pi/2 + m*pi
            # t = (pi/2 + m*pi - phi) / (2*pi*freq)
            
            phi = np.angle(A)
            # Find the nearest zero crossing to t=0.5 (which is x0)
            # t_zero = (pi/2 + m*pi - phi) / (2*pi*freq)
            # We want t_zero closest to 0.5
            # m = round((2*pi*freq*0.5 + phi - pi/2) / pi)
            m = int(round((2*pi*freq*0.5 + phi - pi/2) / pi))
            # Try m and m±1
            for mm in [m-1, m, m+1]:
                t_zero = (pi/2 + mm*pi - phi) / (2*pi*freq)
                # Convert to x
                x_zero = x0 + window * (t_zero - 0.5)
                fx_zero = f(x_zero)
                if abs(fx_zero) < abs(best_fx):
                    best_fx = fx_zero
                    best_x = x_zero
                if abs(fx_zero) <= 1e-8:
                    return x_zero
        
        # Also try a simple secant step from x0
        # Use the two closest samples
        # Actually, let's just keep the best so far
    
    # --- Step 2: Refine with secant method ---
    # Start with best_x and a nearby point
    x1 = best_x
    fx1 = best_fx
    # Find a second point
    h = 1e-4 * max(1.0, abs(x1))
    x2 = x1 + h
    fx2 = f(x2)
    
    # Secant iterations
    for _ in range(50):
        if abs(fx1) <= 1e-8:
            return x1
        if abs(fx2) <= 1e-8:
            return x2
        if fx1 == fx2:
            break
        x_new = x1 - fx1 * (x1 - x2) / (fx1 - fx2)
        fx_new = f(x_new)
        if abs(fx_new) < abs(fx1):
            x1, fx1 = x2, fx2
            x2, fx2 = x_new, fx_new
        else:
            x2, fx2 = x_new, fx_new
        if abs(fx2) <= 1e-8:
            return x2
    
    # Return the best found
    return x1 if abs(fx1) <= abs(fx2) else x2
# EVOLVE-BLOCK-END
