import numpy as np
import time

N = 100
DEADLINE = 140.0  # Safety margin for the 150s limit

def compute_max_overlap_periodic(h):
    """
    Compute the maximum overlap for periodic shifts.
    Overlap(k) = sum_i h_i * (1 - h_{(i+k) % N})
    """
    # Using FFT for efficient computation of cross-correlation
    # We want sum_i h_i * (1 - h_{i+k}) for all k.
    # Let b = 1 - h.
    # Overlap(k) = sum_i h_i * b_{i+k}
    # This is the cross-correlation of h and b at lag k.
    # Cross-correlation can be computed via FFT.
    
    # Note: np.correlate or fftfreq usage.
    # For periodic correlation, we can use:
    # corr[k] = sum_{i=0}^{N-1} h[i] * b[(i+k) % N]
    # This is equivalent to the circular cross-correlation.
    
    # Using FFT:
    # H = fft(h)
    # B = fft(b)
    # corr = ifft(H * conj(B))  -> This gives sum h[i] * b[i-k] or similar depending on convention.
    # Let's be careful.
    # Circular correlation r_{hb}[k] = sum_n h[n] * b[n-k] (convention varies).
    # We want sum_i h[i] * b[i+k].
    # Let's just compute it directly for N=100. It's fast enough.
    # N^2 = 10,000 operations. Very fast.
    
    b = 1.0 - h
    max_val = 0.0
    for k in range(N):
        # h * b shifted by k
        # b_shifted[i] = b[(i+k) % N]
        # We can use np.roll
        b_shifted = np.roll(b, -k)
        s = np.sum(h * b_shifted)
        if s > max_val:
            max_val = s
    return max_val

def construct():
    start_time = time.time()
    
    # 1. Theoretical Baseline
    # h = 0.5 everywhere. Sum = 50.
    # Overlap(k) = sum 0.5 * 0.5 = 0.25 * N = 25.
    # Max Overlap = 25.
    # Score = 25.
    
    # 2. Improved Strategy: Low-frequency Cosine Wave
    # h[i] = 0.5 + A * cos(2*pi*i/N)
    # Sum is still 50.
    # We need to choose A to minimize the max overlap.
    # The "inspiration" suggests this lowers the global max below 25.
    
    # Let's optimize A and phase.
    # Actually, due to symmetry, phase might not matter for the max, but let's check.
    # For h[i] = 0.5 + A * cos(2*pi*i/N + phi),
    # The overlap function will have a specific shape.
    
    # Let's scan A in [0, 0.5] to ensure h stays in [0,1].
    # If A > 0.5, h goes out of bounds. So A in [0, 0.5].
    
    best_val = 25.0
    best_h = np.full(N, 0.5)
    
    # Scan amplitudes
    # We can be fine-grained.
    num_steps = 100
    for i in range(num_steps + 1):
        if time.time() > DEADLINE:
            break
            
        A = 0.5 * i / num_steps
        # Phase 0 is a good guess.
        # h = 0.5 + A * cos(2*pi*arange(N)/N)
        h_candidate = 0.5 + A * np.cos(2 * np.pi * np.arange(N) / N)
        
        # Check bounds (should be fine for A<=0.5)
        if np.min(h_candidate) < -1e-9 or np.max(h_candidate) > 1 + 1e-9:
            continue
            
        val = compute_max_overlap_periodic(h_candidate)
        
        if val < best_val:
            best_val = val
            best_h = h_candidate.copy()

    # 3. Local Refinement
    # If we found a better A, we can do a local search around it.
    # But the cosine family might not be the only one.
    # The directive says "KEEP this design... investigate improvements within its family".
    # The "family" of the current program is the constant sequence.
    # The "inspiration" adds the cosine wave.
    # So the new family is "Cosine waves with varying amplitude and phase".
    
    # Let's refine the best cosine wave found.
    if best_val < 25.0:
        # Determine A and phase from best_h?
        # Or just restart the scan with finer resolution around the best A?
        # Let's just do a finer scan over A and Phase.
        
        # Extract rough A from best_h
        # h - 0.5 = A cos(...)
        # Max deviation from 0.5 is A.
        A_est = np.max(np.abs(best_h - 0.5))
        
        # Fine scan around A_est
        A_range = np.linspace(max(0, A_est - 0.05), min(0.5, A_est + 0.05), 50)
        phase_range = np.linspace(0, 2*np.pi, 10)
        
        for A in A_range:
            if time.time() > DEADLINE:
                break
            for phi in phase_range:
                if time.time() > DEADLINE:
                    break
                h_candidate = 0.5 + A * np.cos(2 * np.pi * np.arange(N) / N + phi)
                if np.min(h_candidate) < -1e-9 or np.max(h_candidate) > 1 + 1e-9:
                    continue
                val = compute_max_overlap_periodic(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
                    
    return best_h
